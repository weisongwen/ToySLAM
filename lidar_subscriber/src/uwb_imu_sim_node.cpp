#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
#include <random>
#include <deque>
#include <nav_msgs/Odometry.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <geometry_msgs/PointStamped.h>

#include <vector>

class SensorSimulator {
public:
    SensorSimulator() : nh_("~") {
        // Initialize publishers
        imu_pub_ = nh_.advertise<sensor_msgs::Imu>("imu_data", 10);
        uwb_pubs_.resize(5);
        for(int i=0; i<5; i++) {
            uwb_pubs_[i] = nh_.advertise<sensor_msgs::Range>("/uwb_beacon_" + std::to_string(i), 10);
        }
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_markers", 1);
        path_pub_ = nh_.advertise<nav_msgs::Path>("receiver_GT_path", 10);
        user_path_pub_ = nh_.advertise<nav_msgs::Path>("user_path", 10);

        pub_latest_odometry = nh_.advertise<nav_msgs::Odometry>("UWBPoistion", 1000);
        pub_latest_odometry_ps = nh_.advertise<geometry_msgs::PointStamped>("UWBPoistionPS", 1000);

        // Beacon positions (x, y, z) in meters
        beacon_positions_ = {
            {5.0, 5.0, 0.0},
            {-5.0, 5.0, 0.0},
            {-5.0, -5.0, 0.0},
            {5.0, -5.0, 0.0},
            {0.0, 0.0, 3.0}
        };

        // Load parameters with more realistic defaults
        nh_.param<double>("sim_freq", sim_freq_, 200.0);
        nh_.param<double>("omega", omega_, 0.1);  // Angular velocity of circular motion
        nh_.param<double>("radius", radius_, 3.0); // Radius of circle in meters

        // Noise parameters - more realistic values for consumer-grade IMU
        nh_.param<double>("accel_noise_std", accel_noise_std_, 0.03);  // m/s²
        nh_.param<double>("gyro_noise_std", gyro_noise_std_, 0.002);   // rad/s
        nh_.param<double>("uwb_noise_std", uwb_noise_std_, 0.05);      // meters

        // Bias parameters - realistic biases for consumer-grade IMU
        nh_.param<double>("accel_bias_x", accel_bias_x_, 0.05);  // m/s²
        nh_.param<double>("accel_bias_y", accel_bias_y_, -0.07); // m/s²
        nh_.param<double>("accel_bias_z", accel_bias_z_, 0.1);   // m/s²
        nh_.param<double>("gyro_bias_x", gyro_bias_x_, 0.002);   // rad/s
        nh_.param<double>("gyro_bias_y", gyro_bias_y_, -0.003);  // rad/s
        nh_.param<double>("gyro_bias_z", gyro_bias_z_, 0.001);   // rad/s

        sim_time_ = 0.0;
        dt_ = 1.0 / sim_freq_;

        // Initialize random generators with realistic noise levels
        std::random_device rd;
        gen_.reset(new std::mt19937(rd()));
        accel_noise_ = std::normal_distribution<double>(0.0, accel_noise_std_);
        gyro_noise_ = std::normal_distribution<double>(0.0, gyro_noise_std_);
        uwb_noise_ = std::normal_distribution<double>(0.0, uwb_noise_std_);

        // Init biases as 3D vectors
        accel_bias_ = Eigen::Vector3d(accel_bias_x_, accel_bias_y_, accel_bias_z_);
        gyro_bias_ = Eigen::Vector3d(gyro_bias_x_, gyro_bias_y_, gyro_bias_z_);

        // Setup timers
        imu_timer_ = nh_.createTimer(ros::Duration(1.0/sim_freq_), &SensorSimulator::publishImu, this);
        uwb_timer_ = nh_.createTimer(ros::Duration(1.0/(sim_freq_*0.1)), &SensorSimulator::publishUwb, this);
        vis_timer_ = nh_.createTimer(ros::Duration(0.1), &SensorSimulator::publishVisualization, this);
        
        ROS_INFO("Sensor Simulator initialized with realistic IMU parameters:");
        ROS_INFO("Accel noise std: %.4f m/s², Gyro noise std: %.4f rad/s", accel_noise_std_, gyro_noise_std_);
        ROS_INFO("Accel bias: [%.4f, %.4f, %.4f] m/s²", accel_bias_x_, accel_bias_y_, accel_bias_z_);
        ROS_INFO("Gyro bias: [%.4f, %.4f, %.4f] rad/s", gyro_bias_x_, gyro_bias_y_, gyro_bias_z_);
        ROS_INFO("UWB noise std: %.4f m", uwb_noise_std_);
    }

public:
    struct RangeResidual {
        RangeResidual(const Eigen::Vector3d& anchor, double measurement)
            : anchor_(anchor), measurement_(measurement) {}

        template <typename T>
        bool operator()(const T* const position, T* residual) const {
            T dx = position[0] - T(anchor_.x());
            T dy = position[1] - T(anchor_.y());
            T dz = position[2] - T(anchor_.z());
            T distance = ceres::sqrt(dx*dx + dy*dy + dz*dz);
            residual[0] = distance - T(measurement_);
            return true;
        }

        Eigen::Vector3d anchor_;
        double measurement_;
    };

private:
    void publishImu(const ros::TimerEvent&) {
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.header.frame_id = "map";

        // Update simulation time
        sim_time_ += dt_;
        double t = sim_time_;

        // Calculate circular motion parameters
        double theta = omega_ * t;

        // Calculate velocity
        Eigen::Vector3d velocity(
            -radius_ * omega_ * sin(theta),
            radius_ * omega_ * cos(theta),
            0.0
        );

        // Calculate centripetal acceleration
        Eigen::Vector3d acceleration(
            -radius_ * omega_ * omega_ * cos(theta),
            -radius_ * omega_ * omega_ * sin(theta),
            0.0
        );

        // Calculate orientation (yaw)
        double yaw = theta + M_PI/2; // Tangent to circle
        Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

        // IMU simulation with proper bias and noise
        // -------------------------------
        // Calculate proper acceleration (world frame)
        Eigen::Vector3d gravity(0.0, 0.0, 9.81);
        Eigen::Vector3d acc_world = acceleration + gravity;

        // Rotate to body frame
        Eigen::Vector3d acc_body = q.conjugate() * acc_world;

        // Add bias and noise
        acc_body += accel_bias_;
        acc_body.x() += accel_noise_(*gen_);
        acc_body.y() += accel_noise_(*gen_);
        acc_body.z() += accel_noise_(*gen_);

        // Angular velocity (body frame) with bias and noise
        Eigen::Vector3d gyro(0.0, 0.0, omega_);
        gyro += gyro_bias_;
        gyro.x() += gyro_noise_(*gen_);
        gyro.y() += gyro_noise_(*gen_);
        gyro.z() += gyro_noise_(*gen_);

        // Set orientation (ground truth for reference)
        imu_msg.orientation.w = q.w();
        imu_msg.orientation.x = q.x();
        imu_msg.orientation.y = q.y();
        imu_msg.orientation.z = q.z();

        // Set angular velocity
        imu_msg.angular_velocity.x = gyro.x();
        imu_msg.angular_velocity.y = gyro.y();
        imu_msg.angular_velocity.z = gyro.z();

        // Set linear acceleration
        imu_msg.linear_acceleration.x = acc_body.x();
        imu_msg.linear_acceleration.y = acc_body.y();
        imu_msg.linear_acceleration.z = acc_body.z();

        // Set covariances (diagonal) based on noise parameters
        for (int i = 0; i < 9; i++) {
            imu_msg.orientation_covariance[i] = 0.0;
            imu_msg.angular_velocity_covariance[i] = 0.0;
            imu_msg.linear_acceleration_covariance[i] = 0.0;
        }
        
        // Diagonal elements represent variance (std^2)
        imu_msg.orientation_covariance[0] = 0.01; // Set to non-default value
        imu_msg.orientation_covariance[4] = 0.01;
        imu_msg.orientation_covariance[8] = 0.01;
        
        imu_msg.angular_velocity_covariance[0] = gyro_noise_std_ * gyro_noise_std_;
        imu_msg.angular_velocity_covariance[4] = gyro_noise_std_ * gyro_noise_std_;
        imu_msg.angular_velocity_covariance[8] = gyro_noise_std_ * gyro_noise_std_;
        
        imu_msg.linear_acceleration_covariance[0] = accel_noise_std_ * accel_noise_std_;
        imu_msg.linear_acceleration_covariance[4] = accel_noise_std_ * accel_noise_std_;
        imu_msg.linear_acceleration_covariance[8] = accel_noise_std_ * accel_noise_std_;

        imu_pub_.publish(imu_msg);

        // Update receiver path
        updateReceiverPath(t);
    }

    void updateReceiverPath(double t) {
        // Simulate receiver trajectory (circular motion)
        current_position_.x = radius_ * std::cos(t * omega_);
        current_position_.y = radius_ * std::sin(t * omega_);
        current_position_.z = 1.0;

        // Add to path
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
        pose.pose.position = current_position_;
        path_.poses.push_back(pose);

        // User position based on Ceres solver least square 
        pose.pose.position.x = user_pos(0);
        pose.pose.position.y = user_pos(1);
        pose.pose.position.z = user_pos(2);
        path_user_position.poses.push_back(pose);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time::now();
        odometry.header.frame_id = "map";
        odometry.pose.pose.position.x = user_pos(0);
        odometry.pose.pose.position.y = user_pos(1);
        odometry.pose.pose.position.z = user_pos(2);
        pub_latest_odometry.publish(odometry);

        // Keep only last 100 poses
        if(path_.poses.size() > 7200) {
            path_.poses.erase(path_.poses.begin());
        }
        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "map";

        path_user_position.header.stamp = ros::Time::now();
        path_user_position.header.frame_id = "map";
    }

    void publishUwb(const ros::TimerEvent&) {
        std::vector<double> measurements;
        for(size_t i=0; i<beacon_positions_.size(); i++) {
            sensor_msgs::Range uwb_msg;
            uwb_msg.header.stamp = ros::Time::now();
            uwb_msg.header.frame_id = "map";
            uwb_msg.radiation_type = sensor_msgs::Range::ULTRASOUND;
            uwb_msg.field_of_view = 0.1;
            uwb_msg.min_range = 0.1;
            uwb_msg.max_range = 30.0;

            // Calculate distance to beacon with noise
            double dx = beacon_positions_[i][0] - current_position_.x;
            double dy = beacon_positions_[i][1] - current_position_.y;
            double dz = beacon_positions_[i][2] - current_position_.z;
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            uwb_msg.range = distance + uwb_noise_(*gen_);

            uwb_pubs_[i].publish(uwb_msg);
            measurements.push_back(uwb_msg.range);
        }

        // Perform least square to estimate position from UWB ranges
        ceres::Problem problem;
        Eigen::Vector3d position(1,0,0); // Initial guess

        for (size_t i = 0; i < beacon_positions_.size(); ++i) {
            Eigen::Vector3d anchor(beacon_positions_[i][0], beacon_positions_[i][1], beacon_positions_[i][2]);
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<RangeResidual, 1, 3>(
                    new RangeResidual(anchor, measurements[i]));
            
            problem.AddResidualBlock(cost_function, nullptr, position.data());
        }

        ceres::Solver::Options options;
        options.use_nonmonotonic_steps = true;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        options.num_threads = 8;
        options.max_num_iterations = 20;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        user_pos = position;

        // Publish UWB position as PointStamped
        geometry_msgs::PointStamped est_msg;
        est_msg.header.stamp = ros::Time::now();
        est_msg.header.frame_id = "map";
        
        est_msg.point.x = user_pos(0);
        est_msg.point.y = user_pos(1);
        est_msg.point.z = user_pos(2);
        pub_latest_odometry_ps.publish(est_msg);
    }

    void publishVisualization(const ros::TimerEvent&) {
        // Publish beacon markers
        visualization_msgs::MarkerArray markers;
        
        // Create beacon spheres
        for(size_t i=0; i<beacon_positions_.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = ros::Time::now();
            marker.ns = "uwb_beacons";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = beacon_positions_[i][0];
            marker.pose.position.y = beacon_positions_[i][1];
            marker.pose.position.z = beacon_positions_[i][2];
            marker.pose.orientation.w = 1.0;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.5;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            marker.lifetime = ros::Duration();
            markers.markers.push_back(marker);
        }

        // Create line markers between receiver and beacons
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = "map";
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "uwb_links";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::Marker::LINE_LIST;
        line_marker.action = visualization_msgs::Marker::ADD;
        line_marker.scale.x = 0.05; // Line width
        line_marker.color.r = 0.0;
        line_marker.color.g = 1.0;
        line_marker.color.b = 0.0;
        line_marker.color.a = 0.5; // Semi-transparent

        // Add points for each beacon-receiver pair
        for(const auto& beacon : beacon_positions_) {
            geometry_msgs::Point receiver_point;
            receiver_point.x = current_position_.x;
            receiver_point.y = current_position_.y;
            receiver_point.z = current_position_.z;
            
            geometry_msgs::Point beacon_point;
            beacon_point.x = beacon[0];
            beacon_point.y = beacon[1];
            beacon_point.z = beacon[2];
            
            line_marker.points.push_back(receiver_point);
            line_marker.points.push_back(beacon_point);
        }

        markers.markers.push_back(line_marker);
        
        // Publish all markers
        marker_pub_.publish(markers);
        
        // Publish receiver trajectory
        path_pub_.publish(path_);
        user_path_pub_.publish(path_user_position);
    }

    ros::NodeHandle nh_;
    ros::Publisher imu_pub_;
    std::vector<ros::Publisher> uwb_pubs_;
    ros::Publisher marker_pub_;
    ros::Publisher path_pub_;
    ros::Publisher user_path_pub_;
    ros::Publisher pub_latest_odometry;
    ros::Publisher pub_latest_odometry_ps;
    ros::Timer imu_timer_;
    ros::Timer uwb_timer_;
    ros::Timer vis_timer_;

    std::unique_ptr<std::mt19937> gen_;
    std::normal_distribution<double> accel_noise_;
    std::normal_distribution<double> gyro_noise_;
    std::normal_distribution<double> uwb_noise_;
    
    std::vector<std::vector<double>> beacon_positions_;
    Eigen::Vector3d accel_bias_;  // Accelerometer bias vector (m/s²)
    Eigen::Vector3d gyro_bias_;   // Gyroscope bias vector (rad/s)

    // Simulation parameters
    double sim_freq_;  // Simulation frequency (Hz)
    double radius_;    // Radius of circle (m)
    double omega_;     // Angular velocity (rad/s)
    double sim_time_;  // Current simulation time (s)
    double dt_;        // Time step (s)
    
    // Noise parameters
    double accel_noise_std_;  // Accelerometer noise standard deviation (m/s²)
    double gyro_noise_std_;   // Gyroscope noise standard deviation (rad/s)
    double uwb_noise_std_;    // UWB distance noise standard deviation (m)
    
    // Bias parameters
    double accel_bias_x_, accel_bias_y_, accel_bias_z_;  // Accelerometer bias (m/s²)
    double gyro_bias_x_, gyro_bias_y_, gyro_bias_z_;     // Gyroscope bias (rad/s)

    geometry_msgs::Point current_position_;
    nav_msgs::Path path_;
    nav_msgs::Path path_user_position;
    Eigen::Vector3d user_pos;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator");
    SensorSimulator sensor_simulator;
    ros::spin();
    return 0;
}