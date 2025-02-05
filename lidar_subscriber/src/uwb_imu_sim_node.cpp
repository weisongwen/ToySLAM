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

        pub_latest_odometry = nh_.advertise<geometry_msgs::PoseStamped>("UWBPoistion", 1000);

        // Beacon positions (x, y, z) in meters
        beacon_positions_ = {
            {5.0, 5.0, 0.0},
            {-5.0, 5.0, 0.0},
            {-5.0, -5.0, 0.0},
            {5.0, -5.0, 0.0},
            {0.0, 0.0, 3.0}
        };

        // Initialize random generators
        std::random_device rd;
        gen_.reset(new std::mt19937(rd()));
        accel_noise_ = std::normal_distribution<double>(0.0, 0.01);
        gyro_noise_ = std::normal_distribution<double>(0.0, 0.005);
        uwb_noise_ = std::normal_distribution<double>(0.0, 0.1);

        // Setup timers
        imu_timer_ = nh_.createTimer(ros::Duration(1.0/500.0), &SensorSimulator::publishImu, this);
        uwb_timer_ = nh_.createTimer(ros::Duration(1.0/10.0), &SensorSimulator::publishUwb, this);
        vis_timer_ = nh_.createTimer(ros::Duration(0.1), &SensorSimulator::publishVisualization, this);
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
            // std::cout<<"residual[0] -> " << residual[0]<<"\n";
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

        // Simulate accelerometer data (m/s²)
        const double t = ros::Time::now().toSec();
        imu_msg.linear_acceleration.x = 0.5 * std::sin(t) + accel_noise_(*gen_);
        imu_msg.linear_acceleration.y = 0.2 * std::cos(0.5 * t) + accel_noise_(*gen_);
        imu_msg.linear_acceleration.z = 9.8 + accel_noise_(*gen_);

        // Simulate gyroscope data (rad/s)
        imu_msg.angular_velocity.x = 0.1 * std::sin(0.3 * t) + gyro_noise_(*gen_);
        imu_msg.angular_velocity.y = 0.05 * std::cos(0.2 * t) + gyro_noise_(*gen_);
        imu_msg.angular_velocity.z = 0.2 * std::sin(0.1 * t) + gyro_noise_(*gen_);

        // Orientation (for visualization)
        tf2::Quaternion q;
        q.setRPY(0, 0, t);
        imu_msg.orientation.x = q.x();
        imu_msg.orientation.y = q.y();
        imu_msg.orientation.z = q.z();
        imu_msg.orientation.w = q.w();

        imu_pub_.publish(imu_msg);

        // Update receiver path
        updateReceiverPath(t);
    }

    void updateReceiverPath(double t) {
        // Simulate receiver trajectory (circular motion)
        current_position_.x = 3.0 * std::cos(t * 0.5);
        current_position_.y = 3.0 * std::sin(t * 0.5);
        current_position_.z = 1.0;

        // Add to path
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
        pose.pose.position = current_position_;
        path_.poses.push_back(pose);

        //user position based on Cere solver least square 
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
        if(path_.poses.size() > 1200) {
            path_.poses.erase(path_.poses.begin());
        }
        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "map";

        // Keep only last 100 poses ()
        // if(path_user_position.poses.size() > 1200) {
        //     path_user_position.poses.erase(path_user_position.poses.begin());
        // }
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

        // when the imu data is availble, do the least square 
        ceres::Problem problem;
        Eigen::Vector3d position(1,0,0); // Perturbed initial guess
        // double positioN[3]={1,0,0};
        for (size_t i = 0; i < beacon_positions_.size(); ++i) {
            Eigen::Vector3d anchor(beacon_positions_[i][0], beacon_positions_[i][1], beacon_positions_[i][2]);
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<RangeResidual, 1, 3>(
                    new RangeResidual(anchor, measurements[i]));
            
            ceres::LossFunction* loss_function = nullptr;
            if (use_huber_loss_) {
                loss_function = new ceres::HuberLoss(huber_loss_threshold_);
            }

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.1), position.data());
            problem.AddResidualBlock(cost_function, NULL, position.data());
            // problem.AddResidualBlock(cost_function, loss_function, positioN);
        }
        ceres::Solver::Options options;
        // options.max_num_iterations = 20; // max_iterations_
        // options.linear_solver_type = ceres::DENSE_QR;
        // options.minimizer_progress_to_stdout = false;
        // options.function_tolerance = solver_tolerance_;

        options.use_nonmonotonic_steps = true;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        options.num_threads = 8;
        options.max_num_iterations = 258;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        ceres::Solve(options, &problem, &summary);

        ROS_DEBUG_COND(summary.termination_type != ceres::CONVERGENCE,
                      "Optimization failed to converge: %s", 
                      summary.FullReport().c_str());
        user_pos = position;
        // std::cout<<"user position-> " << position <<std::endl;
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
    ros::Timer imu_timer_;
    ros::Timer uwb_timer_;
    ros::Timer vis_timer_;

    std::unique_ptr<std::mt19937> gen_;
    std::normal_distribution<double> accel_noise_;
    std::normal_distribution<double> gyro_noise_;
    std::normal_distribution<double> uwb_noise_;
    
    std::vector<std::vector<double>> beacon_positions_;
    // std::vector<Eigen::Vector3d> beacon_positions_;
    //Eigen::Vector3d

    geometry_msgs::Point current_position_;
    nav_msgs::Path path_;
    nav_msgs::Path path_user_position;

    // Optimization parameters
    int max_iterations_;
    double solver_tolerance_;
    std::string linear_solver_type_;
    bool use_huber_loss_;
    double huber_loss_threshold_;
    Eigen::Vector3d user_pos;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator");
    SensorSimulator sensor_simulator;
    ros::spin();
    return 0;
}