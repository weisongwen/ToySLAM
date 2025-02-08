#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>

class DataSimulator {
public:
    DataSimulator() : nh_("~"), gen_(rd_()) {
        // Publishers
        imu_pub_ = nh_.advertise<sensor_msgs::Imu>("/imu/data", 10);
        uwb_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/uwb/position", 10);

        // Parameters
        nh_.param("radius", radius_, 1.0);
        nh_.param("angular_velocity", omega_, 1.0);
        nh_.param("sim_frequency", sim_freq_, 100.0);
        nh_.param("uwb_noise_std", uwb_noise_std_, 0.01);
        nh_.param("accel_noise_std", accel_noise_std_, 0.01);
        nh_.param("gyro_noise_std", gyro_noise_std_, 0.001);

        // Initialize simulation time
        sim_time_ = 0.0;
        dt_ = 1.0 / sim_freq_;

        // Initialize random number generators
        accel_noise_ = std::normal_distribution<double>(0.0, accel_noise_std_);
        gyro_noise_ = std::normal_distribution<double>(0.0, gyro_noise_std_);
        uwb_noise_ = std::normal_distribution<double>(0.0, uwb_noise_std_);

        // Start simulation timer
        timer_ = nh_.createTimer(ros::Duration(dt_), &DataSimulator::simulateData, this);
    }

private:
    void simulateData(const ros::TimerEvent& event) {
        // Calculate current angle
        double theta = omega_ * sim_time_;
        
        // Calculate trajectory
        Eigen::Vector3d position(
            radius_ * cos(theta),
            radius_ * sin(theta),
            0.0
        );
        
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

        // IMU simulation
        // -------------------------------
        // Calculate proper acceleration (world frame)
        Eigen::Vector3d gravity(0.0, 0.0, 9.81);
        Eigen::Vector3d acc_world = acceleration + gravity;

        // Rotate to body frame
        Eigen::Vector3d acc_body = q.conjugate() * acc_world;

        // Add noise
        acc_body.x() += accel_noise_(gen_);
        acc_body.y() += accel_noise_(gen_);
        acc_body.z() += accel_noise_(gen_);

        // Angular velocity (body frame)
        Eigen::Vector3d gyro(0.0, 0.0, omega_);
        gyro.x() += gyro_noise_(gen_);
        gyro.y() += gyro_noise_(gen_);
        gyro.z() += gyro_noise_(gen_);

        // Create IMU message
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.header.frame_id = "imu";

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

        // UWB simulation
        // -------------------------------
        geometry_msgs::PointStamped uwb_msg;
        uwb_msg.header.stamp = imu_msg.header.stamp;
        uwb_msg.header.frame_id = "world";
        
        // Add noise to position
        uwb_msg.point.x = position.x() + uwb_noise_(gen_);
        uwb_msg.point.y = position.y() + uwb_noise_(gen_);
        uwb_msg.point.z = position.z() + uwb_noise_(gen_);

        // Publish messages
        imu_pub_.publish(imu_msg);
        uwb_pub_.publish(uwb_msg);

        // Increment simulation time
        sim_time_ += dt_;
    }

    ros::NodeHandle nh_;
    ros::Publisher imu_pub_, uwb_pub_;
    ros::Timer timer_;

    // Simulation parameters
    double radius_;
    double omega_;
    double sim_freq_;
    double sim_time_;
    double dt_;

    // Noise parameters
    double uwb_noise_std_;
    double accel_noise_std_;
    double gyro_noise_std_;

    // Random number generation
    std::random_device rd_;
    std::mt19937 gen_;
    std::normal_distribution<double> accel_noise_;
    std::normal_distribution<double> gyro_noise_;
    std::normal_distribution<double> uwb_noise_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_simulator");
    DataSimulator simulator;
    ros::spin();
    return 0;
}