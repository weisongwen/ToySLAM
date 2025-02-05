#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <random>
#include <cmath>

class SensorSimulator {
public:
    SensorSimulator() : nh_("~") {
        // Initialize publishers
        imu_pub_ = nh_.advertise<sensor_msgs::Imu>("imu_data", 10);
        uwb_pub_ = nh_.advertise<sensor_msgs::Range>("uwb_data", 10);

        // Initialize random number generators
        std::random_device rd;
        gen_.reset(new std::mt19937(rd()));
        accel_noise_ = std::normal_distribution<double>(0.0, 0.01);
        gyro_noise_ = std::normal_distribution<double>(0.0, 0.005);
        uwb_noise_ = std::normal_distribution<double>(0.0, 0.1);

        // Setup timers
        imu_timer_ = nh_.createTimer(ros::Duration(1.0/100.0), 
                                   &SensorSimulator::publishImu, this);
        uwb_timer_ = nh_.createTimer(ros::Duration(1.0/10.0), 
                                  &SensorSimulator::publishUwb, this);
    }

private:
    void publishImu(const ros::TimerEvent&) {
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.header.frame_id = "imu_link";

        // Simulate accelerometer data (m/s²)
        const double t = ros::Time::now().toSec();
        imu_msg.linear_acceleration.x = 0.5 * std::sin(t) + accel_noise_(*gen_);
        imu_msg.linear_acceleration.y = 0.2 * std::cos(0.5 * t) + accel_noise_(*gen_);
        imu_msg.linear_acceleration.z = 9.8 + accel_noise_(*gen_);  // Gravity

        // Simulate gyroscope data (rad/s)
        imu_msg.angular_velocity.x = 0.1 * std::sin(0.3 * t) + gyro_noise_(*gen_);
        imu_msg.angular_velocity.y = 0.05 * std::cos(0.2 * t) + gyro_noise_(*gen_);
        imu_msg.angular_velocity.z = 0.2 * std::sin(0.1 * t) + gyro_noise_(*gen_);

        // Set orientation (identity quaternion for simplicity)
        imu_msg.orientation.w = 1.0;

        // Set covariance matrices
        imu_msg.linear_acceleration_covariance.fill(0.0);
        imu_msg.angular_velocity_covariance.fill(0.0);
        imu_msg.orientation_covariance.fill(0.0);
        imu_msg.linear_acceleration_covariance[0] = 0.01;
        imu_msg.angular_velocity_covariance[0] = 0.001;

        imu_pub_.publish(imu_msg);
    }

    void publishUwb(const ros::TimerEvent&) {
        sensor_msgs::Range uwb_msg;
        uwb_msg.header.stamp = ros::Time::now();
        uwb_msg.header.frame_id = "uwb_link";
        uwb_msg.radiation_type = sensor_msgs::Range::ULTRASOUND;
        uwb_msg.field_of_view = 0.1;  // radians
        uwb_msg.min_range = 0.1;      // meters
        uwb_msg.max_range = 30.0;     // meters

        // Simulate range with noise
        const double t = ros::Time::now().toSec();
        uwb_msg.range = 5.0 + 2.0 * std::sin(0.5 * t) + uwb_noise_(*gen_);

        uwb_pub_.publish(uwb_msg);
    }

    ros::NodeHandle nh_;
    ros::Publisher imu_pub_;
    ros::Publisher uwb_pub_;
    ros::Timer imu_timer_;
    ros::Timer uwb_timer_;

    std::unique_ptr<std::mt19937> gen_;
    std::normal_distribution<double> accel_noise_;
    std::normal_distribution<double> gyro_noise_;
    std::normal_distribution<double> uwb_noise_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator");
    SensorSimulator sensor_simulator;
    ros::spin();
    return 0;
}