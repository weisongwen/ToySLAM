#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <deque>

/**
 * @brief IMU Integration and Pose Estimation Node
 * 
 * This node subscribes to IMU data, performs integration to estimate
 * position and orientation, and publishes the resulting pose.
 * It implements a simple inertial navigation system with bias correction.
 */
class ImuIntegrationNode {
public:
    /**
     * @brief Constructor
     * 
     * Initializes the node with parameters and sets up subscribers and publishers.
     */
    ImuIntegrationNode() : nh_("~"), is_first_imu_(true), initialized_(false) {
        // Load parameters
        loadParameters();
        
        // Initialize publishers
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odometry", 10);
        
        // Initialize subscriber
        imu_sub_ = nh_.subscribe(imu_topic_, 100, &ImuIntegrationNode::imuCallback, this);
        
        // Timer for status updates
        status_timer_ = nh_.createTimer(ros::Duration(2.0), &ImuIntegrationNode::publishStatus, this);
        
        // Initialize state variables
        position_ = Eigen::Vector3d(initial_position_[0], initial_position_[1], initial_position_[2]);
        velocity_ = Eigen::Vector3d::Zero();
        
        // Initialize orientation from initial RPY
        tf::Quaternion q;
        q.setRPY(initial_rpy_[0] * M_PI/180.0, initial_rpy_[1] * M_PI/180.0, initial_rpy_[2] * M_PI/180.0);
        orientation_ = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
        
        // Initialize bias estimates
        gyro_bias_ = Eigen::Vector3d::Zero();
        accel_bias_ = Eigen::Vector3d::Zero();
        
        ROS_INFO("IMU Integration Node initialized");
        ROS_INFO("Subscribed to topic: %s", imu_topic_.c_str());
        ROS_INFO("Initial position: [%.2f, %.2f, %.2f]", position_[0], position_[1], position_[2]);
        ROS_INFO("Initial orientation (RPY deg): [%.2f, %.2f, %.2f]", initial_rpy_[0], initial_rpy_[1], initial_rpy_[2]);
    }
    
    /**
     * @brief Load ROS parameters
     * 
     * Loads parameters from the parameter server with defaults.
     */
    void loadParameters() {
        // Topics and frames
        nh_.param<std::string>("imu_topic", imu_topic_, "/imu/data");
        nh_.param<std::string>("world_frame", world_frame_, "world");
        nh_.param<std::string>("body_frame", body_frame_, "imu_link");
        
        // Debug and configuration
        nh_.param<bool>("verbose", verbose_, false);
        nh_.param<bool>("publish_tf", publish_tf_, true);
        nh_.param<double>("gravity_magnitude", gravity_magnitude_, 9.81);
        
        // Initial position [x, y, z]
        std::vector<double> default_position = {0.0, 0.0, 0.0};
        nh_.param<std::vector<double>>("initial_position", initial_position_, default_position);
        if (initial_position_.size() != 3) {
            ROS_WARN("initial_position should have 3 elements. Using [0,0,0]");
            initial_position_ = default_position;
        }
        
        // Initial orientation as [roll, pitch, yaw] in degrees
        std::vector<double> default_rpy = {0.0, 0.0, 0.0};
        nh_.param<std::vector<double>>("initial_rpy", initial_rpy_, default_rpy);
        if (initial_rpy_.size() != 3) {
            ROS_WARN("initial_rpy should have 3 elements. Using [0,0,0]");
            initial_rpy_ = default_rpy;
        }
        
        // Integration parameters
        nh_.param<int>("calibration_samples", calibration_samples_, 200);
        nh_.param<bool>("use_magnetometer", use_magnetometer_, false);
        nh_.param<double>("accel_noise_density", accel_noise_density_, 0.01);
        nh_.param<double>("gyro_noise_density", gyro_noise_density_, 0.0002);
        nh_.param<double>("accel_random_walk", accel_random_walk_, 0.0002);
        nh_.param<double>("gyro_random_walk", gyro_random_walk_, 0.00002);
    }
    
    /**
     * @brief IMU Callback
     * 
     * Processes incoming IMU messages and updates state.
     * 
     * @param msg Pointer to the incoming IMU message
     */
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        // Handle first message
        if (is_first_imu_) {
            last_time_ = msg->header.stamp;
            is_first_imu_ = false;
            
            // Start collecting samples for bias estimation if not initialized
            if (!initialized_) {
                ROS_INFO("Started collecting IMU samples for initial bias estimation...");
            }
            return;
        }
        
        // Calculate time difference
        ros::Time current_time = msg->header.stamp;
        double dt = (current_time - last_time_).toSec();
        
        // Skip if dt is too large (indicates discontinuity)
        if (dt > 0.1) {
            ROS_WARN("Large time gap detected (%.3f sec). Resetting integration.", dt);
            last_time_ = current_time;
            return;
        }
        
        // Count received messages
        msg_count_++;
        
        // Extract IMU data
        Eigen::Vector3d accel(msg->linear_acceleration.x, 
                             msg->linear_acceleration.y, 
                             msg->linear_acceleration.z);
                             
        Eigen::Vector3d gyro(msg->angular_velocity.x,
                            msg->angular_velocity.y,
                            msg->angular_velocity.z);
                            
        // Use IMU orientation if available and magnetometer is enabled
        Eigen::Quaterniond measured_orientation;
        if (use_magnetometer_) {
            measured_orientation = Eigen::Quaterniond(msg->orientation.w,
                                                     msg->orientation.x,
                                                     msg->orientation.y,
                                                     msg->orientation.z);
        }
        
        // Calibration phase
        if (!initialized_ && calibration_samples_.size() < calibration_samples_) {
            // Store samples for calibration
            calibration_samples_.push_back(std::make_pair(gyro, accel));
            
            if (calibration_samples_.size() == calibration_samples_) {
                performInitialCalibration();
                initialized_ = true;
            }
            last_time_ = current_time;
            return;
        }
        
        // State estimation via IMU integration if initialization completed
        if (initialized_) {
            // Apply bias correction
            Eigen::Vector3d unbiased_gyro = gyro - gyro_bias_;
            Eigen::Vector3d unbiased_accel = accel - accel_bias_;
            
            // Orientation integration
            integrateOrientation(unbiased_gyro, dt);
            
            // If using magnetometer, apply complementary filter
            if (use_magnetometer_) {
                // Apply complementary filter for orientation (95% gyro, 5% magnetometer)
                orientation_ = orientation_.slerp(0.05, measured_orientation);
            }
            
            // Acceleration integration
            integrateAcceleration(unbiased_accel, dt);
            
            // Publish results
            publishPose(current_time);
        }
        
        // Update for next iteration
        last_time_ = current_time;
        
        // Print status periodically
        if (msg_count_ % 100 == 0 && verbose_) {
            printStatus(accel, gyro);
        }
    }
    
    /**
     * @brief Perform initial calibration
     * 
     * Estimates initial biases from collected samples
     */
    void performInitialCalibration() {
        ROS_INFO("Performing initial calibration with %lu samples...", calibration_samples_.size());
        
        // Calculate average gyro and accel readings during calibration
        Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
        
        for (const auto& sample : calibration_samples_) {
            gyro_sum += sample.first;
            accel_sum += sample.second;
        }
        
        // Calculate average (assuming sensor is stationary)
        gyro_bias_ = gyro_sum / calibration_samples_.size();
        
        // For accelerometer, the average should equal gravity
        Eigen::Vector3d gravity_vector = accel_sum / calibration_samples_.size();
        double measured_gravity = gravity_vector.norm();
        
        // Normalize gravity vector
        Eigen::Vector3d gravity_direction = gravity_vector / measured_gravity;
        
        // Calculate bias as the difference between measured and expected gravity
        accel_bias_ = gravity_vector - gravity_direction * gravity_magnitude_;
        
        // Calculate initial orientation from gravity direction
        // Gravity should point in the negative z direction in the world frame
        Eigen::Vector3d z_axis(0, 0, -1);
        Eigen::Vector3d rotation_axis = z_axis.cross(gravity_direction);
        
        if (rotation_axis.norm() > 1e-6) {
            rotation_axis.normalize();
            double rotation_angle = acos(z_axis.dot(gravity_direction));
            Eigen::AngleAxisd rotation(rotation_angle, rotation_axis);
            orientation_ = Eigen::Quaterniond(rotation);
        }
        
        ROS_INFO("Calibration complete");
        ROS_INFO("Gyro bias: [%.4f, %.4f, %.4f] rad/s", 
                 gyro_bias_[0], gyro_bias_[1], gyro_bias_[2]);
        ROS_INFO("Accel bias: [%.4f, %.4f, %.4f] m/s²", 
                 accel_bias_[0], accel_bias_[1], accel_bias_[2]);
        ROS_INFO("Measured gravity: %.4f m/s² (expected: %.4f)", 
                 measured_gravity, gravity_magnitude_);
                 
        // Clear calibration samples to free memory
        calibration_samples_.clear();
    }
    
    /**
     * @brief Integrate orientation using angular velocity
     * 
     * @param gyro Angular velocity (rad/s)
     * @param dt Time step (s)
     */
    void integrateOrientation(const Eigen::Vector3d& gyro, double dt) {
        // Quaternion integration using first-order approximation
        double angle = gyro.norm();
        
        if (angle > 1e-10) {
            Eigen::Vector3d axis = gyro / angle;
            Eigen::Quaterniond dq(Eigen::AngleAxisd(angle * dt, axis));
            orientation_ = orientation_ * dq;
            orientation_.normalize();
        }
    }
    
    /**
     * @brief Integrate acceleration to update velocity and position
     * 
     * @param accel Linear acceleration in sensor frame (m/s²)
     * @param dt Time step (s)
     */
    void integrateAcceleration(const Eigen::Vector3d& accel, double dt) {
        // Convert acceleration from sensor frame to world frame
        Eigen::Vector3d accel_world = orientation_ * accel;
        
        // Subtract gravity
        Eigen::Vector3d gravity(0, 0, -gravity_magnitude_);
        Eigen::Vector3d accel_without_gravity = accel_world - gravity;
        
        // Integrate acceleration to get velocity
        Eigen::Vector3d velocity_prev = velocity_;
        velocity_ += accel_without_gravity * dt;
        
        // Simple low-pass filter for velocity
        constexpr double velocity_filter_alpha = 0.1;
        velocity_ = velocity_ * (1.0 - velocity_filter_alpha) + velocity_prev * velocity_filter_alpha;
        
        // Zero-velocity update: if acceleration is very small, assume not moving
        double accel_magnitude = accel_without_gravity.norm();
        if (accel_magnitude < 0.05) {
            static int zero_vel_count = 0;
            zero_vel_count++;
            
            if (zero_vel_count > 50) {  // About 0.5 seconds at 100Hz
                velocity_ *= 0.8;  // Gradually reduce velocity when static
                if (velocity_.norm() < 0.01) {
                    velocity_.setZero();
                }
            }
        } else {
            zero_vel_count = 0;
        }
        
        // Integrate velocity to get position
        position_ += velocity_ * dt;
    }
    
    /**
     * @brief Publish estimated pose
     * 
     * @param time Current timestamp
     */
    void publishPose(const ros::Time& time) {
        // Create and publish pose message
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.stamp = time;
        pose_msg.header.frame_id = world_frame_;
        
        // Position
        pose_msg.pose.position.x = position_[0];
        pose_msg.pose.position.y = position_[1];
        pose_msg.pose.position.z = position_[2];
        
        // Orientation
        pose_msg.pose.orientation.w = orientation_.w();
        pose_msg.pose.orientation.x = orientation_.x();
        pose_msg.pose.orientation.y = orientation_.y();
        pose_msg.pose.orientation.z = orientation_.z();
        
        pose_pub_.publish(pose_msg);
        
        // Create and publish odometry message
        nav_msgs::Odometry odom_msg;
        odom_msg.header = pose_msg.header;
        odom_msg.child_frame_id = body_frame_;
        
        // Position and orientation
        odom_msg.pose.pose = pose_msg.pose;
        
        // Velocity in world frame
        odom_msg.twist.twist.linear.x = velocity_[0];
        odom_msg.twist.twist.linear.y = velocity_[1];
        odom_msg.twist.twist.linear.z = velocity_[2];
        
        // Angular velocity in body frame (not world frame)
        // We don't currently track world-frame angular velocity
        
        odom_pub_.publish(odom_msg);
        
        // Broadcast transform if enabled
        if (publish_tf_) {
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(position_[0], position_[1], position_[2]));
            
            tf::Quaternion q(orientation_.x(), orientation_.y(), orientation_.z(), orientation_.w());
            transform.setRotation(q);
            
            tf_broadcaster_.sendTransform(tf::StampedTransform(
                transform, time, world_frame_, body_frame_));
        }
    }
    
    /**
     * @brief Print status
     * 
     * Prints current state and IMU readings
     */
    void printStatus(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro) {
        // Get Euler angles from quaternion
        tf::Quaternion q(orientation_.x(), orientation_.y(), orientation_.z(), orientation_.w());
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        // Convert to degrees
        roll = roll * 180.0 / M_PI;
        pitch = pitch * 180.0 / M_PI;
        yaw = yaw * 180.0 / M_PI;
        
        ROS_INFO("--- IMU Integration Status ---");
        ROS_INFO("Position (m): [%.3f, %.3f, %.3f]", position_[0], position_[1], position_[2]);
        ROS_INFO("Velocity (m/s): [%.3f, %.3f, %.3f]", velocity_[0], velocity_[1], velocity_[2]);
        ROS_INFO("Orientation (deg): [%.1f, %.1f, %.1f]", roll, pitch, yaw);
        ROS_INFO("Acceleration (m/s²): [%.2f, %.2f, %.2f]", accel[0], accel[1], accel[2]);
        ROS_INFO("Angular Velocity (rad/s): [%.2f, %.2f, %.2f]", gyro[0], gyro[1], gyro[2]);
    }
    
    /**
     * @brief Publish status
     * 
     * Timer callback to publish status periodically
     */
    void publishStatus(const ros::TimerEvent&) {
        if (!initialized_) {
            ROS_INFO("Collecting calibration samples: %lu/%d", 
                     calibration_samples_.size(), calibration_samples_);
            return;
        }
        
        // Get Euler angles from quaternion
        tf::Quaternion q(orientation_.x(), orientation_.y(), orientation_.z(), orientation_.w());
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        
        // Convert to degrees
        roll = roll * 180.0 / M_PI;
        pitch = pitch * 180.0 / M_PI;
        yaw = yaw * 180.0 / M_PI;
        
        ROS_INFO("--- Current State ---");
        ROS_INFO("Position (m): [%.3f, %.3f, %.3f]", position_[0], position_[1], position_[2]);
        ROS_INFO("Orientation (deg): [%.1f, %.1f, %.1f]", roll, pitch, yaw);
    }
    
private:
    // ROS components
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher odom_pub_;
    ros::Timer status_timer_;
    tf::TransformBroadcaster tf_broadcaster_;
    
    // Parameters
    std::string imu_topic_;
    std::string world_frame_;
    std::string body_frame_;
    bool verbose_;
    bool publish_tf_;
    double gravity_magnitude_;
    std::vector<double> initial_position_;
    std::vector<double> initial_rpy_;
    int calibration_samples_;
    bool use_magnetometer_;
    double accel_noise_density_;
    double gyro_noise_density_;
    double accel_random_walk_;
    double gyro_random_walk_;
    
    // State tracking
    Eigen::Vector3d position_;
    Eigen::Vector3d velocity_;
    Eigen::Quaterniond orientation_;
    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d accel_bias_;
    
    // Timing
    ros::Time last_time_;
    bool is_first_imu_;
    int msg_count_ = 0;
    
    // Calibration
    bool initialized_;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> calibration_samples_;
};

/**
 * @brief Main function
 * 
 * Initializes the ROS node and runs the IMU integration node.
 */
int main(int argc, char** argv) {
    // Initialize ROS
    ros::init(argc, argv, "imu_integration_node");
    
    // Create instance of ImuIntegrationNode
    ImuIntegrationNode node;
    
    // Spin
    ROS_INFO("IMU Integration Node running. Press Ctrl+C to terminate.");
    ros::spin();
    
    return 0;
}