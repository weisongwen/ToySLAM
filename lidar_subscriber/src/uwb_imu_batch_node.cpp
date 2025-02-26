#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <deque>
#include <mutex>
#include <algorithm>
#include <memory>

// Custom parameterization for pose (position + quaternion)
class PoseParameterization : public ceres::LocalParameterization {
public:
    virtual ~PoseParameterization() {}

    // Position is updated as-is, quaternion through quaternion multiplication
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // Position update: simple addition
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];
        
        // Quaternion update: quaternion multiplication
        const double* q = x + 3;
        Eigen::Quaterniond q_x(q[0], q[1], q[2], q[3]);
        
        // Convert small delta rotation to quaternion
        Eigen::Quaterniond dq;
        const double theta_squared = delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5];
        
        if (theta_squared > 0.0) {
            const double theta = std::sqrt(theta_squared);
            const double half_theta = theta * 0.5;
            const double scale = sin(half_theta) / theta;
            dq = Eigen::Quaterniond(cos(half_theta), 
                                     delta[3] * scale,
                                     delta[4] * scale,
                                     delta[5] * scale);
        } else {
            dq = Eigen::Quaterniond::Identity();
        }
        
        // Apply update: q_new = q_x * dq
        Eigen::Quaterniond q_result = q_x * dq;
        q_result.normalize();
        
        x_plus_delta[3] = q_result.w();
        x_plus_delta[4] = q_result.x();
        x_plus_delta[5] = q_result.y();
        x_plus_delta[6] = q_result.z();
        
        return true;
    }
    
    // Position: 3, Quaternion: 3 (not 4, because of unit quaternion constraint)
    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }
    
    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        // Initialize to zero
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // Position part: identity
        J.block<3, 3>(0, 0).setIdentity();
        
        // Quaternion part: rotation Jacobian
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        
        double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
        J(3, 3) = -qx * 0.5;
        J(3, 4) = -qy * 0.5;
        J(3, 5) = -qz * 0.5;
        
        J(4, 3) =  qw * 0.5;
        J(4, 4) = -qz * 0.5;
        J(4, 5) =  qy * 0.5;
        
        J(5, 3) =  qz * 0.5;
        J(5, 4) =  qw * 0.5;
        J(5, 5) = -qx * 0.5;
        
        J(6, 3) = -qy * 0.5;
        J(6, 4) =  qx * 0.5;
        J(6, 5) =  qw * 0.5;
        
        return true;
    }
};

// Bias random walk constraint factor
class BiasRandomWalkFactor {
public:
    BiasRandomWalkFactor(double acc_bias_sigma, double gyro_bias_sigma)
        : acc_bias_sigma_(acc_bias_sigma), gyro_bias_sigma_(gyro_bias_sigma) {}
    
    template <typename T>
    bool operator()(const T* const bias_i, const T* const bias_j, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(bias_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(bias_i + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(bias_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(bias_j + 3);
        
        // Compute residuals - difference between consecutive biases
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
        
        // Apply weights based on expected bias random walk
        residual.template segment<3>(0) = (ba_j - ba_i) / T(acc_bias_sigma_);
        residual.template segment<3>(3) = (bg_j - bg_i) / T(gyro_bias_sigma_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(double acc_bias_sigma, double gyro_bias_sigma) {
        return new ceres::AutoDiffCostFunction<BiasRandomWalkFactor, 6, 6, 6>(
            new BiasRandomWalkFactor(acc_bias_sigma, gyro_bias_sigma));
    }
    
private:
    double acc_bias_sigma_;
    double gyro_bias_sigma_;
};

// Bias prior factor to provide regularization
class BiasPriorFactor {
public:
    BiasPriorFactor(const Eigen::Vector3d& prior_acc_bias, const Eigen::Vector3d& prior_gyro_bias,
                   double acc_sigma, double gyro_sigma)
        : prior_acc_bias_(prior_acc_bias), prior_gyro_bias_(prior_gyro_bias),
          acc_sigma_(acc_sigma), gyro_sigma_(gyro_sigma) {}
    
    template <typename T>
    bool operator()(const T* const bias, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba(bias);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bias + 3);
        
        // Compute residuals - difference from prior
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
        
        // Apply weights
        residual.template segment<3>(0) = (ba - prior_acc_bias_.cast<T>()) / T(acc_sigma_);
        residual.template segment<3>(3) = (bg - prior_gyro_bias_.cast<T>()) / T(gyro_sigma_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_acc_bias, 
                                       const Eigen::Vector3d& prior_gyro_bias,
                                       double acc_sigma, double gyro_sigma) {
        return new ceres::AutoDiffCostFunction<BiasPriorFactor, 6, 6>(
            new BiasPriorFactor(prior_acc_bias, prior_gyro_bias, acc_sigma, gyro_sigma));
    }
    
private:
    Eigen::Vector3d prior_acc_bias_;
    Eigen::Vector3d prior_gyro_bias_;
    double acc_sigma_;
    double gyro_sigma_;
};

// Roll/Pitch prior factor for planar motion
class RollPitchPriorFactor {
public:
    RollPitchPriorFactor(double weight = 10.0) : weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract quaternion
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        
        // Convert to rotation matrix
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        
        // Get gravity direction in body frame (should point downward, i.e., z-axis)
        Eigen::Matrix<T, 3, 1> z_body = R.col(2);
        
        // In planar motion, z_body should be close to [0,0,1] or [0,0,-1]
        // Penalize deviation of x and y components from zero
        residuals[0] = T(weight_) * z_body.x();
        residuals[1] = T(weight_) * z_body.y();
        
        return true;
    }
    
    static ceres::CostFunction* Create(double weight = 10.0) {
        return new ceres::AutoDiffCostFunction<RollPitchPriorFactor, 2, 7>(
            new RollPitchPriorFactor(weight));
    }
    
private:
    double weight_;
};

class UwbImuFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    UwbImuFusion() {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // Load parameters
        private_nh.param<double>("gravity_magnitude", gravity_magnitude_, 9.81);
        private_nh.param<double>("imu_acc_noise", imu_acc_noise_, 0.01);  // Increased from 0.0001
        private_nh.param<double>("imu_gyro_noise", imu_gyro_noise_, 0.005);  // Increased from 0.00005
        private_nh.param<double>("imu_acc_bias_noise", imu_acc_bias_noise_, 0.001);
        private_nh.param<double>("imu_gyro_bias_noise", imu_gyro_bias_noise_, 0.0005);
        private_nh.param<double>("uwb_position_noise", uwb_position_noise_, 0.05);
        private_nh.param<int>("optimization_window_size", optimization_window_size_, 20);  // Increased from 10
        private_nh.param<std::string>("world_frame_id", world_frame_id_, "world");
        private_nh.param<std::string>("body_frame_id", body_frame_id_, "base_link");
        private_nh.param<double>("optimization_frequency", optimization_frequency_, 10.0);
        private_nh.param<double>("imu_buffer_time_length", imu_buffer_time_length_, 10.0);
        private_nh.param<double>("uwb_alpha_xy", uwb_alpha_xy_, 0.9); 
        private_nh.param<double>("uwb_alpha_z", uwb_alpha_z_, 0.95); 
        private_nh.param<int>("max_iterations", max_iterations_, 15);
        private_nh.param<bool>("enable_bias_estimation", enable_bias_estimation_, true);
        private_nh.param<double>("position_drift_threshold", position_drift_threshold_, 20.0);
        private_nh.param<double>("roll_pitch_weight", roll_pitch_weight_, 15.0);  // Changed from interval to weight
        
        // Initialize with small non-zero biases to help break symmetry in the optimization
        initial_acc_bias_ = Eigen::Vector3d(0.05, 0.05, 0.05);
        initial_gyro_bias_ = Eigen::Vector3d(0.01, 0.01, 0.01);
        
        // Initialize subscribers and publishers
        imu_sub_ = nh.subscribe("/sensor_simulator/imu_data", 1000, &UwbImuFusion::imuCallback, this);
        uwb_sub_ = nh.subscribe("/sensor_simulator/UWBPoistionPS", 100, &UwbImuFusion::uwbCallback, this);
        pose_pub_ = nh.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/pose", 10);
        
        // Initialize state
        is_initialized_ = false;
        has_imu_data_ = false;
        last_z_reset_time_ = 0;
        z_reset_interval_ = 1.0;
        initializeState();
        
        // Setup optimization timer
        optimization_timer_ = nh.createTimer(ros::Duration(1.0/optimization_frequency_), 
                                           &UwbImuFusion::optimizationTimerCallback, this);
        
        ROS_INFO("UWB-IMU Fusion node initialized. Using factor graph optimization with max %d iterations", max_iterations_);
        ROS_INFO("Bias estimation is %s", enable_bias_estimation_ ? "enabled" : "disabled");
        ROS_INFO("UWB noise: %.5f, position drift threshold: %.2f", uwb_position_noise_, position_drift_threshold_);
        ROS_INFO("Roll/pitch prior weight: %.2f", roll_pitch_weight_);
    }

private:
    // ROS subscribers and publishers
    ros::Subscriber imu_sub_;
    ros::Subscriber uwb_sub_;
    ros::Publisher pose_pub_;
    ros::Timer optimization_timer_;

    // Parameters
    double gravity_magnitude_;
    double imu_acc_noise_;
    double imu_gyro_noise_;
    double imu_acc_bias_noise_;
    double imu_gyro_bias_noise_;
    double uwb_position_noise_;
    int optimization_window_size_;
    double optimization_frequency_;
    double imu_buffer_time_length_;
    double uwb_alpha_xy_;
    double uwb_alpha_z_;
    int max_iterations_;
    bool enable_bias_estimation_;
    std::string world_frame_id_;
    std::string body_frame_id_;
    double last_z_reset_time_;
    double z_reset_interval_;
    double position_drift_threshold_;
    double roll_pitch_weight_;
    
    // Initial bias values to help estimation
    Eigen::Vector3d initial_acc_bias_;
    Eigen::Vector3d initial_gyro_bias_;

    // State variables
    struct State {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d velocity;
        Eigen::Vector3d acc_bias;
        Eigen::Vector3d gyro_bias;
        double timestamp;
    };

    State current_state_;
    std::deque<State, Eigen::aligned_allocator<State>> state_window_;
    bool is_initialized_;
    bool has_imu_data_;
    
    // IMU pre-integration
    struct ImuPreintegration {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d delta_position;
        Eigen::Quaterniond delta_orientation;
        Eigen::Vector3d delta_velocity;
        Eigen::Matrix<double, 9, 9> covariance;
        Eigen::Matrix<double, 9, 6> jacobian_bias;
        double start_time;
        double end_time;
        
        // Reference biases used during pre-integration
        Eigen::Vector3d acc_bias_ref;
        Eigen::Vector3d gyro_bias_ref;
        
        // Flag to indicate if this is a valid preintegration result
        bool is_valid;
    };

    std::deque<sensor_msgs::Imu> imu_buffer_;
    
    // UWB measurements
    struct UwbMeasurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        double timestamp;
    };

    std::vector<UwbMeasurement> uwb_measurements_;

    // Mutex for thread safety
    std::mutex data_mutex_;

    // Gravity vector in world frame
    Eigen::Vector3d gravity_world_;

    // Helper function to convert quaternion to Euler angles in degrees
    Eigen::Vector3d quaternionToEulerDegrees(const Eigen::Quaterniond& q) {
        // Convert to rotation matrix first to avoid gimbal lock issues
        Eigen::Matrix3d rot = q.normalized().toRotationMatrix();
        
        // Extract Euler angles - use intrinsic rotations (ZYX convention)
        double roll = atan2(rot(2,1), rot(2,2));
        double pitch = -asin(rot(2,0));
        double yaw = atan2(rot(1,0), rot(0,0));
        
        // Convert to degrees
        Eigen::Vector3d euler_deg;
        euler_deg << roll * 180.0 / M_PI, 
                     pitch * 180.0 / M_PI, 
                     yaw * 180.0 / M_PI;
        
        return euler_deg;
    }

    void initializeState() {
        try {
            current_state_.position = Eigen::Vector3d::Zero();
            current_state_.orientation = Eigen::Quaterniond::Identity();
            current_state_.velocity = Eigen::Vector3d::Zero();
            current_state_.acc_bias = initial_acc_bias_;
            current_state_.gyro_bias = initial_gyro_bias_;
            current_state_.timestamp = 0;
            
            state_window_.clear();
            uwb_measurements_.clear();
            imu_buffer_.clear();
            
            // Initialize gravity vector
            gravity_world_ = Eigen::Vector3d(0, 0, -gravity_magnitude_);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeState: %s", e.what());
        }
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            has_imu_data_ = true;
            
            static int imu_count = 0;
            imu_count++;
            if (imu_count % 1000 == 0) {
                ROS_INFO_THROTTLE(1.0, "Received %d IMU messages", imu_count);
            }
            
            // Store IMU measurements for pre-integration
            imu_buffer_.push_back(*msg);
            
            // Process IMU data for state propagation
            if (is_initialized_) {
                processImu(*msg);
            }
            
            // Limit the size of the IMU buffer
            double oldest_allowed_time = ros::Time::now().toSec() - imu_buffer_time_length_;
            while (!imu_buffer_.empty() && imu_buffer_.front().header.stamp.toSec() < oldest_allowed_time) {
                imu_buffer_.pop_front();
            }
            
            // Limit IMU buffer size to prevent memory issues
            if (imu_buffer_.size() > 5000) {
                imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + 1000);
                ROS_WARN("IMU buffer too large, removing oldest 1000 measurements");
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in imuCallback: %s", e.what());
        }
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            // Store UWB position measurement
            UwbMeasurement measurement;
            measurement.position = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);
            measurement.timestamp = msg->header.stamp.toSec();
            
            static int uwb_count = 0;
            uwb_count++;
            if (uwb_count % 20 == 0) {
                ROS_INFO_THROTTLE(1.0, "Received UWB message #%d: [%.2f, %.2f, %.2f] at %.6f", 
                           uwb_count, measurement.position.x(), measurement.position.y(), 
                           measurement.position.z(), measurement.timestamp);
            }
            
            // Ensure timestamp is valid
            if (measurement.timestamp <= 0) {
                ROS_WARN("Invalid UWB timestamp: %f, using current time", measurement.timestamp);
                measurement.timestamp = ros::Time::now().toSec();
            }
            
            // Keep uwb_measurements_ limited to a reasonable size
            if (uwb_measurements_.size() > 50) {
                uwb_measurements_.erase(uwb_measurements_.begin(), uwb_measurements_.begin() + 25);
            }
            
            uwb_measurements_.push_back(measurement);
            
            // Initialize if not yet initialized
            if (!is_initialized_) {
                ROS_INFO("Initializing system with UWB measurement at timestamp: %f", measurement.timestamp);
                initializeFromUwb(measurement);
                is_initialized_ = true;
                return;
            }
            
            // Even if we have IMU data, also update the state window during initialization
            if (state_window_.size() < 2) {
                State new_state = current_state_;
                new_state.position = measurement.position;
                new_state.timestamp = measurement.timestamp;
                
                if (std::abs(new_state.timestamp - current_state_.timestamp) > 1e-6) {
                    current_state_ = new_state;
                    state_window_.push_back(current_state_);
                    ROS_INFO("Added new state from UWB. State window size: %zu", state_window_.size());
                }
            }
            
            // Check for excessive drift from UWB position and reset if needed
            if (!state_window_.empty()) {
                // Calculate drift between current position estimate and UWB measurement
                double position_error = (current_state_.position - measurement.position).norm();
                
                if (position_error > position_drift_threshold_) {
                    ROS_WARN("Detected excessive position drift (%.2f m). Resetting to UWB position.", position_error);
                    
                    // Reset position to UWB measurement
                    current_state_.position = measurement.position;
                    current_state_.velocity = Eigen::Vector3d::Zero(); // Reset velocity too
                    
                    // Also update the state window to prevent optimization from pulling back to incorrect position
                    for (auto& state : state_window_) {
                        // Adjust positions and velocities in the state window toward UWB
                        // The closer to the current time, the stronger the correction
                        double time_ratio = (state.timestamp - state_window_.front().timestamp) / 
                                          (current_state_.timestamp - state_window_.front().timestamp);
                        if (time_ratio > 0) {
                            state.position = state.position + time_ratio * (measurement.position - current_state_.position);
                            state.velocity = state.velocity * (1.0 - time_ratio * 0.8); // Dampen velocity
                        }
                    }
                }
            }
            
            // Quick periodic height correction to prevent Z drift
            double now = ros::Time::now().toSec();
            if (now - last_z_reset_time_ > z_reset_interval_) {
                // Directly update current state height using this UWB measurement
                current_state_.position.z() = measurement.position.z();
                current_state_.velocity.z() = 0; // Zero vertical velocity to reduce drift
                
                // Also fix state window Z values to prevent optimization from going back to drifted values
                for (auto& state : state_window_) {
                    state.position.z() = measurement.position.z();
                    state.velocity.z() = 0;
                }
                
                last_z_reset_time_ = now;
                ROS_INFO("Applied height correction to Z = %.3f meters", measurement.position.z());
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in uwbCallback: %s", e.what());
        }
    }

    void optimizationTimerCallback(const ros::TimerEvent& event) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            if (!is_initialized_) {
                ROS_WARN_THROTTLE(5.0, "System not yet initialized, waiting for first UWB measurement");
                return;
            }
            
            ROS_INFO_THROTTLE(1.0, "State window size: %zu, UWB measurements: %zu, IMU buffer: %zu", 
                            state_window_.size(), uwb_measurements_.size(), imu_buffer_.size());
            
            if (state_window_.size() < 2) {
                ROS_WARN_THROTTLE(1.0, "Not enough states for optimization");
                
                // If not enough states, create a new one from UWB
                if (!uwb_measurements_.empty() && state_window_.size() == 1) {
                    auto latest_uwb = uwb_measurements_.back();
                    
                    State new_state = state_window_.back();
                    new_state.position = latest_uwb.position;
                    new_state.timestamp = latest_uwb.timestamp;
                    
                    if (std::abs(new_state.timestamp - state_window_.back().timestamp) > 1e-6) {
                        state_window_.push_back(new_state);
                        current_state_ = new_state;
                        ROS_INFO("Added synthetic state from UWB. State window size: %zu", state_window_.size());
                    }
                }
                return;
            }
            
            if (uwb_measurements_.empty()) {
                ROS_WARN_THROTTLE(5.0, "No UWB measurements available for optimization");
                return;
            }

            // First, do a simple UWB update to anchor the trajectory
            updateWithUwb();
            
            // Then try factor graph optimization
            bool success = false;
            
            try {
                success = optimizeFactorGraph();
            } catch (const std::exception& e) {
                ROS_ERROR("Exception during factor graph optimization: %s", e.what());
                success = false;
            }
            
            // If factor graph optimization failed, our simple UWB update still applies
            if (!success) {
                ROS_WARN("Factor graph optimization failed, using simple UWB update only");
            }
            
            // Ensure we have valid state before publishing
            if (isStateValid(current_state_)) {
                publishState();
            } else {
                ROS_ERROR("Invalid current state after update");
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in optimizationTimerCallback: %s", e.what());
        }
    }

    void initializeFromUwb(const UwbMeasurement& uwb) {
        try {
            // Initialize state using UWB position
            current_state_.position = uwb.position;
            current_state_.orientation = Eigen::Quaterniond::Identity();
            current_state_.velocity = Eigen::Vector3d::Zero();
            current_state_.acc_bias = initial_acc_bias_;
            current_state_.gyro_bias = initial_gyro_bias_;
            current_state_.timestamp = uwb.timestamp;
            last_z_reset_time_ = uwb.timestamp;
            
            // Make sure we have a valid timestamp
            if (current_state_.timestamp <= 0) {
                ROS_WARN("Invalid timestamp in UWB initialization: %f", current_state_.timestamp);
                current_state_.timestamp = ros::Time::now().toSec();
                ROS_INFO("Using current time instead: %f", current_state_.timestamp);
            }
            
            state_window_.clear(); // Ensure clean state window
            state_window_.push_back(current_state_);
            
            ROS_INFO("State initialized at t=%f with position [%f, %f, %f]", 
                    current_state_.timestamp, 
                    current_state_.position.x(), 
                    current_state_.position.y(), 
                    current_state_.position.z());
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeFromUwb: %s", e.what());
        }
    }

    void processImu(const sensor_msgs::Imu& imu_msg) {
        try {
            double timestamp = imu_msg.header.stamp.toSec();
            
            // Error if timestamp is invalid
            if (timestamp <= 0) {
                ROS_WARN_THROTTLE(1.0, "Invalid IMU timestamp: %f", timestamp);
                return;
            }
            
            // Extract IMU measurements
            Eigen::Vector3d acc(imu_msg.linear_acceleration.x,
                               imu_msg.linear_acceleration.y,
                               imu_msg.linear_acceleration.z);
            
            Eigen::Vector3d gyro(imu_msg.angular_velocity.x,
                                imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z);
            
            // Check for NaN/Inf values
            if (!acc.allFinite() || !gyro.allFinite()) {
                ROS_WARN_THROTTLE(1.0, "Non-finite IMU values detected");
                return;
            }
            
            // State propagation
            double dt = timestamp - current_state_.timestamp;
            
            if (dt > 0 && dt < 0.1 && current_state_.timestamp > 0) {
                // Correct for biases
                Eigen::Vector3d acc_corrected = acc - current_state_.acc_bias;
                Eigen::Vector3d gyro_corrected = gyro - current_state_.gyro_bias;
                
                // Remove gravity from acceleration
                Eigen::Vector3d gravity_body = current_state_.orientation.inverse() * gravity_world_;
                Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_body;
                
                // Update orientation using gyro
                Eigen::Quaterniond dq;
                Eigen::Vector3d angle_axis = gyro_corrected * dt;
                double angle = angle_axis.norm();
                if (angle > 1e-10) {
                    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, angle_axis.normalized()));
                } else {
                    dq = Eigen::Quaterniond::Identity();
                }
                
                // Integrate state
                current_state_.orientation = (current_state_.orientation * dq).normalized();
                
                // Rotate acceleration to world frame
                Eigen::Vector3d acc_world = current_state_.orientation * acc_without_gravity;
                
                // Update velocity and position
                current_state_.velocity += acc_world * dt;
                
                // Add a damping factor to vertical velocity to reduce drift
                current_state_.velocity.z() *= 0.95; // Stronger damping (0.95 instead of 0.99)
                
                current_state_.position += current_state_.velocity * dt + 0.5 * acc_world * dt * dt;
                current_state_.timestamp = timestamp;
                
                // Ensure state values are finite
                if (!isStateValid(current_state_)) {
                    ROS_WARN("Non-finite state values after IMU integration. Resetting to previous state.");
                    if (!state_window_.empty()) {
                        current_state_ = state_window_.back();
                    }
                    return;
                }
                
                // Add to state window
                if (state_window_.size() >= optimization_window_size_) {
                    state_window_.pop_front();
                }
                state_window_.push_back(current_state_);
                
                // Debug logging for state window growth
                static int prev_size = 0;
                if (state_window_.size() != prev_size) {
                    ROS_INFO("State window size changed: %zu", state_window_.size());
                    prev_size = state_window_.size();
                }
            } else if (dt > 0.1) {
                ROS_WARN("Large time gap detected in IMU data: %.3f seconds. Skipping integration.", dt);
                current_state_.timestamp = timestamp;
            } else if (dt <= 0) {
                ROS_WARN_THROTTLE(1.0, "Invalid dt in IMU processing: %.6f. Current: %.6f, IMU: %.6f", 
                                dt, current_state_.timestamp, timestamp);
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in processImu: %s", e.what());
        }
    }

    // Helper function to check if a state is valid (no NaN/Inf values)
    bool isStateValid(const State& state) {
        return state.position.allFinite() && 
               state.velocity.allFinite() && 
               state.orientation.coeffs().allFinite() &&
               state.acc_bias.allFinite() && 
               state.gyro_bias.allFinite();
    }

    // Simple UWB update with separate horizontal and vertical weights
    void updateWithUwb() {
        // Find relevant UWB measurements for each state in the window
        for (auto& state : state_window_) {
            // Find closest UWB measurement in time
            const UwbMeasurement* best_uwb = nullptr;
            double min_time_diff = 0.2; // Maximum allowed time difference
            
            for (const auto& uwb : uwb_measurements_) {
                double time_diff = std::abs(state.timestamp - uwb.timestamp);
                if (time_diff < min_time_diff) {
                    min_time_diff = time_diff;
                    best_uwb = &uwb;
                }
            }
            
            // If we found a close enough UWB measurement, update position
            if (best_uwb) {
                // Separate weights for horizontal (x,y) and vertical (z) components
                // Higher weights to trust UWB more
                
                // X and Y components
                state.position.x() = uwb_alpha_xy_ * best_uwb->position.x() + (1.0 - uwb_alpha_xy_) * state.position.x();
                state.position.y() = uwb_alpha_xy_ * best_uwb->position.y() + (1.0 - uwb_alpha_xy_) * state.position.y();
                
                // Z component with higher weight to prevent height drift
                state.position.z() = uwb_alpha_z_ * best_uwb->position.z() + (1.0 - uwb_alpha_z_) * state.position.z();
                
                // Check if Z is drifting significantly
                double z_error = std::abs(state.position.z() - best_uwb->position.z());
                if (z_error > 0.5) { // Reduced threshold for Z correction
                    ROS_WARN("Large Z error: %f meters. Resetting height to UWB value.", z_error);
                    state.position.z() = best_uwb->position.z();
                    state.velocity.z() = 0.0; // Reset vertical velocity
                }
                
                // Also reduce velocity slightly when UWB update occurs
                state.velocity *= 0.95; // 5% velocity damping on UWB update
            }
        }
        
        // Update current state to match the latest state in the window
        if (!state_window_.empty()) {
            current_state_ = state_window_.back();
        }
        
        // Convert orientation to RPY in degrees for logging
        Eigen::Vector3d orientation_deg = quaternionToEulerDegrees(current_state_.orientation);
        
        ROS_INFO_THROTTLE(1.0, "UWB update complete. Position: [%f, %f, %f], Orientation (RPY): [%.1f, %.1f, %.1f], Acc bias: [%f, %f, %f], Gyro bias: [%f, %f, %f]",
                 current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                 orientation_deg.x(), orientation_deg.y(), orientation_deg.z(),
                 current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
                 current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
    }

    // Perform IMU pre-integration between two timestamps
    ImuPreintegration integrateImuMeasurements(double start_time, double end_time, 
                                             const Eigen::Vector3d& acc_bias, 
                                             const Eigen::Vector3d& gyro_bias) {
        try {
            ImuPreintegration result;
            result.delta_position = Eigen::Vector3d::Zero();
            result.delta_orientation = Eigen::Quaterniond::Identity();
            result.delta_velocity = Eigen::Vector3d::Zero();
            result.covariance = Eigen::Matrix<double, 9, 9>::Identity() * 1e-3; // Initialize to small identity matrix
            result.jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();
            result.start_time = start_time;
            result.end_time = end_time;
            result.acc_bias_ref = acc_bias;
            result.gyro_bias_ref = gyro_bias;
            result.is_valid = false; // Default to invalid
            
            // Validate inputs
            if (start_time >= end_time) {
                ROS_WARN("Invalid time interval: start(%f) >= end(%f)", start_time, end_time);
                return result;
            }
            
            if (!acc_bias.allFinite() || !gyro_bias.allFinite()) {
                ROS_WARN("Non-finite bias values provided to IMU integration");
                return result;
            }
            
            // Find relevant IMU measurements in the buffer
            std::vector<sensor_msgs::Imu> relevant_imus;
            for (const auto& imu : imu_buffer_) {
                double timestamp = imu.header.stamp.toSec();
                if (timestamp >= start_time && timestamp <= end_time) {
                    relevant_imus.push_back(imu);
                }
            }
            
            if (relevant_imus.empty()) {
                ROS_WARN("No IMU measurements found between %f and %f", start_time, end_time);
                return result; // Return invalid result
            }
            
            // Found at least one valid IMU measurement
            result.is_valid = true;
            
            // Sort IMU measurements by timestamp to ensure proper integration
            std::sort(relevant_imus.begin(), relevant_imus.end(), 
                     [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                         return a.header.stamp.toSec() < b.header.stamp.toSec();
                     });
            
            // Noise covariance for IMU measurements - higher weights for better bias estimation
            Eigen::Matrix<double, 6, 6> noise_covariance = Eigen::Matrix<double, 6, 6>::Zero();
            noise_covariance.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * imu_acc_noise_ * imu_acc_noise_;
            noise_covariance.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * imu_gyro_noise_ * imu_gyro_noise_;
            
            // Initialize pre-integration
            Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
            Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
            Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
            
            // Initialize covariance and Jacobians
            Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();
            Eigen::Matrix<double, 9, 6> jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();
            
            // Pre-integration starts with reference biases
            double prev_time = start_time;
            for (size_t i = 0; i < relevant_imus.size(); ++i) {
                const auto& imu = relevant_imus[i];
                double curr_time = imu.header.stamp.toSec();
                double dt = curr_time - prev_time;
                
                if (dt <= 0 || dt > 0.1) {
                    // Skip invalid time differences or large gaps
                    prev_time = curr_time;
                    continue;
                }
                
                // Extract IMU measurements
                Eigen::Vector3d acc(imu.linear_acceleration.x,
                                   imu.linear_acceleration.y,
                                   imu.linear_acceleration.z);
                
                Eigen::Vector3d gyro(imu.angular_velocity.x,
                                    imu.angular_velocity.y,
                                    imu.angular_velocity.z);
                
                // Check IMU values are finite
                if (!acc.allFinite() || !gyro.allFinite()) {
                    ROS_WARN("Non-finite IMU values encountered during integration");
                    continue;
                }
                
                // Correct for biases
                Eigen::Vector3d acc_corrected = acc - acc_bias;
                Eigen::Vector3d gyro_corrected = gyro - gyro_bias;
                
                // Simple Euler integration 
                // Integrate rotation
                Eigen::Vector3d angle_axis = gyro_corrected * dt;
                Eigen::Quaterniond dq;
                
                double angle = angle_axis.norm();
                if (angle > 1e-10) {
                    dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, angle_axis / angle));
                } else {
                    dq = Eigen::Quaterniond::Identity();
                }
                
                // Current rotation estimate from reference frame to current IMU frame
                Eigen::Quaterniond delta_q_next = delta_q * dq;
                delta_q_next.normalize(); // Ensure unit quaternion
                
                // Integrate velocity and position
                Eigen::Vector3d acc_rotated = delta_q * acc_corrected;
                Eigen::Vector3d delta_v_next = delta_v + acc_rotated * dt;
                Eigen::Vector3d delta_p_next = delta_p + delta_v * dt + 0.5 * acc_rotated * dt * dt;
                
                // State transition matrix for covariance propagation
                Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
                F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
                F.block<3, 3>(3, 6) = delta_q.toRotationMatrix() * dt;
                F.block<3, 3>(0, 6) = 0.5 * delta_q.toRotationMatrix() * dt * dt;
                
                // Noise propagation matrix
                Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
                G.block<3, 3>(3, 0) = delta_q.toRotationMatrix();
                G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
                
                // Update bias Jacobians - these are critical for bias estimation
                // Effect of acc bias change on delta_p, delta_v
                Eigen::Matrix<double, 9, 6> dF_db = Eigen::Matrix<double, 9, 6>::Zero();
                
                // Improved acc bias Jacobian calculation
                dF_db.block<3, 3>(0, 0) = -0.5 * delta_q.toRotationMatrix() * dt * dt;
                dF_db.block<3, 3>(3, 0) = -delta_q.toRotationMatrix() * dt;
                
                // Effect of gyro bias change on rotation, which indirectly affects delta_p, delta_v
                // This is the skew-symmetric matrix using the rotated acc_corrected
                Eigen::Matrix3d dR_db_gyro = -dt * delta_q.toRotationMatrix() * skewSymmetric(acc_corrected);
                dF_db.block<3, 3>(0, 3) = 0.5 * dR_db_gyro * dt;
                dF_db.block<3, 3>(3, 3) = dR_db_gyro;
                
                // Direct effect of gyro bias on orientation (missing in original code)
                dF_db.block<3, 3>(6, 3) = -dt * Eigen::Matrix3d::Identity();
                
                jacobian_bias = F * jacobian_bias + dF_db;
                
                // Covariance propagation
                covariance = F * covariance * F.transpose() + G * noise_covariance * G.transpose();
                
                // Check intermediate results are finite
                if (!delta_p_next.allFinite() || !delta_v_next.allFinite() || 
                    !delta_q_next.coeffs().allFinite() || 
                    !covariance.allFinite() || !jacobian_bias.allFinite()) {
                    ROS_WARN("Non-finite values during IMU integration");
                    result.is_valid = false;
                    return result;
                }
                
                // Update integration
                delta_p = delta_p_next;
                delta_v = delta_v_next;
                delta_q = delta_q_next;
                
                prev_time = curr_time;
            }
            
            // Ensure numerical stability of covariance
            // Add small values to diagonal to ensure positive definiteness
            for (int i = 0; i < 9; ++i) {
                covariance(i, i) += 1e-6;
            }
            
            // Store results
            result.delta_position = delta_p;
            result.delta_velocity = delta_v;
            result.delta_orientation = delta_q;
            result.covariance = covariance;
            result.jacobian_bias = jacobian_bias;
            
            return result;
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in integrateImuMeasurements: %s", e.what());
            ImuPreintegration result;
            result.delta_position = Eigen::Vector3d::Zero();
            result.delta_orientation = Eigen::Quaterniond::Identity();
            result.delta_velocity = Eigen::Vector3d::Zero();
            result.covariance = Eigen::Matrix<double, 9, 9>::Identity() * 1e-3;
            result.jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();
            result.start_time = start_time;
            result.end_time = end_time;
            result.acc_bias_ref = acc_bias;
            result.gyro_bias_ref = gyro_bias;
            result.is_valid = false;
            return result;
        }
    }

    // IMU pre-integration factor for Ceres
    class ImuFactor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ImuFactor(const ImuPreintegration& preint, const Eigen::Vector3d& gravity, 
                 double pos_weight = 10.0, double ori_weight = 10.0, 
                 double vel_weight = 10.0, double bias_weight = 5.0)
            : preint_(preint), gravity_(gravity), 
              pos_weight_(pos_weight), ori_weight_(ori_weight), 
              vel_weight_(vel_weight), bias_weight_(bias_weight) {}
        
        template <typename T>
        bool operator()(const T* const pose_i, const T* const vel_i, const T* const bias_i,
                       const T* const pose_j, const T* const vel_j, const T* const bias_j,
                       T* residuals) const {
            
            // Extract states
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pose_i);
            Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i(vel_i);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(bias_i);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(bias_i + 3);
            
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(pose_j);
            Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j(vel_j);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(bias_j);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(bias_j + 3);
            
            // Time delta
            T dt = T(preint_.end_time - preint_.start_time);
            
            // Convert pre-integrated measurements to T
            Eigen::Matrix<T, 3, 1> delta_p = preint_.delta_position.cast<T>();
            Eigen::Quaternion<T> delta_q = preint_.delta_orientation.cast<T>();
            Eigen::Matrix<T, 3, 1> delta_v = preint_.delta_velocity.cast<T>();
            
            // Reference biases used during pre-integration
            Eigen::Matrix<T, 3, 1> ba_ref = preint_.acc_bias_ref.cast<T>();
            Eigen::Matrix<T, 3, 1> bg_ref = preint_.gyro_bias_ref.cast<T>();
            
            // Bias corrections - use average for stability
            Eigen::Matrix<T, 3, 1> dba = (ba_i + ba_j) * T(0.5) - ba_ref;
            Eigen::Matrix<T, 3, 1> dbg = (bg_i + bg_j) * T(0.5) - bg_ref;
            
            // Apply first-order bias correction
            Eigen::Matrix<T, 9, 6> jacobian_bias = preint_.jacobian_bias.cast<T>();
            
            // Bias correction vector
            Eigen::Matrix<T, 6, 1> bias_correction_vec;
            bias_correction_vec.template segment<3>(0) = dba;
            bias_correction_vec.template segment<3>(3) = dbg;
            
            // Apply correction to the full state
            Eigen::Matrix<T, 9, 1> bias_correction = jacobian_bias * bias_correction_vec;
            
            // Apply corrections to delta_p and delta_v
            delta_p = delta_p + bias_correction.template segment<3>(0);
            delta_v = delta_v + bias_correction.template segment<3>(3);
            
            // Correction for delta_q using small angle approximation
            Eigen::Matrix<T, 3, 1> dq_correction = bias_correction.template segment<3>(6);
            Eigen::Quaternion<T> q_correction = Eigen::Quaternion<T>(
                T(1.0), dq_correction(0) / T(2.0), dq_correction(1) / T(2.0), dq_correction(2) / T(2.0)).normalized();
            delta_q = (delta_q * q_correction).normalized();
            
            // Gravity as T
            Eigen::Matrix<T, 3, 1> g = gravity_.cast<T>();
            
            // Predict state at time j
            Eigen::Matrix<T, 3, 1> p_j_pred = p_i + v_i * dt + T(0.5) * g * dt * dt + q_i * delta_p;
            Eigen::Quaternion<T> q_j_pred = (q_i * delta_q).normalized();
            Eigen::Matrix<T, 3, 1> v_j_pred = v_i + g * dt + q_i * delta_v;
            
            // Compute residuals
            Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residuals);
            
            // Position residual
            residual.template segment<3>(0) = p_j - p_j_pred;
            
            // Orientation residual using quaternion difference
            Eigen::Quaternion<T> q_error = q_j_pred.conjugate() * q_j;
            residual.template segment<3>(3) = T(2.0) * q_error.vec();
            
            // Velocity residual
            residual.template segment<3>(6) = v_j - v_j_pred;
            
            // Add bias consistency terms - bias should change slowly over time
            residual.template segment<3>(9) = ba_j - ba_i;
            residual.template segment<3>(12) = bg_j - bg_i;
            
            // Apply weights for better conditioning
            residual.template segment<3>(0) *= T(pos_weight_); 
            residual.template segment<3>(3) *= T(ori_weight_);
            residual.template segment<3>(6) *= T(vel_weight_);
            residual.template segment<3>(9) *= T(bias_weight_);
            residual.template segment<3>(12) *= T(bias_weight_);
            
            return true;
        }
        
        static ceres::CostFunction* Create(const ImuPreintegration& preint, const Eigen::Vector3d& gravity,
                                          double pos_weight = 10.0, double ori_weight = 10.0, 
                                          double vel_weight = 10.0, double bias_weight = 5.0) {
            return new ceres::AutoDiffCostFunction<ImuFactor, 15, 7, 3, 6, 7, 3, 6>(
                new ImuFactor(preint, gravity, pos_weight, ori_weight, vel_weight, bias_weight));
        }
        
    private:
        const ImuPreintegration preint_;
        const Eigen::Vector3d gravity_;
        const double pos_weight_;
        const double ori_weight_;
        const double vel_weight_;
        const double bias_weight_;
    };

    // UWB position factor for Ceres
    class UwbPositionFactor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        UwbPositionFactor(const Eigen::Vector3d& measured_position, double noise_xy, double noise_z)
            : measured_position_(measured_position), noise_xy_(noise_xy), noise_z_(noise_z) {}
        
        template <typename T>
        bool operator()(const T* const pose, T* residuals) const {
            // Extract position from pose (position + quaternion)
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(pose);
            
            // Compute position residuals - separate weighting for xy and z
            residuals[0] = (position[0] - T(measured_position_[0])) / T(noise_xy_);
            residuals[1] = (position[1] - T(measured_position_[1])) / T(noise_xy_);
            residuals[2] = (position[2] - T(measured_position_[2])) / T(noise_z_);
            
            return true;
        }
        
        static ceres::CostFunction* Create(const Eigen::Vector3d& measured_position, double noise_xy, double noise_z) {
            return new ceres::AutoDiffCostFunction<UwbPositionFactor, 3, 7>(
                new UwbPositionFactor(measured_position, noise_xy, noise_z));
        }
        
    private:
        const Eigen::Vector3d measured_position_;
        const double noise_xy_;
        const double noise_z_;
    };

    // State constraint factor - used when no IMU is available between states
    class StateConstraintFactor {
    public:
        StateConstraintFactor(double orientation_weight, double velocity_weight)
            : orientation_weight_(orientation_weight), velocity_weight_(velocity_weight) {}
        
        template <typename T>
        bool operator()(const T* const pose_i, const T* const vel_i, const T* const bias_i,
                      const T* const pose_j, const T* const vel_j, const T* const bias_j,
                      T* residuals) const {
            
            // Extract states
            Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i(vel_i);
            Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j(vel_j);
            
            // Residuals
            Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
            
            // Orientation should be similar (small rotation between consecutive states)
            Eigen::Quaternion<T> q_error = q_i.conjugate() * q_j;
            residual.template segment<3>(0) = T(orientation_weight_) * T(2.0) * q_error.vec();
            
            // Velocity should be similar (small acceleration between consecutive states)
            residual.template segment<3>(3) = T(velocity_weight_) * (v_j - v_i);
            
            return true;
        }
        
        static ceres::CostFunction* Create(double orientation_weight, double velocity_weight) {
            return new ceres::AutoDiffCostFunction<StateConstraintFactor, 6, 7, 3, 6, 7, 3, 6>(
                new StateConstraintFactor(orientation_weight, velocity_weight));
        }
        
    private:
        double orientation_weight_;
        double velocity_weight_;
    };
    
    bool optimizeFactorGraph() {
        if (state_window_.size() < 2) {
            ROS_WARN_THROTTLE(1.0, "Not enough states for optimization");
            return false;
        }
        
        // Create Ceres problem
        ceres::Problem problem;
        
        // Use a fresh parameterization for each optimization run
        // Important: Ceres will take ownership of this, so we don't delete it
        ceres::LocalParameterization* pose_parameterization = new PoseParameterization();
        
        // Structure for storing state variables for Ceres
        struct OptVariables {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            double pose[7]; // position (3) + quaternion (4)
            double velocity[3];
            double bias[6]; // acc_bias (3) + gyro_bias (3)
        };
        
        try {
            // Create a vector to store optimization variables 
            std::vector<OptVariables, Eigen::aligned_allocator<OptVariables>> variables(state_window_.size());
            
            // Initialize variables from state window
            for (size_t i = 0; i < state_window_.size(); ++i) {
                const auto& state = state_window_[i];
                
                // Position
                variables[i].pose[0] = state.position.x();
                variables[i].pose[1] = state.position.y();
                variables[i].pose[2] = state.position.z();
                
                // Orientation (quaternion): w, x, y, z
                variables[i].pose[3] = state.orientation.w();
                variables[i].pose[4] = state.orientation.x();
                variables[i].pose[5] = state.orientation.y();
                variables[i].pose[6] = state.orientation.z();
                
                // Velocity
                variables[i].velocity[0] = state.velocity.x();
                variables[i].velocity[1] = state.velocity.y();
                variables[i].velocity[2] = state.velocity.z();
                
                // Biases
                variables[i].bias[0] = state.acc_bias.x();
                variables[i].bias[1] = state.acc_bias.y();
                variables[i].bias[2] = state.acc_bias.z();
                variables[i].bias[3] = state.gyro_bias.x();
                variables[i].bias[4] = state.gyro_bias.y();
                variables[i].bias[5] = state.gyro_bias.z();
            }
            
            // Add pose parameterization for all pose variables
            for (size_t i = 0; i < state_window_.size(); ++i) {
                problem.AddParameterBlock(variables[i].pose, 7, pose_parameterization);
            }
            
            // If bias estimation is disabled, set biases constant
            if (!enable_bias_estimation_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    problem.SetParameterBlockConstant(variables[i].bias);
                }
            }
            
            // Add UWB position factors - more important and with higher weight
            size_t uwb_factors_added = 0;
            for (const auto& uwb : uwb_measurements_) {
                // Find the closest state in time
                size_t state_idx = 0;
                double min_time_diff = std::numeric_limits<double>::max();
                
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    double time_diff = std::abs(state_window_[i].timestamp - uwb.timestamp);
                    if (time_diff < min_time_diff) {
                        min_time_diff = time_diff;
                        state_idx = i;
                    }
                }
                
                // Skip if the measurement is too far in time from any state
                if (min_time_diff > 0.1) {
                    continue;
                }
                
                try {
                    // Use different noise for horizontal and vertical components
                    // Lower noise (= higher weight) for UWB
                    double noise_xy = uwb_position_noise_;
                    double noise_z = uwb_position_noise_ * 0.5; // Higher weight for Z
                    
                    ceres::CostFunction* uwb_factor = UwbPositionFactor::Create(
                        uwb.position, noise_xy, noise_z);
                    
                    // Use HuberLoss to handle outliers
                    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.5); // Smaller parameter = more robust
                    problem.AddResidualBlock(uwb_factor, loss_function, variables[state_idx].pose);
                    uwb_factors_added++;
                } catch (const std::exception& e) {
                    ROS_ERROR("Exception while adding UWB factor: %s", e.what());
                }
            }
            
            if (uwb_factors_added == 0) {
                ROS_WARN("No UWB factors could be added to the optimization problem");
                return false;
            }
            
            // Add roll/pitch prior to all states
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* roll_pitch_prior = RollPitchPriorFactor::Create(roll_pitch_weight_);
                problem.AddResidualBlock(roll_pitch_prior, nullptr, variables[i].pose);
            }
            
            // Add bias prior to all states for regularization
            if (enable_bias_estimation_) {
                // Lower sigma (= higher weight) to regularize biases more strongly
                double acc_bias_sigma = 0.2;  // Tighter constraint on biases
                double gyro_bias_sigma = 0.05;
                
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* bias_prior = BiasPriorFactor::Create(
                        initial_acc_bias_, initial_gyro_bias_, acc_bias_sigma, gyro_bias_sigma);
                    problem.AddResidualBlock(bias_prior, nullptr, variables[i].bias);
                }
            }
            
            // Add IMU pre-integration factors
            bool added_imu_factors = false;
            
            for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                double start_time = state_window_[i].timestamp;
                double end_time = state_window_[i+1].timestamp;
                
                // Skip if the time interval is too short
                if (end_time - start_time < 1e-6) continue;
                
                Eigen::Vector3d acc_bias = state_window_[i].acc_bias;
                Eigen::Vector3d gyro_bias = state_window_[i].gyro_bias;
                
                ImuPreintegration preint = integrateImuMeasurements(start_time, end_time, acc_bias, gyro_bias);
                
                // Only add IMU factor if we have valid IMU measurements for this interval
                if (preint.is_valid) {
                    try {
                        // Increased weights for orientation to better constrain roll/pitch
                        double pos_weight = 10.0;
                        double ori_weight = 15.0;  // Increased from 10.0
                        double vel_weight = 10.0;
                        double bias_weight = 10.0;  // Increased from 5.0
                        
                        ceres::CostFunction* imu_factor = ImuFactor::Create(
                            preint, gravity_world_, pos_weight, ori_weight, vel_weight, bias_weight);
                        
                        problem.AddResidualBlock(imu_factor, nullptr,
                                               variables[i].pose, variables[i].velocity, variables[i].bias,
                                               variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                        added_imu_factors = true;
                        
                        // Add bias random walk constraint if bias estimation is enabled
                        if (enable_bias_estimation_) {
                            // Bias should change smoothly between consecutive states
                            // Allow flexibility for bias changes but constrain them
                            double acc_bias_sigma = imu_acc_bias_noise_ * sqrt(end_time - start_time) * 5.0;
                            double gyro_bias_sigma = imu_gyro_bias_noise_ * sqrt(end_time - start_time) * 5.0;
                            
                            // Looser constraints on bias changes
                            acc_bias_sigma = std::max(acc_bias_sigma, 0.005);   // Increased from 0.002
                            gyro_bias_sigma = std::max(gyro_bias_sigma, 0.002);  // Increased from 0.001
                            
                            ceres::CostFunction* bias_walk = BiasRandomWalkFactor::Create(
                                acc_bias_sigma, gyro_bias_sigma);
                            
                            problem.AddResidualBlock(bias_walk, nullptr, 
                                                   variables[i].bias, variables[i+1].bias);
                        }
                    } catch (const std::exception& e) {
                        ROS_ERROR("Exception while adding IMU factor: %s", e.what());
                    }
                } else {
                    // If no IMU data, add a constraint to maintain reasonable state transitions
                    try {
                        ceres::CostFunction* state_constraint = StateConstraintFactor::Create(5.0, 5.0);
                        problem.AddResidualBlock(state_constraint, nullptr,
                                               variables[i].pose, variables[i].velocity, variables[i].bias,
                                               variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                    } catch (const std::exception& e) {
                        ROS_ERROR("Exception while adding state constraint: %s", e.what());
                    }
                }
            }
            
            // Anchor the first and last states with UWB if available
            if (!uwb_measurements_.empty()) {
                // Find UWB closest to first state
                const UwbMeasurement* first_uwb = nullptr;
                double min_first_time_diff = 0.5; // Relaxed time difference threshold
                
                // Find UWB closest to last state
                const UwbMeasurement* last_uwb = nullptr;
                double min_last_time_diff = 0.5;
                
                for (const auto& uwb : uwb_measurements_) {
                    // For first state
                    double first_time_diff = std::abs(state_window_.front().timestamp - uwb.timestamp);
                    if (first_time_diff < min_first_time_diff) {
                        min_first_time_diff = first_time_diff;
                        first_uwb = &uwb;
                    }
                    
                    // For last state
                    double last_time_diff = std::abs(state_window_.back().timestamp - uwb.timestamp);
                    if (last_time_diff < min_last_time_diff) {
                        min_last_time_diff = last_time_diff;
                        last_uwb = &uwb;
                    }
                }
                
                // Add strong prior to first state if UWB available
                if (first_uwb) {
                    double noise_xy = uwb_position_noise_ * 0.5; // Even stronger weight
                    double noise_z = uwb_position_noise_ * 0.25;
                    
                    ceres::CostFunction* first_uwb_factor = UwbPositionFactor::Create(
                        first_uwb->position, noise_xy, noise_z);
                    
                    problem.AddResidualBlock(first_uwb_factor, nullptr, variables[0].pose);
                }
                
                // Add strong prior to last state if UWB available
                if (last_uwb) {
                    double noise_xy = uwb_position_noise_ * 0.5;
                    double noise_z = uwb_position_noise_ * 0.25;
                    
                    ceres::CostFunction* last_uwb_factor = UwbPositionFactor::Create(
                        last_uwb->position, noise_xy, noise_z);
                    
                    problem.AddResidualBlock(last_uwb_factor, nullptr, variables[state_window_.size()-1].pose);
                }
            }
            
            // Configure solver
            ceres::Solver::Options options;
            options.max_num_iterations = max_iterations_;
            options.linear_solver_type = ceres::DENSE_QR; // More stable for small problems
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 2;
            options.function_tolerance = 1e-5;
            options.gradient_tolerance = 1e-5;
            options.parameter_tolerance = 1e-5;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // More stable for our problem
            
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            if (!summary.IsSolutionUsable()) {
                ROS_WARN("Optimization failed: %s", summary.BriefReport().c_str());
                return false;
            }
            
            // Debug outputs
            std::ostringstream debug_str;
            debug_str << "Initial biases: acc=[" << initial_acc_bias_.x() << ", " << initial_acc_bias_.y() 
                     << ", " << initial_acc_bias_.z() << "], gyro=[" << initial_gyro_bias_.x() << ", " 
                     << initial_gyro_bias_.y() << ", " << initial_gyro_bias_.z() << "]";
            ROS_INFO_THROTTLE(5.0, "%s", debug_str.str().c_str());
            
            // Update state with optimized values
            for (size_t i = 0; i < state_window_.size(); ++i) {
                // Update position
                state_window_[i].position = Eigen::Vector3d(
                    variables[i].pose[0], variables[i].pose[1], variables[i].pose[2]);
                
                // Update orientation
                state_window_[i].orientation = Eigen::Quaterniond(
                    variables[i].pose[3], variables[i].pose[4], variables[i].pose[5], variables[i].pose[6]).normalized();
                
                // Update velocity
                state_window_[i].velocity = Eigen::Vector3d(
                    variables[i].velocity[0], variables[i].velocity[1], variables[i].velocity[2]);
                
                // Update biases if enabled
                if (enable_bias_estimation_) {
                    state_window_[i].acc_bias = Eigen::Vector3d(
                        variables[i].bias[0], variables[i].bias[1], variables[i].bias[2]);
                    
                    state_window_[i].gyro_bias = Eigen::Vector3d(
                        variables[i].bias[3], variables[i].bias[4], variables[i].bias[5]);
                }
            }
            
            // Verify state values are finite
            for (const auto& state : state_window_) {
                if (!isStateValid(state)) {
                    ROS_ERROR("Non-finite state values after optimization!");
                    return false;
                }
            }
            
            // Update current state to the latest state in the window
            if (!state_window_.empty()) {
                current_state_ = state_window_.back();
                
                // Update initial biases for future initializations
                if (enable_bias_estimation_) {
                    initial_acc_bias_ = current_state_.acc_bias;
                    initial_gyro_bias_ = current_state_.gyro_bias;
                }
            }
            
            // Clear old measurements, but keep a few recent ones
            if (uwb_measurements_.size() > 5) {
                uwb_measurements_.erase(uwb_measurements_.begin(), uwb_measurements_.end() - 5);
            }
            
            // Convert orientation to RPY in degrees for logging
            Eigen::Vector3d orientation_deg = quaternionToEulerDegrees(current_state_.orientation);
            
            ROS_INFO_THROTTLE(1.0, "Optimization complete with %zu iterations. Position: [%f, %f, %f], Orientation (RPY): [%.1f, %.1f, %.1f], Acc bias: [%f, %f, %f], Gyro bias: [%f, %f, %f]",
                     summary.iterations.size(), 
                     current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                     orientation_deg.x(), orientation_deg.y(), orientation_deg.z(),
                     current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
                     current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
                     
            return true;
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during optimization: %s", e.what());
            return false;
        }
    }

    void publishState() {
        try {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(current_state_.timestamp);
            odom_msg.header.frame_id = world_frame_id_;
            odom_msg.child_frame_id = body_frame_id_;
            
            // Position
            odom_msg.pose.pose.position.x = current_state_.position.x();
            odom_msg.pose.pose.position.y = current_state_.position.y();
            odom_msg.pose.pose.position.z = current_state_.position.z();
            
            // Orientation
            odom_msg.pose.pose.orientation.w = current_state_.orientation.w();
            odom_msg.pose.pose.orientation.x = current_state_.orientation.x();
            odom_msg.pose.pose.orientation.y = current_state_.orientation.y();
            odom_msg.pose.pose.orientation.z = current_state_.orientation.z();
            
            // Velocity
            odom_msg.twist.twist.linear.x = current_state_.velocity.x();
            odom_msg.twist.twist.linear.y = current_state_.velocity.y();
            odom_msg.twist.twist.linear.z = current_state_.velocity.z();
            
            // Simple diagonal covariance
            odom_msg.pose.covariance[0] = 0.01;  // x
            odom_msg.pose.covariance[7] = 0.01;  // y
            odom_msg.pose.covariance[14] = 0.01; // z
            odom_msg.pose.covariance[21] = 0.01; // roll
            odom_msg.pose.covariance[28] = 0.01; // pitch
            odom_msg.pose.covariance[35] = 0.01; // yaw
            
            // Publish the message
            pose_pub_.publish(odom_msg);
            
            // Print orientation in degrees for debugging
            Eigen::Vector3d orientation_deg = quaternionToEulerDegrees(current_state_.orientation);
            ROS_INFO_THROTTLE(1.0, "Published state: Position [%.2f, %.2f, %.2f], Orientation (RPY) [%.1f°, %.1f°, %.1f°]",
                      current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                      orientation_deg.x(), orientation_deg.y(), orientation_deg.z());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishState: %s", e.what());
        }
    }

    // Skew-symmetric matrix helper
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d skew;
        skew << 0, -v(2), v(1),
                v(2), 0, -v(0),
               -v(1), v(0), 0;
        return skew;
    }
};

int main(int argc, char **argv) {
    try {
        ros::init(argc, argv, "uwb_imu_fusion_node");
        
        {
            UwbImuFusion fusion;
            ros::spin();
        }
        
        ROS_INFO("UWB-IMU fusion node shutting down normally");
        return 0;
    } catch (const std::exception& e) {
        ROS_ERROR("Fatal exception in main: %s", e.what());
        return 1;
    } catch (...) {
        ROS_ERROR("Unknown fatal exception in main");
        return 1;
    }
}