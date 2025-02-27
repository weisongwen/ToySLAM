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
#include <cmath>
#include <limits>
#include <chrono> // For timing the optimization

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

// New factor to dampen high velocities
class VelocityDampingFactor {
public:
    VelocityDampingFactor(double max_velocity, double weight) 
        : max_velocity_(max_velocity), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const velocity, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vel(velocity);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
        
        // Calculate velocity magnitude
        T vel_magnitude = vel.norm();
        
        // If velocity exceeds max threshold, add a penalty
        if (vel_magnitude > T(max_velocity_)) {
            // Residual is the amount by which velocity exceeds the threshold
            residual = T(weight_) * (vel - vel * T(max_velocity_) / vel_magnitude);
        } else {
            residual.setZero();
        }
        
        return true;
    }
    
    static ceres::CostFunction* Create(double max_velocity, double weight) {
        return new ceres::AutoDiffCostFunction<VelocityDampingFactor, 3, 3>(
            new VelocityDampingFactor(max_velocity, weight));
    }
    
private:
    double max_velocity_;
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
        private_nh.param<double>("imu_acc_noise", imu_acc_noise_, 0.05);
        private_nh.param<double>("imu_gyro_noise", imu_gyro_noise_, 0.01);
        private_nh.param<double>("imu_acc_bias_noise", imu_acc_bias_noise_, 0.005);
        private_nh.param<double>("imu_gyro_bias_noise", imu_gyro_bias_noise_, 0.001);
        private_nh.param<double>("uwb_position_noise", uwb_position_noise_, 0.1);
        private_nh.param<int>("optimization_window_size", optimization_window_size_, 12);
        private_nh.param<std::string>("world_frame_id", world_frame_id_, "world");
        private_nh.param<std::string>("body_frame_id", body_frame_id_, "base_link");
        private_nh.param<double>("optimization_frequency", optimization_frequency_, 10.0);
        private_nh.param<double>("imu_buffer_time_length", imu_buffer_time_length_, 10.0);
        private_nh.param<int>("max_iterations", max_iterations_, 15);
        private_nh.param<bool>("enable_bias_estimation", enable_bias_estimation_, true);
        private_nh.param<double>("roll_pitch_weight", roll_pitch_weight_, 15.0);
        private_nh.param<double>("imu_pose_pub_frequency", imu_pose_pub_frequency_, 100.0);
        private_nh.param<double>("max_imu_dt", max_imu_dt_, 0.5);
        private_nh.param<double>("max_velocity", max_velocity_, 5.0);
        private_nh.param<double>("velocity_damping_weight", velocity_damping_weight_, 8.0);
        
        // Initialize with small non-zero biases to help break symmetry in the optimization
        initial_acc_bias_ = Eigen::Vector3d(0.05, 0.05, 0.05);
        initial_gyro_bias_ = Eigen::Vector3d(0.01, 0.01, 0.01);
        
        // Initialize subscribers and publishers
        imu_sub_ = nh.subscribe("/sensor_simulator/imu_data", 1000, &UwbImuFusion::imuCallback, this);
        uwb_sub_ = nh.subscribe("/sensor_simulator/UWBPoistionPS", 100, &UwbImuFusion::uwbCallback, this);
        optimized_pose_pub_ = nh.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/optimized_pose", 10);
        imu_pose_pub_ = nh.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/imu_pose", 200);
        
        // Initialize state
        is_initialized_ = false;
        has_imu_data_ = false;
        last_imu_timestamp_ = 0;
        last_processed_timestamp_ = 0;
        just_optimized_ = false;
        
        initializeState();
        
        // Setup optimization timer
        optimization_timer_ = nh.createTimer(ros::Duration(1.0/optimization_frequency_), 
                                           &UwbImuFusion::optimizationTimerCallback, this);
        
        // Setup high-frequency IMU pose publisher timer
        imu_pose_pub_timer_ = nh.createTimer(ros::Duration(1.0/imu_pose_pub_frequency_), 
                                           &UwbImuFusion::imuPoseTimerCallback, this);
        
        ROS_INFO("UWB-IMU Fusion node initialized. Using factor graph optimization with max %d iterations", max_iterations_);
        ROS_INFO("Bias estimation is %s", enable_bias_estimation_ ? "enabled" : "disabled");
        ROS_INFO("UWB noise: %.5f", uwb_position_noise_);
        ROS_INFO("IMU acc noise: %.5f, gyro noise: %.5f", imu_acc_noise_, imu_gyro_noise_);
        ROS_INFO("IMU acc bias noise: %.5f, gyro bias noise: %.5f", imu_acc_bias_noise_, imu_gyro_bias_noise_);
        ROS_INFO("Roll/pitch prior weight: %.2f", roll_pitch_weight_);
        ROS_INFO("IMU pose publishing frequency: %.1f Hz", imu_pose_pub_frequency_);
        ROS_INFO("Maximum allowed IMU dt: %.3f seconds", max_imu_dt_);
        ROS_INFO("Maximum expected velocity: %.1f m/s with damping weight %.1f", max_velocity_, velocity_damping_weight_);
        ROS_INFO("Window size for optimization: %d", optimization_window_size_);
    }

private:
    // ROS subscribers and publishers
    ros::Subscriber imu_sub_;
    ros::Subscriber uwb_sub_;
    ros::Publisher optimized_pose_pub_;
    ros::Publisher imu_pose_pub_;
    ros::Timer optimization_timer_;
    ros::Timer imu_pose_pub_timer_;

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
    int max_iterations_;
    bool enable_bias_estimation_;
    std::string world_frame_id_;
    std::string body_frame_id_;
    double roll_pitch_weight_;
    double imu_pose_pub_frequency_;
    double max_imu_dt_;
    double max_velocity_;
    double velocity_damping_weight_;
    
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
    double last_imu_timestamp_;
    double last_processed_timestamp_;
    bool just_optimized_;
    
    // NEW: IMU Preintegration between keyframes
    struct ImuPreintegrationBetweenKeyframes {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        // Pre-integrated IMU measurements
        Eigen::Vector3d delta_position;
        Eigen::Quaterniond delta_orientation;
        Eigen::Vector3d delta_velocity;
        
        // Covariance and Jacobians
        Eigen::Matrix<double, 9, 9> covariance;
        Eigen::Matrix<double, 9, 6> jacobian_bias;
        
        // Reference biases
        Eigen::Vector3d acc_bias_ref;
        Eigen::Vector3d gyro_bias_ref;
        
        // Keyframe timestamps for this preintegration
        double start_time;
        double end_time;
        double sum_dt;
        
        // Store entire IMU measurements for potential reintegration after bias updates
        std::vector<sensor_msgs::Imu> imu_measurements;
        
        ImuPreintegrationBetweenKeyframes() {
            reset();
        }
        
        void reset() {
            delta_position.setZero();
            delta_orientation = Eigen::Quaterniond::Identity();
            delta_velocity.setZero();
            covariance = Eigen::Matrix<double, 9, 9>::Identity() * 1e-8;
            jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();
            acc_bias_ref.setZero();
            gyro_bias_ref.setZero();
            start_time = 0;
            end_time = 0;
            sum_dt = 0;
            imu_measurements.clear();
        }
    };
    
    // NEW: Map to store preintegration data between consecutive keyframes
    std::map<std::pair<double, double>, ImuPreintegrationBetweenKeyframes> preintegration_map_;
    
    // IMU pre-integration structure (for on-the-fly integration, not for optimization)
    struct ImuPreintegration {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        // Pre-integrated IMU measurements
        Eigen::Vector3d delta_position;
        Eigen::Quaterniond delta_orientation;
        Eigen::Vector3d delta_velocity;
        
        // Covariance and Jacobians
        Eigen::Matrix<double, 9, 9> covariance;
        Eigen::Matrix<double, 9, 6> jacobian_bias;
        
        // Timestamps 
        double start_time;
        double end_time;
        double sum_dt;
        
        // Reference biases
        Eigen::Vector3d acc_bias_ref;
        Eigen::Vector3d gyro_bias_ref;
        
        bool is_valid;
        
        struct ImuMeasurement {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            double timestamp;
            Eigen::Vector3d linear_acceleration;
            Eigen::Vector3d angular_velocity;
        };
        std::vector<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>> imu_measurements;
        
        ImuPreintegration() {
            reset();
        }
        
        void reset() {
            delta_position.setZero();
            delta_orientation = Eigen::Quaterniond::Identity();
            delta_velocity.setZero();
            covariance = Eigen::Matrix<double, 9, 9>::Identity() * 1e-8;
            jacobian_bias = Eigen::Matrix<double, 9, 6>::Zero();
            start_time = 0;
            end_time = 0;
            sum_dt = 0;
            acc_bias_ref.setZero();
            gyro_bias_ref.setZero();
            is_valid = false;
            imu_measurements.clear();
        }
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

    // Gravity vector in world frame (consistent definition)
    // IMPORTANT: We'll define this as pointing DOWN along the Z axis (negative Z)
    // but we need to remember that IMU measurements have gravity along POSITIVE Z
    Eigen::Vector3d gravity_world_;
    
    // Parameterization that can be reused between optimizations
    static ceres::LocalParameterization* pose_parameterization_;
    
    // Helper functions
    
    // Compute quaternion for small angle rotation - overloaded for different types
    template <typename T>
    static Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
        T theta_norm = theta.norm();
        
        Eigen::Quaternion<T> dq;
        if (theta_norm > T(1e-5)) {
            Eigen::Matrix<T, 3, 1> a = theta / theta_norm;
            dq = Eigen::Quaternion<T>(cos(theta_norm / T(2.0)), 
                                       a.x() * sin(theta_norm / T(2.0)),
                                       a.y() * sin(theta_norm / T(2.0)),
                                       a.z() * sin(theta_norm / T(2.0)));
        } else {
            dq = Eigen::Quaternion<T>(T(1.0), theta.x() / T(2.0), theta.y() / T(2.0), theta.z() / T(2.0));
            dq.normalize();
        }
        return dq;
    }
    
    // Non-template version for direct use with double type
    static Eigen::Quaterniond deltaQ(const Eigen::Vector3d& theta) {
        double theta_norm = theta.norm();
        
        Eigen::Quaterniond dq;
        if (theta_norm > 1e-5) {
            Eigen::Vector3d a = theta / theta_norm;
            dq = Eigen::Quaterniond(cos(theta_norm / 2.0), 
                                    a.x() * sin(theta_norm / 2.0),
                                    a.y() * sin(theta_norm / 2.0),
                                    a.z() * sin(theta_norm / 2.0));
        } else {
            dq = Eigen::Quaterniond(1.0, theta.x() / 2.0, theta.y() / 2.0, theta.z() / 2.0);
            dq.normalize();
        }
        return dq;
    }
    
    // Skew-symmetric matrix helper
    template <typename T>
    static Eigen::Matrix<T, 3, 3> skewSymmetric(const Eigen::Matrix<T, 3, 1>& v) {
        Eigen::Matrix<T, 3, 3> skew;
        skew << T(0), -v(2), v(1),
                v(2), T(0), -v(0),
               -v(1), v(0), T(0);
        return skew;
    }
    
    // Non-template version for double
    static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d skew;
        skew << 0, -v(2), v(1),
                v(2), 0, -v(0),
               -v(1), v(0), 0;
        return skew;
    }

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
            preintegration_map_.clear();
            
            // Initialize gravity vector consistently - pointing down in Z
            // This is used for world-frame calculations
            gravity_world_ = Eigen::Vector3d(0, 0, -gravity_magnitude_);
            
            // Reset timestamp tracking
            last_imu_timestamp_ = 0;
            last_processed_timestamp_ = 0;
            just_optimized_ = false;
            
            ROS_INFO("State initialized with gravity pointing DOWN along Z-axis in world frame: [0, 0, -%.2f]", gravity_magnitude_);
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
            
            double timestamp = msg->header.stamp.toSec();
            
            // Store IMU measurements for pre-integration
            imu_buffer_.push_back(*msg);
            
            // Critical check - don't process messages with duplicate timestamps
            // or timestamps that are earlier than what we've already processed
            if (timestamp <= last_processed_timestamp_) {
                return;
            }
            
            // Update tracking timestamps
            last_imu_timestamp_ = timestamp;
            last_processed_timestamp_ = timestamp;
            
            // Process IMU data for real-time state propagation (not for keyframes)
            if (is_initialized_) {
                propagateStateWithImu(*msg);
            }
            
            // Clean up old IMU messages - more efficient algorithm
            if (imu_buffer_.size() > 1000) { // Only clean when buffer is large
                double oldest_allowed_time = ros::Time::now().toSec() - 2.0 * imu_buffer_time_length_;
                while (!imu_buffer_.empty() && imu_buffer_.front().header.stamp.toSec() < oldest_allowed_time) {
                    imu_buffer_.pop_front();
                }
            }
            
            // Keep buffer size reasonable but larger to prevent data loss
            if (imu_buffer_.size() > 10000) {
                imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + 5000);
                ROS_WARN("IMU buffer very large (%zu), removing oldest 5000 measurements", imu_buffer_.size());
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
            
            // Keep uwb_measurements_ limited to a reasonable size - only when necessary
            if (uwb_measurements_.size() > 100) {
                uwb_measurements_.erase(uwb_measurements_.begin(), uwb_measurements_.begin() + 50);
            }
            
            // Add to UWB measurements list
            uwb_measurements_.push_back(measurement);
            
            // Initialize if not yet initialized
            if (!is_initialized_) {
                ROS_INFO("Initializing system with UWB measurement at timestamp: %f", measurement.timestamp);
                initializeFromUwb(measurement);
                is_initialized_ = true;
                return;
            }
            
            // NEW: Create a new keyframe at each UWB measurement
            if (is_initialized_ && has_imu_data_) {
                createKeyframe(measurement);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in uwbCallback: %s", e.what());
        }
    }
    
    // NEW: Create a keyframe (state) based on UWB measurement
    void createKeyframe(const UwbMeasurement& uwb) {
        try {
            // Skip if we already have a keyframe at this time (within a small tolerance)
            for (const auto& state : state_window_) {
                if (std::abs(state.timestamp - uwb.timestamp) < 0.005) {
                    return; // Already have a keyframe for this UWB measurement
                }
            }
            
            // Skip if the state window is empty
            if (state_window_.empty()) {
                // Create initial keyframe from current state
                State new_state = current_state_;
                new_state.position = uwb.position; // Use UWB position directly
                new_state.timestamp = uwb.timestamp;
                state_window_.push_back(new_state);
                ROS_INFO("Added first UWB-based keyframe at t=%.3f", uwb.timestamp);
                return;
            }
            
            // Calculate the time difference from the previous keyframe
            double dt = uwb.timestamp - state_window_.back().timestamp;
            
            // Skip if the time difference is too small
            if (dt < 0.01) {
                return;
            }
            
            // Log the time interval between keyframes (but not too frequently)
            static double last_log_time = 0;
            if (ros::Time::now().toSec() - last_log_time > 1.0) {
                ROS_INFO_THROTTLE(2.0, "Time interval between keyframes: %.3f seconds", dt);
                last_log_time = ros::Time::now().toSec();
            }
            
            // Compute the propagated state at the UWB timestamp
            State propagated_state = propagateState(state_window_.back(), uwb.timestamp);
            
            // Set the UWB position while keeping the propagated orientation and velocity
            propagated_state.position = uwb.position;
            propagated_state.timestamp = uwb.timestamp;
            
            // Add to state window, managing window size
            if (state_window_.size() >= optimization_window_size_) {
                state_window_.pop_front();
            }
            state_window_.push_back(propagated_state);
            
            // Propagate the current state to the latest time for continuous state tracking
            current_state_ = propagated_state;
            
            // Update preintegration between the last two keyframes
            if (state_window_.size() >= 2) {
                size_t n = state_window_.size();
                double start_time = state_window_[n-2].timestamp;
                double end_time = state_window_[n-1].timestamp;
                
                // Store preintegration data for optimization
                performPreintegrationBetweenKeyframes(start_time, end_time, state_window_[n-2].acc_bias, state_window_[n-2].gyro_bias);
            }
            
            // Throttle logging to avoid log spam
            ROS_INFO_THROTTLE(1.0, "Added new UWB-based keyframe at t=%.3f, window size: %zu", uwb.timestamp, state_window_.size());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in createKeyframe: %s", e.what());
        }
    }
    
    // NEW: Propagate state from a reference state to a target time using IMU data
    State propagateState(const State& reference_state, double target_time) {
        State result = reference_state;
        
        // If target_time is earlier than reference time, just return the reference state
        if (target_time <= reference_state.timestamp) {
            return reference_state;
        }
        
        // Find IMU measurements between reference_state.timestamp and target_time
        std::vector<sensor_msgs::Imu> relevant_imu_msgs;
        relevant_imu_msgs.reserve(100); // Reasonable pre-allocation
        
        for (const auto& imu : imu_buffer_) {
            double timestamp = imu.header.stamp.toSec();
            if (timestamp > reference_state.timestamp && timestamp <= target_time) {
                relevant_imu_msgs.push_back(imu);
            }
        }
        
        // Sort by timestamp just to be safe (with our IMU buffer this should be unnecessary)
        if (relevant_imu_msgs.size() > 1) {
            std::sort(relevant_imu_msgs.begin(), relevant_imu_msgs.end(), 
                     [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                         return a.header.stamp.toSec() < b.header.stamp.toSec();
                     });
        }
        
        // Propagate state using IMU messages
        double prev_time = reference_state.timestamp;
        
        for (const auto& imu_msg : relevant_imu_msgs) {
            double timestamp = imu_msg.header.stamp.toSec();
            double dt = timestamp - prev_time;
            
            // Skip invalid dt
            if (dt <= 0 || dt > max_imu_dt_) {
                prev_time = timestamp;
                continue;
            }
            
            // Extract IMU data
            Eigen::Vector3d acc(imu_msg.linear_acceleration.x,
                                imu_msg.linear_acceleration.y,
                                imu_msg.linear_acceleration.z);
            
            Eigen::Vector3d gyro(imu_msg.angular_velocity.x,
                                 imu_msg.angular_velocity.y,
                                 imu_msg.angular_velocity.z);
            
            // Correct for biases
            Eigen::Vector3d acc_corrected = acc - result.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
            
            // Update orientation
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            Eigen::Quaterniond orientation_new = (result.orientation * dq).normalized();
            
            // Remove gravity in sensor frame
            Eigen::Vector3d sensor_gravity(0, 0, gravity_magnitude_);
            Eigen::Vector3d acc_without_gravity = acc_corrected - sensor_gravity;
            
            // Rotate to world frame using midpoint rotation
            Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
            Eigen::Quaterniond orientation_mid = result.orientation * deltaQ(half_angle_axis);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity and position
            Eigen::Vector3d velocity_new = result.velocity + acc_world * dt;
            Eigen::Vector3d position_new = result.position + 0.5 * (result.velocity + velocity_new) * dt;
            
            // Update state
            result.orientation = orientation_new;
            result.velocity = velocity_new;
            result.position = position_new;
            result.timestamp = timestamp;
            
            prev_time = timestamp;
        }
        
        // If we didn't propagate all the way to target_time, do one final integration step
        if (prev_time < target_time && !relevant_imu_msgs.empty()) {
            double dt = target_time - prev_time;
            
            // Use the last IMU measurement for prediction
            const auto& last_imu = relevant_imu_msgs.back();
            
            Eigen::Vector3d acc(last_imu.linear_acceleration.x,
                               last_imu.linear_acceleration.y,
                               last_imu.linear_acceleration.z);
            
            Eigen::Vector3d gyro(last_imu.angular_velocity.x,
                                last_imu.angular_velocity.y,
                                last_imu.angular_velocity.z);
            
            // Correct for biases
            Eigen::Vector3d acc_corrected = acc - result.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
            
            // Update orientation
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            Eigen::Quaterniond orientation_new = (result.orientation * dq).normalized();
            
            // Remove gravity in sensor frame
            Eigen::Vector3d sensor_gravity(0, 0, gravity_magnitude_);
            Eigen::Vector3d acc_without_gravity = acc_corrected - sensor_gravity;
            
            // Rotate to world frame
            Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
            Eigen::Quaterniond orientation_mid = result.orientation * deltaQ(half_angle_axis);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity and position
            Eigen::Vector3d velocity_new = result.velocity + acc_world * dt;
            Eigen::Vector3d position_new = result.position + 0.5 * (result.velocity + velocity_new) * dt;
            
            // Update state
            result.orientation = orientation_new;
            result.velocity = velocity_new;
            result.position = position_new;
            result.timestamp = target_time;
        }
        
        // Limit velocity if needed
        double velocity_magnitude = result.velocity.norm();
        if (velocity_magnitude > max_velocity_) {
            result.velocity *= (max_velocity_ / velocity_magnitude);
        }
        
        return result;
    }
    
    // Real-time state propagation with IMU for visualization/feedback
    void propagateStateWithImu(const sensor_msgs::Imu& imu_msg) {
        try {
            double timestamp = imu_msg.header.stamp.toSec();
            
            // Special handling if we just ran optimization
            if (just_optimized_) {
                // Just update timestamp without integration
                current_state_.timestamp = timestamp;
                just_optimized_ = false;
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
            
            // Calculate time difference
            double dt = 0;
            if (current_state_.timestamp > 0) {
                dt = timestamp - current_state_.timestamp;
            } else {
                // First IMU measurement after initialization
                current_state_.timestamp = timestamp;
                return;  // Skip integration for the first IMU message
            }
            
            // Skip integration for invalid dt
            if (dt <= 0 || dt > max_imu_dt_) {
                current_state_.timestamp = timestamp;
                return;
            }
            
            // Correct for biases
            Eigen::Vector3d acc_corrected = acc - current_state_.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - current_state_.gyro_bias;
            
            // Update orientation using gyro - integrate using midpoint rule
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            
            // Update orientation
            Eigen::Quaterniond orientation_new = (current_state_.orientation * dq).normalized();
            
            // CRITICAL: In your IMU measurements, gravity is reported along +Z axis in sensor frame
            // We need to define the sensor-frame gravity vector and then subtract it
            Eigen::Vector3d sensor_gravity(0, 0, gravity_magnitude_); // POSITIVE Z in sensor frame
            
            // Remove gravity in sensor frame FIRST
            Eigen::Vector3d acc_without_gravity = acc_corrected - sensor_gravity;
            
            // Rotate acceleration to world frame using midpoint rotation
            Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
            Eigen::Quaterniond orientation_mid = current_state_.orientation * deltaQ(half_angle_axis);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity and position (using midpoint integration)
            Eigen::Vector3d velocity_new = current_state_.velocity + acc_world * dt;
            
            // Trapezoidal integration for position
            Eigen::Vector3d position_new = current_state_.position + 
                                          0.5 * (current_state_.velocity + velocity_new) * dt;
            
            // Update state
            current_state_.orientation = orientation_new;
            current_state_.velocity = velocity_new;
            current_state_.position = position_new;
            current_state_.timestamp = timestamp;
            
            // Ensure state values are finite
            if (!isStateValid(current_state_)) {
                ROS_WARN("Non-finite state values after IMU integration. Resetting to previous state.");
                if (!state_window_.empty()) {
                    current_state_ = state_window_.back();
                }
                return;
            }
            
            // Apply velocity limit (if enabled)
            double velocity_magnitude = current_state_.velocity.norm();
            if (velocity_magnitude > max_velocity_) {
                current_state_.velocity *= (max_velocity_ / velocity_magnitude);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in propagateStateWithImu: %s", e.what());
        }
    }

    // Timer callback for high-frequency IMU pose publishing
    void imuPoseTimerCallback(const ros::TimerEvent& event) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            if (is_initialized_ && has_imu_data_) {
                publishImuPose();
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in imuPoseTimerCallback: %s", e.what());
        }
    }

    void optimizationTimerCallback(const ros::TimerEvent& event) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            if (!is_initialized_) {
                ROS_WARN_THROTTLE(5.0, "System not yet initialized, waiting for first UWB measurement");
                return;
            }
            
            // Need at least 2 states for optimization
            if (state_window_.size() < 2) {
                ROS_WARN_THROTTLE(5.0, "Not enough keyframes for optimization, need at least 2 but have %zu", state_window_.size());
                return;
            }
            
            // Throttle info log
            ROS_INFO_THROTTLE(2.0, "Performing optimization with %zu keyframes and %zu UWB measurements", 
                              state_window_.size(), uwb_measurements_.size());
            
            // Check for extreme velocity
            double velocity_magnitude = current_state_.velocity.norm();
            if (velocity_magnitude > 2.0 * max_velocity_) {
                ROS_WARN("Extreme velocity detected: %.2f m/s. Scaling down velocity.", velocity_magnitude);
                current_state_.velocity *= (max_velocity_ / velocity_magnitude);
            }
            
            // Reset to UWB position if we detect a large position drift
            if (!uwb_measurements_.empty() && !state_window_.empty()) {
                const auto& latest_uwb = uwb_measurements_.back();
                const auto& latest_state = state_window_.back();
                
                double position_error_z = std::abs(latest_state.position.z() - latest_uwb.position.z());
                double position_error_xy = (latest_state.position.head<2>() - latest_uwb.position.head<2>()).norm();
                
                // If Z drift is extreme (more than 20 meters), reset to the UWB position
                if (position_error_z > 20.0 || position_error_xy > 50.0) {
                    ROS_WARN("Extreme position drift detected: Z=%.2f meters, XY=%.2f meters. Resetting to UWB position.", 
                             position_error_z, position_error_xy);
                    
                    // Create a reset state that keeps orientation but uses UWB position and resets velocity
                    State reset_state = latest_state;
                    reset_state.position = latest_uwb.position;
                    
                    // IMPORTANT: Also reset velocity when position is reset to maintain consistency
                    reset_state.velocity.setZero();
                    
                    // Replace all states in the window with this reset state at their respective timestamps
                    for (auto& state : state_window_) {
                        State new_state = reset_state;
                        new_state.timestamp = state.timestamp;
                        state = new_state;
                    }
                    
                    current_state_ = reset_state;
                    ROS_INFO("Reset state to position [%.2f, %.2f, %.2f] with zero velocity", 
                             reset_state.position.x(), reset_state.position.y(), reset_state.position.z());
                    
                    // Also clear the preintegration map since states have changed
                    preintegration_map_.clear();
                }
            }
            
            // Ensure we have preintegration data between all consecutive keyframes
            for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                double start_time = state_window_[i].timestamp;
                double end_time = state_window_[i+1].timestamp;
                
                std::pair<double, double> key(start_time, end_time);
                
                if (preintegration_map_.find(key) == preintegration_map_.end()) {
                    // Perform preintegration for this interval (with throttled logging)
                    ROS_DEBUG("Computing preintegration between keyframes at %.3f and %.3f (dt = %.3f s)", 
                             start_time, end_time, end_time - start_time);
                    performPreintegrationBetweenKeyframes(start_time, end_time, 
                                                         state_window_[i].acc_bias, 
                                                         state_window_[i].gyro_bias);
                }
            }

            // Time the factor graph optimization
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Directly use factor graph optimization
            bool success = false;
            
            try {
                success = optimizeFactorGraph();
            } catch (const std::exception& e) {
                ROS_ERROR("Exception during factor graph optimization: %s", e.what());
                success = false;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_time - start_time;
            
            if (!success) {
                ROS_WARN("Factor graph optimization failed (took %.1f ms)", duration.count());
            } else {
                // Set optimization flag
                just_optimized_ = true;
                
                // Log optimization time
                ROS_INFO("Factor graph optimization completed in %.1f ms", duration.count());
                
                // Ensure we have valid state before publishing
                if (isStateValid(current_state_)) {
                    publishOptimizedPose();
                } else {
                    ROS_ERROR("Invalid current state after update");
                }
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

    // Helper function to check if a state is valid (no NaN/Inf values)
    bool isStateValid(const State& state) {
        return state.position.allFinite() && 
               state.velocity.allFinite() && 
               state.orientation.coeffs().allFinite() &&
               state.acc_bias.allFinite() && 
               state.gyro_bias.allFinite();
    }

    // NEW: Perform IMU pre-integration between keyframes and store the result
    void performPreintegrationBetweenKeyframes(double start_time, double end_time, 
                                             const Eigen::Vector3d& acc_bias, 
                                             const Eigen::Vector3d& gyro_bias) {
        try {
            // Create key for the map
            std::pair<double, double> key(start_time, end_time);
            
            // Check if we already have this preintegration
            if (preintegration_map_.find(key) != preintegration_map_.end()) {
                // Check if bias reference is the same
                const auto& existing = preintegration_map_[key];
                if ((existing.acc_bias_ref - acc_bias).norm() < 1e-8 && 
                    (existing.gyro_bias_ref - gyro_bias).norm() < 1e-8) {
                    // Bias reference is the same, no need to recompute
                    return;
                }
            }
            
            // Create new preintegration data
            ImuPreintegrationBetweenKeyframes preint;
            preint.reset();
            preint.start_time = start_time;
            preint.end_time = end_time;
            preint.acc_bias_ref = acc_bias;
            preint.gyro_bias_ref = gyro_bias;
            
            // Find relevant IMU measurements - with better performance
            preint.imu_measurements.reserve(100); // Pre-allocate for better performance
            
            for (const auto& imu : imu_buffer_) {
                double timestamp = imu.header.stamp.toSec();
                if (timestamp >= start_time && timestamp <= end_time) {
                    preint.imu_measurements.push_back(imu);
                }
            }
            
            if (preint.imu_measurements.empty()) {
                ROS_WARN_THROTTLE(2.0, "No IMU data found between keyframes %.6f and %.6f", start_time, end_time);
                return;
            }
            
            // Only log extensive information at DEBUG level
            double time_interval = end_time - start_time;
            ROS_DEBUG("Integrating %zu IMU measurements over %.3f seconds between keyframes", 
                     preint.imu_measurements.size(), time_interval);
            
            // Sort IMU measurements by timestamp
            if (preint.imu_measurements.size() > 1) {
                std::sort(preint.imu_measurements.begin(), preint.imu_measurements.end(), 
                         [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                             return a.header.stamp.toSec() < b.header.stamp.toSec();
                         });
            }
            
            // Perform preintegration
            double prev_time = start_time;
            Eigen::Vector3d acc_prev, gyro_prev;
            bool first_imu = true;
            
            // CRITICAL: In your IMU measurements, gravity is reported along +Z axis in sensor frame
            // We need to define the sensor-frame gravity vector and then subtract it
            Eigen::Vector3d sensor_gravity(0, 0, gravity_magnitude_); // POSITIVE Z in sensor frame
            
            // Precompute IMU noise matrix once
            Eigen::Matrix<double, 6, 6> noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;
            noise_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * imu_acc_noise_ * imu_acc_noise_;
            noise_cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * imu_gyro_noise_ * imu_gyro_noise_;
            
            for (const auto& imu_msg : preint.imu_measurements) {
                double timestamp = imu_msg.header.stamp.toSec();
                double dt = timestamp - prev_time;
                
                // Skip invalid time differences
                if (dt <= 0 || dt > max_imu_dt_) {
                    prev_time = timestamp;
                    continue;
                }
                
                // Extract IMU data
                Eigen::Vector3d acc(imu_msg.linear_acceleration.x,
                                   imu_msg.linear_acceleration.y,
                                   imu_msg.linear_acceleration.z);
                
                Eigen::Vector3d gyro(imu_msg.angular_velocity.x,
                                    imu_msg.angular_velocity.y,
                                    imu_msg.angular_velocity.z);
                
                // Apply bias correction
                Eigen::Vector3d acc_corrected = acc - acc_bias;
                Eigen::Vector3d gyro_corrected = gyro - gyro_bias;
                
                if (first_imu) {
                    // First measurement - store and continue
                    acc_prev = acc_corrected;
                    gyro_prev = gyro_corrected;
                    first_imu = false;
                    prev_time = timestamp;
                    continue;
                }
                
                // Midpoint integration approach
                
                // 1. Use average gyro for orientation update
                Eigen::Vector3d gyro_mid = 0.5 * (gyro_prev + gyro_corrected);
                Eigen::Vector3d angle_axis = gyro_mid * dt;
                Eigen::Quaterniond dq = deltaQ(angle_axis);
                
                // 2. Update orientation
                Eigen::Quaterniond new_delta_q = preint.delta_orientation * dq;
                
                // 3. Compute midpoint rotation
                Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
                Eigen::Quaterniond delta_q_half = preint.delta_orientation * deltaQ(half_angle_axis);
                
                // 4. Average acceleration and remove gravity in sensor frame
                Eigen::Vector3d acc_mid = 0.5 * (acc_prev + acc_corrected);
                Eigen::Vector3d acc_mid_without_gravity = acc_mid - sensor_gravity;
                
                // 5. Rotate to integration frame
                Eigen::Vector3d acc_integration_frame = delta_q_half * acc_mid_without_gravity;
                
                // 6. Update velocity and position
                preint.delta_velocity += acc_integration_frame * dt;
                preint.delta_position += preint.delta_velocity * dt + 
                                       0.5 * acc_integration_frame * dt * dt;
                
                // 7. Update orientation
                preint.delta_orientation = new_delta_q.normalized();
                
                // 8. Calculate Jacobians for noise propagation
                Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
                F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
                F.block<3, 3>(3, 6) = delta_q_half.toRotationMatrix() * dt;
                F.block<3, 3>(0, 6) = 0.5 * delta_q_half.toRotationMatrix() * dt * dt;
                
                // 9. Noise propagation matrix
                Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
                G.block<3, 3>(3, 0) = delta_q_half.toRotationMatrix();
                G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
                
                // 11. Update bias Jacobians
                Eigen::Matrix<double, 9, 6> dF_db = Eigen::Matrix<double, 9, 6>::Zero();
                
                // Jacobian for accelerometer bias
                dF_db.block<3, 3>(0, 0) = -0.5 * delta_q_half.toRotationMatrix() * dt * dt;
                dF_db.block<3, 3>(3, 0) = -delta_q_half.toRotationMatrix() * dt;
                
                // Jacobian for gyroscope bias
                Eigen::Matrix3d dR_dbg_times_a = -dt * skewSymmetric(delta_q_half * acc_mid_without_gravity);
                dF_db.block<3, 3>(0, 3) = 0.5 * dR_dbg_times_a * dt;
                dF_db.block<3, 3>(3, 3) = dR_dbg_times_a;
                dF_db.block<3, 3>(6, 3) = -dt * Eigen::Matrix3d::Identity();
                
                // Update the Jacobian for bias
                preint.jacobian_bias = F * preint.jacobian_bias + dF_db;
                
                // Update the covariance
                preint.covariance = F * preint.covariance * F.transpose() + G * noise_cov * G.transpose();
                
                // Store current values for next iteration
                acc_prev = acc_corrected;
                gyro_prev = gyro_corrected;
                prev_time = timestamp;
                preint.sum_dt += dt;
            }
            
            // Ensure numerical stability of covariance
            for (int i = 0; i < 9; ++i) {
                preint.covariance(i, i) = std::max(preint.covariance(i, i), 1e-8);
            }
            
            // Store in map
            preintegration_map_[key] = preint;
            
            // Only log at debug level
            ROS_DEBUG("Completed preintegration with delta_t=%.3f seconds, using %zu IMU measurements", 
                     preint.sum_dt, preint.imu_measurements.size());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in performPreintegrationBetweenKeyframes: %s", e.what());
        }
    }

    // IMU pre-integration factor for Ceres (for on-the-fly optimization)
    class ImuFactor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ImuFactor(const ImuPreintegrationBetweenKeyframes& preint, const Eigen::Vector3d& gravity) 
            : preint_(preint), gravity_(gravity) {}
        
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
            T sum_dt = T(preint_.sum_dt);
            
            // Pre-integrated IMU measurements
            Eigen::Matrix<T, 3, 1> delta_p = preint_.delta_position.cast<T>();
            Eigen::Quaternion<T> delta_q = preint_.delta_orientation.cast<T>();
            Eigen::Matrix<T, 3, 1> delta_v = preint_.delta_velocity.cast<T>();
            
            // Reference biases used during pre-integration
            Eigen::Matrix<T, 3, 1> ba_ref = preint_.acc_bias_ref.cast<T>();
            Eigen::Matrix<T, 3, 1> bg_ref = preint_.gyro_bias_ref.cast<T>();
            
            // Bias corrections - now using current biases
            Eigen::Matrix<T, 3, 1> dba = ba_i - ba_ref;
            Eigen::Matrix<T, 3, 1> dbg = bg_i - bg_ref;
            
            // Jacobian w.r.t bias changes
            Eigen::Matrix<T, 9, 6> jacobian_bias = preint_.jacobian_bias.cast<T>();
            
            // Bias correction vector
            Eigen::Matrix<T, 6, 1> bias_correction_vec;
            bias_correction_vec.template segment<3>(0) = dba;
            bias_correction_vec.template segment<3>(3) = dbg;
            
            // Corrections to pre-integrated measurements
            Eigen::Matrix<T, 9, 1> delta_bias_correction = jacobian_bias * bias_correction_vec;
            
            Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p + delta_bias_correction.template segment<3>(0);
            Eigen::Matrix<T, 3, 1> corrected_delta_v = delta_v + delta_bias_correction.template segment<3>(3);
            
            // Correction to delta_q
            Eigen::Matrix<T, 3, 1> corrected_delta_q_vec = delta_bias_correction.template segment<3>(6);
            Eigen::Quaternion<T> corrected_delta_q = delta_q * deltaQ(corrected_delta_q_vec);
            
            // Gravity vector in world frame (pointing DOWN)
            Eigen::Matrix<T, 3, 1> g = gravity_.cast<T>();
            
            // Compute residuals in body frame of state i
            Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residuals);
            
            // Position residual: transform the difference into the body frame of state i
            residual.template segment<3>(0) = q_i.inverse() * ((p_j - p_i - v_i * sum_dt) - 
                                                           T(0.5) * g * sum_dt * sum_dt) - corrected_delta_p;
            
            // Orientation residual: q_i^-1 * q_j should equal delta_q with corrected biases
            Eigen::Quaternion<T> q_i_inverse_times_q_j = q_i.conjugate() * q_j;
            Eigen::Quaternion<T> delta_q_residual = corrected_delta_q.conjugate() * q_i_inverse_times_q_j;
            
            // Convert to angle-axis representation for the residual (small angle approximation)
            // Using safe division
            T dq_w = delta_q_residual.w();
            if (dq_w < T(1e-5)) {
                dq_w = T(1e-5);  // Prevent division by very small number
            }
            residual.template segment<3>(3) = T(2.0) * delta_q_residual.vec() / dq_w;
            
            // Velocity residual: transform the difference into the body frame of state i
            residual.template segment<3>(6) = q_i.inverse() * (v_j - v_i - g * sum_dt) - corrected_delta_v;
            
            // Bias residuals - use fixed scale factors adjusted for better bias estimation
            residual.template segment<3>(9) = (ba_j - ba_i) / T(0.02);   // Adjusted from 0.05
            residual.template segment<3>(12) = (bg_j - bg_i) / T(0.005); // Adjusted from 0.01
            
            // Weight the residuals by the information matrix
            Eigen::Matrix<T, 9, 9> sqrt_information = preint_.covariance.cast<T>().inverse().llt().matrixL().transpose();
            
            // Scale the position, orientation, and velocity residuals with information matrix
            residual.template segment<3>(0) = sqrt_information.template block<3, 3>(0, 0) * residual.template segment<3>(0);
            residual.template segment<3>(3) = sqrt_information.template block<3, 3>(3, 3) * residual.template segment<3>(3);
            residual.template segment<3>(6) = sqrt_information.template block<3, 3>(6, 6) * residual.template segment<3>(6);
            
            return true;
        }
        
        static ceres::CostFunction* Create(const ImuPreintegrationBetweenKeyframes& preint, const Eigen::Vector3d& gravity) {
            return new ceres::AutoDiffCostFunction<ImuFactor, 15, 7, 3, 6, 7, 3, 6>(
                new ImuFactor(preint, gravity));
        }
        
    private:
        const ImuPreintegrationBetweenKeyframes preint_;
        const Eigen::Vector3d gravity_;
        
        // Helper function to compute quaternion for small angle-axis rotation
        template <typename T>
        static Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
            T theta_norm = theta.norm();
            
            Eigen::Quaternion<T> dq;
            if (theta_norm > T(1e-5)) {
                Eigen::Matrix<T, 3, 1> a = theta / theta_norm;
                dq = Eigen::Quaternion<T>(cos(theta_norm / T(2.0)), 
                                        a.x() * sin(theta_norm / T(2.0)),
                                        a.y() * sin(theta_norm / T(2.0)),
                                        a.z() * sin(theta_norm / T(2.0)));
            } else {
                dq = Eigen::Quaternion<T>(T(1.0), theta.x() / T(2.0), theta.y() / T(2.0), theta.z() / T(2.0));
                dq.normalize();
            }
            return dq;
        }
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
        
        // Create Ceres problem with more efficient options
        ceres::Problem::Options problem_options;
        problem_options.enable_fast_removal = true;  // Speeds up parameter block removal
        ceres::Problem problem(problem_options);
        
        // Initialize pose parameterization if not already done
        if (!pose_parameterization_) {
            pose_parameterization_ = new PoseParameterization();
        }
        
        // Structure for storing state variables for Ceres
        struct OptVariables {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            double pose[7]; // position (3) + quaternion (4)
            double velocity[3];
            double bias[6]; // acc_bias (3) + gyro_bias (3)
        };
        
        try {
            // Preallocate with reserve instead of resize to avoid unnecessary initialization
            std::vector<OptVariables, Eigen::aligned_allocator<OptVariables>> variables;
            variables.reserve(state_window_.size());
            
            // Initialize variables from state window
            for (size_t i = 0; i < state_window_.size(); ++i) {
                OptVariables var;
                const auto& state = state_window_[i];
                
                // Position
                var.pose[0] = state.position.x();
                var.pose[1] = state.position.y();
                var.pose[2] = state.position.z();
                
                // Orientation (quaternion): w, x, y, z
                var.pose[3] = state.orientation.w();
                var.pose[4] = state.orientation.x();
                var.pose[5] = state.orientation.y();
                var.pose[6] = state.orientation.z();
                
                // Velocity
                var.velocity[0] = state.velocity.x();
                var.velocity[1] = state.velocity.y();
                var.velocity[2] = state.velocity.z();
                
                // Biases
                var.bias[0] = state.acc_bias.x();
                var.bias[1] = state.acc_bias.y();
                var.bias[2] = state.acc_bias.z();
                var.bias[3] = state.gyro_bias.x();
                var.bias[4] = state.gyro_bias.y();
                var.bias[5] = state.gyro_bias.z();
                
                variables.push_back(var);
            }
            
            // Add pose parameterization for all pose variables
            for (size_t i = 0; i < state_window_.size(); ++i) {
                problem.AddParameterBlock(variables[i].pose, 7, pose_parameterization_);
            }
            
            // If bias estimation is disabled, set biases constant
            if (!enable_bias_estimation_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    problem.SetParameterBlockConstant(variables[i].bias);
                }
            }
            
            // Check for extreme drift and add emergency constraints if needed
            double max_position_z = 0;
            double min_position_z = std::numeric_limits<double>::max();
            
            for (size_t i = 0; i < state_window_.size(); ++i) {
                max_position_z = std::max(max_position_z, variables[i].pose[2]);
                min_position_z = std::min(min_position_z, variables[i].pose[2]);
            }
            
            bool emergency_mode = false;
            if (max_position_z > 50.0 || min_position_z < -50.0) {
                ROS_WARN("Detected large Z drift: min=%.2f, max=%.2f. Using emergency mode.", 
                         min_position_z, max_position_z);
                emergency_mode = true;
            }
            
            // Add UWB position factors for each keyframe
            int uwb_factors_added = 0;
            for (size_t i = 0; i < state_window_.size(); ++i) {
                double keyframe_time = state_window_[i].timestamp;
                
                // Find any UWB measurement that matches this keyframe's timestamp
                for (const auto& uwb : uwb_measurements_) {
                    if (std::abs(uwb.timestamp - keyframe_time) < 0.01) { // 10ms tolerance
                        // Use smaller noise values in emergency mode
                        double noise_xy = emergency_mode ? uwb_position_noise_ * 0.01 : uwb_position_noise_ * 0.1;
                        double noise_z = emergency_mode ? uwb_position_noise_ * 0.005 : uwb_position_noise_ * 0.05;
                        
                        ceres::CostFunction* uwb_factor = UwbPositionFactor::Create(
                            uwb.position, noise_xy, noise_z);
                        
                        // Use HuberLoss to handle outliers
                        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.5);
                        problem.AddResidualBlock(uwb_factor, loss_function, variables[i].pose);
                        uwb_factors_added++;
                        break;
                    }
                }
            }
            
            // Add roll/pitch prior to all states
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* roll_pitch_prior = RollPitchPriorFactor::Create(roll_pitch_weight_);
                problem.AddResidualBlock(roll_pitch_prior, nullptr, variables[i].pose);
            }
            
            // Add velocity damping to all states
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* vel_damping = VelocityDampingFactor::Create(max_velocity_, velocity_damping_weight_);
                problem.AddResidualBlock(vel_damping, nullptr, variables[i].velocity);
            }
            
            // Add bias prior to all states for regularization
            if (enable_bias_estimation_) {
                // Adjusted constraints on bias estimation
                double acc_bias_sigma = 0.2;
                double gyro_bias_sigma = 0.1;
                
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* bias_prior = BiasPriorFactor::Create(
                        initial_acc_bias_, initial_gyro_bias_, acc_bias_sigma, gyro_bias_sigma);
                    problem.AddResidualBlock(bias_prior, nullptr, variables[i].bias);
                }
            }
            
            // Add IMU pre-integration factors between keyframes
            int imu_factors_added = 0;
            for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                double start_time = state_window_[i].timestamp;
                double end_time = state_window_[i+1].timestamp;
                
                // Skip if the time interval is too short
                if (end_time - start_time < 1e-6) continue;
                
                std::pair<double, double> key(start_time, end_time);
                
                if (preintegration_map_.find(key) != preintegration_map_.end()) {
                    const auto& preint = preintegration_map_[key];
                    
                    ceres::CostFunction* imu_factor = ImuFactor::Create(preint, gravity_world_);
                    
                    problem.AddResidualBlock(imu_factor, nullptr,
                                           variables[i].pose, variables[i].velocity, variables[i].bias,
                                           variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                    imu_factors_added++;
                    
                    // Add bias random walk constraint if bias estimation is enabled
                    if (enable_bias_estimation_) {
                        // Bias should change smoothly between consecutive states
                        double time_diff = end_time - start_time;
                        double acc_bias_sigma = imu_acc_bias_noise_ * sqrt(time_diff) * 2.0;
                        double gyro_bias_sigma = imu_gyro_bias_noise_ * sqrt(time_diff) * 2.0;
                        
                        // Ensure minimum values
                        acc_bias_sigma = std::max(acc_bias_sigma, 0.002);
                        gyro_bias_sigma = std::max(gyro_bias_sigma, 0.0005);
                        
                        ceres::CostFunction* bias_walk = BiasRandomWalkFactor::Create(
                            acc_bias_sigma, gyro_bias_sigma);
                        
                        problem.AddResidualBlock(bias_walk, nullptr, 
                                               variables[i].bias, variables[i+1].bias);
                    }
                } else {
                    // If no preintegration data, add a state constraint
                    ROS_WARN_THROTTLE(5.0, "No preintegration data between keyframes at %.3f and %.3f", start_time, end_time);
                    // Increased weights for state constraints
                    ceres::CostFunction* state_constraint = StateConstraintFactor::Create(10.0, 8.0);
                    problem.AddResidualBlock(state_constraint, nullptr,
                                           variables[i].pose, variables[i].velocity, variables[i].bias,
                                           variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                }
            }
            
            // Configure solver - Proper settings
            ceres::Solver::Options options;
            options.max_num_iterations = max_iterations_;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // Much faster sparse solver
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 4;  // Explicit thread count
            options.function_tolerance = 1e-6;
            options.gradient_tolerance = 1e-8;
            options.parameter_tolerance = 1e-6;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            
            // Enable line search if we're in emergency mode for better stability
            if (emergency_mode) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.dogleg_type = ceres::SUBSPACE_DOGLEG;
                options.use_nonmonotonic_steps = true;
                options.max_num_iterations = max_iterations_ * 2; // Allow more iterations
            }
            
            // Solve the optimization problem
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            if (!summary.IsSolutionUsable()) {
                ROS_WARN("Optimization failed: %s", summary.BriefReport().c_str());
                return false;
            }
            
            // Debug outputs - use throttled logging to reduce overhead
            ROS_INFO_THROTTLE(5.0, "Initial biases: acc=[%.3f, %.3f, %.3f], gyro=[%.3f, %.3f, %.3f]",
                     initial_acc_bias_.x(), initial_acc_bias_.y(), initial_acc_bias_.z(),
                     initial_gyro_bias_.x(), initial_gyro_bias_.y(), initial_gyro_bias_.z());
            
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
            
            // Update preintegration data if biases changed significantly
            if (enable_bias_estimation_) {
                bool bias_changed = false;
                for (const auto& state : state_window_) {
                    if ((state.acc_bias - initial_acc_bias_).norm() > 0.05 || 
                        (state.gyro_bias - initial_gyro_bias_).norm() > 0.02) {
                        bias_changed = true;
                        break;
                    }
                }
                
                if (bias_changed) {
                    // Clear preintegration map to force recomputation with new biases
                    preintegration_map_.clear();
                    ROS_INFO("Biases changed significantly, will recompute preintegration data");
                }
            }
            
            // Clean up old UWB measurements
            if (uwb_measurements_.size() > state_window_.size() * 2) {
                size_t to_keep = state_window_.size() * 2;
                if (uwb_measurements_.size() > to_keep) {
                    uwb_measurements_.erase(uwb_measurements_.begin(), 
                                          uwb_measurements_.end() - to_keep);
                }
            }
            
            // Log optimization results
            ROS_INFO("Optimization completed in %.2f ms with %d iterations. Final cost: %.6f", 
                    summary.total_time_in_seconds * 1000.0,
                    summary.iterations.size(), 
                    summary.final_cost);
            
            return true;
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during optimization: %s", e.what());
            return false;
        }
    }

    // Publish IMU-predicted pose (high frequency)
    void publishImuPose() {
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
            odom_msg.pose.covariance[0] = 0.05;  // Higher uncertainty for IMU-only pose
            odom_msg.pose.covariance[7] = 0.05;
            odom_msg.pose.covariance[14] = 0.05;
            odom_msg.pose.covariance[21] = 0.02;
            odom_msg.pose.covariance[28] = 0.02;
            odom_msg.pose.covariance[35] = 0.02;
            
            // Publish the message
            imu_pose_pub_.publish(odom_msg);
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishImuPose: %s", e.what());
        }
    }

    // Publish optimized state (lower frequency)
    void publishOptimizedPose() {
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
            odom_msg.pose.covariance[0] = 0.01;  // Lower uncertainty for optimized pose
            odom_msg.pose.covariance[7] = 0.01;
            odom_msg.pose.covariance[14] = 0.01;
            odom_msg.pose.covariance[21] = 0.01;
            odom_msg.pose.covariance[28] = 0.01;
            odom_msg.pose.covariance[35] = 0.01;
            
            // Publish the message
            optimized_pose_pub_.publish(odom_msg);
            
            // Print position for debugging (throttled)
            ROS_INFO_THROTTLE(1.0, "Published optimized state: Position [%.2f, %.2f, %.2f], Velocity [%.2f, %.2f, %.2f]",
                      current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                      current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z());
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishOptimizedPose: %s", e.what());
        }
    }
};

// Initialize static members
ceres::LocalParameterization* UwbImuFusion::pose_parameterization_ = nullptr;

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