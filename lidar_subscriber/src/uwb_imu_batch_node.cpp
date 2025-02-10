/*
 * UWB-IMU Fusion using Factor Graph Optimization
 * This implements loose coupling of UWB and IMU measurements
 * using factor graph optimization with IMU preintegration
 */

 #include <ros/ros.h>
 #include <sensor_msgs/Imu.h>
 #include <geometry_msgs/PointStamped.h>
 #include <nav_msgs/Odometry.h>
 #include <Eigen/Dense>
 #include <ceres/ceres.h>
 #include <deque>
 #include <memory>

 #include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
 
 namespace uwb_imu_fusion {
 
 // Constants
 const double GRAVITY_MAGNITUDE = 9.81;
 const Eigen::Vector3d GRAVITY_VECTOR(0, 0, -GRAVITY_MAGNITUDE);
 
 // Measurement structures
 struct ImuMeasurement {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     Eigen::Vector3d acc;
     Eigen::Vector3d gyro;
     double timestamp;
 };
 
 struct UwbMeasurement {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     Eigen::Vector3d position;
     double timestamp;
 };
 
 // IMU Preintegration class
 class ImuPreintegration {
 public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW

     struct PreintegrationResult {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        double dt_sum;
        Eigen::Vector3d delta_p;    // Preintegrated position
        Eigen::Vector3d delta_v;    // Preintegrated velocity
        Eigen::Quaterniond delta_q; // Preintegrated rotation
        
        // Jacobians w.r.t. bias
        Eigen::Matrix<double, 3, 3> jacobian_p_ba;  // Position w.r.t. acc bias
        Eigen::Matrix<double, 3, 3> jacobian_p_bg;  // Position w.r.t. gyro bias
        Eigen::Matrix<double, 3, 3> jacobian_v_ba;  // Velocity w.r.t. acc bias
        Eigen::Matrix<double, 3, 3> jacobian_v_bg;  // Velocity w.r.t. gyro bias
        Eigen::Matrix<double, 3, 3> jacobian_q_bg;  // Rotation w.r.t. gyro bias
    
        PreintegrationResult() {
            dt_sum = 0.0;
            delta_p.setZero();
            delta_v.setZero();
            delta_q.setIdentity();
            jacobian_p_ba.setZero();
            jacobian_p_bg.setZero();
            jacobian_v_ba.setZero();
            jacobian_v_bg.setZero();
            jacobian_q_bg.setZero();
        }
    };
 
     ImuPreintegration(const Eigen::Vector3d& acc_bias, const Eigen::Vector3d& gyro_bias)
         : acc_bias_(acc_bias), gyro_bias_(gyro_bias) {
         acc_noise_ = 0.01;
         gyro_noise_ = 0.01;
         acc_bias_noise_ = 0.0001;
         gyro_bias_noise_ = 0.0001;
         reset();
     }
 
     void reset() {
         delta_p_.setZero();
         delta_v_.setZero();
         delta_q_.setIdentity();
         jacobian_bias_.setZero();
         covariance_.setZero();

         jacobian_p_ba_.setZero();
        jacobian_p_bg_.setZero();
        jacobian_v_ba_.setZero();
        jacobian_v_bg_.setZero();
        jacobian_q_bg_.setZero();

         dt_sum_ = 0;
 
         Eigen::Matrix<double, 15, 15> noise_cov = Eigen::Matrix<double, 15, 15>::Zero();
         noise_cov.block<3, 3>(0, 0) = acc_noise_ * Eigen::Matrix3d::Identity();
         noise_cov.block<3, 3>(3, 3) = gyro_noise_ * Eigen::Matrix3d::Identity();
         noise_cov.block<3, 3>(6, 6) = acc_bias_noise_ * Eigen::Matrix3d::Identity();
         noise_cov.block<3, 3>(9, 9) = gyro_bias_noise_ * Eigen::Matrix3d::Identity();
         noise_cov_ = noise_cov;
     }
 
     void integrate(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt) {
         const Eigen::Vector3d acc_unbiased = acc - acc_bias_;
         const Eigen::Vector3d gyro_unbiased = gyro - gyro_bias_;
 
         Eigen::Matrix3d rot_k = delta_q_.toRotationMatrix();
 
         // Midpoint integration
         Eigen::Vector3d delta_angle = gyro_unbiased * dt;
         Eigen::Quaterniond dq;
         if (delta_angle.norm() > 1e-12) {
             dq = Eigen::Quaterniond(Eigen::AngleAxisd(delta_angle.norm(), delta_angle.normalized()));
         } else {
             dq = Eigen::Quaterniond::Identity();
         }
 
         // Update delta measurements
         Eigen::Vector3d delta_p_k = delta_v_ * dt + 0.5 * rot_k * acc_unbiased * dt * dt;
         Eigen::Vector3d delta_v_k = rot_k * acc_unbiased * dt;
         Eigen::Quaterniond delta_q_k = delta_q_ * dq;
 
         // Update Jacobian and covariance
         updateJacobianAndCovariance(acc_unbiased, gyro_unbiased, dt, rot_k);
 
         // Update states
         delta_p_ += delta_p_k;
         delta_v_ += delta_v_k;
         delta_q_ = delta_q_k;
         dt_sum_ += dt;
     }

     PreintegrationResult getResult() const {
        PreintegrationResult result;
        result.dt_sum = dt_sum_;
        result.delta_p = delta_p_;
        result.delta_v = delta_v_;
        result.delta_q = delta_q_;
        result.jacobian_p_ba = jacobian_p_ba_;
        result.jacobian_p_bg = jacobian_p_bg_;
        result.jacobian_v_ba = jacobian_v_ba_;
        result.jacobian_v_bg = jacobian_v_bg_;
        result.jacobian_q_bg = jacobian_q_bg_;
        return result;
    }
 
 private:
     void updateJacobianAndCovariance(const Eigen::Vector3d& acc_unbiased,
                                     const Eigen::Vector3d& gyro_unbiased,
                                     double dt,
                                     const Eigen::Matrix3d& rot_k) {
         // State transition matrix
         Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
         F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
         F.block<3, 3>(0, 6) = -0.25 * rot_k * skewSymmetric(acc_unbiased) * dt * dt;
         F.block<3, 3>(0, 9) = -0.5 * rot_k * dt * dt;
         F.block<3, 3>(3, 6) = -rot_k * skewSymmetric(acc_unbiased) * dt;
         F.block<3, 3>(3, 9) = -rot_k * dt;
         F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - skewSymmetric(gyro_unbiased * dt);
         F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;
 
         // Noise input matrix
         Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
         G.block<3, 3>(0, 0) = 0.5 * rot_k * dt * dt;
         G.block<3, 3>(3, 0) = rot_k * dt;
         G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;
         G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity() * dt;
         G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity() * dt;
 
         // Noise covariance
         Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();
         Q.block<3, 3>(0, 0) = acc_noise_ * Eigen::Matrix3d::Identity();
         Q.block<3, 3>(3, 3) = gyro_noise_ * Eigen::Matrix3d::Identity();
         Q.block<3, 3>(6, 6) = acc_bias_noise_ * Eigen::Matrix3d::Identity();
         Q.block<3, 3>(9, 9) = gyro_bias_noise_ * Eigen::Matrix3d::Identity();
 
         covariance_ = F * covariance_ * F.transpose() + G * Q * G.transpose();
 
         // Update bias Jacobians
         jacobian_bias_.block<3, 3>(0, 0) = -0.5 * rot_k * dt * dt;
         jacobian_bias_.block<3, 3>(0, 3) = -0.25 * rot_k * skewSymmetric(acc_unbiased) * dt * dt * dt;
         jacobian_bias_.block<3, 3>(3, 0) = -rot_k * dt;
         jacobian_bias_.block<3, 3>(3, 3) = -rot_k * skewSymmetric(acc_unbiased) * dt * dt;
         jacobian_bias_.block<3, 3>(6, 0) = Eigen::Matrix3d::Zero();
         jacobian_bias_.block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity() * dt;
     }
 
     static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
         Eigen::Matrix3d m;
         m << 0, -v(2), v(1),
              v(2), 0, -v(0),
              -v(1), v(0), 0;
         return m;
     }
 
     Eigen::Vector3d delta_p_;
     Eigen::Vector3d delta_v_;
     Eigen::Quaterniond delta_q_;
     Eigen::Matrix<double, 9, 6> jacobian_bias_;
     Eigen::Matrix<double, 15, 15> covariance_;
     Eigen::Matrix<double, 15, 15> noise_cov_;
     double dt_sum_;
 
     Eigen::Vector3d acc_bias_;
     Eigen::Vector3d gyro_bias_;
     double acc_noise_;
     double gyro_noise_;
     double acc_bias_noise_;
     double gyro_bias_noise_;

     Eigen::Matrix<double, 3, 3> jacobian_p_ba_;
     Eigen::Matrix<double, 3, 3> jacobian_p_bg_;
     Eigen::Matrix<double, 3, 3> jacobian_v_ba_;
     Eigen::Matrix<double, 3, 3> jacobian_v_bg_;
     Eigen::Matrix<double, 3, 3> jacobian_q_bg_;
 };
 
 // Factor classes for optimization
 struct ImuFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit ImuFactor(const ImuPreintegration::PreintegrationResult& preintegration)
        : preintegration_(preintegration) {
        sqrt_information_ = Eigen::Matrix<double, 15, 15>::Identity();
        sqrt_information_.block<3,3>(0,0) *= 1.0 / 0.1;   // position weight
        sqrt_information_.block<3,3>(3,3) *= 1.0 / 0.2;   // velocity weight
        sqrt_information_.block<3,3>(6,6) *= 1.0 / 0.1;   // rotation weight
        sqrt_information_.block<3,3>(9,9) *= 1.0 / 0.01;  // acc bias weight
        sqrt_information_.block<3,3>(12,12) *= 1.0 / 0.01;// gyro bias weight
    }

    template <typename T>
    bool operator()(const T* const state1, const T* const state2, T* residuals) const {
        // Extract states at time i
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(state1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i(state1 + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(state1 + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(state1 + 10);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(state1 + 13);

        // Extract states at time j
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(state2);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j(state2 + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(state2 + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(state2 + 10);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(state2 + 13);

        // Get preintegrated measurements
        const Eigen::Matrix<T, 3, 1> alpha = preintegration_.delta_p.cast<T>();
        const Eigen::Matrix<T, 3, 1> beta = preintegration_.delta_v.cast<T>();
        const Eigen::Quaternion<T> gamma = preintegration_.delta_q.cast<T>();
        const T delta_t = T(preintegration_.dt_sum);

        // Get bias Jacobians
        const Eigen::Matrix<T, 3, 3> J_p_ba = preintegration_.jacobian_p_ba.cast<T>();
        const Eigen::Matrix<T, 3, 3> J_p_bg = preintegration_.jacobian_p_bg.cast<T>();
        const Eigen::Matrix<T, 3, 3> J_v_ba = preintegration_.jacobian_v_ba.cast<T>();
        const Eigen::Matrix<T, 3, 3> J_v_bg = preintegration_.jacobian_v_bg.cast<T>();
        const Eigen::Matrix<T, 3, 3> J_q_bg = preintegration_.jacobian_q_bg.cast<T>();

        // Bias corrections
        const Eigen::Matrix<T, 3, 1> dba = ba_j - ba_i;
        const Eigen::Matrix<T, 3, 1> dbg = bg_j - bg_i;

        // Corrected measurements
        const Eigen::Matrix<T, 3, 1> alpha_corrected = alpha + J_p_ba * dba + J_p_bg * dbg;
        const Eigen::Matrix<T, 3, 1> beta_corrected = beta + J_v_ba * dba + J_v_bg * dbg;
        
        // Compute rotation correction
        Eigen::Quaternion<T> gamma_corrected = gamma;
        Eigen::Matrix<T, 3, 1> theta_correction = J_q_bg * dbg;
        if (theta_correction.norm() > T(1e-12)) {
            gamma_corrected = gamma * Eigen::Quaternion<T>(
                Eigen::AngleAxis<T>(theta_correction.norm(), theta_correction.normalized()));
        }

        // Gravity vector
        const Eigen::Matrix<T, 3, 1> g(T(0), T(0), T(-9.81));

        Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residuals);

        // Position residual (simplified)
        residual.template segment<3>(0) = p_j - (p_i + v_i * delta_t + 
            T(0.5) * g * delta_t * delta_t + q_i * alpha_corrected);

        // Velocity residual (simplified)
        residual.template segment<3>(3) = v_j - (v_i + g * delta_t + 
            q_i * beta_corrected);

        // Rotation residual (corrected)
        Eigen::Quaternion<T> q_error = (q_i * gamma_corrected).conjugate() * q_j;
        residual.template segment<3>(6) = T(2.0) * q_error.vec();

        // Scale residuals
        const T pos_scale = T(1.0);
        const T vel_scale = T(0.1);
        const T rot_scale = T(1.0);
        
        residual.template segment<3>(0) *= pos_scale;
        residual.template segment<3>(3) *= vel_scale;
        residual.template segment<3>(6) *= rot_scale;

        // Apply information matrix
        residual = sqrt_information_.cast<T>() * residual;

        return true;
    }

    ImuPreintegration::PreintegrationResult preintegration_;
    Eigen::Matrix<double, 15, 15> sqrt_information_;
};
 
 struct UwbFactor {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
     UwbFactor(const Eigen::Vector3d& measurement, const Eigen::Matrix3d& covariance)
         : measurement_(measurement), information_(covariance.inverse()) {}
 
     template <typename T>
     bool operator()(const T* const state, T* residuals) const {
         Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(state);
         Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
 
         residual = position - measurement_.cast<T>();
         residual = information_.cast<T>() * residual;
        //  std::cout<<"residuals of the UWB-> " << residual[0] <<"\n";
 
         return true;
     }
 
     Eigen::Vector3d measurement_;
     Eigen::Matrix3d information_;
 };

 struct PositionDriftFactor {
    explicit PositionDriftFactor(double max_drift) : max_drift_(max_drift) {}

    template <typename T>
    bool operator()(const T* const state1, const T* const state2, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(state1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(state2);
        
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
        
        Eigen::Matrix<T, 3, 1> drift = p2 - p1;
        T drift_norm = drift.norm();
        
        if (drift_norm > T(max_drift_)) {
            residual = drift * (T(1.0) - T(max_drift_) / drift_norm);
        } else {
            residual.setZero();
        }
        
        return true;
    }

    double max_drift_;
};
 
 // Main fusion class
// Add this to the previous code, replacing the incomplete UwbImuFusion class

class UwbImuFusion {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        struct State {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            double timestamp;
            Eigen::Vector3d position;
            Eigen::Vector3d velocity;
            Eigen::Quaterniond orientation;
            Eigen::Vector3d acc_bias;
            Eigen::Vector3d gyro_bias;
        };
    
        UwbImuFusion() {
            initializeParameters();
            
            // ROS subscribers and publishers
            // imu_sub_ = nh_.subscribe("/imu/data", 1000, &UwbImuFusion::imuCallback, this);
            // uwb_sub_ = nh_.subscribe("/vins_estimator/UWBPoistionPS", 1000, &UwbImuFusion::uwbCallback, this);

            imu_sub_ = nh_.subscribe("/sensor_simulator/imu_data", 1000, &UwbImuFusion::imuCallback, this);
            uwb_sub_ = nh_.subscribe("/sensor_simulator/UWBPoistionPS", 1000, &UwbImuFusion::uwbCallback, this);


            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/optimized_pose", 100);
    
            // Initialize IMU preintegration
            imu_preintegration_ = std::make_unique<ImuPreintegration>(
                current_state_.acc_bias, current_state_.gyro_bias);
    
            ROS_INFO("UWB-IMU Fusion node initialized.");
        }
    
    private:
        // Add these members

        struct BatchState {

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            
            double timestamp;
            
            bool fixed; // Whether this state is fixed in optimization
            
            Eigen::Vector3d position;
            
            Eigen::Vector3d velocity;
            
            Eigen::Quaterniond orientation;
            
            Eigen::Vector3d acc_bias;
            
            Eigen::Vector3d gyro_bias;
            
        };
        std::vector<BatchState> trajectory_states_;
        ros::Publisher path_pub_;
        ros::Publisher batch_pose_pub_;
        
        // Add these members
        std::vector<State> batch_states_;
        bool batch_initialized_;
        double batch_start_time_;
        double batch_duration_;
        size_t batch_size_;

        // Batch parameters
        double state_dt_;           // Time between states in batch
        size_t num_states_batch_;   // Number of states in each batch

        void initializeParameters() {

            // Batch processing parameters
            batch_duration_ = 1.0;     // Process 5 seconds of data at a time
            state_dt_ = 0.1;          // State every 0.1 seconds, 0.1
            num_states_batch_ = static_cast<size_t>(batch_duration_ / state_dt_);
            batch_initialized_ = false;
            // Add batch parameters
            batch_size_ = 100;  // Number of states in batch
            
            // Publishers
            path_pub_ = nh_.advertise<nav_msgs::Path>("/uwb_imu_fusion/trajectory", 1);
            batch_pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/batch_pose", 1);
            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/pose", 1);


            // System parameters
            window_size_ = 50;  // Reduced from 10 5
            min_uwb_measurements_ = 2;  // Reduced from 3

            // Adjusted noise parameters
            imu_acc_noise_ = 0.01;    // Reduced
            imu_gyro_noise_ = 0.001;  // Reduced
            imu_acc_bias_noise_ = 0.0001;
            imu_gyro_bias_noise_ = 0.0001;
            uwb_noise_ = 0.1;  // Increased for stronger position constraint

            // Initialize current state
            // Initialize current state
            current_state_.timestamp = 0.0;
            current_state_.position.setZero();
            current_state_.velocity.setZero();
            current_state_.orientation.setIdentity();
            current_state_.acc_bias.setZero();
            current_state_.gyro_bias.setZero();

            initialized_ = false;
            last_imu_time_ = 0.0;
            
            // Add these parameters
            max_imu_queue_size_ = 1000;  // About 2.5s of IMU data at 400Hz
            max_uwb_queue_size_ = 30;    // About 3s of UWB data at 10Hz
            min_imu_between_uwb_ = 20;   // Minimum IMU measurements between UWB updates

            
        }
    
        void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
            ImuMeasurement imu_data;
            imu_data.timestamp = msg->header.stamp.toSec();
            imu_data.acc = Eigen::Vector3d(msg->linear_acceleration.x,
                                          msg->linear_acceleration.y,
                                          msg->linear_acceleration.z);
            imu_data.gyro = Eigen::Vector3d(msg->angular_velocity.x,
                                           msg->angular_velocity.y,
                                           msg->angular_velocity.z);
    
            processImuData(imu_data);
        }
    
        void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
            UwbMeasurement uwb_data;
            uwb_data.timestamp = msg->header.stamp.toSec();
            uwb_data.position = Eigen::Vector3d(msg->point.x,
                                              msg->point.y,
                                              msg->point.z);
    
            processUwbData(uwb_data);
        }
    
        void processImuData(const ImuMeasurement& imu_data) {
            if (!initialized_) {
                initialized_ = true;
                last_imu_time_ = imu_data.timestamp;
                
                if (!batch_initialized_) {
                    batch_start_time_ = imu_data.timestamp;
                    batch_initialized_ = true;
                    batch_states_.clear();
                    
                    // Initialize first state
                    State initial_state;
                    initial_state.timestamp = imu_data.timestamp;
                    initial_state.position.setZero();
                    initial_state.velocity.setZero();
                    initial_state.orientation.setIdentity();
                    initial_state.acc_bias.setZero();
                    initial_state.gyro_bias.setZero();
                    
                    batch_states_.push_back(initial_state);
                }
                return;
            }
        
            // Store IMU measurements
            imu_buffer_.push_back(imu_data);
            
            // Check if batch duration is reached
            if (imu_data.timestamp - batch_start_time_ >= batch_duration_) {
                performBatchOptimization();
            }
        }
    
        void processUwbData(const UwbMeasurement& uwb_data) {
            if (!initialized_ || imu_buffer_.size() < min_imu_between_uwb_) {
                return;
            }
        
            // Check if UWB position is within reasonable bounds
            const double MAX_POSITION = 100000.0;  // meters
            if (uwb_data.position.norm() > MAX_POSITION) {
                ROS_WARN("UWB position too large, ignoring measurement");
                return;
            }
        
            uwb_buffer_.push_back(uwb_data);
        
            // Limit buffer size
            while (uwb_buffer_.size() > max_uwb_queue_size_) {
                uwb_buffer_.pop_front();
            }
        
            // Trigger optimization when enough measurements are collected
            if (uwb_buffer_.size() >= min_uwb_measurements_) {
                optimize();
            }
        }
    
        void propagateState(const ImuMeasurement& imu_data, double dt) {
            // Remove bias and integrate IMU data
            Eigen::Vector3d acc_unbias = imu_data.acc - current_state_.acc_bias;
            Eigen::Vector3d gyro_unbias = imu_data.gyro - current_state_.gyro_bias;
        
            // Add strict limits
            const double MAX_ACC = 20.0;  // Reduced from 50.0 m/s^2
            const double MAX_GYRO = 50.0;  // rad/s
            acc_unbias = acc_unbias.array().min(MAX_ACC).max(-MAX_ACC);
            gyro_unbias = gyro_unbias.array().min(MAX_GYRO).max(-MAX_GYRO);
        
            // First update orientation
            Eigen::Vector3d angle_axis = gyro_unbias * dt;
            if (angle_axis.norm() > 1e-8) {  // Changed from 1e-12
                current_state_.orientation = current_state_.orientation * 
                    Eigen::Quaterniond(Eigen::AngleAxisd(angle_axis.norm(), angle_axis.normalized()));
            }
            current_state_.orientation.normalize();  // Important!
        
            // Then update position and velocity in world frame
            Eigen::Vector3d acc_world = current_state_.orientation * acc_unbias + GRAVITY_VECTOR;
            
            // Stricter velocity limits
            const double MAX_VELOCITY = 50.0;  // Reduced from 10.0 m/s
            
            // Integration with velocity limiting
            Eigen::Vector3d delta_v = acc_world * dt;
            delta_v = delta_v.array().min(MAX_VELOCITY * dt).max(-MAX_VELOCITY * dt);
            
            Eigen::Vector3d next_velocity = current_state_.velocity + delta_v;
            next_velocity = next_velocity.array().min(MAX_VELOCITY).max(-MAX_VELOCITY);
            
            // Position update with smaller time step
            current_state_.position += current_state_.velocity * dt + 0.5 * delta_v * dt;
            current_state_.velocity = next_velocity;
        
            // Update timestamp
            current_state_.timestamp = imu_data.timestamp;
        
            // Integrate IMU preintegration
            imu_preintegration_->integrate(imu_data.acc, imu_data.gyro, dt);
        }

        void addImuFactor(ceres::Problem& problem, double* state_i, double* state_j) {
            auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor), // Changed from 9 to 15
                new ceres::HuberLoss(1.0),
                state_i,
                state_j
            );
        }

        void performBatchOptimization() {
            if (imu_buffer_.empty() || uwb_buffer_.empty()) {
                return;
            }
        
            // Initialize batch states if needed
            std::vector<BatchState> batch_states;
            initializeBatchStates(batch_states);
        
            // Setup optimization problem
            ceres::Problem problem;
            std::vector<double*> parameter_blocks;
        
            // Create parameter blocks for all states
            for (size_t i = 0; i < batch_states.size(); ++i) {
                double* state_ptr = new double[16];
                batchStateToArray(batch_states[i], state_ptr);
                parameter_blocks.push_back(state_ptr);
                
                // Fix states that are marked as fixed
                if (batch_states[i].fixed) {
                    problem.SetParameterBlockConstant(state_ptr);
                }
            }
        
            // Add IMU factors
            for (size_t i = 0; i < batch_states.size() - 1; ++i) {
                double t1 = batch_states[i].timestamp;
                double t2 = batch_states[i + 1].timestamp;
                
                imu_preintegration_->reset();
                
                // Integrate IMU measurements between consecutive states
                for (const auto& imu : imu_buffer_) {
                    if (imu.timestamp >= t1 && imu.timestamp < t2) {
                        double dt = imu.timestamp - t1;
                        imu_preintegration_->integrate(imu.acc, imu.gyro, dt);
                        t1 = imu.timestamp;
                    }
                }
                
                // Add IMU factor
                auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor),
                    new ceres::HuberLoss(1.0),
                    parameter_blocks[i],
                    parameter_blocks[i + 1]
                );
            }
        
            // Add UWB factors
            for (const auto& uwb : uwb_buffer_) {
                size_t closest_idx = findClosestStateIndex(uwb.timestamp, batch_states);
                
                Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity();
                auto* uwb_factor = new UwbFactor(uwb.position, uwb_cov);
                
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor),
                    new ceres::CauchyLoss(0.1),
                    parameter_blocks[closest_idx]
                );
            }
        
            // Solve optimization problem
            ceres::Solver::Options options;
            options.minimizer_progress_to_stdout = true;

            options.linear_solver_type = ceres::DENSE_SCHUR;
            //options.num_threads = 2;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.max_num_iterations = 50;
        
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        
            if (summary.termination_type == ceres::CONVERGENCE) {
                // Update batch states with optimization results
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    arrayToBatchState(parameter_blocks[i], batch_states[i]);
                }
                
                // Update trajectory and publish
                updateTrajectory(batch_states);
                publishTrajectory();
            }
        
            // Cleanup
            for (auto ptr : parameter_blocks) {
                delete[] ptr;
            }
        
            // Prepare for next batch
            prepareNextBatch(batch_states);
        }

        void initializeBatchStates(std::vector<BatchState>& batch_states) {
            batch_states.clear();
            
            double current_time = batch_start_time_;
            for (size_t i = 0; i < num_states_batch_; ++i) {
                BatchState state;
                state.timestamp = current_time;
                state.fixed = (i == 0 && !trajectory_states_.empty());
                
                if (i == 0 && !trajectory_states_.empty()) {
                    // Initialize from last state of previous batch
                    state = trajectory_states_.back();
                } else {
                    // Initialize from measurements
                    state.position = interpolatePosition(current_time);
                    state.velocity.setZero();
                    state.orientation.setIdentity();
                    state.acc_bias.setZero();
                    state.gyro_bias.setZero();
                }
                
                batch_states.push_back(state);
                current_time += state_dt_;
            }
        }

        void updateTrajectory(const std::vector<BatchState>& batch_states) {
            // Remove overlap with previous batch (keep one state for continuity)
            if (!trajectory_states_.empty()) {
                trajectory_states_.pop_back();
            }
            
            // Append new batch states
            trajectory_states_.insert(trajectory_states_.end(), 
                                    batch_states.begin(), batch_states.end());
        }
        
        void publishTrajectory() {
            nav_msgs::Path path_msg;
            path_msg.header.stamp = ros::Time::now();
            path_msg.header.frame_id = "map";
            
            for (const auto& state : trajectory_states_) {
                geometry_msgs::PoseStamped pose;
                pose.header.stamp = ros::Time(state.timestamp);
                pose.header.frame_id = "map";
                
                pose.pose.position.x = state.position.x();
                pose.pose.position.y = state.position.y();
                pose.pose.position.z = state.position.z();
                
                pose.pose.orientation.w = state.orientation.w();
                pose.pose.orientation.x = state.orientation.x();
                pose.pose.orientation.y = state.orientation.y();
                pose.pose.orientation.z = state.orientation.z();
                
                path_msg.poses.push_back(pose);
            }
            
            path_pub_.publish(path_msg);
            
            // Also publish the latest state
            if (!trajectory_states_.empty()) {
                publishBatchState(trajectory_states_.back());
            }
        }
        
        void publishBatchState(const BatchState& state) {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(state.timestamp);
            odom_msg.header.frame_id = "map";
            
            odom_msg.pose.pose.position.x = state.position.x();
            odom_msg.pose.pose.position.y = state.position.y();
            odom_msg.pose.pose.position.z = state.position.z();
            
            odom_msg.pose.pose.orientation.w = state.orientation.w();
            odom_msg.pose.pose.orientation.x = state.orientation.x();
            odom_msg.pose.pose.orientation.y = state.orientation.y();
            odom_msg.pose.pose.orientation.z = state.orientation.z();
            
            odom_msg.twist.twist.linear.x = state.velocity.x();
            odom_msg.twist.twist.linear.y = state.velocity.y();
            odom_msg.twist.twist.linear.z = state.velocity.z();
            
            batch_pose_pub_.publish(odom_msg);
        }


        void prepareNextBatch(const std::vector<BatchState>& current_batch) {
            batch_start_time_ = current_batch.back().timestamp;
            imu_buffer_.clear();
            uwb_buffer_.clear();
        }
        
        Eigen::Vector3d interpolatePosition(double timestamp) {
            if (uwb_buffer_.empty()) {
                return Eigen::Vector3d::Zero();
            }
            
            // Find surrounding UWB measurements
            size_t i = 0;
            while (i < uwb_buffer_.size() - 1 && uwb_buffer_[i + 1].timestamp < timestamp) {
                i++;
            }
            
            if (i >= uwb_buffer_.size() - 1) {
                return uwb_buffer_.back().position;
            }
            
            // Linear interpolation
            double t0 = uwb_buffer_[i].timestamp;
            double t1 = uwb_buffer_[i + 1].timestamp;
            double alpha = (timestamp - t0) / (t1 - t0);
            
            return (1 - alpha) * uwb_buffer_[i].position + alpha * uwb_buffer_[i + 1].position;
        }
        
        size_t findClosestState(double timestamp) {
            size_t closest_idx = 0;
            double min_dt = std::numeric_limits<double>::max();
            
            for (size_t i = 0; i < batch_states_.size(); ++i) {
                double dt = std::abs(timestamp - batch_states_[i].timestamp);
                if (dt < min_dt) {
                    min_dt = dt;
                    closest_idx = i;
                }
            }
            
            return closest_idx;
        }
        
        void publishState(const State& state) {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(state.timestamp);
            odom_msg.header.frame_id = "map";
            
            odom_msg.pose.pose.position.x = state.position.x();
            odom_msg.pose.pose.position.y = state.position.y();
            odom_msg.pose.pose.position.z = state.position.z();
            
            odom_msg.pose.pose.orientation.w = state.orientation.w();
            odom_msg.pose.pose.orientation.x = state.orientation.x();
            odom_msg.pose.pose.orientation.y = state.orientation.y();
            odom_msg.pose.pose.orientation.z = state.orientation.z();
            
            odom_msg.twist.twist.linear.x = state.velocity.x();
            odom_msg.twist.twist.linear.y = state.velocity.y();
            odom_msg.twist.twist.linear.z = state.velocity.z();
            
            pose_pub_.publish(odom_msg);
        }
    
        void optimize() {
            if (imu_buffer_.empty() || uwb_buffer_.size() < min_uwb_measurements_) {
                return;
            }
        
            // Setup optimization problem
            ceres::Problem problem;
            std::vector<State> state_window;
            std::vector<double*> parameter_blocks;
        
            // Initialize state window
            initializeStateWindow(state_window, parameter_blocks);
        
            // Add IMU factors with proper weighting
            double imu_weight = 1.0;
            // Update IMU factor dimensions from 9 to 15
            for (size_t i = 0; i < state_window.size() - 1; ++i) {
                auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor), // Changed from 9 to 15
                    loss_function,
                    parameter_blocks[i],
                    parameter_blocks[i + 1]
                );
            }
        
            // Increase UWB weight significantly
            double uwb_weight = 100.0;  // Increased from 10.0
            for (const auto& uwb_data : uwb_buffer_) {
                Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity() / uwb_weight;
                auto* uwb_factor = new UwbFactor(uwb_data.position, uwb_cov);
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor),
                    new ceres::CauchyLoss(0.1),  // Changed loss function
                    parameter_blocks[0]
                );
            }

            // Add position drift constraint
            for (size_t i = 1; i < parameter_blocks.size(); ++i) {
                auto* drift_factor = new PositionDriftFactor(5.0);  // 5.0 meters max drift
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PositionDriftFactor, 3, 16, 16>(drift_factor),
                    new ceres::CauchyLoss(1.0),
                    parameter_blocks[0],
                    parameter_blocks[i]
                );
            }
        
            // Set solver options
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations = 20;
            options.function_tolerance = 1e-3;
            options.gradient_tolerance = 1e-3;
            options.parameter_tolerance = 1e-4;
        
            // Solve
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        
            if (summary.termination_type == ceres::CONVERGENCE) {
                // Update states only if optimization converged
                updateStates(state_window, parameter_blocks);
            } else {
                ROS_WARN("Optimization failed to converge");
            }
        
            // Cleanup
            for (auto ptr : parameter_blocks) {
                delete[] ptr;
            }
        
            // Reset IMU preintegration and clear UWB buffer
            imu_preintegration_->reset();
            uwb_buffer_.clear();
        
            // Publish results
            publishResults();
        }
    
        void initializeStateWindow(std::vector<State>& state_window,
                                 std::vector<double*>& parameter_blocks) {
            state_window.clear();
            parameter_blocks.clear();
    
            // First state is current state
            state_window.push_back(current_state_);
    
            // Initialize other states using IMU propagation
            for (size_t i = 1; i < window_size_; ++i) {
                State state = state_window.back();
                // Simple propagation (could be improved)
                state.position += state.velocity * 0.01;  // Assume 100Hz
                state_window.push_back(state);
            }
    
            // Create parameter blocks
            for (size_t i = 0; i < state_window.size(); ++i) {
                double* state_ptr = new double[16];  // [p, v, q, ba, bg]
                stateToArray(state_window[i], state_ptr);
                parameter_blocks.push_back(state_ptr);
            }
        }

        void batchStateToArray(const BatchState& state, double* arr) {
            Eigen::Map<Eigen::Vector3d> p(arr);
            Eigen::Map<Eigen::Vector3d> v(arr + 3);
            Eigen::Map<Eigen::Quaterniond> q(arr + 6);
            Eigen::Map<Eigen::Vector3d> ba(arr + 10);
            Eigen::Map<Eigen::Vector3d> bg(arr + 13);
        
            p = state.position;
            v = state.velocity;
            q = state.orientation;
            ba = state.acc_bias;
            bg = state.gyro_bias;
        }
        
        void arrayToBatchState(const double* arr, BatchState& state) {
            Eigen::Map<const Eigen::Vector3d> p(arr);
            Eigen::Map<const Eigen::Vector3d> v(arr + 3);
            Eigen::Map<const Eigen::Quaterniond> q(arr + 6);
            Eigen::Map<const Eigen::Vector3d> ba(arr + 10);
            Eigen::Map<const Eigen::Vector3d> bg(arr + 13);
        
            state.position = p;
            state.velocity = v;
            state.orientation = q;
            state.acc_bias = ba;
            state.gyro_bias = bg;
        }

        size_t findClosestStateIndex(double timestamp, const std::vector<BatchState>& states) {
            size_t closest_idx = 0;
            double min_dt = std::numeric_limits<double>::max();
            
            for (size_t i = 0; i < states.size(); ++i) {
                double dt = std::abs(timestamp - states[i].timestamp);
                if (dt < min_dt) {
                    min_dt = dt;
                    closest_idx = i;
                }
            }
            
            return closest_idx;
        }
    
        void stateToArray(const State& state, double* arr) {
            Eigen::Map<Eigen::Vector3d> p(arr);
            Eigen::Map<Eigen::Vector3d> v(arr + 3);
            Eigen::Map<Eigen::Quaterniond> q(arr + 6);
            Eigen::Map<Eigen::Vector3d> ba(arr + 10);
            Eigen::Map<Eigen::Vector3d> bg(arr + 13);
    
            p = state.position;
            v = state.velocity;
            q = state.orientation;
            ba = state.acc_bias;
            bg = state.gyro_bias;
        }
    
        void arrayToState(const double* arr, State& state) {
            Eigen::Map<const Eigen::Vector3d> p(arr);
            Eigen::Map<const Eigen::Vector3d> v(arr + 3);
            Eigen::Map<const Eigen::Quaterniond> q(arr + 6);
            Eigen::Map<const Eigen::Vector3d> ba(arr + 10);
            Eigen::Map<const Eigen::Vector3d> bg(arr + 13);
    
            state.position = p;
            state.velocity = v;
            state.orientation = q;
            state.acc_bias = ba;
            state.gyro_bias = bg;
        }
    
        void addUwbFactor(ceres::Problem& problem, 
                         const std::vector<double*>& parameter_blocks,
                         const UwbMeasurement& uwb_data) {
            // Find closest state
            size_t closest_idx = 0;
            double min_dt = std::numeric_limits<double>::max();
            
            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                double dt = std::abs(uwb_data.timestamp - 
                                   (last_imu_time_ - (parameter_blocks.size() - 1 - i) * 0.01));
                if (dt < min_dt) {
                    min_dt = dt;
                    closest_idx = i;
                }
            }
    
            Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity();
            auto* uwb_factor = new UwbFactor(uwb_data.position, uwb_cov);
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor),
                new ceres::HuberLoss(10.0),
                parameter_blocks[closest_idx]
            );
            std::cout<<"add the UWB factors \n";
        }
    
        void updateStates(const std::vector<State>& state_window,
                         const std::vector<double*>& parameter_blocks) {
            // Update current state with the latest optimized state
            arrayToState(parameter_blocks.back(), current_state_);
        }
    
        void publishResults() {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(current_state_.timestamp);
            odom_msg.header.frame_id = "map";
            
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
            
            pose_pub_.publish(odom_msg);
        }
    
        // ROS members
        ros::NodeHandle nh_;
        ros::Subscriber imu_sub_;
        ros::Subscriber uwb_sub_;
        ros::Publisher pose_pub_;
    
        // State and measurements
        State current_state_;
        std::deque<ImuMeasurement> imu_buffer_;
        std::deque<UwbMeasurement> uwb_buffer_;
        std::unique_ptr<ImuPreintegration> imu_preintegration_;
    
        // Parameters
        double imu_acc_noise_;
        double imu_gyro_noise_;
        double imu_acc_bias_noise_;
        double imu_gyro_bias_noise_;
        double uwb_noise_;
        
        size_t window_size_;
        size_t min_uwb_measurements_;
        
        bool initialized_;
        double last_imu_time_;
    
    // Add these members to the class
    private:
        size_t max_imu_queue_size_;
        size_t max_uwb_queue_size_;
        size_t min_imu_between_uwb_;
    };
 
 } // namespace uwb_imu_fusion
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "uwb_imu_fusion_node");
     uwb_imu_fusion::UwbImuFusion fusion;
     ros::spin();
     return 0;
 }