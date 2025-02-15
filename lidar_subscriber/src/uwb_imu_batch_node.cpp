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
 
 // Main fusion class
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
            // Initialize all flags and parameters first
            initialized_ = false;
            batch_initialized_ = false;
            
            // Initialize timestamps
            trajectory_start_time_ = 0.0;
            batch_start_time_ = 0.0;
            last_imu_time_ = 0.0;
            state_interval_ = 0.1;
            
            // Batch processing parameters
            batch_duration_ = 1.0;     // Process 5 seconds of data at a time
            state_dt_ = 0.1;          // State every 0.1 seconds, 0.1
            num_states_batch_ = static_cast<size_t>(batch_duration_ / state_dt_);

            // Adjusted noise parameters
            imu_acc_noise_ = 0.01;    // Reduced
            imu_gyro_noise_ = 0.001;  // Reduced
            imu_acc_bias_noise_ = 0.0001;
            imu_gyro_bias_noise_ = 0.0001;
            uwb_noise_ = 0.1;  // Increased for stronger position constraint
            
            // Add these parameters
            max_imu_queue_size_ = 10000;  // About 2.5s of IMU data at 400Hz (1000)
            max_uwb_queue_size_ = 30;    // About 3s of UWB data at 10Hz
            min_imu_between_uwb_ = 20;   // Minimum IMU measurements between UWB updates
            
            // Initialize state
            current_state_.timestamp = 0.0;
            current_state_.position.setZero();
            current_state_.velocity.setZero();
            current_state_.orientation.setIdentity();
            current_state_.acc_bias.setZero();
            current_state_.gyro_bias.setZero();
            
            // Initialize IMU preintegration
            imu_preintegration_ = std::make_unique<ImuPreintegration>(
                current_state_.acc_bias, current_state_.gyro_bias);
                        
            // Set up subscribers with proper queue sizes and transport hints
            ros::TransportHints transportHints;
            transportHints.tcpNoDelay();
            
            imu_sub_ = nh_.subscribe("/sensor_simulator/imu_data", 2000,
                &UwbImuFusion::imuCallback, this, transportHints);
            uwb_sub_ = nh_.subscribe("/sensor_simulator/UWBPoistionPS", 2000,
                &UwbImuFusion::uwbCallback, this, transportHints);
            
            // Set up publishers
            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/optimized_pose", 100); // high frequency from imu pre-integration
            path_pub_ = nh_.advertise<nav_msgs::Path>("/uwb_imu_fusion/trajectory", 100);
            batch_pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/batch_pose", 100);
            
            ROS_INFO("UWB-IMU Fusion node initialized.");
            
        }
    
    private:
        // Add these members

        struct BatchState {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            double timestamp;
            bool fixed;
            Eigen::Vector3d position;
            Eigen::Vector3d velocity;
            Eigen::Quaterniond orientation;
            Eigen::Vector3d acc_bias;
            Eigen::Vector3d gyro_bias;
            
            // Add constructor for easy initialization
            BatchState() {
                timestamp = 0.0;
                fixed = false;
                position.setZero();
                velocity.setZero();
                orientation.setIdentity();
                acc_bias.setZero();
                gyro_bias.setZero();
            }
        };
        
        std::vector<BatchState> trajectory_states_;
        std::vector<BatchState> full_trajectory_;
        double trajectory_start_time_;
        double state_interval_;  // Time interval between states
        std::deque<ImuMeasurement> full_imu_buffer_;
        std::deque<UwbMeasurement> full_uwb_buffer_;


        ros::Publisher path_pub_;
        ros::Publisher batch_pose_pub_;
        
        // Add these members
        std::vector<State> batch_states_;
        bool batch_initialized_;
        double batch_start_time_;
        double batch_duration_;
        // size_t batch_size_;

        // Batch parameters
        double state_dt_;           // Time between states in batch
        size_t num_states_batch_;   // Number of states in each batch

        // Add mutex protection for shared data
        std::mutex state_mutex_;
        std::mutex buffer_mutex_;
    
        void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
            // std::cout<<"imuCallback \n";
            static int imu_count = 0;
            static ros::Time last_print_time = ros::Time::now();
            
            imu_count++;
            ros::Time current_time = ros::Time::now();
            
            if ((current_time - last_print_time).toSec() >= 1.0) {
                // ROS_INFO("IMU frequency: %d Hz", imu_count);
                imu_count = 0;
                last_print_time = current_time;
            }
            
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            
            ImuMeasurement imu_data;
            imu_data.timestamp = msg->header.stamp.toSec();
            imu_data.acc = Eigen::Vector3d(msg->linear_acceleration.x,
                                          msg->linear_acceleration.y,
                                          msg->linear_acceleration.z);
            imu_data.gyro = Eigen::Vector3d(msg->angular_velocity.x,
                                           msg->angular_velocity.y,
                                           msg->angular_velocity.z);
            
            // Check for valid data
            if (std::isnan(imu_data.acc.norm()) || std::isnan(imu_data.gyro.norm())) {
                ROS_WARN("Received NaN in IMU data");
                return;
            }
            
            // Store in both buffers
            imu_buffer_.push_back(imu_data);
            full_imu_buffer_.push_back(imu_data);
            
            // Limit buffer sizes
            while (imu_buffer_.size() > max_imu_queue_size_) {
                imu_buffer_.pop_front();
            }
            while (full_imu_buffer_.size() > max_imu_queue_size_) {
                full_imu_buffer_.pop_front();
            }
            
            ROS_DEBUG("IMU buffers - Main: %zu, Full: %zu", 
                      imu_buffer_.size(), full_imu_buffer_.size());
            
            processImuData(imu_data);
        }
    
        void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(buffer_mutex_);  // Add lock here
            // std::cout<<"uwbCallback \n";
            UwbMeasurement uwb_data;
            uwb_data.timestamp = msg->header.stamp.toSec();
            uwb_data.position = Eigen::Vector3d(msg->point.x,
                                              msg->point.y,
                                              msg->point.z);
    
            processUwbData(uwb_data);
        }
    
        void processImuData(const ImuMeasurement& imu_data) {
            // ROS_INFO_THROTTLE(1.0, "Processing IMU data, initialized: %d", initialized_);
            
            if (!initialized_) {
                if (imu_buffer_.size() >= min_imu_between_uwb_) {
                    initialized_ = true;
                    last_imu_time_ = imu_data.timestamp;

                    // add by Weisong
                    trajectory_start_time_ = imu_buffer_.front().timestamp;
                    
                    ROS_INFO("System initialized with IMU data");
                    
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
                        current_state_ = initial_state;  // Also initialize current state
                        ROS_INFO("Batch processing initialized");
                    }
                }
                return;
            }
            
            double dt = imu_data.timestamp - last_imu_time_;
            if (dt > 0) {
                propagateStateWithImu(imu_data, dt);  // Changed from propagateState
                last_imu_time_ = imu_data.timestamp;
                
                publishState(current_state_);
            }
            
            // Check if batch duration is reached
            if (imu_data.timestamp - batch_start_time_ >= batch_duration_) {
                // ROS_INFO("Starting batch optimization");
                performBatchOptimization();
                // ROS_INFO("Completed batch optimization");
            }
        }
    
        void processUwbData(const UwbMeasurement& uwb_data) {
            // std::cout<<"processUwbData \n";
            if (!initialized_) {
                if (imu_buffer_.size() < min_imu_between_uwb_) {
                    return;
                }
                initialized_ = true;
                trajectory_start_time_ = imu_buffer_.front().timestamp;
            }
        
            // Store UWB measurement
            uwb_buffer_.push_back(uwb_data);
            full_uwb_buffer_.push_back(uwb_data);
            
            // Perform batch optimization (to be test)
            performFullBatchOptimization(uwb_data.timestamp);
        }

        void performFullBatchOptimization(double current_time) {
            // Calculate number of states needed
            double trajectory_duration = current_time - trajectory_start_time_;
            size_t num_states = static_cast<size_t>(trajectory_duration / state_interval_) + 1;
            
            // Initialize or extend trajectory
            if (full_trajectory_.empty()) {
                initializeFullTrajectory(num_states);
            } else {
                extendTrajectory(num_states);
            }
        
            // Setup optimization problem
            ceres::Problem problem;
            std::vector<double*> parameter_blocks;
            
            // Create and add parameter blocks first
            for (size_t i = 0; i < full_trajectory_.size(); ++i) {
                double* state_ptr = new double[16];
                batchStateToArray(full_trajectory_[i], state_ptr);
                
                // Add parameter block to problem BEFORE setting it constant
                problem.AddParameterBlock(state_ptr, 16);
                
                // Now we can safely set it constant if it's the first state
                if (i == 0) {
                    problem.SetParameterBlockConstant(state_ptr);
                }
                
                parameter_blocks.push_back(state_ptr);
            }
        
            // Add IMU factors
            for (size_t i = 0; i < full_trajectory_.size() - 1; ++i) {
                addImuFactorBetweenStates(problem, i, i + 1, parameter_blocks);
            }
        
            // Add UWB factors
            for (const auto& uwb : full_uwb_buffer_) {
                size_t closest_idx = findClosestStateIndex(uwb.timestamp, full_trajectory_);
                addUwbFactor(problem, parameter_blocks[closest_idx], uwb);
            }
        
            // Solve optimization problem
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.max_num_iterations = 100;
            options.minimizer_progress_to_stdout = true;
        
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        
            // Update trajectory if optimization succeeded
            if (summary.termination_type == ceres::CONVERGENCE) {
                for (size_t i = 0; i < full_trajectory_.size(); ++i) {
                    arrayToBatchState(parameter_blocks[i], full_trajectory_[i]);
                }
                publishTrajectory();
            }
        
            // Cleanup
            for (auto ptr : parameter_blocks) {
                delete[] ptr;
            }
        }


        void initializeFullTrajectory(size_t num_states) {
            full_trajectory_.clear();
            
            for (size_t i = 0; i < num_states; ++i) {
                BatchState state;
                state.timestamp = trajectory_start_time_ + i * state_interval_;
                
                if (i == 0) {
                    state.position = full_uwb_buffer_.front().position;
                } else {
                    propagateBatchState(state, full_trajectory_.back(), state_interval_);  // Changed from propagateState
                }
                
                full_trajectory_.push_back(state);
            }
        }
        
        void extendTrajectory(size_t new_size) {
            if (new_size <= full_trajectory_.size()) {
                return;
            }
            
            size_t current_size = full_trajectory_.size();
            for (size_t i = current_size; i < new_size; ++i) {
                BatchState state;
                state.timestamp = trajectory_start_time_ + i * state_interval_;
                propagateBatchState(state, full_trajectory_.back(), state_interval_);  // Changed from propagateState
                full_trajectory_.push_back(state);
            }
        }
        
        void addImuFactorBetweenStates(ceres::Problem& problem, 
            size_t idx1, 
            size_t idx2,
            const std::vector<double*>& parameter_blocks) {
                // Validate indices
                if (idx1 >= parameter_blocks.size() || idx2 >= parameter_blocks.size()) {
                ROS_ERROR("Invalid indices in addImuFactorBetweenStates");
                return;
                }

                double t1 = full_trajectory_[idx1].timestamp;
                double t2 = full_trajectory_[idx2].timestamp;

                imu_preintegration_->reset();

                // Integrate IMU measurements between states
                for (const auto& imu : full_imu_buffer_) {
                if (imu.timestamp >= t1 && imu.timestamp < t2) {
                double dt = imu.timestamp - t1;
                imu_preintegration_->integrate(imu.acc, imu.gyro, dt);
                t1 = imu.timestamp;
                }
                }

                auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
                problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor),
                new ceres::HuberLoss(1.0),
                parameter_blocks[idx1],
                parameter_blocks[idx2]
                );
            }
    
        void propagateStateWithImu(const ImuMeasurement& imu_data, double dt) {
            // Check for numerical stability
            if (std::isnan(dt) || std::isinf(dt)) {
                ROS_ERROR("Invalid dt in propagateState");
                return;
            }
        
            // Check for NaN in state
            if (current_state_.position.hasNaN() || 
                current_state_.velocity.hasNaN() ||
                current_state_.orientation.coeffs().hasNaN()) {
                ROS_ERROR("NaN detected in state");
                return;
            }
        
            // Remove bias and integrate IMU data
            Eigen::Vector3d acc_unbias = imu_data.acc - current_state_.acc_bias;
            Eigen::Vector3d gyro_unbias = imu_data.gyro - current_state_.gyro_bias;
        
            // Add strict limits
            const double MAX_ACC = 20.0;  // m/s^2
            const double MAX_GYRO = 50.0;  // rad/s
            acc_unbias = acc_unbias.array().min(MAX_ACC).max(-MAX_ACC);
            gyro_unbias = gyro_unbias.array().min(MAX_GYRO).max(-MAX_GYRO);
        
            // First update orientation
            Eigen::Vector3d angle_axis = gyro_unbias * dt;
            if (angle_axis.norm() > 1e-8) {
                current_state_.orientation = current_state_.orientation * 
                    Eigen::Quaterniond(Eigen::AngleAxisd(angle_axis.norm(), angle_axis.normalized()));
            }
            current_state_.orientation.normalize();
        
            // Then update position and velocity in world frame
            Eigen::Vector3d acc_world = current_state_.orientation * acc_unbias + GRAVITY_VECTOR;
            
            // Stricter velocity limits
            const double MAX_VELOCITY = 50.0;  // m/s
            
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

        void propagateBatchState(BatchState& next_state, const BatchState& current_state, double dt) {
            // Check for numerical stability
            if (std::isnan(dt) || std::isinf(dt)) {
                ROS_ERROR("Invalid dt in propagateBatchState");
                return;
            }
        
            // Find relevant IMU measurements
            std::vector<ImuMeasurement> imu_measurements;
            for (const auto& imu : full_imu_buffer_) {
                if (imu.timestamp >= current_state.timestamp && 
                    imu.timestamp < current_state.timestamp + dt) {
                    imu_measurements.push_back(imu);
                }
            }
        
            // If no IMU measurements, do simple linear propagation
            if (imu_measurements.empty()) {
                next_state.position = current_state.position + current_state.velocity * dt;
                next_state.velocity = current_state.velocity;
                next_state.orientation = current_state.orientation;
                next_state.acc_bias = current_state.acc_bias;
                next_state.gyro_bias = current_state.gyro_bias;
                return;
            }
        
            // Initialize next state
            next_state = current_state;
        
            // Integrate IMU measurements
            for (size_t i = 0; i < imu_measurements.size(); ++i) {
                const auto& imu = imu_measurements[i];
                double delta_t = (i == imu_measurements.size() - 1) ? 
                    (current_state.timestamp + dt - imu.timestamp) :
                    (imu_measurements[i + 1].timestamp - imu.timestamp);
        
                // Remove bias
                Eigen::Vector3d acc_unbias = imu.acc - current_state.acc_bias;
                Eigen::Vector3d gyro_unbias = imu.gyro - current_state.gyro_bias;
        
                // Update orientation
                Eigen::Vector3d angle_axis = gyro_unbias * delta_t;
                if (angle_axis.norm() > 1e-8) {
                    next_state.orientation = next_state.orientation * 
                        Eigen::Quaterniond(Eigen::AngleAxisd(angle_axis.norm(), angle_axis.normalized()));
                }
                next_state.orientation.normalize();
        
                // Update position and velocity
                Eigen::Vector3d acc_world = next_state.orientation * acc_unbias + GRAVITY_VECTOR;
                next_state.velocity += acc_world * delta_t;
                next_state.position += next_state.velocity * delta_t + 
                                     0.5 * acc_world * delta_t * delta_t;
            }
            
            next_state.timestamp = current_state.timestamp + dt;
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
            std::cout << "imu_buffer_.size() : " <<  imu_buffer_.size()<<std::endl;
            std::cout << "uwb_buffer_.size() : " <<  uwb_buffer_.size()<<std::endl;
            if (imu_buffer_.empty() || uwb_buffer_.empty()) {
                return;
            }

            std::cout << "performBatchOptimization : " <<  std::endl;
        
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
                    new ceres::HuberLoss(0.1),
                    parameter_blocks[closest_idx]
                );
            }
        
            // Solve optimization problem
            ceres::Solver::Options options;
            options.minimizer_progress_to_stdout = false;

            options.linear_solver_type = ceres::DENSE_SCHUR;
            //options.num_threads = 2;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.max_num_iterations = 50;
            
            auto t1 = ros::WallTime::now();
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
            auto t2 = ros::WallTime::now();
            std::cout << "Optimization time : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;
            std::cout << "Window size : " << parameter_blocks.size() << std::endl;
        
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
    
        void addUwbFactor(ceres::Problem& problem, double* state_ptr, 
            const UwbMeasurement& uwb_data) {
            // Make sure state_ptr is valid
            if (state_ptr == nullptr) {
                ROS_ERROR("Invalid state pointer in addUwbFactor");
                return;
                }

                Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity();
                auto* uwb_factor = new UwbFactor(uwb_data.position, uwb_cov);

                // Add residual block with parameter
                problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor),
                new ceres::HuberLoss(0.1),
                state_ptr
            );
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
    ROS_INFO("Starting UWB-IMU fusion node...");
    
    ros::NodeHandle nh;
        
    uwb_imu_fusion::UwbImuFusion fusion;
    
    ROS_INFO("Node initialized, spinning...");
    ros::spin();
    return 0;
}