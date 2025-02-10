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
 
    //  struct PreintegrationResult {
    //      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //      Eigen::Vector3d delta_p;
    //      Eigen::Vector3d delta_v;
    //      Eigen::Quaterniond delta_q;
    //      Eigen::Matrix<double, 9, 6> jacobian_bias;
    //      Eigen::Matrix<double, 15, 15> covariance;
    //  };

     struct PreintegrationResult {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d delta_p;
        Eigen::Vector3d delta_v;
        Eigen::Quaterniond delta_q;
        Eigen::Matrix<double, 9, 6> jacobian_bias;
        Eigen::Matrix<double, 15, 15> covariance;
        double dt_sum;  // Add integration time sum
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
 
    //  PreintegrationResult getResult() const {
    //      PreintegrationResult result;
    //      result.delta_p = delta_p_;
    //      result.delta_v = delta_v_;
    //      result.delta_q = delta_q_;
    //      result.jacobian_bias = jacobian_bias_;
    //      result.covariance = covariance_;
    //      return result;
    //  }

     PreintegrationResult getResult() const {
        PreintegrationResult result;
        result.delta_p = delta_p_;
        result.delta_v = delta_v_;
        result.delta_q = delta_q_;
        result.jacobian_bias = jacobian_bias_;
        result.covariance = covariance_;
        result.dt_sum = dt_sum_;  // Add dt_sum to result
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
 };
 
 // Factor classes for optimization
 struct ImuFactor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit ImuFactor(const ImuPreintegration::PreintegrationResult& preintegration)
        : preintegration_(preintegration) {}

    template <typename T>
    bool operator()(const T* const state1, const T* const state2, T* residuals) const {
        // Extract states
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(state1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v1(state1 + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q1(state1 + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba1(state1 + 10);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg1(state1 + 13);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(state2);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v2(state2 + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q2(state2 + 6);

        // Convert preintegration measurements to T type
        Eigen::Matrix<T, 3, 1> dp = preintegration_.delta_p.cast<T>();
        Eigen::Matrix<T, 3, 1> dv = preintegration_.delta_v.cast<T>();
        Eigen::Quaternion<T> dq = preintegration_.delta_q.cast<T>();

        // Compute residuals
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residual(residuals);
        
        // Position residual
        residual.segment<3>(0) = p2 - (p1 + v1 * T(preintegration_.dt_sum) +  // Use dt_sum
                              T(0.5) * GRAVITY_VECTOR.cast<T>() * T(preintegration_.dt_sum * preintegration_.dt_sum) +
                              q1 * dp);

        // Velocity residual
        residual.segment<3>(3) = v2 - (v1 + GRAVITY_VECTOR.cast<T>() * T(preintegration_.dt_sum) +  // Use dt_sum
                              q1 * dv);

        // Rotation residual
        residual.segment<3>(6) = T(2.0) * ((dq * q1.conjugate() * q2).vec());

        return true;
    }

    ImuPreintegration::PreintegrationResult preintegration_;
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
 
         return true;
     }
 
     Eigen::Vector3d measurement_;
     Eigen::Matrix3d information_;
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
            imu_sub_ = nh_.subscribe("/imu", 1000, &UwbImuFusion::imuCallback, this);
            uwb_sub_ = nh_.subscribe("/uwb", 1000, &UwbImuFusion::uwbCallback, this);
            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/optimized_pose", 100);
    
            // Initialize IMU preintegration
            imu_preintegration_ = std::make_unique<ImuPreintegration>(
                current_state_.acc_bias, current_state_.gyro_bias);
    
            ROS_INFO("UWB-IMU Fusion node initialized.");
        }
    
    private:
        void initializeParameters() {
            // System parameters
            window_size_ = 10;  // Number of states in sliding window
            min_uwb_measurements_ = 3;  // Minimum UWB measurements before optimization
    
            // Noise parameters
            imu_acc_noise_ = 0.01;        // m/s^2
            imu_gyro_noise_ = 0.01;       // rad/s
            imu_acc_bias_noise_ = 0.0001; // m/s^3
            imu_gyro_bias_noise_ = 0.0001;// rad/s^2
            uwb_noise_ = 0.1;             // m
    
            // Initialize current state
            current_state_.timestamp = 0.0;
            current_state_.position.setZero();
            current_state_.velocity.setZero();
            current_state_.orientation.setIdentity();
            current_state_.acc_bias.setZero();
            current_state_.gyro_bias.setZero();
    
            initialized_ = false;
            last_imu_time_ = 0.0;
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
                return;
            }
    
            double dt = imu_data.timestamp - last_imu_time_;
            if (dt <= 0) return;
    
            // Propagate state
            propagateState(imu_data, dt);
    
            // Add to buffer
            imu_buffer_.push_back(imu_data);
    
            // Remove old IMU data
            while (imu_buffer_.size() > window_size_ * 100) {  // Assume 100Hz IMU data
                imu_buffer_.pop_front();
            }
    
            last_imu_time_ = imu_data.timestamp;
        }
    
        void processUwbData(const UwbMeasurement& uwb_data) {
            if (!initialized_) return;
    
            uwb_buffer_.push_back(uwb_data);
    
            // Remove old UWB data
            while (uwb_buffer_.size() > window_size_ * 10) {  // Assume 10Hz UWB data
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
    
            // Position integration
            current_state_.position += current_state_.velocity * dt +
                0.5 * (current_state_.orientation * acc_unbias + GRAVITY_VECTOR) * dt * dt;
    
            // Velocity integration
            current_state_.velocity += (current_state_.orientation * acc_unbias + GRAVITY_VECTOR) * dt;
    
            // Orientation integration
            Eigen::Vector3d angle_axis = gyro_unbias * dt;
            if (angle_axis.norm() > 1e-12) {
                current_state_.orientation *= Eigen::Quaterniond(
                    Eigen::AngleAxisd(angle_axis.norm(), angle_axis.normalized()));
            }
    
            // Normalize quaternion
            current_state_.orientation.normalize();
    
            // Update timestamp
            current_state_.timestamp = imu_data.timestamp;
    
            // Integrate IMU preintegration
            imu_preintegration_->integrate(imu_data.acc, imu_data.gyro, dt);
        }
    
        void optimize() {
            if (imu_buffer_.empty() || uwb_buffer_.empty()) return;
    
            // Setup optimization problem
            ceres::Problem problem;
            std::vector<State> state_window;
            std::vector<double*> parameter_blocks;
    
            // Initialize state window
            initializeStateWindow(state_window, parameter_blocks);
    
            // Add IMU factors
            for (size_t i = 0; i < state_window.size() - 1; ++i) {
                addImuFactor(problem, parameter_blocks[i], parameter_blocks[i + 1]);
            }
    
            // Add UWB factors
            for (const auto& uwb_data : uwb_buffer_) {
                addUwbFactor(problem, parameter_blocks, uwb_data);
            }
    
            // Set solver options
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations = 50;
            options.num_threads = 4;
    
            // Solve
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
    
            // Update states
            updateStates(state_window, parameter_blocks);
    
            // Cleanup
            for (auto ptr : parameter_blocks) {
                delete[] ptr;
            }
    
            // Reset IMU preintegration
            imu_preintegration_->reset();
            
            // Clear UWB buffer
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
    
        void addImuFactor(ceres::Problem& problem, double* state_i, double* state_j) {
            auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ImuFactor, 9, 16, 16>(imu_factor),
                new ceres::HubberdLossFunction(1.0),
                state_i,
                state_j
            );
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
                new ceres::HubberdLossFunction(1.0),
                parameter_blocks[closest_idx]
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
            odom_msg.header.frame_id = "world";
            
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
    };
 
 } // namespace uwb_imu_fusion
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "uwb_imu_fusion_node");
     uwb_imu_fusion::UwbImuFusion fusion;
     ros::spin();
     return 0;
 }