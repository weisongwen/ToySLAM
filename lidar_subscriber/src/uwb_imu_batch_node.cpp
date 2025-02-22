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

// Add these new headers
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
 
 namespace uwb_imu_fusion {
 
 // Constants
 const double GRAVITY_MAGNITUDE = 9.81;
 const Eigen::Vector3d GRAVITY_VECTOR(0, 0, -GRAVITY_MAGNITUDE);
 
 // Measurement structures
 struct ImuMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    Eigen::Quaterniond orientation;  // Add IMU orientation
    double timestamp;
};
 
 struct UwbMeasurement {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     Eigen::Vector3d position;
     double timestamp;
 };

 class MarginalizationInfo {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MarginalizationInfo() {
            valid_ = false;
            linearized_ = false;
            parameter_block_size_ = 16;  // State size
            residual_size_ = 15;         // Residual size
            
            // Initialize matrices with proper sizes
            linearized_jacobian_.resize(residual_size_, parameter_block_size_);
            linearized_residual_.resize(residual_size_);
            linearized_jacobian_.setZero();
            linearized_residual_.setZero();
        }

        ~MarginalizationInfo() {
            for (auto& info : residual_block_infos_) {
                if (info) {
                    delete info->cost_function;
                    delete info->loss_function;
                    delete info;
                }
            }
            residual_block_infos_.clear();
        }

        int getParameterBlockSize() const {
            return parameter_block_size_;
        }

        int getResidualSize() const {
            return residual_size_;
        }

        // Modified to work with mapped types
        void computeResidual(const double* parameters, double* residuals) const {
            if (!valid_ || !linearized_ || !parameters || !residuals) {
                std::fill(residuals, residuals + residual_size_, 0.0);
                return;
            }
            // if (!valid_ || !linearized_) {
            //     std::fill(residuals, residuals + residual_size_, 0.0);
            //     return;
            // }
        
            try {
                Eigen::Map<const Eigen::Matrix<double, 16, 1>> param_vec(parameters);
                Eigen::Map<Eigen::Matrix<double, 15, 1>> residual_vec(residuals);
        
                residual_vec.setZero();
        
                if (linearized_jacobian_.rows() > 0 && linearized_residual_.size() > 0) {
                    int actual_rows = std::min(15, static_cast<int>(linearized_jacobian_.rows()));
                    residual_vec.head(actual_rows) = 
                        linearized_jacobian_.topRows(actual_rows) * param_vec + 
                        linearized_residual_.head(actual_rows);
                }
            } catch (const std::exception& e) {
                ROS_ERROR_STREAM("Exception in computeResidual: " << e.what());
                std::fill(residuals, residuals + residual_size_, 0.0);
            }
        }

        void addResidualBlock(ceres::CostFunction* cost_function,
                            ceres::LossFunction* loss_function,
                            const std::vector<double*>& parameter_blocks,
                            std::vector<int> drop_set) {
            ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(
                cost_function, loss_function, parameter_blocks, drop_set);
            residual_block_infos_.push_back(residual_block_info);
        }
    
        void marginalize() {
            if (valid_) return;
    
            // Construct linear system
            constructLinearSystem();
    
            // Perform Schur complement marginalization
            performSchurComplement();
    
            valid_ = true;
        }

        const Eigen::MatrixXd& getLinearizedJacobian() const {
            return linearized_jacobian_;
        }    
    
    private:
        struct ResidualBlockInfo {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
            ResidualBlockInfo(ceres::CostFunction* _cost_function,
                             ceres::LossFunction* _loss_function,
                             std::vector<double*> _parameter_blocks,
                             std::vector<int> _drop_set)
                : cost_function(_cost_function),
                  loss_function(_loss_function),
                  parameter_blocks(_parameter_blocks),
                  drop_set(_drop_set) {}
    
            ceres::CostFunction* cost_function;
            ceres::LossFunction* loss_function;
            std::vector<double*> parameter_blocks;
            std::vector<int> drop_set;
        };
    
        void constructLinearSystem() {
            if (linearized_) return;
        
            // Pre-calculate total residual size
            int total_residual_size = 0;
            for (const auto& info : residual_block_infos_) {
                if (info && info->cost_function) {
                    total_residual_size += info->cost_function->num_residuals();
                }
            }
        
            if (total_residual_size == 0) {
                ROS_WARN("No residuals to construct linear system");
                linearized_ = true;
                return;
            }
        
            // Create temporary matrices with proper sizes
            Eigen::MatrixXd H_temp(total_residual_size, parameter_block_size_);
            Eigen::VectorXd b_temp(total_residual_size);
            H_temp.setZero();
            b_temp.setZero();
        
            int current_row = 0;
            for (const auto& info : residual_block_infos_) {
                if (!info || !info->cost_function) continue;
        
                int residual_size = info->cost_function->num_residuals();
                if (residual_size <= 0) continue;
        
                // Evaluate the residual
                std::vector<double> residual(residual_size, 0.0);
                std::vector<double*> jacobian_raw(info->parameter_blocks.size(), NULL);
                std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                    jacobian_matrices(info->parameter_blocks.size());
        
                // Prepare jacobian matrices
                for (size_t i = 0; i < info->parameter_blocks.size(); ++i) {
                    jacobian_matrices[i].resize(residual_size, parameter_block_size_);
                    jacobian_matrices[i].setZero();
                    jacobian_raw[i] = jacobian_matrices[i].data();
                }
        
                // Evaluate residual block
                if (!info->cost_function->Evaluate(
                    info->parameter_blocks.data(),
                    residual.data(),
                    jacobian_raw.data())) {
                    ROS_WARN("Residual block evaluation failed");
                    continue;
                }
        
                // Safely copy data to temporary matrices
                if (current_row + residual_size <= total_residual_size) {
                    b_temp.segment(current_row, residual_size) = 
                        Eigen::Map<Eigen::VectorXd>(residual.data(), residual_size);
        
                    for (size_t i = 0; i < info->parameter_blocks.size(); ++i) {
                        int col_offset = i * parameter_block_size_;
                        if (col_offset + parameter_block_size_ <= H_temp.cols()) {
                            H_temp.block(current_row, col_offset, residual_size, parameter_block_size_) = 
                                jacobian_matrices[i];
                        }
                    }
                    current_row += residual_size;
                }
            }
        
            // Resize final matrices to match the actual data
            if (current_row > 0) {
                linearized_jacobian_ = H_temp.topRows(current_row);
                linearized_residual_ = b_temp.head(current_row);
            } else {
                linearized_jacobian_.resize(0, parameter_block_size_);
                linearized_residual_.resize(0);
            }
            
            linearized_ = true;
        }
        
        void performSchurComplement() {
            if (!linearized_) {
                constructLinearSystem();
            }
        
            // Ensure matrices are properly sized
            if (linearized_jacobian_.rows() == 0 || linearized_jacobian_.cols() == 0) {
                ROS_WARN("Empty matrices in Schur complement");
                valid_ = true;
                return;
            }
        
            // Get marginalization size
            int marg_size = std::min(parameter_block_size_, 
                                    static_cast<int>(linearized_jacobian_.cols()));
        
            try {
                // Compute Schur complement
                Eigen::MatrixXd H_mm = linearized_jacobian_.leftCols(marg_size);
                Eigen::VectorXd b_m;
                
                if (linearized_residual_.size() >= marg_size) {
                    b_m = linearized_residual_.head(marg_size);
                } else {
                    b_m.resize(marg_size);
                    b_m.setZero();
                }
        
                // Add damping
                double lambda = 1e-6;
                H_mm.diagonal().array() += lambda;
        
                // Update matrices
                linearized_jacobian_ = H_mm;
                linearized_residual_ = b_m;
        
            } catch (const std::exception& e) {
                ROS_ERROR_STREAM("Exception in performSchurComplement: " << e.what());
                // Set to safe values
                linearized_jacobian_.resize(residual_size_, parameter_block_size_);
                linearized_residual_.resize(residual_size_);
                linearized_jacobian_.setZero();
                linearized_residual_.setZero();
            }
        
            valid_ = true;
        }
    
        std::vector<ResidualBlockInfo*> residual_block_infos_;
        Eigen::MatrixXd linearized_jacobian_;
        Eigen::VectorXd linearized_residual_;
        bool valid_;
        bool linearized_;
        int parameter_block_size_;  // Add these member variables
        int residual_size_;
    };
    
    // Modify MarginalizationFactor to use dynamic sized residuals
    // Modified MarginalizationFactor to use fixed sizes
    class MarginalizationFactor {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
            explicit MarginalizationFactor(const std::shared_ptr<MarginalizationInfo>& _marginalization_info)
                : marginalization_info(_marginalization_info) {
                CHECK(_marginalization_info != NULL);
            }
        
            template <typename T>
            bool operator()(const T* const parameters, T* residuals) const {
                if (!marginalization_info) {
                    // Fill residuals with zeros if no marginalization info
                    for (int i = 0; i < 15; ++i) {
                        residuals[i] = T(0);
                    }
                    return true;
                }
        
                try {
                    std::vector<double> parameters_double(16);
                    std::vector<double> residuals_double(15);
        
                    // Convert parameters to double
                    for (int i = 0; i < 16; ++i) {
                        parameters_double[i] = getValue(parameters[i]);
                    }
        
                    // Compute residuals using double values
                    marginalization_info->computeResidual(parameters_double.data(), residuals_double.data());
        
                    // Convert residuals back and apply jacobian
                    const Eigen::MatrixXd& jac = marginalization_info->getLinearizedJacobian();
                    Eigen::Map<const Eigen::Matrix<double, 15, 16, Eigen::RowMajor>> J(jac.data());
        
                    for (int i = 0; i < 15; ++i) {
                        // Initialize with the base residual
                        residuals[i] = T(residuals_double[i]);
        
                        // Add jacobian contributions
                        for (int j = 0; j < 16; ++j) {
                            T diff = parameters[j] - T(parameters_double[j]);
                            residuals[i] += T(J(i, j)) * diff;
                        }
                    }
        
                    return true;
                } catch (const std::exception& e) {
                    // Handle any exceptions
                    for (int i = 0; i < 15; ++i) {
                        residuals[i] = T(0);
                    }
                    return false;
                }
            }
        
        private:
            // Helper function to get value from either Jet or double
            template <typename T>
            static double getValue(const T& x) {
                return static_cast<double>(x);
            }
        
            // Specialization for Jet types
            template <typename T, int N>
            static double getValue(const ceres::Jet<T, N>& x) {
                return static_cast<double>(x.a);
            }
        
            std::shared_ptr<MarginalizationInfo> marginalization_info;
        };
        
        // Helper class to create the cost function
        class MarginalizationFactorAutoDiff {
        public:
            explicit MarginalizationFactorAutoDiff(const std::shared_ptr<MarginalizationInfo>& info)
                : factor_(info) {}
        
            static ceres::CostFunction* Create(const std::shared_ptr<MarginalizationInfo>& info) {
                return new ceres::AutoDiffCostFunction<MarginalizationFactor, 15, 16>(
                    new MarginalizationFactor(info));
            }
        
        private:
            MarginalizationFactor factor_;
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
    
        Eigen::Matrix<double, 15, 15> covariance;  // Add covariance matrix
        
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
            covariance.setZero();  // Initialize covariance
        }
    };
 
    ImuPreintegration(const Eigen::Vector3d& acc_bias, const Eigen::Vector3d& gyro_bias)
    : acc_bias_(acc_bias), gyro_bias_(gyro_bias) {
        acc_noise_ = 0.01 * 0.01;      // Square the noise values
        gyro_noise_ = 0.001 * 0.001;   // Square the noise values
        acc_bias_noise_ = 0.0001 * 0.0001;
        gyro_bias_noise_ = 0.0001 * 0.0001;
        reset();
    }

     
 
     void reset() {
        delta_p_.setZero();
        delta_v_.setZero();
        delta_q_.setIdentity();
        
        // Initialize all jacobians
        jacobian_p_ba_.setZero();
        jacobian_p_bg_.setZero();
        jacobian_v_ba_.setZero();
        jacobian_v_bg_.setZero();
        jacobian_q_bg_.setZero();
        
        covariance_.setZero();
        dt_sum_ = 0;
    
        // Set initial noise covariance
        Eigen::Matrix<double, 15, 15> noise_cov = Eigen::Matrix<double, 15, 15>::Zero();
        noise_cov.block<3, 3>(0, 0) = acc_noise_ * acc_noise_ * Eigen::Matrix3d::Identity();
        noise_cov.block<3, 3>(3, 3) = gyro_noise_ * gyro_noise_ * Eigen::Matrix3d::Identity();
        noise_cov.block<3, 3>(6, 6) = acc_bias_noise_ * acc_bias_noise_ * Eigen::Matrix3d::Identity();
        noise_cov.block<3, 3>(9, 9) = gyro_bias_noise_ * gyro_bias_noise_ * Eigen::Matrix3d::Identity();
        noise_cov_ = noise_cov;
    }
 
    void integrate(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt) {
        if (dt <= 0) return;
    
        // Remove bias
        Eigen::Vector3d acc_unbiased = acc - acc_bias_;
        Eigen::Vector3d gyro_unbiased = gyro - gyro_bias_;
    
        // Current rotation matrix
        Eigen::Matrix3d R_k = delta_q_.toRotationMatrix();
    
        // Integrate rotation using exponential map
        Eigen::Vector3d omega = gyro_unbiased;
        double theta = omega.norm() * dt;
        Eigen::Matrix3d Omega = skewSymmetric(omega);
        Eigen::Matrix3d exp_omega;
        
        if (theta < 1e-10) {
            exp_omega = Eigen::Matrix3d::Identity();
        } else {
            exp_omega = Eigen::Matrix3d::Identity() + 
                       (sin(theta) / theta) * Omega +
                       ((1 - cos(theta)) / (theta * theta)) * Omega * Omega;
        }
    
        // Update orientation
        Eigen::Matrix3d R_kp1 = R_k * exp_omega;
        delta_q_ = Eigen::Quaterniond(R_kp1);
        delta_q_.normalize();
    
        // Compute midpoint rotation (average between R_k and R_kp1)
        double half_theta = theta * 0.5;
        Eigen::Matrix3d R_mid;
        if (half_theta < 1e-10) {
            R_mid = R_k;
        } else {
            Eigen::Matrix3d half_exp = Eigen::Matrix3d::Identity() + 
                                      (sin(half_theta) / half_theta) * (Omega * 0.5) +
                                      ((1 - cos(half_theta)) / (half_theta * half_theta)) * (Omega * 0.5) * (Omega * 0.5);
            R_mid = R_k * half_exp;
        }
    
        // Update velocity and position
        Eigen::Vector3d acc_world = R_mid * acc_unbiased;
        delta_v_ += acc_world * dt;
        delta_p_ += delta_v_ * dt + 0.5 * acc_world * dt * dt;
    
        // Update bias Jacobians
        Eigen::Matrix3d acc_skew = skewSymmetric(acc_unbiased);
        jacobian_v_ba_ += -R_mid * dt;
        jacobian_p_ba_ += -R_mid * dt * dt / 2.0;
        
        Eigen::Matrix3d right_jacobian;
        if (theta < 1e-10) {
            right_jacobian = Eigen::Matrix3d::Identity();
        } else {
            right_jacobian = Eigen::Matrix3d::Identity() -
                            ((1 - cos(theta)) / (theta * theta)) * Omega +
                            ((theta - sin(theta)) / (theta * theta * theta)) * Omega * Omega;
        }
        
        jacobian_q_bg_ = -right_jacobian * dt;
        jacobian_v_bg_ += -R_mid * acc_skew * jacobian_q_bg_;
        jacobian_p_bg_ += jacobian_v_bg_ * dt;
    
        // Update timestamp
        dt_sum_ += dt;
    
        // Update covariance
        updateJacobianAndCovariance(acc_unbiased, gyro_unbiased, dt, R_k);
    }

    Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& omega, double dt) {
        double theta = omega.norm() * dt;
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity();
        }
        
        Eigen::Matrix3d Omega = skewSymmetric(omega);
        return Eigen::Matrix3d::Identity() -
               ((1 - cos(theta)) / (theta * theta)) * Omega +
               ((theta - sin(theta)) / (theta * theta * theta)) * Omega * Omega;
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
            // Use smaller weights initially for better convergence
            sqrt_information_ = Eigen::Matrix<double, 15, 15>::Identity();
            sqrt_information_.block<3,3>(0,0) *= 0.1;     // position weight
            sqrt_information_.block<3,3>(3,3) *= 0.1;     // velocity weight
            sqrt_information_.block<3,3>(6,6) *= 1.0;     // rotation weight
            sqrt_information_.block<3,3>(9,9) *= 0.1;     // acc bias weight
            sqrt_information_.block<3,3>(12,12) *= 0.1;   // gyro bias weight

            // Initialize gravity
            gravity_ = Eigen::Vector3d(0, 0, -GRAVITY_MAGNITUDE);
        }

        template <typename T>
        bool operator()(const T* const state1, const T* const state2, T* residuals) const {
            try {
                // Map states
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(state1);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> v1(state1 + 3);
                Eigen::Map<const Eigen::Quaternion<T>> q1(state1 + 6);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba1(state1 + 10);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg1(state1 + 13);

                Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(state2);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> v2(state2 + 3);
                Eigen::Map<const Eigen::Quaternion<T>> q2(state2 + 6);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba2(state2 + 10);
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg2(state2 + 13);

                // Normalize quaternions
                Eigen::Quaternion<T> q1_normalized = q1.normalized();
                Eigen::Quaternion<T> q2_normalized = q2.normalized();

                // Time delta
                T delta_t = T(preintegration_.dt_sum);
                if (delta_t <= T(0)) {
                    setZeroResiduals(residuals);
                    return true;
                }

                // Bias deltas
                Eigen::Matrix<T, 3, 1> dba = ba2 - ba1;
                Eigen::Matrix<T, 3, 1> dbg = bg2 - bg1;

                // Get preintegrated measurements with bias corrections
                Eigen::Matrix<T, 3, 1> delta_p = preintegration_.delta_p.cast<T>();
                Eigen::Matrix<T, 3, 1> delta_v = preintegration_.delta_v.cast<T>();
                Eigen::Quaternion<T> delta_q = preintegration_.delta_q.cast<T>();

                // Apply first-order corrections for bias updates
                delta_p = delta_p + 
                        preintegration_.jacobian_p_ba.cast<T>() * dba + 
                        preintegration_.jacobian_p_bg.cast<T>() * dbg;
                
                delta_v = delta_v + 
                        preintegration_.jacobian_v_ba.cast<T>() * dba + 
                        preintegration_.jacobian_v_bg.cast<T>() * dbg;

                // Gravity vector
                Eigen::Matrix<T, 3, 1> gravity = gravity_.cast<T>();

                // Compute residuals
                Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residuals);

                // Position residual
                residual.template segment<3>(0) = q1_normalized.inverse() * (p2 - p1 - v1 * delta_t - 
                    T(0.5) * gravity * delta_t * delta_t) - delta_p;

                // Velocity residual
                residual.template segment<3>(3) = q1_normalized.inverse() * (v2 - v1 - gravity * delta_t) - 
                    delta_v;

                // Rotation residual
                Eigen::Quaternion<T> q_error = (delta_q.inverse() * (q1_normalized.inverse() * q2_normalized));
                residual.template segment<3>(6) = T(2.0) * q_error.vec();

                // Bias residuals
                residual.template segment<3>(9) = dba;
                residual.template segment<3>(12) = dbg;

                // Apply information matrix
                residual = sqrt_information_.cast<T>() * residual;

                return true;
            } catch (...) {
                setZeroResiduals(residuals);
                return false;
            }
        }

    private:
        template <typename T>
        void setZeroResiduals(T* residuals) const {
            for (int i = 0; i < 15; ++i) {
                residuals[i] = T(0);
            }
        }

        ImuPreintegration::PreintegrationResult preintegration_;
        Eigen::Matrix<double, 15, 15> sqrt_information_;
        Eigen::Vector3d gravity_;
    };


    struct ImuOrientationFactor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        ImuOrientationFactor(const Eigen::Quaterniond& measured_orientation, double weight = 1.0)
            : measured_orientation_(measured_orientation), weight_(weight) {
                sqrt_information_ = weight * Eigen::Matrix3d::Identity();
        }
    
        template <typename T>
        bool operator()(const T* const state, T* residuals) const {
            Eigen::Map<const Eigen::Quaternion<T>> q_state(state + 6);
            Eigen::Quaternion<T> q_state_normalized = q_state.normalized();
            Eigen::Quaternion<T> q_measured = measured_orientation_.cast<T>();
    
            // Compute orientation error
            Eigen::Quaternion<T> q_error = q_measured.conjugate() * q_state_normalized;
            
            // Convert to angle-axis representation
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
            residual = T(2.0) * q_error.vec();
    
            // Apply weight
            residual = sqrt_information_.cast<T>() * residual;
    
            return true;
        }
    
    private:
        Eigen::Quaterniond measured_orientation_;
        Eigen::Matrix3d sqrt_information_;
        double weight_;
    };

    // Add this new factor
    struct OrientationRegularizationFactor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        OrientationRegularizationFactor(double weight = 1.0) : weight_(weight) {}
    
        template <typename T>
        bool operator()(const T* const state1, const T* const state2, T* residuals) const {
            Eigen::Map<const Eigen::Quaternion<T>> q1(state1 + 6);
            Eigen::Map<const Eigen::Quaternion<T>> q2(state2 + 6);
    
            // Normalize quaternions
            Eigen::Quaternion<T> q1_norm = q1.normalized();
            Eigen::Quaternion<T> q2_norm = q2.normalized();
    
            // Compute relative rotation
            Eigen::Quaternion<T> q_diff = q1_norm.conjugate() * q2_norm;
    
            // Extract vector part as residual (simpler approach)
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
            
            // Scale the vector part of the quaternion
            residual = q_diff.vec() * T(2.0) * T(weight_);
    
            return true;
        }
    
    private:
        double weight_;
    };

    struct UwbFactor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        UwbFactor(const Eigen::Vector3d& measurement, const Eigen::Matrix3d& covariance)
            : measurement_(measurement) {
            // Use Cholesky decomposition for numerical stability
            Eigen::LLT<Eigen::Matrix3d> llt_of_info(covariance.inverse());
            sqrt_information_ = llt_of_info.matrixL().transpose();
        }

        template <typename T>
        bool operator()(const T* const state, T* residuals) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(state);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);

            // Compute position error
            residual = position - measurement_.cast<T>();
            
            // Scale by square root information matrix
            residual = sqrt_information_.cast<T>() * residual;

            return true;
        }

        private:
            Eigen::Vector3d measurement_;
            Eigen::Matrix3d sqrt_information_;
    };

    struct BiasRegularizationFactor {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        BiasRegularizationFactor(double acc_weight = 1.0, double gyro_weight = 1.0) 
            : acc_weight_(acc_weight), gyro_weight_(gyro_weight) {}

        template <typename T>
        bool operator()(const T* const state, T* residuals) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> acc_bias(state + 10);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> gyro_bias(state + 13);
            
            Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residuals);
            
            // Limit maximum bias values
            const T max_acc_bias = T(0.5);
            const T max_gyro_bias = T(0.2);
            
            residual.template head<3>() = (acc_bias.array().min(max_acc_bias).max(-max_acc_bias)) * T(acc_weight_);
            residual.template tail<3>() = (gyro_bias.array().min(max_gyro_bias).max(-max_gyro_bias)) * T(gyro_weight_);
            
            return true;
        }

        private:
            double acc_weight_;
            double gyro_weight_;
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

        
    
        UwbImuFusion() : running_(true)  // Initialize atomic bool in constructor initializer list
        {
            // running_ = true;  // Add this before starting the thread
            initializeParameters();

            // Start processing thread
            imu_process_thread_ = std::thread(&UwbImuFusion::processImuLoop, this);
            
            // ROS subscribers and publishers
            // imu_sub_ = nh_.subscribe("/imu/data", 1000, &UwbImuFusion::imuCallback, this);
            // uwb_sub_ = nh_.subscribe("/vins_estimator/UWBPoistionPS", 1000, &UwbImuFusion::uwbCallback, this);

            imu_sub_ = nh_.subscribe("/sensor_simulator/imu_data", 1000, &UwbImuFusion::imuCallback, this);
            uwb_sub_ = nh_.subscribe("/sensor_simulator/UWBPoistionPS", 1000, &UwbImuFusion::uwbCallback, this);


            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/optimized_pose", 100);

            // In initializeParameters()
            high_freq_pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/high_freq_pose", 200);
    
            // Initialize IMU preintegration
            imu_preintegration_ = std::make_unique<ImuPreintegration>(
                current_state_.acc_bias, current_state_.gyro_bias);
    
            ROS_INFO("UWB-IMU Fusion node initialized.");
        }

        ~UwbImuFusion() {
            running_ = false;
            if (imu_process_thread_.joinable()) {
                imu_process_thread_.join();
            }
            // Clean up marginalization resources
            last_marginalization_info_.reset();
            for (auto ptr : last_marginalization_parameter_blocks_) {
                delete[] ptr;
            }
            last_marginalization_parameter_blocks_.clear();
        }
    
    private:
        // Add these members
        

        struct BatchState {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            
            double timestamp;
            bool fixed;
            
            // State variables in the same order as the parameter block
            Eigen::Vector3d position;      // Indices 0-2
            Eigen::Vector3d velocity;      // Indices 3-5
            Eigen::Quaterniond orientation; // Indices 6-9
            Eigen::Vector3d acc_bias;      // Indices 10-12
            Eigen::Vector3d gyro_bias;     // Indices 13-15
        };
        std::vector<BatchState> trajectory_states_;
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

        void initializeParameters() {

            // Batch processing parameters
            batch_duration_ = 2.0;     // Process 5 seconds of data at a time
            state_dt_ = 0.1;          // State every 0.1 seconds, 0.1
            num_states_batch_ = static_cast<size_t>(batch_duration_ / state_dt_);
            batch_initialized_ = false;
            // Add batch parameters
            // batch_size_ = 100;  // Number of states in batch
            
            // Publishers
            path_pub_ = nh_.advertise<nav_msgs::Path>("/uwb_imu_fusion/trajectory", 1);
            batch_pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/batch_pose", 1);
            pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/uwb_imu_fusion/pose", 1);


            // System parameters
            min_uwb_measurements_ = 2;  // Reduced from 3

            // Adjusted noise parameters
            // imu_acc_noise_ = 0.01;    // Reduced
            // imu_gyro_noise_ = 0.001;  // Reduced
            // imu_acc_bias_noise_ = 0.0001;
            // imu_gyro_bias_noise_ = 0.0001;
            // uwb_noise_ = 0.0001;  // Increased for stronger position constraint

            // Adjusted noise parameters (reduced for better stability)
            imu_acc_noise_ = 0.005;        // Reduced from 0.01
            imu_gyro_noise_ = 0.0005;      // Reduced from 0.001
            imu_acc_bias_noise_ = 0.00001; // Reduced from 0.0001
            imu_gyro_bias_noise_ = 0.00001;// Reduced from 0.0001
            uwb_noise_ = 0.01;             // Increased for looser position constraint

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
            max_imu_queue_size_ = 10000;  // 1000 About 2.5s of IMU data at 400Hz
            max_uwb_queue_size_ = 3000;    // 30 About 3s of UWB data at 10Hz
            min_imu_between_uwb_ = 20;   // 20 Minimum IMU measurements between UWB updates

            // Initialize pose history
            pose_history_.clear();
            
        }

        void cleanupMarginalizationInfo() {
            for (auto ptr : last_marginalization_parameter_blocks_) {
                delete[] ptr;
            }
            last_marginalization_parameter_blocks_.clear();
            last_marginalization_info_.reset();
        }
    
        double* copyParameterBlock(const double* source) {
            double* dest = new double[16];
            std::memcpy(dest, source, 16 * sizeof(double));
            return dest;
        }

        bool initializeOrientation(const std::deque<ImuMeasurement>& imu_buffer) {
            if (imu_buffer.size() < 200) {
                return false;
            }
        
            // Use initial period to estimate gravity direction and initial heading
            Eigen::Vector3d avg_acc = Eigen::Vector3d::Zero();
            Eigen::Vector3d avg_gyro = Eigen::Vector3d::Zero();
            size_t valid_samples = 0;
            
            // Compute mean and variance of measurements
            std::vector<double> acc_norms;
            for (const auto& imu : imu_buffer) {
                double acc_norm = imu.acc.norm();
                acc_norms.push_back(acc_norm);
                
                if (std::abs(acc_norm - GRAVITY_MAGNITUDE) < 0.5) {
                    avg_acc += imu.acc;
                    avg_gyro += imu.gyro;
                    valid_samples++;
                }
            }
        
            if (valid_samples < 100) {
                ROS_WARN("Not enough static samples for initialization");
                return false;
            }
        
            avg_acc /= valid_samples;
            avg_gyro /= valid_samples;
        
            // Estimate initial gyro bias
            current_state_.gyro_bias = avg_gyro;
        
            // Compute gravity direction in body frame
            Eigen::Vector3d g_body = -avg_acc.normalized();
            Eigen::Vector3d g_world(0, 0, -1);  // Assuming Z-up coordinate system
        
            // Construct initial orientation that aligns gravity
            Eigen::Vector3d rot_axis = g_body.cross(g_world);
            if (rot_axis.norm() > 1e-8) {
                double rot_angle = std::acos(g_body.dot(g_world));
                Eigen::AngleAxisd rot_align(rot_angle, rot_axis.normalized());
                current_state_.orientation = Eigen::Quaterniond(rot_align);
            } else {
                current_state_.orientation.setIdentity();
            }
        
            // Initialize covariance
            double acc_var = 0;
            for (const auto& norm : acc_norms) {
                acc_var += std::pow(norm - GRAVITY_MAGNITUDE, 2);
            }
            acc_var /= acc_norms.size();
            
            ROS_INFO("Initialization complete:");
            ROS_INFO("  Initial gyro bias: [%.3f, %.3f, %.3f]", 
                     current_state_.gyro_bias.x(),
                     current_state_.gyro_bias.y(),
                     current_state_.gyro_bias.z());
            ROS_INFO("  Accelerometer variance: %.3f", acc_var);
        
            return true;
        }

        void estimateGyroBias() {
            if (imu_buffer_.size() < 100) return;
    
            Eigen::Vector3d avg_gyro = Eigen::Vector3d::Zero();
            size_t count = 0;
    
            // Use samples when acceleration is close to gravity (static periods)
            for (const auto& imu : imu_buffer_) {
                if (std::abs(imu.acc.norm() - GRAVITY_MAGNITUDE) < 0.1) {
                    avg_gyro += imu.gyro;
                    count++;
                }
            }
    
            if (count > 50) {  // Only update if enough static samples
                Eigen::Vector3d new_gyro_bias = avg_gyro / count;
                
                // Smooth update
                const double alpha = 0.1;  // Smoothing factor
                current_state_.gyro_bias = (1-alpha) * current_state_.gyro_bias + 
                                         alpha * new_gyro_bias;
                
                ROS_DEBUG("Updated gyro bias: [%.6f, %.6f, %.6f]", 
                         current_state_.gyro_bias.x(),
                         current_state_.gyro_bias.y(),
                         current_state_.gyro_bias.z());
            }
        }

        
        bool checkOrientationConsistency(const Eigen::Quaterniond& q1, 
            const Eigen::Quaterniond& q2) {
        // Check for large orientation jumps
        double angle_diff = std::abs(2.0 * std::acos(std::abs(q1.dot(q2))));
        if (angle_diff > M_PI/4) {  // More than 45 degrees
        ROS_WARN("Large orientation jump detected: %.1f degrees", 
        angle_diff * 180/M_PI);
        return false;
        }
        return true;
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
            imu_data.orientation = Eigen::Quaterniond(msg->orientation.w,
                                                    msg->orientation.x,
                                                    msg->orientation.y,
                                                    msg->orientation.z);
            
            imu_queue_.push(imu_data);
        }

        void processImuLoop() {
            ROS_INFO("IMU processing thread started");
            while (running_) {
                ImuMeasurement imu_data;
                if (imu_queue_.pop(imu_data)) {
                    try {
                        processImuData(imu_data);
                        
                        // Check if it's time for batch optimization
                        if (initialized_ && imu_data.timestamp - batch_start_time_ >= batch_duration_) 
                        {
                            performBatchOptimization();
                        }
                    } catch (const std::exception& e) {
                        ROS_ERROR_STREAM("Error processing IMU data: " << e.what());
                    }
                } else {
                    // Sleep a bit if queue is empty
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            ROS_INFO("IMU processing thread stopped");
        }
    
        void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
            static int count = 0;
            if (count++ % 10 == 0) {  // Print every 10th message
                // ROS_INFO("Received UWB message, timestamp: %.3f", msg->header.stamp.toSec());
            }
            
            UwbMeasurement uwb_data;
            uwb_data.timestamp = msg->header.stamp.toSec();
            uwb_data.position = Eigen::Vector3d(msg->point.x,
                                              msg->point.y,
                                              msg->point.z);
        
            processUwbData(uwb_data);
        }
    
        void processImuData(const ImuMeasurement& imu_data) {
            std::lock_guard<std::mutex> lock(state_mutex_);  // Protect state access
            if (!initialized_) {
                imu_buffer_.push_back(imu_data);
                
                if (imu_buffer_.size() >= 200) {
                    if (initializeOrientation(imu_buffer_)) {
                        initialized_ = true;
                        last_imu_time_ = imu_data.timestamp;
                        batch_start_time_ = imu_data.timestamp;
                        ROS_INFO("System initialized with orientation alignment");
                    }
                }
                return;
            }
        
            // Propagate state using IMU
            // Propagate state using IMU
            double dt = imu_data.timestamp - last_imu_time_;
            if (dt > 0) {
                propagateState(imu_data, dt);
                last_imu_time_ = imu_data.timestamp;
                
                // Publish high-frequency pose
                nav_msgs::Odometry odom_msg;
                odom_msg.header.stamp = ros::Time(imu_data.timestamp);
                odom_msg.header.frame_id = "map";
                
                odom_msg.pose.pose.position.x = current_state_.position.x();
                odom_msg.pose.pose.position.y = current_state_.position.y();
                odom_msg.pose.pose.position.z = current_state_.position.z();
                
                odom_msg.pose.pose.orientation.w = current_state_.orientation.w();
                odom_msg.pose.pose.orientation.x = current_state_.orientation.x();
                odom_msg.pose.pose.orientation.y = current_state_.orientation.y();
                odom_msg.pose.pose.orientation.z = current_state_.orientation.z();
                
                odom_msg.twist.twist.linear.x = current_state_.velocity.x();
                odom_msg.twist.twist.linear.y = current_state_.velocity.y();
                odom_msg.twist.twist.linear.z = current_state_.velocity.z();
                
                high_freq_pose_pub_.publish(odom_msg);
            }

            // Store IMU measurements for optimization
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                imu_buffer_.push_back(imu_data);
                
                // Limit buffer size
                while (imu_buffer_.size() > max_imu_queue_size_) {
                    imu_buffer_.pop_front();
                }
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
            
            // Remove this part since we're not using optimize() anymore
            // Trigger optimization when enough measurements are collected
            // if (uwb_buffer_.size() >= min_uwb_measurements_) {
            //     // optimize();
            // }
        }
    
        void propagateState(const ImuMeasurement& imu_data, double dt) {
            if (dt <= 0) return;
        
            // Apply bias and threshold corrections
            Eigen::Vector3d acc_unbias = imu_data.acc - current_state_.acc_bias;
            Eigen::Vector3d gyro_unbias = imu_data.gyro - current_state_.gyro_bias;
        
            // Thresholds for noise reduction
            const double GYRO_THRESHOLD = 0.001;  // rad/s
            const double ACC_THRESHOLD = 0.05;    // m/s^2
        
            // Apply thresholds to remove small noisy measurements
            for (int i = 0; i < 3; i++) {
                if (std::abs(gyro_unbias[i]) < GYRO_THRESHOLD) gyro_unbias[i] = 0.0;
                if (std::abs(acc_unbias[i]) < ACC_THRESHOLD) acc_unbias[i] = 0.0;
            }
        
            // Update orientation using exponential map
            Eigen::Vector3d omega = gyro_unbias;
            double omega_norm = omega.norm();
            
            if (omega_norm > 1e-10) {
                Eigen::Vector3d omega_normalized = omega / omega_norm;
                double half_angle = omega_norm * dt * 0.5;
                double cos_half = std::cos(half_angle);
                double sin_half = std::sin(half_angle);
                
                Eigen::Quaterniond dq(cos_half, 
                                     sin_half * omega_normalized.x(),
                                     sin_half * omega_normalized.y(),
                                     sin_half * omega_normalized.z());
                
                current_state_.orientation = (current_state_.orientation * dq).normalized();
            }
        
            // Transform acceleration to world frame
            Eigen::Matrix3d R = current_state_.orientation.toRotationMatrix();
            Eigen::Vector3d acc_world = R * acc_unbias;
        
            // Remove gravity
            acc_world += GRAVITY_VECTOR;
        
            // Velocity integration with bounds
            const double MAX_VELOCITY = 10.0;  // m/s
            Eigen::Vector3d delta_v = acc_world * dt;
            current_state_.velocity += delta_v;
            
            // Apply velocity bounds using min/max instead of clamp
            for (int i = 0; i < 3; i++) {
                current_state_.velocity[i] = std::min(std::max(current_state_.velocity[i], 
                                                              -MAX_VELOCITY), 
                                                    MAX_VELOCITY);
            }
        
            // Position integration
            current_state_.position += current_state_.velocity * dt + 
                                      0.5 * acc_world * dt * dt;
        
            // Debug orientation changes
            static Eigen::Vector3d last_euler = 
                current_state_.orientation.toRotationMatrix().eulerAngles(0, 1, 2);
            Eigen::Vector3d current_euler = 
                current_state_.orientation.toRotationMatrix().eulerAngles(0, 1, 2);
            
            // Detect large orientation changes
            Eigen::Vector3d euler_diff = (current_euler - last_euler).array().abs();
            if (euler_diff.maxCoeff() > 0.1) {  // > ~5.7 degrees
                ROS_DEBUG("Large orientation change detected:");
                ROS_DEBUG("  Gyro input: [%.3f, %.3f, %.3f]", 
                          gyro_unbias.x(), gyro_unbias.y(), gyro_unbias.z());
                ROS_DEBUG("  Delta orientation (deg): [%.1f, %.1f, %.1f]",
                          euler_diff.x() * 180/M_PI,
                          euler_diff.y() * 180/M_PI,
                          euler_diff.z() * 180/M_PI);
            }
            last_euler = current_euler;
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

        void clearOldTrajectoryData() {
            double current_time = ros::Time::now().toSec();
            while (!pose_history_.empty()) {
                if (current_time - pose_history_.front().header.stamp.toSec() > 60.0) {  // Remove poses older than 60 seconds
                    pose_history_.pop_front();
                } else {
                    break;
                }
            }
        }

        void performBatchOptimization() {
            std::lock_guard<std::mutex> lock(marginalization_mutex_);
            if (imu_buffer_.empty() || uwb_buffer_.empty()) {
                ROS_INFO("Empty buffers, skipping optimization");
                return;
            }
        
            // Initialize batch states
            std::vector<BatchState> batch_states;
            initializeBatchStates(batch_states);
        
            ROS_INFO("Number of batch states: %zu", batch_states.size());
            if (batch_states.empty()) {
                ROS_WARN("No batch states, skipping optimization");
                return;
            }
        
            // Setup optimization problem
            ceres::Problem::Options problem_options;
            problem_options.enable_fast_removal = true;
            ceres::Problem problem(problem_options);
            
            std::vector<double*> parameter_blocks;
            auto* quaternion_parameterization = new ceres::QuaternionParameterization();
        
            try {
                // Step 1: Create all parameter blocks first
                parameter_blocks.reserve(batch_states.size());
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    double* state_ptr = new double[16];
                    batchStateToArray(batch_states[i], state_ptr);
                    parameter_blocks.push_back(state_ptr);
                }
        
                // Step 2: Add parameter blocks to the problem
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    // First add the parameter block
                    problem.AddParameterBlock(parameter_blocks[i], 16);
        
                    if (batch_states[i].fixed) {
                        // problem.SetParameterBlockConstant(parameter_blocks[i]);
                    }
                }

                // Add marginalization factor from last optimization if available
                if (last_marginalization_info_ && !last_marginalization_parameter_blocks_.empty()) {
                    ceres::CostFunction* marginalization_cost = 
                        MarginalizationFactorAutoDiff::Create(last_marginalization_info_);
                    
                    problem.AddResidualBlock(
                        marginalization_cost,
                        NULL,
                        parameter_blocks[0]  // Connect to first state
                    );
                }
        
                // Add bias regularization factors
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    auto* bias_factor = new BiasRegularizationFactor(0.1, 0.1);
                    auto* cost_function = 
                        new ceres::AutoDiffCostFunction<BiasRegularizationFactor, 6, 16>(bias_factor);
                    problem.AddResidualBlock(cost_function, NULL, parameter_blocks[i]);
                }
        
                // Add IMU factors
                for (size_t i = 0; i < batch_states.size() - 1; ++i) {
                    double t1 = batch_states[i].timestamp;
                    double t2 = batch_states[i + 1].timestamp;
                    
                    imu_preintegration_->reset();
                    
                    int imu_count = 0;
                    for (const auto& imu : imu_buffer_) {
                        if (imu.timestamp >= t1 && imu.timestamp < t2) {
                            double dt = imu.timestamp - t1;
                            imu_preintegration_->integrate(imu.acc, imu.gyro, dt);
                            t1 = imu.timestamp;
                            imu_count++;
                        }
                    }
                    
                    if (imu_count > 0) {
                        auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
                        auto* cost_function = 
                            new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor);
                        
                        problem.AddResidualBlock(
                            cost_function,
                            new ceres::HuberLoss(1.0),
                            parameter_blocks[i],
                            parameter_blocks[i + 1]
                        );
                    }
                }

                
                // Add IMU orientation factors
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    double state_time = batch_states[i].timestamp;
                    
                    // Find closest IMU measurement
                    auto it = std::lower_bound(imu_buffer_.begin(), imu_buffer_.end(), state_time,
                        [](const ImuMeasurement& imu, double time) {
                            return imu.timestamp < time;
                        });
                    
                    if (it != imu_buffer_.end() && 
                        std::abs(it->timestamp - state_time) < state_dt_) {
                        auto* orientation_factor = 
                            new ImuOrientationFactor(it->orientation, 100.0); // Adjust weight as needed
                        
                        auto* cost_function = 
                            new ceres::AutoDiffCostFunction<ImuOrientationFactor, 3, 16>(orientation_factor);
                        
                        problem.AddResidualBlock(
                            cost_function,
                            new ceres::HuberLoss(1.0),
                            parameter_blocks[i]
                        );
                    }
                }

                // Add orientation regularization between consecutive states
                // const double orientation_weight = 1.0;
                // for (size_t i = 0; i < batch_states.size() - 1; ++i) {
                //     ceres::CostFunction* orientation_cost = 
                //         new ceres::AutoDiffCostFunction<OrientationRegularizationFactor, 3, 16, 16>(
                //             new OrientationRegularizationFactor(orientation_weight));
                    
                //     problem.AddResidualBlock(
                //         orientation_cost,
                //         new ceres::HuberLoss(0.1),
                //         parameter_blocks[i],
                //         parameter_blocks[i + 1]
                //     );
                // }
        
                // Add UWB factors
                for (const auto& uwb : uwb_buffer_) {
                    size_t closest_idx = findClosestStateIndex(uwb.timestamp, batch_states);
                    
                    Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity();
                    auto* uwb_factor = new UwbFactor(uwb.position, uwb_cov);
                    
                    auto* cost_function = 
                        new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor);
                    
                    problem.AddResidualBlock(
                        cost_function,
                        new ceres::HuberLoss(0.1),
                        parameter_blocks[closest_idx]
                    );
                }
        
                // Create a local parameterization for the quaternion part
                class StateParameterization : public ceres::LocalParameterization {
                public:
                    virtual ~StateParameterization() {}
        
                    virtual bool Plus(const double* x,
                                    const double* delta,
                                    double* x_plus_delta) const {
                        // Copy position and velocity
                        for (int i = 0; i < 6; ++i) {
                            x_plus_delta[i] = x[i] + delta[i];
                        }
        
                        // Handle quaternion (indices 6-9)
                        Eigen::Map<const Eigen::Quaterniond> q(x + 6);
                        Eigen::Map<const Eigen::Vector3d> dq(delta + 6);
                        Eigen::Quaterniond q_plus_delta = q * Eigen::Quaterniond(Eigen::AngleAxisd(dq.norm(), dq.normalized()));
                        Eigen::Map<Eigen::Quaterniond> q_out(x_plus_delta + 6);
                        q_out = q_plus_delta.normalized();
        
                        // Copy biases
                        for (int i = 10; i < 16; ++i) {
                            x_plus_delta[i] = x[i] + delta[i];
                        }
        
                        return true;
                    }
        
                    virtual bool ComputeJacobian(const double* x,
                                               double* jacobian) const {
                        Eigen::Map<Eigen::Matrix<double, 16, 15, Eigen::RowMajor>> J(jacobian);
                        J.setZero();
                        
                        // Position and velocity blocks
                        J.block<6,6>(0,0).setIdentity();
                        
                        // Quaternion block
                        J.block<4,3>(6,6) = Eigen::Matrix<double,4,3>::Identity();
                        
                        // Bias blocks
                        J.block<6,6>(10,9).setIdentity();
                        
                        return true;
                    }
        
                    virtual int GlobalSize() const { return 16; }
                    virtual int LocalSize() const { return 15; }
                };
        
                // Add the custom parameterization to all state blocks
                auto* state_parameterization = new StateParameterization();
                for (size_t i = 0; i < batch_states.size(); ++i) {
                    problem.SetParameterization(parameter_blocks[i], state_parameterization);
                }
        
                // Solve optimization problem
                ceres::Solver::Options options;
                options.minimizer_progress_to_stdout = false;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = 50;
                
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
        
                ROS_INFO_STREAM("Optimization report: " << summary.BriefReport());
        
                // After optimization converges, prepare marginalization for next iteration
                if (summary.termination_type == ceres::CONVERGENCE) {
                    // Create new marginalization info
                    auto new_marginalization_info = std::make_shared<MarginalizationInfo>();
                    
                    if (batch_states.size() > 1) {
                        // Add factors to marginalization
                        // IMU factor
                        if (!imu_buffer_.empty()) {
                            imu_preintegration_->reset();
                            double t1 = batch_states[0].timestamp;
                            double t2 = batch_states[1].timestamp;
                            
                            for (const auto& imu : imu_buffer_) {
                                if (imu.timestamp >= t1 && imu.timestamp < t2) {
                                    imu_preintegration_->integrate(imu.acc, imu.gyro, 
                                                                 imu.timestamp - t1);
                                    t1 = imu.timestamp;
                                }
                            }
        
                            auto* imu_factor = new ImuFactor(imu_preintegration_->getResult());
                            std::vector<double*> marg_parameter_blocks{
                                parameter_blocks[0],
                                parameter_blocks[1]
                            };
                            
                            new_marginalization_info->addResidualBlock(
                                new ceres::AutoDiffCostFunction<ImuFactor, 15, 16, 16>(imu_factor),
                                NULL,
                                marg_parameter_blocks,
                                std::vector<int>{0}  // Marginalize first state
                            );
                        }
        
                        // UWB factors
                        for (const auto& uwb : uwb_buffer_) {
                            if (std::abs(uwb.timestamp - batch_states[0].timestamp) < state_dt_) {
                                Eigen::Matrix3d uwb_cov = uwb_noise_ * Eigen::Matrix3d::Identity();
                                auto* uwb_factor = new UwbFactor(uwb.position, uwb_cov);
                                
                                std::vector<double*> marg_parameter_blocks{parameter_blocks[0]};
                                
                                new_marginalization_info->addResidualBlock(
                                    new ceres::AutoDiffCostFunction<UwbFactor, 3, 16>(uwb_factor),
                                    NULL,
                                    marg_parameter_blocks,
                                    std::vector<int>{0}
                                );
                            }
                        }
        
                        // Perform marginalization
                        new_marginalization_info->marginalize();
        
                        // Update marginalization info for next iteration
                        last_marginalization_info_ = new_marginalization_info;
                        last_marginalization_parameter_blocks_.clear();
                        
                        // Deep copy the parameter block for the next iteration
                        double* next_state = new double[16];
                        std::memcpy(next_state, parameter_blocks[1], 16 * sizeof(double));
                        last_marginalization_parameter_blocks_.push_back(next_state);
                    }
        
                    // Update states
                    for (size_t i = 0; i < batch_states.size(); ++i) {
                        arrayToBatchState(parameter_blocks[i], batch_states[i]);
                    }
                    
                    updateTrajectory(batch_states);
                    clearOldTrajectoryData();
                    publishTrajectory();

                    // Update current state with latest optimized state
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        const BatchState& latest_state = batch_states.back();
                        current_state_.position = latest_state.position;
                        current_state_.velocity = latest_state.velocity;
                        current_state_.orientation = latest_state.orientation;
                        current_state_.acc_bias = latest_state.acc_bias;
                        current_state_.gyro_bias = latest_state.gyro_bias;
                        current_state_.timestamp = latest_state.timestamp;

                        geometry_msgs::Vector3 euler_msg;
                        auto euler = current_state_.orientation.toRotationMatrix().eulerAngles(0, 1, 2);
                        euler_msg.x = euler.x() * 180/M_PI;  // Convert to degrees
                        euler_msg.y = euler.y() * 180/M_PI;
                        euler_msg.z = euler.z() * 180/M_PI;
                        std::cout<<"euler_msg.x-> " << euler_msg.x<<"\n";
                        std::cout<<"euler_msg.y-> " << euler_msg.y<<"\n";
                        std::cout<<"euler_msg.z-> " << euler_msg.z<<"\n";
                    }
                }
        
                // Cleanup parameter blocks
                for (auto ptr : parameter_blocks) {
                    delete[] ptr;
                }
                parameter_blocks.clear();
        
            } catch (const std::exception& e) {
                ROS_ERROR_STREAM("Exception in optimization: " << e.what());
                // Cleanup on error
                for (auto ptr : parameter_blocks) {
                    delete[] ptr;
                }
                throw;
            }
        
            prepareNextBatch(batch_states);
        }

        void initializeBatchStates(std::vector<BatchState>& batch_states) {
            batch_states.clear();
            
            double current_time = batch_start_time_;
            ROS_INFO_STREAM("Initializing batch states from time: " << current_time);
            
            for (size_t i = 0; i < num_states_batch_; ++i) {
                BatchState state;
                state.timestamp = current_time;
                state.fixed = (i == 0 && !trajectory_states_.empty());
                
                if (i == 0 && !trajectory_states_.empty()) {
                    state = trajectory_states_.back();
                    ROS_INFO("Using previous state for initialization");
                } else {
                    state.position = interpolatePosition(current_time);
                    state.velocity.setZero();
                    state.orientation.setIdentity();
                    state.acc_bias.setZero();
                    state.gyro_bias.setZero();
                    
                    // Add small random noise to avoid perfect zero initialization
                    state.position += Eigen::Vector3d::Random() * 0.01;
                    state.velocity += Eigen::Vector3d::Random() * 0.01;
                    state.orientation = Eigen::Quaterniond::UnitRandom();
                    state.acc_bias += Eigen::Vector3d::Random() * 0.001;
                    state.gyro_bias += Eigen::Vector3d::Random() * 0.001;
                    
                    // ROS_INFO_STREAM("Initialized new state at time " << current_time);
                }
                
                state.orientation.normalize();
                batch_states.push_back(state);
                current_time += state_dt_;
            }
            
            ROS_INFO_STREAM("Created " << batch_states.size() << " batch states");
        }

        void updateTrajectory(const std::vector<BatchState>& batch_states) {
            // Keep a fixed number of previous states for marginalization
            const size_t keep_states = 20; // 2
            // const size_t keep_states = std::max(size_t(2), num_states_batch_ / 4) * 8;
            
            if (trajectory_states_.size() > keep_states) {
                trajectory_states_.erase(
                    trajectory_states_.begin(),
                    trajectory_states_.end() - keep_states);
            }
            
            // Append new batch states
            trajectory_states_.insert(trajectory_states_.end(), 
                                    batch_states.begin(), batch_states.end());
        }
        
        void publishTrajectory() {
            if (trajectory_states_.empty()) {
                return;
            }
        
            // Get the latest state
            const BatchState& latest_state = trajectory_states_.back();
            
            // Create pose stamped message for the latest state
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time(latest_state.timestamp);
            pose.header.frame_id = "map";
            
            pose.pose.position.x = latest_state.position.x();
            pose.pose.position.y = latest_state.position.y();
            pose.pose.position.z = latest_state.position.z();
            
            pose.pose.orientation.w = latest_state.orientation.w();
            pose.pose.orientation.x = latest_state.orientation.x();
            pose.pose.orientation.y = latest_state.orientation.y();
            pose.pose.orientation.z = latest_state.orientation.z();

            std::cout<<"latest_state.acc_bias-> " <<latest_state.acc_bias<<std::endl;
            std::cout<<"latest_state.gyro_bias-> " <<latest_state.gyro_bias<<std::endl;
            std::cout<<"latest_state.velocity-> " <<latest_state.velocity<<std::endl;
            std::cout<<"latest_state.position-> " <<latest_state.position<<std::endl;
            std::cout<<"latest_state.orientation-> " <<latest_state.orientation.x()<<std::endl;
        
            // Add to pose history
            pose_history_.push_back(pose);
        
            // Maintain maximum size of pose history
            while (pose_history_.size() > max_pose_history_size_) {
                pose_history_.pop_front();
            }
        
            // Create and publish path message
            nav_msgs::Path path_msg;
            path_msg.header.stamp = ros::Time::now();
            path_msg.header.frame_id = "map";
            path_msg.poses.insert(path_msg.poses.end(), 
                                 pose_history_.begin(), 
                                 pose_history_.end());
            
            path_pub_.publish(path_msg);
            
            // Also publish the latest state
            publishBatchState(latest_state);
        
            // Publish latest state as current pose
            nav_msgs::Odometry current_pose_msg;
            current_pose_msg.header = pose.header;
            current_pose_msg.pose.pose = pose.pose;
            current_pose_msg.twist.twist.linear.x = latest_state.velocity.x();
            current_pose_msg.twist.twist.linear.y = latest_state.velocity.y();
            current_pose_msg.twist.twist.linear.z = latest_state.velocity.z();
            
            pose_pub_.publish(current_pose_msg);
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
            // Clear old marginalization parameter blocks
            for (auto ptr : last_marginalization_parameter_blocks_) {
                delete[] ptr;
            }
            last_marginalization_parameter_blocks_.clear();
        
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
            // Position (0-2)
            arr[0] = state.position.x();
            arr[1] = state.position.y();
            arr[2] = state.position.z();
            
            // Velocity (3-5)
            arr[3] = state.velocity.x();
            arr[4] = state.velocity.y();
            arr[5] = state.velocity.z();
            
            // Quaternion (6-9) - Note: Eigen quaternion order (x,y,z,w)
            arr[6] = state.orientation.x();
            arr[7] = state.orientation.y();
            arr[8] = state.orientation.z();
            arr[9] = state.orientation.w();
            
            // Accelerometer bias (10-12)
            arr[10] = state.acc_bias.x();
            arr[11] = state.acc_bias.y();
            arr[12] = state.acc_bias.z();
            
            // Gyroscope bias (13-15)
            arr[13] = state.gyro_bias.x();
            arr[14] = state.gyro_bias.y();
            arr[15] = state.gyro_bias.z();
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
    
        // ROS members
        ros::NodeHandle nh_;
        ros::Subscriber imu_sub_;
        ros::Subscriber uwb_sub_;
        ros::Publisher pose_pub_;

        // Add to private members
        ros::Publisher high_freq_pose_pub_;
    
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
        
        size_t min_uwb_measurements_; 
        
        bool initialized_;
        double last_imu_time_;
    
    // Add these members to the class
    private:
        size_t max_imu_queue_size_;
        size_t max_uwb_queue_size_;
        size_t min_imu_between_uwb_;

        // Add this to the private member variables
        std::deque<geometry_msgs::PoseStamped> pose_history_;
        size_t max_pose_history_size_ = 1000;  // Store last 1000 poses 

        std::shared_ptr<MarginalizationInfo> last_marginalization_info_;
        std::vector<double*> last_marginalization_parameter_blocks_;

        mutable std::mutex marginalization_mutex_;

        // Add to private members
        struct ThreadSafeQueue {
            std::mutex mutex;
            std::condition_variable cond;
            std::deque<ImuMeasurement> queue;
            
            void push(const ImuMeasurement& data) {
                std::lock_guard<std::mutex> lock(mutex);
                queue.push_back(data);
            }
            
            bool pop(ImuMeasurement& data) {
                std::lock_guard<std::mutex> lock(mutex);
                if (queue.empty()) return false;
                data = queue.front();
                queue.pop_front();
                return true;
            }
        };

        ThreadSafeQueue imu_queue_;
        std::thread imu_process_thread_;
        std::atomic<bool> running_;
    };
 
 } // namespace uwb_imu_fusion
 
 int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_imu_fusion_node");
    ros::NodeHandle nh;
    
    ROS_INFO("Starting UWB-IMU fusion node");
    uwb_imu_fusion::UwbImuFusion fusion;
    
    ros::AsyncSpinner spinner(4);  // Use 4 threads
    spinner.start();
    ros::waitForShutdown();
    
    return 0;
}