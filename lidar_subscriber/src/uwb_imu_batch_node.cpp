#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_broadcaster.h>
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
#include <chrono>
#include <novatel_msgs/INSPVAX.h>  // Added INSPVAX message header

// Added for visualization
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <iomanip>  // For std::setprecision
#include <sstream>  // For std::stringstream

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

// CRITICAL: Hard constraint on bias magnitude
class BiasMagnitudeConstraint {
public:
    BiasMagnitudeConstraint(double acc_max = 0.1, double gyro_max = 0.01, double weight = 1000.0) 
        : acc_max_(acc_max), gyro_max_(gyro_max), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const bias, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba(bias);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bias + 3);
        
        // Compute bias magnitudes
        T ba_norm = ba.norm();
        T bg_norm = bg.norm();
        
        // Residuals: penalty proportional to how much bias exceeds maximum
        // For accelerometer bias
        residuals[0] = T(0.0);
        if (ba_norm > T(acc_max_)) {
            residuals[0] = T(weight_) * (ba_norm - T(acc_max_));
        }
        
        // CRITICAL: Much higher weight for gyro bias constraint
        residuals[1] = T(0.0);
        if (bg_norm > T(gyro_max_)) {
            residuals[1] = T(weight_ * 10.0) * (bg_norm - T(gyro_max_));
        }
        
        return true;
    }
    
    static ceres::CostFunction* Create(double acc_max = 0.1, double gyro_max = 0.01, double weight = 1000.0) {
        return new ceres::AutoDiffCostFunction<BiasMagnitudeConstraint, 2, 6>(
            new BiasMagnitudeConstraint(acc_max, gyro_max, weight));
    }
    
private:
    double acc_max_;
    double gyro_max_;
    double weight_;
};

// IMPROVED: Adaptive velocity magnitude constraint for high-speed scenarios
class VelocityMagnitudeConstraint {
public:
    VelocityMagnitudeConstraint(double max_velocity = 25.0, double weight = 300.0)
        : max_velocity_(max_velocity), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const velocity, T* residuals) const {
        // Compute velocity magnitude
        T vx = velocity[0];
        T vy = velocity[1];
        T vz = velocity[2];
        T magnitude = ceres::sqrt(vx*vx + vy*vy + vz*vz);
        
        // Only penalize if velocity exceeds maximum - with smoother penalty
        residuals[0] = T(0.0);
        if (magnitude > T(max_velocity_)) {
            // Use quadratic penalty for more gradual constraint
            T excess = magnitude - T(max_velocity_);
            residuals[0] = T(weight_) * excess * excess;
        }
        
        return true;
    }
    
    static ceres::CostFunction* Create(double max_velocity = 25.0, double weight = 300.0) {
        return new ceres::AutoDiffCostFunction<VelocityMagnitudeConstraint, 1, 3>(
            new VelocityMagnitudeConstraint(max_velocity, weight));
    }
    
private:
    double max_velocity_;
    double weight_;
};

// FIXED: Better horizontal velocity incentive factor - numerically stable
class HorizontalVelocityIncentiveFactor {
public:
    HorizontalVelocityIncentiveFactor(double min_velocity = 0.2, double weight = 10.0)
        : min_velocity_(min_velocity), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const velocity, const T* const pose, T* residuals) const {
        // Extract velocity
        T vx = velocity[0];
        T vy = velocity[1];
        
        // Compute horizontal velocity magnitude with numerical stability safeguard
        T h_vel_sq = vx*vx + vy*vy;
        T h_vel_mag = ceres::sqrt(h_vel_sq + T(1e-10)); // Add small epsilon to avoid numerical issues
        
        // Only encourage minimum velocity if below threshold, with smooth response
        residuals[0] = T(0.0);
        if (h_vel_mag < T(min_velocity_)) {
            // Using smoothed residual function to improve numerical stability
            T diff = T(min_velocity_) - h_vel_mag;
            residuals[0] = T(weight_) * diff * diff / (diff + T(0.01));
        }
        
        return true;
    }
    
    static ceres::CostFunction* Create(double min_velocity = 0.2, double weight = 10.0) {
        return new ceres::AutoDiffCostFunction<HorizontalVelocityIncentiveFactor, 1, 3, 7>(
            new HorizontalVelocityIncentiveFactor(min_velocity, weight));
    }
    
private:
    double min_velocity_;
    double weight_;
};

// Roll/Pitch prior factor for planar motion
class RollPitchPriorFactor {
public:
    RollPitchPriorFactor(double weight = 300.0) : weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract quaternion
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        
        // Convert to rotation matrix
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        
        // Get gravity direction in body frame
        Eigen::Matrix<T, 3, 1> z_body = R.col(2);
        
        // In planar motion with ENU frame, z_body should be close to [0,0,1]
        residuals[0] = T(weight_) * z_body.x();
        residuals[1] = T(weight_) * z_body.y();
        
        return true;
    }
    
    static ceres::CostFunction* Create(double weight = 300.0) {
        return new ceres::AutoDiffCostFunction<RollPitchPriorFactor, 2, 7>(
            new RollPitchPriorFactor(weight));
    }
    
private:
    double weight_;
};

// FIXED: Orientation smoothness factor to enforce smooth orientation changes
class OrientationSmoothnessFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    OrientationSmoothnessFactor(double weight = 150.0) : weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // Extract orientations
        Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);
        
        // Normalize quaternions for numerical stability
        Eigen::Quaternion<T> q_i_normalized = q_i.normalized();
        Eigen::Quaternion<T> q_j_normalized = q_j.normalized();
        
        // Compute dot product between quaternions
        T dot = q_i_normalized.w() * q_j_normalized.w() + 
                q_i_normalized.x() * q_j_normalized.x() + 
                q_i_normalized.y() * q_j_normalized.y() + 
                q_i_normalized.z() * q_j_normalized.z();
        
        // Make sure dot product is in valid range for acos
        dot = ceres::abs(dot) < T(1.0) ? dot : (dot > T(0.0) ? T(0.999999) : T(-0.999999));
        
        // Compute angle between orientations (safer than previous implementation)
        T angle = T(2.0) * ceres::acos(dot);
        
        // Set residual proportional to angular change with safety check
        residuals[0] = angle < T(1e-6) ? T(0.0) : T(weight_) * angle;
        
        return true;
    }
    
    static ceres::CostFunction* Create(double weight = 150.0) {
        return new ceres::AutoDiffCostFunction<OrientationSmoothnessFactor, 1, 7, 7>(
            new OrientationSmoothnessFactor(weight));
    }
    
private:
    double weight_;
};

// Gravity alignment factor - uses accelerometer to align with world gravity
class GravityAlignmentFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GravityAlignmentFactor(const Eigen::Vector3d& measured_acc, double weight = 200.0)
        : measured_acc_(measured_acc), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract orientation
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        
        // Normalized accelerometer measurement
        Eigen::Matrix<T, 3, 1> acc_normalized = measured_acc_.normalized().cast<T>();
        
        // World gravity direction (negative Z in ENU)
        Eigen::Matrix<T, 3, 1> gravity_world(T(0), T(0), T(-1));
        
        // Rotate world gravity to sensor frame using inverse rotation
        Eigen::Matrix<T, 3, 1> expected_acc = q.conjugate() * gravity_world;
        
        // Residuals: difference between expected and measured normalized acceleration
        residuals[0] = T(weight_) * (expected_acc[0] - acc_normalized[0]);
        residuals[1] = T(weight_) * (expected_acc[1] - acc_normalized[1]);
        residuals[2] = T(weight_) * (expected_acc[2] - acc_normalized[2]);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_acc, double weight = 200.0) {
        return new ceres::AutoDiffCostFunction<GravityAlignmentFactor, 3, 7>(
            new GravityAlignmentFactor(measured_acc, weight));
    }
    
private:
    Eigen::Vector3d measured_acc_;
    double weight_;
};

// FIXED: Numerically stable YawOnlyOrientationFactor 
class YawOnlyOrientationFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    YawOnlyOrientationFactor(const Eigen::Quaterniond& measured_orientation, double weight)
        : weight_(weight) {
        // Ensure measured orientation is normalized
        Eigen::Quaterniond normalized_orientation = measured_orientation.normalized();
        
        // Extract yaw from measured orientation with safety checks
        double q_x = normalized_orientation.x();
        double q_y = normalized_orientation.y();
        double q_z = normalized_orientation.z();
        double q_w = normalized_orientation.w();
        
        // Convert to yaw angle with safety check
        double term1 = 2.0 * (q_w * q_z + q_x * q_y);
        double term2 = 1.0 - 2.0 * (q_y * q_y + q_z * q_z);
        double yaw = atan2(term1, term2);
        
        // Create quaternion with only yaw (roll=pitch=0)
        yaw_only_quat_ = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        yaw_only_quat_.normalize(); // Ensure normalized
    }
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract orientation quaternion from pose
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        Eigen::Quaternion<T> q_norm = q.normalized(); // Ensure normalized
        
        // Extract yaw with numerical stability
        T q_x = q_norm.x();
        T q_y = q_norm.y();
        T q_z = q_norm.z();
        T q_w = q_norm.w();
        
        // Ensure values are in valid range
        T term1 = T(2.0) * (q_w * q_z + q_x * q_y);
        T term2 = T(1.0) - T(2.0) * (q_y * q_y + q_z * q_z);
        
        // Add small epsilon to avoid division by zero
        T epsilon = T(1e-10);
        term2 = ceres::abs(term2) < epsilon ? (term2 >= T(0.0) ? epsilon : -epsilon) : term2;
        
        T yaw = ceres::atan2(term1, term2);
        
        // Create yaw-only quaternion using stable construction
        T cy = ceres::cos(yaw * T(0.5));
        T sy = ceres::sin(yaw * T(0.5));
        Eigen::Quaternion<T> pose_yaw_only(cy, T(0), T(0), sy);
        
        // Compare with measured yaw-only quaternion
        Eigen::Quaternion<T> q_measured = yaw_only_quat_.cast<T>();
        
        // Compute difference angle safely
        T dot_product = pose_yaw_only.w() * q_measured.w() + 
                        pose_yaw_only.x() * q_measured.x() + 
                        pose_yaw_only.y() * q_measured.y() +
                        pose_yaw_only.z() * q_measured.z();
                        
        // Clamp to valid domain with extra safety margin
        dot_product = ceres::abs(dot_product) < T(1.0) ? dot_product : 
                     (dot_product > T(0.0) ? T(0.999) : T(-0.999));
        
        // Compute angular difference and scale by weight
        T angle = T(2.0) * ceres::acos(dot_product);
        
        // Return zero if angle is very small to avoid numerical issues
        residuals[0] = angle < T(1e-6) ? T(0.0) : T(weight_) * angle;
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Quaterniond& measured_orientation, double weight) {
        return new ceres::AutoDiffCostFunction<YawOnlyOrientationFactor, 1, 7>(
            new YawOnlyOrientationFactor(measured_orientation, weight));
    }
    
private:
    Eigen::Quaterniond yaw_only_quat_;
    double weight_;
};

// NEW: GPS Orientation Factor for full orientation constraint
class GpsOrientationFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GpsOrientationFactor(const Eigen::Quaterniond& measured_orientation, double noise_std)
        : measured_orientation_(measured_orientation.normalized()), noise_std_(noise_std) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract orientation quaternion from pose
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        Eigen::Quaternion<T> q_normalized = q.normalized();
        
        // Convert measured orientation to template type
        Eigen::Quaternion<T> q_measured = measured_orientation_.template cast<T>();
        
        // Compute orientation difference
        Eigen::Quaternion<T> q_diff = q_normalized.conjugate() * q_measured;
        
        // Convert to axis-angle representation
        T angle = T(2.0) * acos(std::min(std::max(q_diff.w(), T(-1.0)), T(1.0)));
        
        // Extract rotation axis from quaternion difference
        Eigen::Matrix<T, 3, 1> axis;
        T axis_norm = q_diff.vec().norm();
        
        if (axis_norm > T(1e-10)) {
            axis = q_diff.vec() / axis_norm;
        } else {
            axis = Eigen::Matrix<T, 3, 1>(T(1.0), T(0.0), T(0.0));
        }
        
        // Residuals are proportional to rotation angle along each axis
        residuals[0] = (angle * axis[0]) / T(noise_std_);
        residuals[1] = (angle * axis[1]) / T(noise_std_);
        residuals[2] = (angle * axis[2]) / T(noise_std_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Quaterniond& measured_orientation, double noise_std) {
        return new ceres::AutoDiffCostFunction<GpsOrientationFactor, 3, 7>(
            new GpsOrientationFactor(measured_orientation, noise_std));
    }
    
private:
    const Eigen::Quaterniond measured_orientation_;
    const double noise_std_;
};

// ==================== GPS-RELATED FACTORS ====================

// GPS position factor for Ceres
class GpsPositionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GpsPositionFactor(const Eigen::Vector3d& measured_position, double noise_std)
        : measured_position_(measured_position), noise_std_(noise_std) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract position from pose (position + quaternion)
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(pose);
        
        // Compute position residuals - weighted by noise standard deviation
        residuals[0] = (position[0] - T(measured_position_[0])) / T(noise_std_);
        residuals[1] = (position[1] - T(measured_position_[1])) / T(noise_std_);
        residuals[2] = (position[2] - T(measured_position_[2])) / T(noise_std_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_position, double noise_std) {
        return new ceres::AutoDiffCostFunction<GpsPositionFactor, 3, 7>(
            new GpsPositionFactor(measured_position, noise_std));
    }
    
private:
    const Eigen::Vector3d measured_position_;
    const double noise_std_;
};

// GPS velocity factor for Ceres
class GpsVelocityFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GpsVelocityFactor(const Eigen::Vector3d& measured_velocity, double noise_std)
        : measured_velocity_(measured_velocity), noise_std_(noise_std) {}
    
    template <typename T>
    bool operator()(const T* const velocity, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vel(velocity);
        
        // Compute velocity residuals
        residuals[0] = (vel[0] - T(measured_velocity_[0])) / T(noise_std_);
        residuals[1] = (vel[1] - T(measured_velocity_[1])) / T(noise_std_);
        residuals[2] = (vel[2] - T(measured_velocity_[2])) / T(noise_std_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_velocity, double noise_std) {
        return new ceres::AutoDiffCostFunction<GpsVelocityFactor, 3, 3>(
            new GpsVelocityFactor(measured_velocity, noise_std));
    }
    
private:
    const Eigen::Vector3d measured_velocity_;
    const double noise_std_;
};

// ==================== MARGINALIZATION CLASSES ====================

// Marginalization information class - handles Schur complement
class MarginalizationInfo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Structure to hold residual block information
    struct ResidualBlockInfo {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ResidualBlockInfo(ceres::CostFunction* _cost_function, 
                         ceres::LossFunction* _loss_function,
                         std::vector<double*>& _parameter_blocks,
                         std::vector<int>& _drop_set)
            : cost_function(_cost_function), loss_function(_loss_function),
              parameter_blocks(_parameter_blocks), drop_set(_drop_set) {
            
            // Calculate sizes
            num_residuals = cost_function->num_residuals();
            parameter_block_sizes.clear();
            
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                parameter_block_sizes.push_back(cost_function->parameter_block_sizes()[i]);
            }
            
            // Allocate memory
            raw_jacobians = new double*[parameter_blocks.size()];
            jacobians.resize(parameter_blocks.size());
            
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                jacobians[i].resize(num_residuals, parameter_block_sizes[i]);
                jacobians[i].setZero();
            }
            
            residuals.resize(num_residuals);
            residuals.setZero();
        }
        
        ~ResidualBlockInfo() {
            delete[] raw_jacobians;
            
            // Only delete the cost function if we own it
            if (cost_function) {
                delete cost_function;
                cost_function = nullptr;
            }
                
            // Only delete the loss function if we own it  
            if (loss_function) {
                delete loss_function;
                loss_function = nullptr;
            }
        }
        
        void Evaluate() {
            // Skip evaluation if we don't have all parameters
            if (parameter_blocks_data.size() != parameter_blocks.size()) {
                ROS_WARN("Parameter blocks data size mismatch in Evaluate()");
                return;
            }
            
            // Allocate memory for parameters and residuals
            double** parameters = new double*[parameter_blocks.size()];
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                parameters[i] = parameter_blocks_data[i];
                raw_jacobians[i] = new double[num_residuals * parameter_block_sizes[i]];
                memset(raw_jacobians[i], 0, sizeof(double) * num_residuals * parameter_block_sizes[i]);
            }
            
            double* raw_residuals = new double[num_residuals];
            memset(raw_residuals, 0, sizeof(double) * num_residuals);
            
            // Evaluate the cost function
            cost_function->Evaluate(parameters, raw_residuals, raw_jacobians);
            
            // Apply loss function if needed
            if (loss_function) {
                double residual_scaling = 1.0;
                double alpha_sq_norm = 0.0;
                
                for (int i = 0; i < num_residuals; i++) {
                    alpha_sq_norm += raw_residuals[i] * raw_residuals[i];
                }
                
                double sqrt_rho1 = 1.0;
                if (alpha_sq_norm > 0) {
                    double rho[3];
                    loss_function->Evaluate(alpha_sq_norm, rho);
                    sqrt_rho1 = sqrt(rho[1]);
                    
                    if (sqrt_rho1 == 0) {
                        residual_scaling = 0.0;
                    } else {
                        residual_scaling = sqrt_rho1 / alpha_sq_norm;
                    }
                }
                
                for (int i = 0; i < num_residuals; i++) {
                    raw_residuals[i] *= sqrt_rho1;
                }
                
                for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                    for (int j = 0; j < parameter_block_sizes[i] * num_residuals; j++) {
                        raw_jacobians[i][j] *= residual_scaling;
                    }
                }
            }
            
            // Copy raw residuals to Eigen vector
            for (int i = 0; i < num_residuals; i++) {
                residuals(i) = raw_residuals[i];
            }
            
            // Copy raw jacobians to Eigen matrices
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                    mat_jacobian(raw_jacobians[i], num_residuals, parameter_block_sizes[i]);
                jacobians[i] = mat_jacobian;
            }
            
            // Clean up
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                delete[] raw_jacobians[i];
            }
            delete[] parameters;
            delete[] raw_residuals;
        }
        
        ceres::CostFunction* cost_function;
        ceres::LossFunction* loss_function;
        std::vector<double*> parameter_blocks;
        std::vector<int> parameter_block_sizes;
        int num_residuals;
        std::vector<int> drop_set;
        
        std::vector<double*> parameter_blocks_data;
        double** raw_jacobians;
        std::vector<Eigen::MatrixXd> jacobians;
        Eigen::VectorXd residuals;
    };
    
    MarginalizationInfo() {
        keep_block_size = 0;
        keep_block_idx.clear();
        keep_block_data.clear();
        keep_block_addr.clear();
    }
    
    ~MarginalizationInfo() {
        // Clean up parameter block data - this needs to be done before residual_block_infos
        for (auto& it : parameter_block_data) {
            if (it.second) {
                delete[] it.second;
                it.second = nullptr;
            }
        }
        parameter_block_data.clear();
        
        // Clean up residual blocks
        for (auto& it : residual_block_infos) {
            delete it;
        }
        residual_block_infos.clear();
    }
    
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info) {
        if (!residual_block_info) {
            ROS_WARN("Trying to add null ResidualBlockInfo");
            return;
        }
        
        residual_block_infos.emplace_back(residual_block_info);
        
        // Add all parameter blocks to our tracking
        for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
            double* addr = residual_block_info->parameter_blocks[i];
            // Skip null parameter blocks
            if (!addr) {
                ROS_WARN("Null parameter block address in ResidualBlockInfo");
                continue;
            }
            
            int size = residual_block_info->parameter_block_sizes[i];
            if (size <= 0) {
                ROS_WARN("Invalid parameter block size: %d", size);
                continue;
            }
            
            parameter_block_size[addr] = size;
            
            // If this is a new parameter block, make a copy of its data
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                double* data = new double[size];
                memcpy(data, addr, sizeof(double) * size);
                parameter_block_data[addr] = data;
                parameter_block_idx[addr] = 0;
            }
        }
    }
    
    void preMarginalize() {
        // Evaluate all residual blocks (compute Jacobians and residuals)
        for (auto it : residual_block_infos) {
            if (!it) continue;
            
            it->parameter_blocks_data.clear();
            
            for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
                double* addr = it->parameter_blocks[i];
                // Skip null parameter blocks
                if (!addr) continue;
                
                if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                    ROS_ERROR("Parameter block %p not found in marginalization info", addr);
                    continue;
                }
                
                it->parameter_blocks_data.push_back(parameter_block_data[addr]);
            }
            
            // Only evaluate if we have all parameter blocks
            if (it->parameter_blocks_data.size() == it->parameter_blocks.size()) {
                it->Evaluate();
            }
        }
    }
    
    void marginalize() {
        // Count total parameters size and index them
        int total_block_size = 0;
        for (const auto& it : parameter_block_size) {
            total_block_size += it.second;
        }
        
        // Map parameters to indices
        int idx = 0;
        for (auto& it : parameter_block_idx) {
            it.second = idx;
            idx += parameter_block_size[it.first];
        }
        
        // Get parameters to keep (not in any drop set)
        keep_block_size = 0;
        keep_block_idx.clear();
        keep_block_data.clear();
        keep_block_addr.clear();
        
        for (const auto& it : parameter_block_idx) {
            double* addr = it.first;
            
            if (!addr) continue; // Skip null addresses
            
            int size = parameter_block_size[addr];
            if (size <= 0) continue; // Skip invalid sizes
            
            // Check if this parameter should be dropped (marginalized)
            bool is_dropped = false;
            for (const auto& rbi : residual_block_infos) {
                if (!rbi) continue;
                
                for (int i = 0; i < static_cast<int>(rbi->parameter_blocks.size()); i++) {
                    if (rbi->parameter_blocks[i] == addr && 
                        std::find(rbi->drop_set.begin(), rbi->drop_set.end(), i) != rbi->drop_set.end()) {
                        is_dropped = true;
                        break;
                    }
                }
                if (is_dropped) break;
            }
            
            if (!is_dropped) {
                // This parameter is kept
                keep_block_size += size;
                
                if (parameter_block_data.find(addr) != parameter_block_data.end()) {
                    keep_block_data.push_back(parameter_block_data[addr]);
                    keep_block_addr.push_back(addr);
                    keep_block_idx.push_back(parameter_block_idx[addr]);
                }
            }
        }
        
        if (keep_block_size == 0) {
            ROS_WARN("No parameters to keep after marginalization");
            return;
        }
        
        // Calculate marginalized block size
        int marg_block_size = total_block_size - keep_block_size;
        if (marg_block_size <= 0) {
            ROS_WARN("No parameters to marginalize");
            return;
        }
        
        // Calculate total residual size
        int total_residual_size = 0;
        for (const auto& rbi : residual_block_infos) {
            if (!rbi) continue;
            total_residual_size += rbi->num_residuals;
        }
        
        if (total_residual_size == 0) {
            ROS_WARN("No residuals in marginalization");
            return;
        }
        
        // Construct the linearized system: Jacobian and residuals
        Eigen::MatrixXd linearized_jacobians(total_residual_size, total_block_size);
        linearized_jacobians.setZero();
        Eigen::VectorXd linearized_residuals(total_residual_size);
        linearized_residuals.setZero();
        
        // Fill the jacobian and residual
        int residual_idx = 0;
        for (const auto& rbi : residual_block_infos) {
            if (!rbi) continue;
            
            // Copy residuals
            linearized_residuals.segment(residual_idx, rbi->num_residuals) = rbi->residuals;
            
            // Copy jacobians for each parameter block
            for (int i = 0; i < static_cast<int>(rbi->parameter_blocks.size()); i++) {
                double* addr = rbi->parameter_blocks[i];
                // Skip null parameter blocks
                if (!addr) continue;
                
                if (parameter_block_idx.find(addr) == parameter_block_idx.end()) {
                    ROS_ERROR("Parameter block %p index not found during linearization", addr);
                    continue;
                }
                
                int idx = parameter_block_idx[addr];
                int size = parameter_block_size[addr];
                
                // Safety check for bounds
                if (residual_idx + rbi->num_residuals > linearized_jacobians.rows() ||
                    idx + size > linearized_jacobians.cols()) {
                    ROS_ERROR("Jacobian index out of bounds: residual_idx=%d, num_residuals=%d, idx=%d, size=%d",
                             residual_idx, rbi->num_residuals, idx, size);
                    continue;
                }
                
                // Copy jacobian block
                linearized_jacobians.block(residual_idx, idx, rbi->num_residuals, size) = rbi->jacobians[i];
            }
            
            residual_idx += rbi->num_residuals;
        }
        
        // Reorder the Jacobian to have [kept_params | marg_params]
        Eigen::MatrixXd reordered_jacobians = Eigen::MatrixXd::Zero(total_residual_size, total_block_size);
        
        // First, copy kept parameters
        int col_idx = 0;
        for (int i = 0; i < static_cast<int>(keep_block_addr.size()); i++) {
            double* addr = keep_block_addr[i];
            if (!addr) continue;
            
            int idx = keep_block_idx[i];
            int size = parameter_block_size[addr];
            
            // Safety check for bounds
            if (idx + size > linearized_jacobians.cols() || 
                col_idx + size > reordered_jacobians.cols()) {
                ROS_ERROR("Reordering jacobian index out of bounds");
                continue;
            }
            
            reordered_jacobians.block(0, col_idx, total_residual_size, size) = 
                linearized_jacobians.block(0, idx, total_residual_size, size);
            
            col_idx += size;
        }
        
        // Then, copy marginalized parameters
        for (const auto& it : parameter_block_idx) {
            double* addr = it.first;
            if (!addr) continue;
            
            // Skip if this parameter is kept
            if (std::find(keep_block_addr.begin(), keep_block_addr.end(), addr) != keep_block_addr.end()) {
                continue;
            }
            
            int idx = it.second;
            int size = parameter_block_size[addr];
            
            // Safety check for bounds
            if (idx + size > linearized_jacobians.cols() || 
                col_idx + size > reordered_jacobians.cols()) {
                ROS_ERROR("Reordering marg jacobian index out of bounds");
                continue;
            }
            
            reordered_jacobians.block(0, col_idx, total_residual_size, size) = 
                linearized_jacobians.block(0, idx, total_residual_size, size);
            
            col_idx += size;
        }
        
        // Split into kept and marginalized parts
        Eigen::MatrixXd jacobian_keep = reordered_jacobians.leftCols(keep_block_size);
        Eigen::MatrixXd jacobian_marg = reordered_jacobians.rightCols(marg_block_size);
        
        // Form the normal equations: J^T * J * delta_x = -J^T * r
        Eigen::MatrixXd H_marg = jacobian_marg.transpose() * jacobian_marg;
        Eigen::MatrixXd H_keep_marg = jacobian_keep.transpose() * jacobian_marg;
        Eigen::VectorXd b = -reordered_jacobians.transpose() * linearized_residuals;
        
        Eigen::VectorXd b_keep = b.head(keep_block_size);
        Eigen::VectorXd b_marg = b.tail(marg_block_size);
        
        // Add regularization to H_marg for numerical stability
        double lambda = 1e-4;
        for (int i = 0; i < H_marg.rows(); i++) {
            H_marg(i, i) += lambda;
        }
        
        // Compute Schur complement with regularization for numerical stability
        // First, compute eigendecomposition of H_marg
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H_marg);
        Eigen::VectorXd S = saes.eigenvalues();
        Eigen::MatrixXd V = saes.eigenvectors();
        
        // Apply regularization to eigenvalues
        Eigen::VectorXd S_inv = Eigen::VectorXd::Zero(S.size());
        double lambda_threshold = 1e-5;
        for (int i = 0; i < S.size(); i++) {
            if (S(i) > lambda_threshold) {
                S_inv(i) = 1.0 / S(i);
            } else {
                S_inv(i) = 0.0;
            }
        }
        
        // Compute inverse of H_marg using eigendecomposition
        Eigen::MatrixXd H_marg_inv = V * S_inv.asDiagonal() * V.transpose();
        
        // Compute Schur complement for prior
        Eigen::MatrixXd schur_complement = H_keep_marg * H_marg_inv * H_keep_marg.transpose();
        
        // Final linearized system for prior
        linearized_jacobians_ = jacobian_keep.transpose() * jacobian_keep - schur_complement;
        linearized_residuals_ = b_keep - H_keep_marg * H_marg_inv * b_marg;
    }
    
    // Interface for getting data needed by MarginalizationFactor
    const Eigen::MatrixXd& getLinearizedJacobians() const {
        return linearized_jacobians_;
    }
    
    const Eigen::VectorXd& getLinearizedResiduals() const {
        return linearized_residuals_;
    }
    
private:
    // Residual blocks to be marginalized
    std::vector<ResidualBlockInfo*> residual_block_infos;
    
    // Parameter block information
    std::map<double*, int> parameter_block_size;
    std::map<double*, int> parameter_block_idx;
    std::map<double*, double*> parameter_block_data;
    
    // Kept parameter block information
    int keep_block_size;
    std::vector<double*> keep_block_data;
    std::vector<double*> keep_block_addr;
    std::vector<int> keep_block_idx;
    
    // Linearized system after marginalization
    Eigen::MatrixXd linearized_jacobians_;
    Eigen::VectorXd linearized_residuals_;
};

// Fixed MarginalizationFactor with fixed parameter structure
class MarginalizationFactor : public ceres::CostFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    MarginalizationFactor(MarginalizationInfo* _marginalization_info) 
        : marginalization_info(_marginalization_info) {
        
        const Eigen::VectorXd& r = marginalization_info->getLinearizedResiduals();
        
        // Set residual size
        set_num_residuals(r.size());
        
        // CRITICAL: Always expect exactly 6 parameter blocks with fixed sizes
        mutable_parameter_block_sizes()->clear();
        mutable_parameter_block_sizes()->push_back(7); // pose1 (position + quaternion)
        mutable_parameter_block_sizes()->push_back(3); // velocity1
        mutable_parameter_block_sizes()->push_back(6); // bias1 (acc + gyro)
        mutable_parameter_block_sizes()->push_back(7); // pose2
        mutable_parameter_block_sizes()->push_back(3); // velocity2
        mutable_parameter_block_sizes()->push_back(6); // bias2
    }
    
    virtual bool Evaluate(double const* const* parameters, 
                         double* residuals, 
                         double** jacobians) const {
        
        const Eigen::VectorXd& linearized_residuals = marginalization_info->getLinearizedResiduals();
        
        // Fill residuals
        for (int i = 0; i < num_residuals() && i < linearized_residuals.size(); i++) {
            residuals[i] = linearized_residuals(i);
        }
        
        // Fill Jacobians if requested
        if (jacobians) {
            int param_sizes[6] = {7, 3, 6, 7, 3, 6}; // Fixed sizes for our parameter blocks
            
            for (int p = 0; p < 6; p++) {
                if (jacobians[p]) {
                    // Initialize jacobian for this parameter block to zero
                    memset(jacobians[p], 0, sizeof(double) * num_residuals() * param_sizes[p]);
                    
                    // Simple diagonal approximation of Jacobian for stability
                    double weight = (p < 3) ? 1.0 : 0.1; // Stronger weight for first state
                    for (int r = 0; r < std::min(num_residuals(), param_sizes[p]); r++) {
                        jacobians[p][r * param_sizes[p] + r] = weight;
                    }
                }
            }
        }
        
        return true;
    }
    
private:
    MarginalizationInfo* marginalization_info;
};

// UWB position factor for Ceres
class UwbPositionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    UwbPositionFactor(const Eigen::Vector3d& measured_position, double noise_std)
        : measured_position_(measured_position), noise_std_(noise_std) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract position from pose (position + quaternion)
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(pose);
        
        // Compute position residuals
        residuals[0] = (position[0] - T(measured_position_[0])) / T(noise_std_);
        residuals[1] = (position[1] - T(measured_position_[1])) / T(noise_std_);
        residuals[2] = (position[2] - T(measured_position_[2])) / T(noise_std_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_position, double noise_std) {
        return new ceres::AutoDiffCostFunction<UwbPositionFactor, 3, 7>(
            new UwbPositionFactor(measured_position, noise_std));
    }
    
private:
    const Eigen::Vector3d measured_position_;
    const double noise_std_;
};

// IMPROVED: IMU pre-integration factor for Ceres with bias correction validity check
class ImuFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
        
        // Store IMU measurements for possible recomputation
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
    
    ImuFactor(const ImuPreintegrationBetweenKeyframes& preint, const Eigen::Vector3d& gravity,
             double bias_correction_threshold = 0.05) 
        : preint_(preint), gravity_(gravity), bias_correction_threshold_(bias_correction_threshold) {}
    
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
        
        // IMPROVED: Bias corrections with validity check
        Eigen::Matrix<T, 3, 1> dba = ba_i - ba_ref;
        Eigen::Matrix<T, 3, 1> dbg = bg_i - bg_ref;
        
        // Check if bias corrections are small enough for linear approximation
        T dba_norm = dba.norm();
        T dbg_norm = dbg.norm();
        
        // IMPROVED: Add nonlinear correction if bias change is too large
        // This is a simplified model for the case study - in production,
        // you would recompute the entire preintegration
        if (dba_norm > T(bias_correction_threshold_) || dbg_norm > T(bias_correction_threshold_)) {
            // Apply nonlinear scaling to limit extreme corrections
            T dba_scale = ceres::fmin(T(bias_correction_threshold_) / dba_norm, T(1.0));
            T dbg_scale = ceres::fmin(T(bias_correction_threshold_) / dbg_norm, T(1.0));
            
            dba *= dba_scale;
            dbg *= dbg_scale;
        }
        
        // Limit bias correction magnitude for numerical stability
        const T max_bias_correction = T(0.1);
        for (int i = 0; i < 3; ++i) {
            dba(i) = ceres::fmin(ceres::fmax(dba(i), -max_bias_correction), max_bias_correction);
            dbg(i) = ceres::fmin(ceres::fmax(dbg(i), -max_bias_correction), max_bias_correction);
        }
        
        // Jacobian w.r.t bias changes
        Eigen::Matrix<T, 9, 6> jacobian_bias = preint_.jacobian_bias.cast<T>();
        
        // Bias correction vector
        Eigen::Matrix<T, 6, 1> bias_correction_vec;
        bias_correction_vec.template segment<3>(0) = dba;
        bias_correction_vec.template segment<3>(3) = dbg;
        
        // Corrections to pre-integrated measurements using Jacobians
        Eigen::Matrix<T, 9, 1> delta_bias_correction = jacobian_bias * bias_correction_vec;
        
        Eigen::Matrix<T, 3, 1> corrected_delta_p = delta_p + delta_bias_correction.template segment<3>(0);
        Eigen::Matrix<T, 3, 1> corrected_delta_v = delta_v + delta_bias_correction.template segment<3>(3);
        
        // Correction to delta_q (orientation)
        Eigen::Matrix<T, 3, 1> corrected_delta_q_vec = delta_bias_correction.template segment<3>(6);
        
        // Limit orientation correction magnitude
        const T max_angle_correction = T(0.1);  // radians
        T correction_norm = corrected_delta_q_vec.norm();
        if (correction_norm > max_angle_correction) {
            corrected_delta_q_vec *= (max_angle_correction / correction_norm);
        }
        
        Eigen::Quaternion<T> corrected_delta_q = delta_q * deltaQ(corrected_delta_q_vec);
        
        // Gravity vector in world frame
        Eigen::Matrix<T, 3, 1> g = gravity_.cast<T>();
        
        // Compute residuals
        Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(residuals);
        
        // Position residual
        residual.template segment<3>(0) = q_i.inverse() * ((p_j - p_i - v_i * sum_dt) - 
                                                       T(0.5) * g * sum_dt * sum_dt) - corrected_delta_p;
        
        // Orientation residual - IMPROVED with safer handling
        Eigen::Quaternion<T> q_i_inverse_times_q_j = q_i.conjugate() * q_j;
        Eigen::Quaternion<T> delta_q_residual = corrected_delta_q.conjugate() * q_i_inverse_times_q_j;
        
        // Normalize to ensure valid quaternion
        delta_q_residual.normalize();
        
        // Use safer conversion to angle-axis
        T dot_product = delta_q_residual.w();
        // Make sure it's in valid range for acos
        dot_product = ceres::abs(dot_product) < T(1.0) ? dot_product : 
                     (dot_product > T(0.0) ? T(0.999999) : T(-0.999999));
        
        // Handle the case when rotation is nearly zero
        if (dot_product > T(0.999999)) {
            residual.template segment<3>(3).setZero();
        } else {
            T angle = T(2.0) * ceres::acos(dot_product);
            Eigen::Matrix<T, 3, 1> axis;
            
            // Normalize axis safely
            T vec_norm = delta_q_residual.vec().norm();
            if (vec_norm > T(1e-10)) {
                axis = delta_q_residual.vec() / vec_norm;
            } else {
                // If very small rotation, just use x-axis
                axis = Eigen::Matrix<T, 3, 1>(T(1.0), T(0.0), T(0.0));
            }
            
            residual.template segment<3>(3) = angle * axis;
        }
        
        // Velocity residual
        residual.template segment<3>(6) = q_i.inverse() * (v_j - v_i - g * sum_dt) - corrected_delta_v;
        
        // CRITICAL: Bias change residuals - adjusted for high-speed scenario
        residual.template segment<3>(9) = (ba_j - ba_i) / T(0.002);   // Accelerometer bias change
        residual.template segment<3>(12) = (bg_j - bg_i) / T(0.0002); // Gyroscope bias change
        
        // Weight residuals by the information matrix
        Eigen::Matrix<T, 9, 9> sqrt_information = preint_.covariance.cast<T>().inverse().llt().matrixL().transpose();
        
        // Scale the position, orientation, and velocity residuals
        residual.template segment<3>(0) = sqrt_information.template block<3, 3>(0, 0) * residual.template segment<3>(0);
        residual.template segment<3>(3) = sqrt_information.template block<3, 3>(3, 3) * residual.template segment<3>(3);
        residual.template segment<3>(6) = sqrt_information.template block<3, 3>(6, 6) * residual.template segment<3>(6);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const ImuPreintegrationBetweenKeyframes& preint, 
                                      const Eigen::Vector3d& gravity,
                                      double bias_correction_threshold = 0.05) {
        return new ceres::AutoDiffCostFunction<ImuFactor, 15, 7, 3, 6, 7, 3, 6>(
            new ImuFactor(preint, gravity, bias_correction_threshold));
    }
    
private:
    const ImuPreintegrationBetweenKeyframes preint_;
    const Eigen::Vector3d gravity_;
    const double bias_correction_threshold_;
    
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

// Main UWB/GPS-IMU fusion class
class UwbImuFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    UwbImuFusion() {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // Load topic name parameters
        private_nh.param<std::string>("imu_topic", imu_topic_, "/imu/data");
        private_nh.param<std::string>("uwb_topic", uwb_topic_, "/sensor_simulator/UWBPoistionPS");
        private_nh.param<int>("imu_queue_size", imu_queue_size_, 1000);
        private_nh.param<int>("uwb_queue_size", uwb_queue_size_, 100);
        
        // Add GPS/UWB mode selection
        private_nh.param<bool>("use_gps_instead_of_uwb", use_gps_instead_of_uwb_, false);
        
        // Load GPS-related parameters
        private_nh.param<std::string>("gps_topic", gps_topic_, "/novatel_data/inspvax");
        private_nh.param<int>("gps_queue_size", gps_queue_size_, 100);
        private_nh.param<double>("gps_position_noise", gps_position_noise_, 0.01); // 1cm - GPS is more accurate than UWB
        private_nh.param<double>("gps_velocity_noise", gps_velocity_noise_, 0.01); // 1cm/s
        private_nh.param<double>("gps_orientation_noise", gps_orientation_noise_, 0.1); // rad
        
        // GPS data usage configuration
        private_nh.param<bool>("use_gps_orientation_as_initial", use_gps_orientation_as_initial_, true);
        private_nh.param<bool>("use_gps_orientation_as_constraint", use_gps_orientation_as_constraint_, false);
        private_nh.param<bool>("use_gps_velocity", use_gps_velocity_, true);
        
        // Initialize GPS reference point
        has_gps_reference_ = false;
        ref_latitude_ = 0.0;
        ref_longitude_ = 0.0;
        ref_altitude_ = 0.0;
        
        // Load output topic parameters
        private_nh.param<std::string>("optimized_pose_topic", optimized_pose_topic_, "/uwb_imu_fusion/optimized_pose");
        private_nh.param<std::string>("imu_pose_topic", imu_pose_topic_, "/uwb_imu_fusion/imu_pose");
        private_nh.param<int>("optimized_pose_queue_size", optimized_pose_queue_size_, 10);
        private_nh.param<int>("imu_pose_queue_size", imu_pose_queue_size_, 200);
        
        // Load parameters
        private_nh.param<double>("gravity_magnitude", gravity_magnitude_, 9.81);
        
        // Realistic IMU noise parameters
        private_nh.param<double>("imu_acc_noise", imu_acc_noise_, 0.03);    // m/s²
        private_nh.param<double>("imu_gyro_noise", imu_gyro_noise_, 0.002); // rad/s
        
        // CRITICAL: Realistic bias parameters
        private_nh.param<double>("imu_acc_bias_noise", imu_acc_bias_noise_, 0.0001);  // m/s²/sqrt(s)
        private_nh.param<double>("imu_gyro_bias_noise", imu_gyro_bias_noise_, 0.00001); // rad/s/sqrt(s)
        private_nh.param<double>("acc_bias_max", acc_bias_max_, 0.1);   // Maximum allowed acc bias (m/s²)
        private_nh.param<double>("gyro_bias_max", gyro_bias_max_, 0.01); // Maximum allowed gyro bias (rad/s)
        
        // CRITICAL: Initial biases (small realistic values)
        private_nh.param<double>("initial_acc_bias_x", initial_acc_bias_x_, 0.05);
        private_nh.param<double>("initial_acc_bias_y", initial_acc_bias_y_, -0.05);
        private_nh.param<double>("initial_acc_bias_z", initial_acc_bias_z_, 0.05);
        private_nh.param<double>("initial_gyro_bias_x", initial_gyro_bias_x_, 0.001);
        private_nh.param<double>("initial_gyro_bias_y", initial_gyro_bias_y_, -0.001);
        private_nh.param<double>("initial_gyro_bias_z", initial_gyro_bias_z_, 0.001);
        
        private_nh.param<double>("uwb_position_noise", uwb_position_noise_, 0.05);  // m
        private_nh.param<int>("optimization_window_size", optimization_window_size_, 20); // Reduced for stability
        
        // Frame IDs
        private_nh.param<std::string>("world_frame_id", world_frame_id_, "map");
        private_nh.param<std::string>("body_frame_id", body_frame_id_, "base_link");
        
        private_nh.param<double>("optimization_frequency", optimization_frequency_, 10.0);
        private_nh.param<double>("imu_buffer_time_length", imu_buffer_time_length_, 10.0);
        private_nh.param<int>("max_iterations", max_iterations_, 10); // Lower iterations
        
        private_nh.param<bool>("enable_bias_estimation", enable_bias_estimation_, true);
        
        // NEW: Enable marginalization
        private_nh.param<bool>("enable_marginalization", enable_marginalization_, true);
        
        // NEW: Add feature configuration parameters
        private_nh.param<bool>("enable_roll_pitch_constraint", enable_roll_pitch_constraint_, true);
        private_nh.param<bool>("enable_gravity_alignment_factor", enable_gravity_alignment_factor_, true);
        private_nh.param<bool>("enable_orientation_smoothness_factor", enable_orientation_smoothness_factor_, true);
        private_nh.param<bool>("enable_velocity_constraint", enable_velocity_constraint_, true);
        private_nh.param<bool>("enable_horizontal_velocity_incentive", enable_horizontal_velocity_incentive_, true);
        private_nh.param<bool>("enable_imu_orientation_factor", enable_imu_orientation_factor_, true);
        
        // Constraint weights
        private_nh.param<double>("roll_pitch_weight", roll_pitch_weight_, 300.0); // Increased from 100.0
        private_nh.param<double>("max_imu_dt", max_imu_dt_, 0.5);
        private_nh.param<double>("imu_orientation_weight", imu_orientation_weight_, 50.0);
        private_nh.param<double>("bias_constraint_weight", bias_constraint_weight_, 1000.0);
        
        // IMPROVED: Velocity parameters for high-speed scenarios (0-70 km/h)
        private_nh.param<double>("max_velocity", max_velocity_, 25.0); // Maximum velocity (m/s) = 90 km/h
        private_nh.param<double>("velocity_constraint_weight", velocity_constraint_weight_, 150.0);
        private_nh.param<double>("min_horizontal_velocity", min_horizontal_velocity_, 0.5); // Minimum desired velocity
        private_nh.param<double>("horizontal_velocity_weight", horizontal_velocity_weight_, 10.0);
        
        private_nh.param<double>("orientation_smoothness_weight", orientation_smoothness_weight_, 100.0);
        private_nh.param<double>("gravity_alignment_weight", gravity_alignment_weight_, 150.0);
        
        // IMPROVED: RK4 integration parameters
        private_nh.param<double>("max_integration_dt", max_integration_dt_, 0.005); // Reduced for high-speed scenarios
        private_nh.param<double>("min_integration_dt", min_integration_dt_, 1e-8); // Minimum step size
        private_nh.param<double>("bias_correction_threshold", bias_correction_threshold_, 0.05); // Threshold for bias validity check
        
        // Initialize with small non-zero biases that match your simulation
        initial_acc_bias_ = Eigen::Vector3d(initial_acc_bias_x_, initial_acc_bias_y_, initial_acc_bias_z_);
        initial_gyro_bias_ = Eigen::Vector3d(initial_gyro_bias_x_, initial_gyro_bias_y_, initial_gyro_bias_z_);
        
        // Initialize subscribers and publishers based on mode
        imu_sub_ = nh.subscribe(imu_topic_, imu_queue_size_, &UwbImuFusion::imuCallback, this);
        
        if (use_gps_instead_of_uwb_) {
            gps_sub_ = nh.subscribe(gps_topic_, gps_queue_size_, &UwbImuFusion::gpsCallback, this);
            ROS_INFO("Using GPS+IMU fusion mode");
            ROS_INFO("Subscribing to GPS topic: %s (queue: %d)", gps_topic_.c_str(), gps_queue_size_);
            ROS_INFO("GPS noise: position=%.3f m, velocity=%.3f m/s, orientation=%.3f rad", 
                    gps_position_noise_, gps_velocity_noise_, gps_orientation_noise_);
            ROS_INFO("GPS orientation usage: as initial=%s, as constraint=%s", 
                    use_gps_orientation_as_initial_ ? "enabled" : "disabled",
                    use_gps_orientation_as_constraint_ ? "enabled" : "disabled");
            ROS_INFO("GPS velocity usage: %s", use_gps_velocity_ ? "enabled" : "disabled");
        } else {
            uwb_sub_ = nh.subscribe(uwb_topic_, uwb_queue_size_, &UwbImuFusion::uwbCallback, this);
            ROS_INFO("Using UWB+IMU fusion mode");
            ROS_INFO("Subscribing to UWB topic: %s (queue: %d)", uwb_topic_.c_str(), uwb_queue_size_);
            ROS_INFO("UWB position noise: %.3f m", uwb_position_noise_);
        }
        
        optimized_pose_pub_ = nh.advertise<nav_msgs::Odometry>(optimized_pose_topic_, optimized_pose_queue_size_);
        imu_pose_pub_ = nh.advertise<nav_msgs::Odometry>(imu_pose_topic_, imu_pose_queue_size_);
        
        // Initialize visualization publishers
        gps_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/gps_path", 1, true);
        optimized_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/optimized_path", 1, true);
        position_error_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/errors/position", 1);
        velocity_error_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/errors/velocity", 1);

        // Initialize path messages
        gps_path_msg_.header.frame_id = world_frame_id_;
        optimized_path_msg_.header.frame_id = world_frame_id_;

        ROS_INFO("Visualization publishers initialized. Add these topics in RViz:");
        ROS_INFO(" - GPS trajectory: Path display at /trajectory/gps_path");
        ROS_INFO(" - Optimized trajectory: Path display at /trajectory/optimized_path");
        ROS_INFO(" - Position errors: MarkerArray at /errors/position");
        ROS_INFO(" - Velocity errors: MarkerArray at /errors/velocity");
        
        // Initialize state
        is_initialized_ = false;
        has_imu_data_ = false;
        last_imu_timestamp_ = 0;
        last_processed_timestamp_ = 0;
        just_optimized_ = false;
        optimization_count_ = 0;
        
        // Initialize marginalization
        last_marginalization_info_ = nullptr;
        
        initializeState();
        
        // Setup optimization timer
        optimization_timer_ = nh.createTimer(ros::Duration(1.0/optimization_frequency_), 
                                           &UwbImuFusion::optimizationTimerCallback, this);
        
        ROS_INFO("UWB/GPS-IMU Fusion node initialized with IMU-rate pose publishing");
        ROS_INFO("Subscribing to IMU topic: %s (queue: %d)", imu_topic_.c_str(), imu_queue_size_);
        ROS_INFO("Publishing to optimized pose topic: %s", optimized_pose_topic_.c_str());
        ROS_INFO("Publishing to IMU pose topic: %s", imu_pose_topic_.c_str());
        ROS_INFO("IMU pre-integration using RK4 with max step size: %.4f sec", max_integration_dt_);
        ROS_INFO("IMU noise: acc=%.3f m/s², gyro=%.4f rad/s", imu_acc_noise_, imu_gyro_noise_);
        ROS_INFO("Bias noise: acc=%.6f m/s²/sqrt(s), gyro=%.6f rad/s/sqrt(s)", imu_acc_bias_noise_, imu_gyro_bias_noise_);
        ROS_INFO("Bias parameters: max_acc=%.3f m/s², max_gyro=%.4f rad/s", 
                 acc_bias_max_, gyro_bias_max_);
        ROS_INFO("Velocity constraints: max=%.1f m/s (%.1f km/h), min_horizontal=%.1f m/s", 
                 max_velocity_, max_velocity_*3.6, min_horizontal_velocity_);
        ROS_INFO("Initial biases: acc=[%.3f, %.3f, %.3f], gyro=[%.3f, %.3f, %.3f]",
                 initial_acc_bias_.x(), initial_acc_bias_.y(), initial_acc_bias_.z(),
                 initial_gyro_bias_.x(), initial_gyro_bias_.y(), initial_gyro_bias_.z());
        ROS_INFO("Bias estimation is %s", enable_bias_estimation_ ? "enabled" : "disabled");
        ROS_INFO("Marginalization is %s", enable_marginalization_ ? "enabled" : "disabled");
        ROS_INFO("Using RK4 integration with bias correction threshold: %.3f", bias_correction_threshold_);
        
        // Log feature configuration status
        ROS_INFO("Feature configuration: roll_pitch=%s, gravity=%s, orientation_smooth=%s",
                 enable_roll_pitch_constraint_ ? "enabled" : "disabled",
                 enable_gravity_alignment_factor_ ? "enabled" : "disabled",
                 enable_orientation_smoothness_factor_ ? "enabled" : "disabled");
        ROS_INFO("Feature configuration: velocity=%s, horizontal_velocity=%s, imu_orientation=%s",
                 enable_velocity_constraint_ ? "enabled" : "disabled",
                 enable_horizontal_velocity_incentive_ ? "enabled" : "disabled",
                 enable_imu_orientation_factor_ ? "enabled" : "disabled");
        ROS_INFO("Optimized for high-speed scenarios (0-70 km/h)");
    }

    ~UwbImuFusion() {
        // Clean up marginalization resources
        if (last_marginalization_info_) {
            delete last_marginalization_info_;
            last_marginalization_info_ = nullptr;
        }
    }

private:
    // GPS/UWB mode selection
    bool use_gps_instead_of_uwb_;
    
    // GPS data usage configuration
    bool use_gps_orientation_as_initial_;
    bool use_gps_orientation_as_constraint_;
    bool use_gps_velocity_;
    
    // GPS-related members
    ros::Subscriber gps_sub_;
    std::string gps_topic_;
    int gps_queue_size_;
    double gps_position_noise_;
    double gps_velocity_noise_;
    double gps_orientation_noise_;

    // GPS reference for ENU conversion
    bool has_gps_reference_;
    double ref_latitude_;
    double ref_longitude_;
    double ref_altitude_;

    // ROS subscribers and publishers
    ros::Subscriber imu_sub_;
    ros::Subscriber uwb_sub_;
    ros::Publisher optimized_pose_pub_;
    ros::Publisher imu_pose_pub_;
    ros::Timer optimization_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // Visualization publishers
    ros::Publisher gps_path_pub_;
    ros::Publisher optimized_path_pub_;
    ros::Publisher position_error_pub_;
    ros::Publisher velocity_error_pub_;

    // Path messages for visualization
    nav_msgs::Path gps_path_msg_;
    nav_msgs::Path optimized_path_msg_;

    // Error visualization
    visualization_msgs::MarkerArray position_error_markers_;
    visualization_msgs::MarkerArray velocity_error_markers_;

    // Store latest error statistics
    struct ErrorStats {
        double position_error_e = 0.0;
        double position_error_n = 0.0;
        double position_error_u = 0.0;
        double position_error_norm = 0.0;
        double velocity_error_e = 0.0;
        double velocity_error_n = 0.0;
        double velocity_error_u = 0.0;
        double velocity_error_norm = 0.0;
        double timestamp = 0.0;
    };
    ErrorStats latest_error_stats_;

    // Topic names and queue sizes
    std::string imu_topic_;
    std::string uwb_topic_;
    int imu_queue_size_;
    int uwb_queue_size_;
    std::string optimized_pose_topic_;
    std::string imu_pose_topic_;
    int optimized_pose_queue_size_;
    int imu_pose_queue_size_;

    // Parameters
    double gravity_magnitude_;
    double imu_acc_noise_;
    double imu_gyro_noise_;
    double imu_acc_bias_noise_;
    double imu_gyro_bias_noise_;
    double acc_bias_max_;      // Maximum allowed accelerometer bias
    double gyro_bias_max_;     // Maximum allowed gyroscope bias
    double initial_acc_bias_x_, initial_acc_bias_y_, initial_acc_bias_z_;
    double initial_gyro_bias_x_, initial_gyro_bias_y_, initial_gyro_bias_z_;
    double uwb_position_noise_;
    int optimization_window_size_;
    double optimization_frequency_;
    double imu_buffer_time_length_;
    int max_iterations_;
    bool enable_bias_estimation_;
    bool enable_marginalization_;  // Whether to use marginalization
    
    // Feature configuration parameters
    bool enable_roll_pitch_constraint_;
    bool enable_gravity_alignment_factor_;
    bool enable_orientation_smoothness_factor_;
    bool enable_velocity_constraint_;
    bool enable_horizontal_velocity_incentive_;
    bool enable_imu_orientation_factor_;
    
    std::string world_frame_id_;
    std::string body_frame_id_;
    double roll_pitch_weight_;
    double max_imu_dt_;
    double imu_orientation_weight_;
    double bias_constraint_weight_;
    double max_velocity_;      // Maximum allowed velocity - now 25 m/s (90 km/h) for higher speeds
    double velocity_constraint_weight_;
    double min_horizontal_velocity_; // Minimum desired horizontal velocity
    double horizontal_velocity_weight_; // Weight for horizontal velocity incentive
    double orientation_smoothness_weight_; // Weight for orientation smoothness constraints
    double gravity_alignment_weight_;      // Weight for gravity alignment constraint
    
    // IMPROVED: RK4 integration parameters
    double max_integration_dt_; // Maximum step size for RK4 integration
    double min_integration_dt_; // Minimum step size to avoid numerical issues
    double bias_correction_threshold_; // Threshold for bias correction validity check
    
    // Initial bias values
    Eigen::Vector3d initial_acc_bias_;
    Eigen::Vector3d initial_gyro_bias_;

    // Marginalization resources
    MarginalizationInfo* last_marginalization_info_;

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

    // Structure for optimization variables
    struct OptVariables {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        double pose[7]; // position (3) + quaternion (4)
        double velocity[3];
        double bias[6]; // acc_bias (3) + gyro_bias (3)
    };

    State current_state_;
    std::deque<State, Eigen::aligned_allocator<State>> state_window_;
    bool is_initialized_;
    bool has_imu_data_;
    double last_imu_timestamp_;
    double last_processed_timestamp_;
    bool just_optimized_;
    int optimization_count_;
    
    // IMU Preintegration between keyframes
    typedef typename ImuFactor::ImuPreintegrationBetweenKeyframes ImuPreintegrationBetweenKeyframes;
    
    // Map to store preintegration data between consecutive keyframes
    std::map<std::pair<double, double>, ImuPreintegrationBetweenKeyframes> preintegration_map_;
    
    std::deque<sensor_msgs::Imu> imu_buffer_;
    
    // UWB measurements
    struct UwbMeasurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        double timestamp;
    };

    std::vector<UwbMeasurement> uwb_measurements_;

    // GPS measurements
    struct GpsMeasurement {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Quaterniond orientation;
        double timestamp;
    };
    std::vector<GpsMeasurement> gps_measurements_;

    // Mutex for thread safety
    std::mutex data_mutex_;

    // Gravity vector in world frame (ENU, Z-up)
    Eigen::Vector3d gravity_world_;
    
    // ==================== VISUALIZATION METHODS ====================

    // Update optimized path in publishOptimizedPose 
    void updateOptimizedPath() {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(current_state_.timestamp);
        pose_stamped.header.frame_id = world_frame_id_;
        pose_stamped.pose.position.x = current_state_.position.x();
        pose_stamped.pose.position.y = current_state_.position.y();
        pose_stamped.pose.position.z = current_state_.position.z();
        pose_stamped.pose.orientation.w = current_state_.orientation.w();
        pose_stamped.pose.orientation.x = current_state_.orientation.x();
        pose_stamped.pose.orientation.y = current_state_.orientation.y();
        pose_stamped.pose.orientation.z = current_state_.orientation.z();
        
        optimized_path_msg_.header.stamp = ros::Time(current_state_.timestamp);
        optimized_path_msg_.poses.push_back(pose_stamped);
        
        // Limit path size
        if (optimized_path_msg_.poses.size() > 1000) {
            optimized_path_msg_.poses.erase(optimized_path_msg_.poses.begin());
        }
        
        // Publish the optimized path
        optimized_path_pub_.publish(optimized_path_msg_);
    }

    // Calculate and visualize position error between optimized pose and GPS
    void calculateAndVisualizePositionError() {
        position_error_markers_.markers.clear();
        
        if (gps_measurements_.empty() || state_window_.empty()) {
            return;
        }
        
        // Find closest GPS measurement to the current state
        double min_time_diff = std::numeric_limits<double>::max();
        GpsMeasurement closest_gps;
        bool found_gps = false;
        
        double current_time = current_state_.timestamp;
        
        for (const auto& gps : gps_measurements_) {
            double time_diff = std::abs(gps.timestamp - current_time);
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                closest_gps = gps;
                found_gps = true;
            }
        }
        
        // If we found a close GPS measurement (within 0.1s)
        if (found_gps && min_time_diff < 0.1) {
            // Calculate position error vector (optimized - GPS)
            Eigen::Vector3d position_error = current_state_.position - closest_gps.position;
            
            // Update error statistics
            latest_error_stats_.position_error_e = position_error.x();
            latest_error_stats_.position_error_n = position_error.y();
            latest_error_stats_.position_error_u = position_error.z();
            latest_error_stats_.position_error_norm = position_error.norm();
            latest_error_stats_.timestamp = current_time;
            
            // Calculate error magnitude
            double error_norm = position_error.norm();
            
            // Create marker for the total error vector (from GPS to optimized)
            visualization_msgs::Marker error_marker;
            error_marker.header.frame_id = world_frame_id_;
            error_marker.header.stamp = ros::Time(current_time);
            error_marker.ns = "position_error";
            error_marker.id = 0;
            error_marker.type = visualization_msgs::Marker::ARROW;
            error_marker.action = visualization_msgs::Marker::ADD;
            
            // Start of the arrow is at the GPS position
            error_marker.points.resize(2);
            error_marker.points[0].x = closest_gps.position.x();
            error_marker.points[0].y = closest_gps.position.y();
            error_marker.points[0].z = closest_gps.position.z();
            
            // End of the arrow is at the estimated position
            error_marker.points[1].x = current_state_.position.x();
            error_marker.points[1].y = current_state_.position.y();
            error_marker.points[1].z = current_state_.position.z();
            
            // Set the arrow properties
            error_marker.scale.x = 0.05; // shaft diameter
            error_marker.scale.y = 0.1;  // head diameter
            error_marker.scale.z = 0.1;  // head length
            
            // Color the arrow based on error magnitude (green to red)
            error_marker.color.a = 1.0;
            
            // Scale from green (small error) to red (large error)
            double max_expected_error = 5.0; // meters
            double error_ratio = std::min(1.0, error_norm / max_expected_error);
            error_marker.color.r = error_ratio;
            error_marker.color.g = 1.0 - error_ratio;
            error_marker.color.b = 0.0;
            
            // Add to marker array
            position_error_markers_.markers.push_back(error_marker);
            
            // Create text marker to display error value
            visualization_msgs::Marker text_marker;
            text_marker.header = error_marker.header;
            text_marker.ns = "position_error_text";
            text_marker.id = 0;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // Position the text above the error arrow
            text_marker.pose.position.x = (closest_gps.position.x() + current_state_.position.x()) / 2.0;
            text_marker.pose.position.y = (closest_gps.position.y() + current_state_.position.y()) / 2.0;
            text_marker.pose.position.z = (closest_gps.position.z() + current_state_.position.z()) / 2.0 + 0.5;
            text_marker.pose.orientation.w = 1.0;
            
            // Set the text content to show error components
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) 
               << "Error: " << error_norm << "m "
               << "E:" << position_error.x() << " "
               << "N:" << position_error.y() << " "
               << "U:" << position_error.z();
            text_marker.text = ss.str();
            
            // Set text properties
            text_marker.scale.z = 0.3; // text height
            text_marker.color.r = error_ratio;
            text_marker.color.g = 1.0 - error_ratio;
            text_marker.color.b = 0.0;
            text_marker.color.a = 1.0;
            
            // Add to marker array
            position_error_markers_.markers.push_back(text_marker);
            
            // Create component arrows for ENU directions
            std::string components[3] = {"east", "north", "up"};
            
            // Define colors and directions for each component
            struct ComponentInfo {
                Eigen::Vector3d direction;
                std::array<float, 3> color;
                int index;
            };
            
            ComponentInfo enu_components[3] = {
                {Eigen::Vector3d(1, 0, 0), {1.0f, 0.0f, 0.0f}, 0}, // East (Red)
                {Eigen::Vector3d(0, 1, 0), {0.0f, 1.0f, 0.0f}, 1}, // North (Green)
                {Eigen::Vector3d(0, 0, 1), {0.0f, 0.0f, 1.0f}, 2}  // Up (Blue)
            };
            
            // Create markers for each component
            for (int i = 0; i < 3; i++) {
                // Create a new marker for this component
                visualization_msgs::Marker component_marker;
                component_marker.header = error_marker.header;
                component_marker.ns = "position_error_" + components[i];
                component_marker.id = i;
                component_marker.type = visualization_msgs::Marker::ARROW;
                component_marker.action = visualization_msgs::Marker::ADD;
                
                // Start at GPS position
                component_marker.points.resize(2);
                component_marker.points[0].x = closest_gps.position.x();
                component_marker.points[0].y = closest_gps.position.y();
                component_marker.points[0].z = closest_gps.position.z();
                
                // Calculate end point: project the error along this component's direction
                double component_error = position_error(enu_components[i].index);
                
                // End at GPS position + error component in specific direction
                component_marker.points[1] = component_marker.points[0];
                component_marker.points[1].x += component_error * enu_components[i].direction.x();
                component_marker.points[1].y += component_error * enu_components[i].direction.y();
                component_marker.points[1].z += component_error * enu_components[i].direction.z();
                
                // Ensure minimum arrow size for visibility (if there is some error)
                const double min_visible_length = 0.1; // meters
                double arrow_length = std::abs(component_error);
                
                if (arrow_length > 0.001 && arrow_length < min_visible_length) {
                    // Scale up small errors to be visible
                    double scale_factor = min_visible_length / arrow_length;
                    
                    // Apply scaling to make arrow longer
                    component_marker.points[1].x = component_marker.points[0].x + 
                        (component_marker.points[1].x - component_marker.points[0].x) * scale_factor;
                    component_marker.points[1].y = component_marker.points[0].y + 
                        (component_marker.points[1].y - component_marker.points[0].y) * scale_factor;
                    component_marker.points[1].z = component_marker.points[0].z + 
                        (component_marker.points[1].z - component_marker.points[0].z) * scale_factor;
                }
                
                // Set the arrow properties
                component_marker.scale.x = 0.04; // shaft diameter
                component_marker.scale.y = 0.08; // head diameter
                component_marker.scale.z = 0.08; // head length
                
                // Set color based on component (RGB = ENU)
                component_marker.color.r = enu_components[i].color[0];
                component_marker.color.g = enu_components[i].color[1];
                component_marker.color.b = enu_components[i].color[2];
                component_marker.color.a = 0.8;
                
                // Add label with component error value
                visualization_msgs::Marker text_marker;
                text_marker.header = component_marker.header;
                text_marker.ns = "position_error_text_" + components[i];
                text_marker.id = i;
                text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::Marker::ADD;
                
                // Position the text at the end of the component arrow
                text_marker.pose.position = component_marker.points[1];
                text_marker.pose.position.z += 0.15; // offset slightly above the arrow
                text_marker.pose.orientation.w = 1.0;
                
                // Set the text content
                std::stringstream ss;
                ss << components[i] << ": " << std::fixed << std::setprecision(2) << component_error << "m";
                text_marker.text = ss.str();
                
                // Set text properties
                text_marker.scale.z = 0.2; // text height
                text_marker.color.r = enu_components[i].color[0];
                text_marker.color.g = enu_components[i].color[1];
                text_marker.color.b = enu_components[i].color[2];
                text_marker.color.a = 1.0;
                
                // Only add the marker if there is some error in this component
                if (std::abs(component_error) > 0.001 || arrow_length >= min_visible_length) {
                    position_error_markers_.markers.push_back(component_marker);
                    position_error_markers_.markers.push_back(text_marker);
                }
            }
            
            // Publish the position error visualization
            position_error_pub_.publish(position_error_markers_);
            
            // Log error statistics periodically
            static double last_log_time = 0;
            if (current_time - last_log_time > 5.0) {
                ROS_INFO("Position Error (m): %.2f (E:%.2f, N:%.2f, U:%.2f), publishing %zu markers", 
                         error_norm, position_error.x(), position_error.y(), position_error.z(),
                         position_error_markers_.markers.size());
                last_log_time = current_time;
            }
        } else {
            // No matching GPS data found
            if (found_gps) {
                ROS_WARN_THROTTLE(5.0, "Found closest GPS but time difference too large: %.3f seconds", min_time_diff);
            } else {
                ROS_WARN_THROTTLE(5.0, "No GPS measurements available for error calculation");
            }
        }
    }

    // Calculate and visualize velocity error between optimized pose and GPS
    void calculateAndVisualizeVelocityError() {
        velocity_error_markers_.markers.clear();
        
        if (gps_measurements_.empty() || state_window_.empty()) {
            return;
        }
        
        // Find closest GPS measurement to the current state
        double min_time_diff = std::numeric_limits<double>::max();
        GpsMeasurement closest_gps;
        bool found_gps = false;
        
        double current_time = current_state_.timestamp;
        
        for (const auto& gps : gps_measurements_) {
            double time_diff = std::abs(gps.timestamp - current_time);
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                closest_gps = gps;
                found_gps = true;
            }
        }
        
        // If we found a close GPS measurement (within 0.1s)
        if (found_gps && min_time_diff < 0.1) {
            // Calculate velocity error vector
            Eigen::Vector3d velocity_error = current_state_.velocity - closest_gps.velocity;
            
            // Update error statistics
            latest_error_stats_.velocity_error_e = velocity_error.x();
            latest_error_stats_.velocity_error_n = velocity_error.y();
            latest_error_stats_.velocity_error_u = velocity_error.z();
            latest_error_stats_.velocity_error_norm = velocity_error.norm();
            
            // Create markers for the velocity vectors
            // 1. Current estimated velocity
            visualization_msgs::Marker est_vel_marker;
            est_vel_marker.header.frame_id = world_frame_id_;
            est_vel_marker.header.stamp = ros::Time(current_time);
            est_vel_marker.ns = "velocity";
            est_vel_marker.id = 0;
            est_vel_marker.type = visualization_msgs::Marker::ARROW;
            est_vel_marker.action = visualization_msgs::Marker::ADD;
            
            // Start at current position
            est_vel_marker.points.resize(2);
            est_vel_marker.points[0].x = current_state_.position.x();
            est_vel_marker.points[0].y = current_state_.position.y();
            est_vel_marker.points[0].z = current_state_.position.z();
            
            // Scale velocity for visualization (2x scale)
            double vel_scale = 2.0;
            est_vel_marker.points[1].x = current_state_.position.x() + vel_scale * current_state_.velocity.x();
            est_vel_marker.points[1].y = current_state_.position.y() + vel_scale * current_state_.velocity.y();
            est_vel_marker.points[1].z = current_state_.position.z() + vel_scale * current_state_.velocity.z();
            
            // Set marker properties
            est_vel_marker.scale.x = 0.05; // shaft diameter
            est_vel_marker.scale.y = 0.1;  // head diameter
            est_vel_marker.scale.z = 0.1;  // head length
            est_vel_marker.color.r = 0.0;
            est_vel_marker.color.g = 0.8;
            est_vel_marker.color.b = 0.0;
            est_vel_marker.color.a = 1.0;
            
            // 2. GPS velocity
            visualization_msgs::Marker gps_vel_marker = est_vel_marker;
            gps_vel_marker.id = 1;
            
            // Start at GPS position
            gps_vel_marker.points[0].x = closest_gps.position.x();
            gps_vel_marker.points[0].y = closest_gps.position.y();
            gps_vel_marker.points[0].z = closest_gps.position.z();
            
            // End at GPS position + GPS velocity (scaled)
            gps_vel_marker.points[1].x = closest_gps.position.x() + vel_scale * closest_gps.velocity.x();
            gps_vel_marker.points[1].y = closest_gps.position.y() + vel_scale * closest_gps.velocity.y();
            gps_vel_marker.points[1].z = closest_gps.position.z() + vel_scale * closest_gps.velocity.z();
            
            // Set GPS velocity marker color
            gps_vel_marker.color.r = 0.8;
            gps_vel_marker.color.g = 0.0;
            gps_vel_marker.color.b = 0.0;
            
            // Add to marker array
            velocity_error_markers_.markers.push_back(est_vel_marker);
            velocity_error_markers_.markers.push_back(gps_vel_marker);
            
            // Create text marker to display velocity error value
            visualization_msgs::Marker text_marker;
            text_marker.header = est_vel_marker.header;
            text_marker.ns = "velocity_error_text";
            text_marker.id = 0;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // Position the text above the current position
            text_marker.pose.position.x = current_state_.position.x();
            text_marker.pose.position.y = current_state_.position.y();
            text_marker.pose.position.z = current_state_.position.z() + 1.0;
            text_marker.pose.orientation.w = 1.0;
            
            // Set the text content to show velocity error components
            double velocity_error_norm = velocity_error.norm();
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) 
               << "Vel Error: " << velocity_error_norm << "m/s "
               << "E:" << velocity_error.x() << " "
               << "N:" << velocity_error.y() << " "
               << "U:" << velocity_error.z();
            text_marker.text = ss.str();
            
            // Set text properties
            text_marker.scale.z = 0.3; // text height
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            
            // Add to marker array
            velocity_error_markers_.markers.push_back(text_marker);
            
            // Add velocity component visualizations for ENU directions
            std::string components[3] = {"east", "north", "up"};
            Eigen::Vector3d unit_vectors[3] = {
                Eigen::Vector3d(1, 0, 0),
                Eigen::Vector3d(0, 1, 0),
                Eigen::Vector3d(0, 0, 1)
            };
            
            for (int i = 0; i < 3; i++) {
                visualization_msgs::Marker vel_est_component = est_vel_marker;
                vel_est_component.ns = "velocity_est_" + components[i];
                vel_est_component.id = i;
                
                // Start at current position
                vel_est_component.points[0] = est_vel_marker.points[0];
                
                // End at current position + velocity component
                vel_est_component.points[1] = est_vel_marker.points[0];
                vel_est_component.points[1].x += vel_scale * current_state_.velocity(i) * unit_vectors[i](0);
                vel_est_component.points[1].y += vel_scale * current_state_.velocity(i) * unit_vectors[i](1);
                vel_est_component.points[1].z += vel_scale * current_state_.velocity(i) * unit_vectors[i](2);
                
                // Set color based on component (RGB = ENU)
                vel_est_component.color.r = (i == 0) ? 0.8 : 0.0;
                vel_est_component.color.g = (i == 1) ? 0.8 : 0.0;
                vel_est_component.color.b = (i == 2) ? 0.8 : 0.0;
                vel_est_component.color.a = 0.5;
                
                velocity_error_markers_.markers.push_back(vel_est_component);
                
                // Do the same for GPS velocity
                visualization_msgs::Marker vel_gps_component = gps_vel_marker;
                vel_gps_component.ns = "velocity_gps_" + components[i];
                vel_gps_component.id = i;
                
                vel_gps_component.points[0] = gps_vel_marker.points[0];
                vel_gps_component.points[1] = gps_vel_marker.points[0];
                vel_gps_component.points[1].x += vel_scale * closest_gps.velocity(i) * unit_vectors[i](0);
                vel_gps_component.points[1].y += vel_scale * closest_gps.velocity(i) * unit_vectors[i](1);
                vel_gps_component.points[1].z += vel_scale * closest_gps.velocity(i) * unit_vectors[i](2);
                
                vel_gps_component.color.r = (i == 0) ? 0.5 : 0.0;
                vel_gps_component.color.g = (i == 1) ? 0.5 : 0.0;
                vel_gps_component.color.b = (i == 2) ? 0.5 : 0.0;
                vel_gps_component.color.a = 0.5;
                
                velocity_error_markers_.markers.push_back(vel_gps_component);
            }
            
            // Publish the velocity error visualization
            velocity_error_pub_.publish(velocity_error_markers_);
            
            // Log velocity error statistics periodically
            static double last_log_time = 0;
            if (current_time - last_log_time > 5.0) {
                ROS_INFO("Velocity Error (m/s): %.2f (E:%.2f, N:%.2f, U:%.2f)", 
                        velocity_error_norm, velocity_error.x(), velocity_error.y(), velocity_error.z());
                last_log_time = current_time;
            }
        }
    }

    // Reset visualization data
    void resetVisualization() {
        gps_path_msg_.poses.clear();
        optimized_path_msg_.poses.clear();
        position_error_markers_.markers.clear();
        velocity_error_markers_.markers.clear();
        
        // Clear error statistics
        latest_error_stats_ = ErrorStats();
    }
    
    // ==================== GPS-RELATED METHODS ====================

    // Convert GPS lat/lon/alt to ENU coordinates
    Eigen::Vector3d convertGpsToEnu(double latitude, double longitude, double altitude) {
        // Input validation
        if (std::isnan(latitude) || std::isnan(longitude) || std::isnan(altitude) ||
            std::abs(latitude) > 90.0 || std::abs(longitude) > 180.0) {
            ROS_WARN("Invalid GPS coordinates (lat=%.7f, lon=%.7f, alt=%.3f), using zeros", 
                     latitude, longitude, altitude);
            return Eigen::Vector3d::Zero();
        }
        
        // Convert from degrees to radians
        double lat_rad = latitude * M_PI / 180.0;
        double lon_rad = longitude * M_PI / 180.0;
        double ref_lat_rad = ref_latitude_ * M_PI / 180.0;
        double ref_lon_rad = ref_longitude_ * M_PI / 180.0;
        
        // Earth's radius in meters
        const double R = 6378137.0; // WGS84 equatorial radius
        
        // Calculate ENU coordinates with numerical stability check
        double e = (longitude - ref_longitude_) * M_PI / 180.0 * R * cos(ref_lat_rad);
        double n = (latitude - ref_latitude_) * M_PI / 180.0 * R;
        double u = altitude - ref_altitude_;
        
        // Validate output
        if (std::isnan(e) || std::isnan(n) || std::isnan(u) ||
            std::abs(e) > 1e6 || std::abs(n) > 1e6 || std::abs(u) > 1e5) {
            ROS_WARN("ENU conversion produced invalid results [%.2f, %.2f, %.2f], using zeros", e, n, u);
            return Eigen::Vector3d::Zero();
        }
        
        return Eigen::Vector3d(e, n, u);
    }

    // FIXED: Robust GPS to Unix time conversion
    double gpsToUnixTime(uint32_t gps_week, double gps_seconds) {
        // Debug input parameters
        ROS_DEBUG("Converting GPS time: week=%u, sec=%.3f", gps_week, gps_seconds);
        
        // Detect high-precision time format and scale appropriately
        if (gps_seconds > 1000000.0) {
            // Check if it's likely microseconds (common GPS format)
            if (gps_seconds < 604800000000.0) { // Less than a week in microseconds
                ROS_INFO_ONCE("Converting GPS time from microseconds format");
                gps_seconds /= 1000000.0;  // Convert from microseconds to seconds
            }
        }
        
        // GPS epoch started on January 6, 1980 00:00:00 UTC
        // Unix epoch started on January 1, 1970 00:00:00 UTC
        const double GPS_UNIX_OFFSET = 315964800.0; // Seconds between Unix and GPS epochs
        const double SECONDS_IN_WEEK = 604800.0;
        const double LEAP_SECONDS = 18.0;
        
        // Input validation (after possible scaling)
        if (gps_week > 4000 || gps_seconds < 0 || gps_seconds >= SECONDS_IN_WEEK) {
            ROS_WARN("Invalid GPS time (week=%u, sec=%.3f)", gps_week, gps_seconds);
            return 0;
        }
        
        // Calculate seconds since GPS epoch
        double gps_time = gps_week * SECONDS_IN_WEEK + gps_seconds;
        
        // Convert to Unix time by adding the offset and subtracting leap seconds
        double unix_time = gps_time + GPS_UNIX_OFFSET - LEAP_SECONDS;
        
        // Log successful conversion
        ROS_DEBUG("GPS time converted: week=%u, sec=%.3f -> unix=%.3f", 
                 gps_week, gps_seconds, unix_time);
                 
        return unix_time;
    }

    // GPS callback function
    void gpsCallback(const novatel_msgs::INSPVAX::ConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            // Convert GPS time to Unix time with fixed handling
            double unix_timestamp = gpsToUnixTime(msg->header.gps_week, msg->header.gps_week_seconds/1000);
            
            // Debug output to verify conversion
            static bool first_conversion = true;
            if (first_conversion) {
                ROS_INFO("First GPS time conversion: week=%u, sec=%.3f → unix=%.3f",
                        msg->header.gps_week, msg->header.gps_week_seconds, unix_timestamp);
                first_conversion = false;
            }
            
            // CRITICAL: When playing back bags, NEVER use ros::Time::now()
            // Instead, use the converted timestamps directly from the bag
            
            // Set reference point if not set yet
            if (!has_gps_reference_) {
                ref_latitude_ = msg->latitude;
                ref_longitude_ = msg->longitude;
                ref_altitude_ = msg->altitude;
                has_gps_reference_ = true;
                ROS_INFO("Set GPS reference: lat=%.7f, lon=%.7f, alt=%.3f", 
                        ref_latitude_, ref_longitude_, ref_altitude_);
            }
            
            // Convert GPS to ENU with safety checks
            Eigen::Vector3d enu_position = convertGpsToEnu(msg->latitude, msg->longitude, msg->altitude);
            
            // Get velocity (convert from NED to ENU)
            Eigen::Vector3d enu_velocity(msg->east_velocity, msg->north_velocity, -msg->up_velocity);
            
            // Get orientation with safety checks
            double roll_rad = msg->roll * M_PI / 180.0;
            double pitch_rad = msg->pitch * M_PI / 180.0;
            double azimuth_rad = msg->azimuth * M_PI / 180.0;
            
            // Convert NED azimuth to ENU yaw
            double yaw_enu = M_PI/2.0 - azimuth_rad;
            
            // Create quaternion from euler angles
            Eigen::Quaterniond orientation = 
                Eigen::AngleAxisd(yaw_enu, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX());
            
            // Ensure quaternion is normalized
            orientation.normalize();
            
            // Create GPS measurement with timestamps from the bag
            GpsMeasurement measurement;
            measurement.position = enu_position;
            measurement.velocity = enu_velocity;
            measurement.orientation = orientation;
            measurement.timestamp = unix_timestamp;  // Use the bag's timestamp

            // Add to GPS path for visualization
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time(unix_timestamp);
            pose_stamped.header.frame_id = world_frame_id_;
            pose_stamped.pose.position.x = enu_position.x();
            pose_stamped.pose.position.y = enu_position.y();
            pose_stamped.pose.position.z = enu_position.z();
            pose_stamped.pose.orientation.w = orientation.w();
            pose_stamped.pose.orientation.x = orientation.x();
            pose_stamped.pose.orientation.y = orientation.y();
            pose_stamped.pose.orientation.z = orientation.z();

            gps_path_msg_.header.stamp = ros::Time(unix_timestamp);
            gps_path_msg_.poses.push_back(pose_stamped);

            // Limit path size for performance
            if (gps_path_msg_.poses.size() > 1000) {
                gps_path_msg_.poses.erase(gps_path_msg_.poses.begin());
            }

            // Publish the GPS path
            gps_path_pub_.publish(gps_path_msg_);
            
            // Limit GPS buffer size
            if (gps_measurements_.size() > 100) {
                gps_measurements_.erase(gps_measurements_.begin(), gps_measurements_.begin() + 50);
            }
            
            // Add to GPS measurements
            gps_measurements_.push_back(measurement);
            
            // Initialize if not initialized and using GPS
            if (!is_initialized_ && use_gps_instead_of_uwb_) {
                // Wait for some IMU data before initializing
                if (imu_buffer_.size() >= 5) {
                    ROS_INFO("Initializing system with GPS measurement at timestamp: %.3f", unix_timestamp);
                    initializeFromGps(measurement);
                    is_initialized_ = true;
                } else {
                    ROS_INFO_THROTTLE(1.0, "Waiting for IMU data before GPS initialization...");
                }
                return;
            }
            
            // Create keyframe if initialized and using GPS
            if (is_initialized_ && has_imu_data_ && use_gps_instead_of_uwb_) {
                // Check if we have IMU data covering this GPS timestamp
                bool has_surrounding_imu_data = false;
                double closest_imu_time = 0;
                double closest_time_diff = std::numeric_limits<double>::max();
                
                for (const auto& imu : imu_buffer_) {
                    double imu_time = imu.header.stamp.toSec();
                    double time_diff = std::abs(imu_time - unix_timestamp);
                    
                    if (time_diff < closest_time_diff) {
                        closest_time_diff = time_diff;
                        closest_imu_time = imu_time;
                    }
                    
                    // Consider IMU data within 50ms of the GPS timestamp
                    if (time_diff < 0.05) {
                        has_surrounding_imu_data = true;
                        break;
                    }
                }
                
                if (!has_surrounding_imu_data) {
                    if (!imu_buffer_.empty()) {
                        ROS_WARN("GPS-IMU time mismatch: GPS=%.3f, closest IMU=%.3f (diff=%.3f sec), buffer range [%.3f to %.3f]", 
                               unix_timestamp, closest_imu_time, closest_time_diff,
                               imu_buffer_.front().header.stamp.toSec(), 
                               imu_buffer_.back().header.stamp.toSec());
                        
                        // If the time difference is small enough, still create a keyframe
                        if (closest_time_diff < 0.2) {  // Accept up to 200ms difference
                            has_surrounding_imu_data = true;
                            ROS_INFO("Using nearby IMU data (%.3f sec offset) for keyframe", closest_time_diff);
                            
                            // Fix: Keep the GPS timestamp but know we have nearby IMU data
                        }
                    }
                }
                
                if (has_surrounding_imu_data) {
                    // Create keyframe from GPS
                    createKeyframeFromGps(measurement);
                    ROS_INFO("Created GPS keyframe at timestamp %.3f", measurement.timestamp);
                } else {
                    ROS_WARN("Skipping GPS keyframe at %.3f - no surrounding IMU data", unix_timestamp);
                }
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in gpsCallback: %s", e.what());
        }
    }

    // Initialize from GPS measurement
    void initializeFromGps(const GpsMeasurement& gps) {
        try {
            // Initialize state using GPS position
            current_state_.position = gps.position;
            
            // Use GPS orientation only if configured
            if (use_gps_orientation_as_initial_) {
                current_state_.orientation = gps.orientation;
            } else {
                // Use a default orientation (or keep previous if available)
                if (current_state_.orientation.w() == 0) {
                    current_state_.orientation = Eigen::Quaterniond::Identity();
                }
                // Try to find IMU orientation if available
                sensor_msgs::Imu closest_imu = findClosestImuMeasurement(gps.timestamp);
                if (closest_imu.header.stamp.toSec() > 0 && 
                    closest_imu.orientation_covariance[0] != -1) {
                    current_state_.orientation = Eigen::Quaterniond(
                        closest_imu.orientation.w,
                        closest_imu.orientation.x,
                        closest_imu.orientation.y,
                        closest_imu.orientation.z
                    ).normalized();
                    ROS_INFO("Using IMU orientation for initialization");
                }
            }
            
            // Use GPS velocity only if configured
            if (use_gps_velocity_) {
                current_state_.velocity = gps.velocity;
            } else {
                // Initialize with minimum horizontal velocity in current orientation direction
                double yaw = atan2(2.0 * (current_state_.orientation.w() * current_state_.orientation.z() + 
                                current_state_.orientation.x() * current_state_.orientation.y()),
                               1.0 - 2.0 * (current_state_.orientation.y() * current_state_.orientation.y() + 
                                          current_state_.orientation.z() * current_state_.orientation.z()));
                
                current_state_.velocity = Eigen::Vector3d(
                    min_horizontal_velocity_ * cos(yaw),
                    min_horizontal_velocity_ * sin(yaw),
                    0.0
                );
            }
            
            // Initialize with proper non-zero biases
            current_state_.acc_bias = initial_acc_bias_;
            current_state_.gyro_bias = initial_gyro_bias_;
            
            current_state_.timestamp = gps.timestamp;
            
            state_window_.clear(); // Ensure clean state window
            state_window_.push_back(current_state_);
            
            // Reset marginalization
            if (last_marginalization_info_) {
                delete last_marginalization_info_;
                last_marginalization_info_ = nullptr;
            }
            
            // Reset optimization count
            optimization_count_ = 0;
            
            // Reset visualization
            resetVisualization();
            
            ROS_INFO("State initialized from GPS at position [%.2f, %.2f, %.2f]", 
                    current_state_.position.x(), 
                    current_state_.position.y(), 
                    current_state_.position.z());
            ROS_INFO("Initial velocity [%.2f, %.2f, %.2f] m/s (%s)",
                    current_state_.velocity.x(), 
                    current_state_.velocity.y(), 
                    current_state_.velocity.z(),
                    use_gps_velocity_ ? "from GPS" : "estimated");
            ROS_INFO("Initial orientation (roll, pitch, yaw) [%.2f, %.2f, %.2f] deg (%s)",
                    quaternionToEulerDegrees(current_state_.orientation).x(), 
                    quaternionToEulerDegrees(current_state_.orientation).y(), 
                    quaternionToEulerDegrees(current_state_.orientation).z(),
                    use_gps_orientation_as_initial_ ? "from GPS" : "from IMU/default");
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeFromGps: %s", e.what());
        }
    }

    // Create keyframe from GPS measurement
    void createKeyframeFromGps(const GpsMeasurement& gps) {
        try {
            // Skip if we already have a keyframe at this time
            for (const auto& state : state_window_) {
                if (std::abs(state.timestamp - gps.timestamp) < 0.005) {
                    return; // Already have a keyframe for this GPS measurement
                }
            }
            
            // Skip if the state window is empty
            if (state_window_.empty()) {
                // Create initial keyframe from current state
                State new_state = current_state_;
                new_state.position = gps.position;
                
                if (use_gps_orientation_as_initial_) {
                    new_state.orientation = gps.orientation;
                } else {
                    // Keep current orientation or try to use IMU
                    sensor_msgs::Imu closest_imu = findClosestImuMeasurement(gps.timestamp);
                    if (closest_imu.header.stamp.toSec() > 0 && 
                        closest_imu.orientation_covariance[0] != -1) {
                        new_state.orientation = Eigen::Quaterniond(
                            closest_imu.orientation.w,
                            closest_imu.orientation.x,
                            closest_imu.orientation.y,
                            closest_imu.orientation.z
                        ).normalized();
                    }
                }
                
                if (use_gps_velocity_) {
                    new_state.velocity = gps.velocity;
                }
                
                new_state.timestamp = gps.timestamp;
                
                state_window_.push_back(new_state);
                ROS_DEBUG("Added first GPS-based keyframe at t=%.3f", gps.timestamp);
                return;
            }
            
            // Calculate the time difference from the previous keyframe
            double dt = gps.timestamp - state_window_.back().timestamp;
            
            // Skip if the time difference is too small
            if (dt < 0.01) {
                return;
            }
            
            // Compute the propagated state at the GPS timestamp
            State propagated_state = propagateState(state_window_.back(), gps.timestamp);
            
            // Set the GPS position
            propagated_state.position = gps.position;
            
            // Set orientation only if configured to use GPS orientation
            if (use_gps_orientation_as_initial_) {
                propagated_state.orientation = gps.orientation;
            }
            
            // Set velocity only if configured to use GPS velocity
            if (use_gps_velocity_) {
                propagated_state.velocity = gps.velocity;
            }
            
            propagated_state.timestamp = gps.timestamp;
            
            // CRITICAL: Ensure biases are reasonable in the new keyframe
            clampBiases(propagated_state.acc_bias, propagated_state.gyro_bias);
            
            // Add to state window, with marginalization if needed
            if (state_window_.size() >= optimization_window_size_) {
                if (enable_marginalization_) {
                    // Prepare marginalization before removing the oldest state
                    prepareMarginalization();
                }
                state_window_.pop_front();
            }
            
            state_window_.push_back(propagated_state);
            
            // Update current state
            current_state_ = propagated_state;
            
            // Update preintegration between the last two keyframes
            if (state_window_.size() >= 2) {
                size_t n = state_window_.size();
                double start_time = state_window_[n-2].timestamp;
                double end_time = state_window_[n-1].timestamp;
                
                // Store preintegration data for optimization
                performPreintegrationBetweenKeyframes(start_time, end_time, 
                                                     state_window_[n-2].acc_bias, 
                                                     state_window_[n-2].gyro_bias);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in createKeyframeFromGps: %s", e.what());
        }
    }
    
    // ==================== EXISTING METHODS WITH GPS SUPPORT ====================
    
    // Helper functions
    
    // IMPROVED: RK4 integration for quaternion
    void rk4IntegrateOrientation(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2, 
                                const double dt, Eigen::Quaterniond& q) {
        // Implement 4th-order Runge-Kutta integration for quaternion
        Eigen::Vector3d k1 = omega1;
        Eigen::Vector3d k2 = omega1 + 0.5 * dt * omegaDot(omega1, omega2);
        Eigen::Vector3d k3 = omega1 + 0.5 * dt * omegaDot(omega1, k2);
        Eigen::Vector3d k4 = omega2;
        
        // Combined angular velocity update
        Eigen::Vector3d omega_integrated = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 * dt;
        
        // Apply quaternion update
        if (omega_integrated.norm() > 1e-8) {
            q = q * deltaQ(omega_integrated);
        }
        q.normalize();  // Ensure quaternion stays normalized
    }
    
    // Helper for RK4 integration - compute omega_dot (angular acceleration)
    Eigen::Vector3d omegaDot(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2) {
        // Simple linear approximation of angular acceleration
        return (omega2 - omega1);
    }
    
    // Compute quaternion for small angle rotation
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
    
    // Non-template version for double type
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
        // Normalize the quaternion to ensure proper conversion
        Eigen::Quaterniond quat = q.normalized();
        
        // Extract Euler angles
        double roll = atan2(2.0 * (quat.w() * quat.x() + quat.y() * quat.z()),
                         1.0 - 2.0 * (quat.x() * quat.x() + quat.y() * quat.y()));
        
        // Use asin for pitch, but clamp input to avoid numerical issues
        double sinp = 2.0 * (quat.w() * quat.y() - quat.z() * quat.x());
        double pitch = (std::abs(sinp) >= 1) ? 
                      copysign(M_PI / 2, sinp) : // use 90° if out of range
                      asin(sinp);
        
        double yaw = atan2(2.0 * (quat.w() * quat.z() + quat.x() * quat.y()),
                       1.0 - 2.0 * (quat.y() * quat.y() + quat.z() * quat.z()));
        
        // Convert to degrees
        Eigen::Vector3d euler_deg;
        euler_deg << roll * 180.0 / M_PI, 
                     pitch * 180.0 / M_PI, 
                     yaw * 180.0 / M_PI;
        
        return euler_deg;
    }

    // Helper to find closest IMU measurement to a given timestamp
    sensor_msgs::Imu findClosestImuMeasurement(double timestamp) {
        sensor_msgs::Imu closest_imu;
        double min_time_diff = std::numeric_limits<double>::max();
        
        for (const auto& imu : imu_buffer_) {
            double imu_time = imu.header.stamp.toSec();
            double time_diff = std::abs(imu_time - timestamp);
            
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                closest_imu = imu;
            }
        }
        
        return closest_imu;
    }

    // Helper to add IMU orientation factor
    void addImuOrientationFactor(ceres::Problem& problem, 
                               double* pose_param, 
                               const sensor_msgs::Imu& imu_msg) {
        // Create quaternion from IMU message
        Eigen::Quaterniond q_imu(
            imu_msg.orientation.w,
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z
        );
        
        // Ensure quaternion is normalized
        q_imu.normalize();
        
        // Only constrain yaw rotation
        ceres::CostFunction* yaw_factor = 
            YawOnlyOrientationFactor::Create(q_imu, imu_orientation_weight_);
        problem.AddResidualBlock(yaw_factor, nullptr, pose_param);
    }

    // Check if bias values are within reasonable limits
    bool areBiasesReasonable(const Eigen::Vector3d& acc_bias, const Eigen::Vector3d& gyro_bias) {
        // Check accelerometer bias
        if (acc_bias.norm() > acc_bias_max_) {
            return false;
        }
        
        // Check gyroscope bias
        if (gyro_bias.norm() > gyro_bias_max_) {
            return false;
        }
        
        return true;
    }
    
    // CRITICAL: Ensure biases stay within reasonable limits
    void clampBiases(Eigen::Vector3d& acc_bias, Eigen::Vector3d& gyro_bias) {
        double acc_bias_norm = acc_bias.norm();
        double gyro_bias_norm = gyro_bias.norm();
        
        if (acc_bias_norm > acc_bias_max_) {
            acc_bias *= (acc_bias_max_ / acc_bias_norm);
        }
        
        if (gyro_bias_norm > gyro_bias_max_) {
            gyro_bias *= (gyro_bias_max_ / gyro_bias_norm);
        }
    }
    
    // IMPROVED: Better velocity clamping that preserves direction for high-speed scenarios
    void clampVelocity(Eigen::Vector3d& velocity, double max_velocity = 25.0) {
        double velocity_norm = velocity.norm();
        
        if (velocity_norm > max_velocity) {
            // Scale velocity proportionally to keep direction but limit magnitude
            velocity *= (max_velocity / velocity_norm);
        }
        
        // Only enforce a minimum horizontal velocity if it's very small and we're not moving vertically
        double h_vel_norm = std::sqrt(velocity.x()*velocity.x() + velocity.y()*velocity.y());
        double v_vel_abs = std::abs(velocity.z());
        
        // If horizontal velocity is small but vertical velocity is significant, don't enforce minimum
        if (h_vel_norm < 0.05 && v_vel_abs < 0.5) {
            // Set a minimum velocity in the current horizontal direction or along x-axis if zero
            if (h_vel_norm > 1e-6) {
                // Scale up existing direction to minimum
                double scale = min_horizontal_velocity_ * 0.2 / h_vel_norm;
                velocity.x() *= scale;
                velocity.y() *= scale;
            } else {
                // Add a small default velocity along x-axis
                velocity.x() = min_horizontal_velocity_ * 0.2;
                velocity.y() = 0.0;
            }
            
            // Re-verify total velocity is within bounds
            velocity_norm = velocity.norm();
            if (velocity_norm > max_velocity) {
                velocity *= (max_velocity / velocity_norm);
            }
        }
    }

    // Estimate velocity magnitude from IMU data
    double estimateMaxVelocityFromImu() {
        // Default value if we don't have enough data
        double estimated_max_velocity = max_velocity_;
        
        // Look at recent IMU data to estimate reasonable velocity bound
        if (imu_buffer_.size() >= 10) {
            double max_acc = 0.0;
            
            // Find maximum acceleration magnitude in recent data
            for (size_t i = imu_buffer_.size() - 10; i < imu_buffer_.size(); i++) {
                const auto& imu = imu_buffer_[i];
                double acc_mag = std::sqrt(
                    imu.linear_acceleration.x * imu.linear_acceleration.x +
                    imu.linear_acceleration.y * imu.linear_acceleration.y +
                    imu.linear_acceleration.z * imu.linear_acceleration.z
                );
                max_acc = std::max(max_acc, acc_mag - gravity_magnitude_);
            }
            
            // Use a reasonable time period for acceleration (1-5 seconds)
            // v = a * t, assuming constant acceleration
            double assumed_acc_time = 3.0;
            double potential_max_vel = max_acc * assumed_acc_time;
            
            // Ensure a reasonable range: between 5 m/s and 35 m/s (18-126 km/h)
            estimated_max_velocity = std::max(5.0, std::min(35.0, potential_max_vel));
        }
        
        return estimated_max_velocity;
    }

    void initializeState() {
        try {
            current_state_.position = Eigen::Vector3d::Zero();
            current_state_.orientation = Eigen::Quaterniond::Identity();
            current_state_.velocity = Eigen::Vector3d::Zero();
            
            // CRITICAL: Initialize with sane non-zero biases
            current_state_.acc_bias = initial_acc_bias_;
            current_state_.gyro_bias = initial_gyro_bias_;
            
            current_state_.timestamp = 0;
            
            state_window_.clear();
            uwb_measurements_.clear();
            gps_measurements_.clear();  // Clear GPS measurements
            imu_buffer_.clear();
            preintegration_map_.clear();
            
            // Reset GPS reference point
            has_gps_reference_ = false;
            ref_latitude_ = 0.0;
            ref_longitude_ = 0.0;
            ref_altitude_ = 0.0;
            
            // Initialize gravity vector in world frame (ENU, Z points up)
            // In ENU frame, gravity points downward along negative Z axis
            gravity_world_ = Eigen::Vector3d(0, 0, -gravity_magnitude_);
            
            // Reset timestamp tracking
            last_imu_timestamp_ = 0;
            last_processed_timestamp_ = 0;
            just_optimized_ = false;
            
            // Reset optimization count
            optimization_count_ = 0;
            
            // Reset marginalization
            if (last_marginalization_info_) {
                delete last_marginalization_info_;
                last_marginalization_info_ = nullptr;
            }
            
            // Reset visualization
            resetVisualization();
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeState: %s", e.what());
        }
    }

    // IMPROVED: Better IMU measurement finding with increased time tolerance
    std::vector<sensor_msgs::Imu> findIMUMeasurementsBetweenTimes(double start_time, double end_time) {
        std::vector<sensor_msgs::Imu> measurements;
        
        // For 400Hz IMU, we expect about 400 messages per second
        measurements.reserve(static_cast<size_t>((end_time - start_time) * 400));
        
        // FIXED: Apply a larger time tolerance to account for potential timestamp mismatches
        const double time_tolerance = 0.05; // 50ms tolerance (up from 20ms)
        
        // Log the search parameters for debugging
        ROS_DEBUG("Searching for IMU data between %.6f and %.6f (with %.3fs tolerance)",
                start_time, end_time, time_tolerance);
        
        // Search the buffer
        size_t count = 0;
        double earliest_found = std::numeric_limits<double>::max();
        double latest_found = 0;
        
        for (const auto& imu : imu_buffer_) {
            double timestamp = imu.header.stamp.toSec();
            
            // Update time range statistics for debugging
            if (timestamp < earliest_found) earliest_found = timestamp;
            if (timestamp > latest_found) latest_found = timestamp;
            
            if (timestamp >= (start_time - time_tolerance) && timestamp <= (end_time + time_tolerance)) {
                measurements.push_back(imu);
                count++;
            }
        }
        
        // Debug output for diagnostics
        if (measurements.empty()) {
            // Print the buffer time range to help diagnose the issue
            if (!imu_buffer_.empty()) {
                double buffer_start = imu_buffer_.front().header.stamp.toSec();
                double buffer_end = imu_buffer_.back().header.stamp.toSec();
                
                ROS_WARN("No IMU data found between %.6f and %.6f. Buffer timespan: [%.6f to %.6f] (%zu messages)", 
                        start_time, end_time, buffer_start, buffer_end, imu_buffer_.size());
            } else {
                ROS_WARN("No IMU data found between %.6f and %.6f. IMU buffer is empty!", 
                        start_time, end_time);
            }
        } else {
            ROS_DEBUG("Found %zu IMU messages between %.6f and %.6f", 
                    measurements.size(), start_time, end_time);
        }
        
        return measurements;
    }

    // Modify the imuCallback function to handle 400Hz IMU data
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        try {
            static int imu_count = 0;
            static double last_report_time = 0;
            
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            double timestamp = msg->header.stamp.toSec();
            has_imu_data_ = true;
            imu_count++;
            
            // For the first IMU message, print the timestamp for debugging
            static bool first_imu = true;
            if (first_imu) {
                ROS_INFO("First IMU timestamp: %.3f", timestamp);
                first_imu = false;
                last_report_time = timestamp;
            }
            
            // Store IMU measurements with original bag timestamps
            imu_buffer_.push_back(*msg);
            
            // Skip messages with duplicate or old timestamps 
            if (timestamp <= last_processed_timestamp_) {
                return;
            }
            
            // Update tracking timestamps
            last_imu_timestamp_ = timestamp;
            last_processed_timestamp_ = timestamp;
            
            // Process IMU data for real-time state propagation
            if (is_initialized_) {
                propagateStateWithImu(*msg);
                publishImuPose();
            }
            
            // Report IMU statistics periodically based on message timestamps, not system time
            if (timestamp - last_report_time > 5.0) {  // Every 5 seconds in bag time
                double rate = imu_count / (timestamp - last_report_time);
                
                if (!imu_buffer_.empty()) {
                    double buffer_start = imu_buffer_.front().header.stamp.toSec();
                    double buffer_end = imu_buffer_.back().header.stamp.toSec();
                    
                    ROS_INFO("IMU stats: %.1f Hz, buffer: %zu msgs spanning %.3f sec [%.3f to %.3f]", 
                            rate, imu_buffer_.size(), buffer_end - buffer_start, buffer_start, buffer_end);
                }
                
                imu_count = 0;
                last_report_time = timestamp;
            }
            
            // Modified IMU buffer cleanup based on time difference from latest timestamp
            if (imu_buffer_.size() > 6000) {  // Larger buffer for 400Hz IMU
                double latest_time = imu_buffer_.back().header.stamp.toSec();
                double oldest_allowed_time = latest_time - 15.0;  // Keep 15 seconds of data
                
                int count_before = imu_buffer_.size();
                while (imu_buffer_.size() > 1000 && imu_buffer_.front().header.stamp.toSec() < oldest_allowed_time) {
                    imu_buffer_.pop_front();
                }
                
                int count_after = imu_buffer_.size();
                if (count_before - count_after > 100) {
                    ROS_INFO("Cleaned %d old IMU messages, remaining: %d", count_before - count_after, count_after);
                }
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in imuCallback: %s", e.what());
        }
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            // Skip if we're in GPS mode
            if (use_gps_instead_of_uwb_) {
                return;
            }
            
            // Store UWB position measurement
            UwbMeasurement measurement;
            measurement.position = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);
            measurement.timestamp = msg->header.stamp.toSec();
            
            // Ensure timestamp is valid
            if (measurement.timestamp <= 0) {
                ROS_WARN("Invalid UWB timestamp: %f, using current time", measurement.timestamp);
                measurement.timestamp = ros::Time::now().toSec();
            }
            
            // Limit size of UWB measurements buffer
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
            
            // Create a new keyframe at each UWB measurement
            if (is_initialized_ && has_imu_data_) {
                createKeyframe(measurement);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in uwbCallback: %s", e.what());
        }
    }
    
    // Create a keyframe (state) based on UWB measurement
    void createKeyframe(const UwbMeasurement& uwb) {
        try {
            // Skip if we already have a keyframe at this time
            for (const auto& state : state_window_) {
                if (std::abs(state.timestamp - uwb.timestamp) < 0.005) {
                    return; // Already have a keyframe for this UWB measurement
                }
            }
            
            // Skip if the state window is empty
            if (state_window_.empty()) {
                // Create initial keyframe from current state
                State new_state = current_state_;
                new_state.position = uwb.position;
                new_state.timestamp = uwb.timestamp;
                
                // Initialize with a reasonable velocity but don't force direction
                if (new_state.velocity.norm() < min_horizontal_velocity_ * 0.5) {
                    new_state.velocity = Eigen::Vector3d(min_horizontal_velocity_, 0, 0);
                }
                
                // Find closest IMU measurement for orientation
                sensor_msgs::Imu closest_imu = findClosestImuMeasurement(uwb.timestamp);
                
                // If we have a valid IMU message, use its orientation
                if (closest_imu.header.stamp.toSec() > 0) {
                    new_state.orientation = Eigen::Quaterniond(
                        closest_imu.orientation.w,
                        closest_imu.orientation.x,
                        closest_imu.orientation.y,
                        closest_imu.orientation.z
                    ).normalized();
                }
                
                state_window_.push_back(new_state);
                ROS_DEBUG("Added first UWB-based keyframe at t=%.3f", uwb.timestamp);
                return;
            }
            
            // Calculate the time difference from the previous keyframe
            double dt = uwb.timestamp - state_window_.back().timestamp;
            
            // Skip if the time difference is too small
            if (dt < 0.01) {
                return;
            }
            
            // Compute the propagated state at the UWB timestamp
            State propagated_state = propagateState(state_window_.back(), uwb.timestamp);
            
            // Set the UWB position while keeping the propagated orientation and velocity
            propagated_state.position = uwb.position;
            propagated_state.timestamp = uwb.timestamp;
            
            // Find closest IMU measurement for orientation updating
            sensor_msgs::Imu closest_imu = findClosestImuMeasurement(uwb.timestamp);
            
            // If the IMU measurement is close enough, use its orientation
            if (closest_imu.header.stamp.toSec() > 0) {
                double imu_time_diff = std::abs(closest_imu.header.stamp.toSec() - uwb.timestamp);
                if (imu_time_diff < 0.05) { // 50ms threshold
                    propagated_state.orientation = Eigen::Quaterniond(
                        closest_imu.orientation.w,
                        closest_imu.orientation.x,
                        closest_imu.orientation.y,
                        closest_imu.orientation.z
                    ).normalized();
                }
            }
            
            // CRITICAL: Ensure biases are reasonable in the new keyframe
            clampBiases(propagated_state.acc_bias, propagated_state.gyro_bias);
            
            // CRITICAL: Ensure velocity is reasonable but preserve direction
            double adaptive_max_velocity = max_velocity_;
            // For high-speed scenarios, estimate max velocity from IMU data
            if (imu_buffer_.size() > 10) {
                adaptive_max_velocity = estimateMaxVelocityFromImu();
            }
            clampVelocity(propagated_state.velocity, adaptive_max_velocity);
            
            // Add to state window, with marginalization if needed
            if (state_window_.size() >= optimization_window_size_) {
                if (enable_marginalization_) {
                    // Prepare marginalization before removing the oldest state
                    prepareMarginalization();
                }
                state_window_.pop_front();
            }
            
            state_window_.push_back(propagated_state);
            
            // Update current state
            current_state_ = propagated_state;
            
            // Update preintegration between the last two keyframes
            if (state_window_.size() >= 2) {
                size_t n = state_window_.size();
                double start_time = state_window_[n-2].timestamp;
                double end_time = state_window_[n-1].timestamp;
                
                // Store preintegration data for optimization
                performPreintegrationBetweenKeyframes(start_time, end_time, 
                                                     state_window_[n-2].acc_bias, 
                                                     state_window_[n-2].gyro_bias);
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in createKeyframe: %s", e.what());
        }
    }
    
    // Prepare marginalization by adding factors connected to the oldest state
    void prepareMarginalization() {
        try {
            if (!enable_marginalization_ || state_window_.size() < 2) {
                return;
            }
            
            // Create a new marginalization info object
            MarginalizationInfo* marginalization_info = new MarginalizationInfo();
            
            // Local tracking of allocated memory for cleanup in case of exception
            std::vector<double*> local_allocations;
            
            try {
                // The oldest state is being marginalized
                const State& oldest_state = state_window_.front();
                const State& next_state = state_window_[1];
                
                // Create parameter blocks for the two states involved
                double* pose_param1 = new double[7];
                double* vel_param1 = new double[3];
                double* bias_param1 = new double[6];
                double* pose_param2 = new double[7];
                double* vel_param2 = new double[3];
                double* bias_param2 = new double[6];
                
                // Add to local tracking for cleanup in case of exception
                local_allocations.push_back(pose_param1);
                local_allocations.push_back(vel_param1);
                local_allocations.push_back(bias_param1);
                local_allocations.push_back(pose_param2);
                local_allocations.push_back(vel_param2);
                local_allocations.push_back(bias_param2);
                
                // Copy the state data to the parameters
                // Oldest state
                pose_param1[0] = oldest_state.position.x();
                pose_param1[1] = oldest_state.position.y();
                pose_param1[2] = oldest_state.position.z();
                pose_param1[3] = oldest_state.orientation.w();
                pose_param1[4] = oldest_state.orientation.x();
                pose_param1[5] = oldest_state.orientation.y();
                pose_param1[6] = oldest_state.orientation.z();
                
                vel_param1[0] = oldest_state.velocity.x();
                vel_param1[1] = oldest_state.velocity.y();
                vel_param1[2] = oldest_state.velocity.z();
                
                bias_param1[0] = oldest_state.acc_bias.x();
                bias_param1[1] = oldest_state.acc_bias.y();
                bias_param1[2] = oldest_state.acc_bias.z();
                bias_param1[3] = oldest_state.gyro_bias.x();
                bias_param1[4] = oldest_state.gyro_bias.y();
                bias_param1[5] = oldest_state.gyro_bias.z();
                
                // Next state
                pose_param2[0] = next_state.position.x();
                pose_param2[1] = next_state.position.y();
                pose_param2[2] = next_state.position.z();
                pose_param2[3] = next_state.orientation.w();
                pose_param2[4] = next_state.orientation.x();
                pose_param2[5] = next_state.orientation.y();
                pose_param2[6] = next_state.orientation.z();
                
                vel_param2[0] = next_state.velocity.x();
                vel_param2[1] = next_state.velocity.y();
                vel_param2[2] = next_state.velocity.z();
                
                bias_param2[0] = next_state.acc_bias.x();
                bias_param2[1] = next_state.acc_bias.y();
                bias_param2[2] = next_state.acc_bias.z();
                bias_param2[3] = next_state.gyro_bias.x();
                bias_param2[4] = next_state.gyro_bias.y();
                bias_param2[5] = next_state.gyro_bias.z();
                
                // Add position factor for the oldest state based on fusion mode
                if (use_gps_instead_of_uwb_) {
                    // Add GPS position factor
                    for (const auto& gps : gps_measurements_) {
                        if (std::abs(gps.timestamp - oldest_state.timestamp) < 0.01) {
                            // Add GPS position factor
                            ceres::CostFunction* gps_factor = GpsPositionFactor::Create(
                                gps.position, gps_position_noise_);
                            
                            std::vector<double*> parameter_blocks = {pose_param1};
                            std::vector<int> drop_set = {0}; // Drop the pose parameter
                            
                            auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                                gps_factor, nullptr, parameter_blocks, drop_set);
                            marginalization_info->addResidualBlockInfo(residual_info);
                            
                            // Also add GPS velocity factor if enabled
                            if (use_gps_velocity_) {
                                ceres::CostFunction* gps_vel_factor = GpsVelocityFactor::Create(
                                    gps.velocity, gps_velocity_noise_);
                                
                                std::vector<double*> vel_parameter_blocks = {vel_param1};
                                std::vector<int> vel_drop_set = {0}; // Drop the velocity parameter
                                
                                auto* vel_residual_info = new MarginalizationInfo::ResidualBlockInfo(
                                    gps_vel_factor, nullptr, vel_parameter_blocks, vel_drop_set);
                                marginalization_info->addResidualBlockInfo(vel_residual_info);
                            }
                            
                            // Add GPS orientation factor if enabled
                            if (use_gps_orientation_as_constraint_) {
                                ceres::CostFunction* orientation_factor = GpsOrientationFactor::Create(
                                    gps.orientation, gps_orientation_noise_);
                                
                                std::vector<double*> ori_parameter_blocks = {pose_param1};
                                std::vector<int> ori_drop_set = {0}; // Drop the pose parameter
                                
                                auto* ori_residual_info = new MarginalizationInfo::ResidualBlockInfo(
                                    orientation_factor, nullptr, ori_parameter_blocks, ori_drop_set);
                                marginalization_info->addResidualBlockInfo(ori_residual_info);
                            }
                            
                            break;
                        }
                    }
                } else {
                    // Add UWB position factor
                    for (const auto& uwb : uwb_measurements_) {
                        if (std::abs(uwb.timestamp - oldest_state.timestamp) < 0.01) {
                            // Create UWB factor
                            ceres::CostFunction* uwb_factor = UwbPositionFactor::Create(
                                uwb.position, uwb_position_noise_);
                            
                            std::vector<double*> parameter_blocks = {pose_param1};
                            std::vector<int> drop_set = {0}; // Drop the pose parameter
                            
                            auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                                uwb_factor, nullptr, parameter_blocks, drop_set);
                            marginalization_info->addResidualBlockInfo(residual_info);
                            break;
                        }
                    }
                }
                
                // Add IMU factor between oldest state and second oldest state
                double start_time = oldest_state.timestamp;
                double end_time = next_state.timestamp;
                std::pair<double, double> key(start_time, end_time);
                
                if (preintegration_map_.find(key) != preintegration_map_.end()) {
                    const auto& preint = preintegration_map_[key];
                    
                    // Create IMU factor
                    ceres::CostFunction* imu_factor = ImuFactor::Create(preint, gravity_world_);
                    
                    std::vector<double*> parameter_blocks = {
                        pose_param1, vel_param1, bias_param1,
                        pose_param2, vel_param2, bias_param2
                    };
                    
                    // Drop only parameters from oldest state
                    std::vector<int> drop_set = {0, 1, 2}; // Pose, velocity, bias of oldest state
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        imu_factor, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                    
                    // Add orientation smoothness factor between states if enabled
                    if (enable_orientation_smoothness_factor_) {
                        ceres::CostFunction* orientation_factor = 
                            OrientationSmoothnessFactor::Create(orientation_smoothness_weight_);
                        
                        std::vector<double*> orientation_params = {pose_param1, pose_param2};
                        std::vector<int> orientation_drop_set = {0}; // Drop only the oldest pose
                        
                        auto* orientation_residual = new MarginalizationInfo::ResidualBlockInfo(
                            orientation_factor, nullptr, orientation_params, orientation_drop_set);
                        marginalization_info->addResidualBlockInfo(orientation_residual);
                    }
                }
                
                // Add roll/pitch prior for oldest state if enabled
                if (enable_roll_pitch_constraint_) {
                    ceres::CostFunction* roll_pitch_factor = RollPitchPriorFactor::Create(roll_pitch_weight_);
                    std::vector<double*> parameter_blocks = {pose_param1};
                    std::vector<int> drop_set = {0}; // Drop the pose parameter
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        roll_pitch_factor, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                }
                
                // Add gravity alignment factor if IMU data is available and enabled
                sensor_msgs::Imu closest_imu = findClosestImuMeasurement(oldest_state.timestamp);
                if (enable_gravity_alignment_factor_ && closest_imu.header.stamp.toSec() > 0) {
                    Eigen::Vector3d acc(closest_imu.linear_acceleration.x,
                                       closest_imu.linear_acceleration.y,
                                       closest_imu.linear_acceleration.z);
                    
                    // Apply bias correction
                    acc -= oldest_state.acc_bias;
                    
                    ceres::CostFunction* gravity_factor = GravityAlignmentFactor::Create(acc, gravity_alignment_weight_);
                    std::vector<double*> parameter_blocks = {pose_param1};
                    std::vector<int> drop_set = {0}; // Drop the pose parameter
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        gravity_factor, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                }
                
                // If IMU has orientation and orientation factor is enabled, add yaw-only factor
                if (enable_imu_orientation_factor_ && closest_imu.header.stamp.toSec() > 0 &&
                    closest_imu.orientation_covariance[0] != -1) {
                    Eigen::Quaterniond q_imu(
                        closest_imu.orientation.w,
                        closest_imu.orientation.x,
                        closest_imu.orientation.y,
                        closest_imu.orientation.z
                    );
                    
                    ceres::CostFunction* yaw_factor = YawOnlyOrientationFactor::Create(q_imu, imu_orientation_weight_);
                    std::vector<double*> yaw_params = {pose_param1};
                    std::vector<int> yaw_drop_set = {0}; // Drop the pose parameter
                    
                    auto* yaw_residual = new MarginalizationInfo::ResidualBlockInfo(
                        yaw_factor, nullptr, yaw_params, yaw_drop_set);
                    marginalization_info->addResidualBlockInfo(yaw_residual);
                }
                
                // Add velocity constraint if enabled
                if (enable_velocity_constraint_) {
                    // Use adaptive max velocity based on IMU data
                    double adaptive_max_velocity = max_velocity_;
                    if (imu_buffer_.size() > 10) {
                        adaptive_max_velocity = estimateMaxVelocityFromImu();
                    }
                    
                    ceres::CostFunction* vel_constraint = VelocityMagnitudeConstraint::Create(
                        adaptive_max_velocity, velocity_constraint_weight_);
                    std::vector<double*> parameter_blocks = {vel_param1};
                    std::vector<int> drop_set = {0}; // Drop the velocity parameter
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        vel_constraint, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                }
                
                // Add horizontal velocity incentive if enabled
                if (enable_horizontal_velocity_incentive_) {
                    ceres::CostFunction* h_vel_incentive = HorizontalVelocityIncentiveFactor::Create(
                        min_horizontal_velocity_, horizontal_velocity_weight_);
                    std::vector<double*> parameter_blocks = {vel_param1, pose_param1};
                    std::vector<int> drop_set = {0, 1}; // Drop both parameters
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        h_vel_incentive, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                }
                
                // Add bias constraint
                if (enable_bias_estimation_) {
                    ceres::CostFunction* bias_constraint = BiasMagnitudeConstraint::Create(
                        acc_bias_max_, gyro_bias_max_, bias_constraint_weight_);
                    std::vector<double*> parameter_blocks = {bias_param1};
                    std::vector<int> drop_set = {0}; // Drop the bias parameter
                    
                    auto* residual_info = new MarginalizationInfo::ResidualBlockInfo(
                        bias_constraint, nullptr, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_info);
                }
                
                // Perform pre-marginalization
                marginalization_info->preMarginalize();
                
                // Perform marginalization
                marginalization_info->marginalize();
                
                // Clean up previous marginalization info
                if (last_marginalization_info_) {
                    delete last_marginalization_info_;
                    last_marginalization_info_ = nullptr;
                }
                
                // Store new marginalization info
                last_marginalization_info_ = marginalization_info;
                
                // Clear local allocations since they're now owned by marginalization_info
                local_allocations.clear();
                
            } catch (const std::exception& e) {
                // Clean up locally allocated memory if exception occurs
                for (auto ptr : local_allocations) {
                    delete[] ptr;
                }
                delete marginalization_info;
                throw;
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in prepareMarginalization: %s", e.what());
        }
    }
    
    // IMPROVED: Perform IMU pre-integration between keyframes with RK4 and bias random walk
    void performPreintegrationBetweenKeyframes(double start_time, double end_time, 
                                              const Eigen::Vector3d& acc_bias, 
                                              const Eigen::Vector3d& gyro_bias) {
        try {
            // Create key for the map
            std::pair<double, double> key(start_time, end_time);
            
            // Check if we already have this preintegration and bias hasn't changed significantly
            if (preintegration_map_.find(key) != preintegration_map_.end()) {
                const auto& existing = preintegration_map_[key];
                
                // Bias correction validity check - recompute if bias difference is large
                Eigen::Vector3d dba = acc_bias - existing.acc_bias_ref;
                Eigen::Vector3d dbg = gyro_bias - existing.gyro_bias_ref;
                
                if (dba.norm() < bias_correction_threshold_ && dbg.norm() < bias_correction_threshold_) {
                    // Bias change is small enough to use linear approximation
                    return;
                }
                
                // Bias change is significant - we'll recompute with the new bias values
                ROS_DEBUG("Recomputing pre-integration with new bias values (norm: acc=%.3f, gyro=%.3f)",
                        dba.norm(), dbg.norm());
            }
            
            // Ensure biases are within reasonable limits
            Eigen::Vector3d clamped_acc_bias = acc_bias;
            Eigen::Vector3d clamped_gyro_bias = gyro_bias;
            clampBiases(clamped_acc_bias, clamped_gyro_bias);
            
            // Create new preintegration data
            ImuPreintegrationBetweenKeyframes preint;
            preint.reset();
            preint.start_time = start_time;
            preint.end_time = end_time;
            preint.acc_bias_ref = clamped_acc_bias;
            preint.gyro_bias_ref = clamped_gyro_bias;
            
            // Find relevant IMU measurements with time tolerance for bag playback
            preint.imu_measurements.reserve(static_cast<size_t>((end_time - start_time) * 400 + 50)); // Reserve for 400Hz + buffer
            const double time_tolerance = 0.02; // 20ms tolerance
            
            // For bag playback, log the timespan we're searching for
            if (!imu_buffer_.empty()) {
                double buffer_start = imu_buffer_.front().header.stamp.toSec();
                double buffer_end = imu_buffer_.back().header.stamp.toSec();
                
                ROS_DEBUG("Looking for IMU data between %.6f and %.6f (span %.3f sec). IMU buffer covers [%.6f to %.6f] (span %.3f sec, %zu msgs)", 
                        start_time, end_time, end_time - start_time,
                        buffer_start, buffer_end, buffer_end - buffer_start,
                        imu_buffer_.size());
            }
            
            // Count IMU measurements in different parts of the time range for diagnostics
            int count_before_start = 0;
            int count_in_range = 0;
            int count_after_end = 0;
            double earliest_found = std::numeric_limits<double>::max();
            double latest_found = 0;
            
            for (const auto& imu : imu_buffer_) {
                double timestamp = imu.header.stamp.toSec();
                
                if (timestamp < (start_time - time_tolerance)) {
                    count_before_start++;
                } else if (timestamp > (end_time + time_tolerance)) {
                    count_after_end++;
                } else {
                    count_in_range++;
                    preint.imu_measurements.push_back(imu);
                    
                    // Track the time range of found measurements
                    if (timestamp < earliest_found) earliest_found = timestamp;
                    if (timestamp > latest_found) latest_found = timestamp;
                }
            }
            
            // Report detailed IMU data distribution for debugging
            if (count_in_range > 0) {
                ROS_DEBUG("Found %d IMU measurements between %.6f and %.6f. Actual range: [%.6f to %.6f]", 
                        count_in_range, start_time, end_time, earliest_found, latest_found);
            } else {
                ROS_WARN("No IMU data found between %.6f and %.6f. Buffer has %d msgs before and %d msgs after this range", 
                        start_time, end_time, count_before_start, count_after_end);
            }
            
            // If no IMU data found, create synthetic data
            if (preint.imu_measurements.empty()) {
                // Provide better diagnostics about what's in the buffer
                if (!imu_buffer_.empty()) {
                    double buffer_start = imu_buffer_.front().header.stamp.toSec();
                    double buffer_end = imu_buffer_.back().header.stamp.toSec();
                    
                    ROS_WARN("IMU buffer doesn't cover the required timespan: buffer [%.6f to %.6f], requested [%.6f to %.6f]",
                          buffer_start, buffer_end, start_time, end_time);
                    
                    // Print some sample timestamps around the desired range
                    ROS_INFO("Sample of available IMU timestamps:");
                    int samples_printed = 0;
                    for (const auto& imu : imu_buffer_) {
                        if (samples_printed >= 10) break;
                        if (std::abs(imu.header.stamp.toSec() - start_time) < 0.5 || 
                            std::abs(imu.header.stamp.toSec() - end_time) < 0.5) {
                            ROS_INFO("  IMU timestamp: %.6f", imu.header.stamp.toSec());
                            samples_printed++;
                        }
                    }
                }
                
                // IMPROVED: Create better synthetic IMU data
                // Calculate time between keyframes
                double dt = end_time - start_time;
                int num_synthetic = std::max(10, static_cast<int>(dt * 400.0));
                
                // Get reference points for interpolation
                State start_state;
                State end_state;
                bool found_start = false;
                bool found_end = false;
                
                // Find states closest to the boundaries
                for (const auto& state : state_window_) {
                    if (std::abs(state.timestamp - start_time) < 0.05) {
                        start_state = state;
                        found_start = true;
                    }
                    if (std::abs(state.timestamp - end_time) < 0.05) {
                        end_state = state;
                        found_end = true;
                    }
                }
                
                // If we have both boundary states, create more realistic synthetic data
                if (found_start && found_end) {
                    ROS_INFO("Creating realistic synthetic IMU from boundary states");
                    
                    // Get orientation and velocity differences
                    Eigen::Quaterniond delta_q = start_state.orientation.inverse() * end_state.orientation;
                    Eigen::Vector3d delta_v = end_state.velocity - start_state.velocity;
                    
                    // Generate synthetic measurements with realistic motion
                    for (int i = 0; i < num_synthetic; i++) {
                        double fraction = static_cast<double>(i) / (num_synthetic - 1);
                        double synthetic_time = start_time + fraction * dt;
                        
                        // Create synthetic IMU measurement
                        sensor_msgs::Imu synthetic_imu;
                        synthetic_imu.header.stamp = ros::Time(synthetic_time);
                        
                        // Interpolate orientation (slerp)
                        Eigen::Quaterniond interp_q = start_state.orientation.slerp(fraction, end_state.orientation);
                        
                        // Interpolate velocity (linear)
                        Eigen::Vector3d interp_v = start_state.velocity + fraction * delta_v;
                        
                        // Get gravity in sensor frame
                        Eigen::Vector3d gravity_sensor = interp_q.inverse() * gravity_world_;
                        
                        // Set realistic accelerometer readings (gravity plus acceleration)
                        // For acceleration, use velocity change over time
                        Eigen::Vector3d accel = delta_v / dt;
                        Eigen::Vector3d accel_sensor = interp_q.inverse() * accel;
                        
                        synthetic_imu.linear_acceleration.x = accel_sensor.x() - gravity_sensor.x();
                        synthetic_imu.linear_acceleration.y = accel_sensor.y() - gravity_sensor.y();
                        synthetic_imu.linear_acceleration.z = accel_sensor.z() - gravity_sensor.z();
                        
                        // Set angular velocity based on orientation change
                        double angle;
                        Eigen::Vector3d axis;
                        Eigen::AngleAxisd angle_axis(delta_q);
                        angle = angle_axis.angle();
                        axis = angle_axis.axis();
                        
                        Eigen::Vector3d angular_velocity = axis * (angle / dt);
                        Eigen::Vector3d gyro_sensor = interp_q.inverse() * angular_velocity;
                        
                        synthetic_imu.angular_velocity.x = gyro_sensor.x();
                        synthetic_imu.angular_velocity.y = gyro_sensor.y();
                        synthetic_imu.angular_velocity.z = gyro_sensor.z();
                        
                        preint.imu_measurements.push_back(synthetic_imu);
                    }
                } else {
                    // Get orientation at start time for gravity direction
                    Eigen::Quaterniond start_orientation = Eigen::Quaterniond::Identity();
                    for (const auto& state : state_window_) {
                        if (std::abs(state.timestamp - start_time) < 0.05) {
                            start_orientation = state.orientation;
                            break;
                        }
                    }
                    
                    // Transform gravity to sensor frame
                    Eigen::Vector3d gravity_sensor = start_orientation.inverse() * gravity_world_;
                    
                    // Create synthetic measurements evenly spaced
                    for (int i = 0; i < num_synthetic; i++) {
                        double fraction = static_cast<double>(i) / (num_synthetic - 1);
                        double synthetic_time = start_time + fraction * dt;
                        
                        // Create synthetic IMU measurement
                        sensor_msgs::Imu synthetic_imu;
                        synthetic_imu.header.stamp = ros::Time(synthetic_time);
                        
                        // Set to gravity-only with zero angular velocity as a fallback
                        synthetic_imu.linear_acceleration.x = -gravity_sensor.x();
                        synthetic_imu.linear_acceleration.y = -gravity_sensor.y();
                        synthetic_imu.linear_acceleration.z = -gravity_sensor.z();
                        
                        synthetic_imu.angular_velocity.x = 0;
                        synthetic_imu.angular_velocity.y = 0;
                        synthetic_imu.angular_velocity.z = 0;
                        
                        preint.imu_measurements.push_back(synthetic_imu);
                    }
                }
                
                ROS_INFO("Created %d synthetic IMU measurements between %.6f and %.6f",
                       (int)preint.imu_measurements.size(), start_time, end_time);
            }
            
            // Sort IMU measurements by timestamp
            if (preint.imu_measurements.size() > 1) {
                std::sort(preint.imu_measurements.begin(), preint.imu_measurements.end(), 
                         [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                             return a.header.stamp.toSec() < b.header.stamp.toSec();
                         });
            }
            
            // Initialize integration variables
            double prev_time = start_time;
            preint.sum_dt = 0;
            
            // Get orientation at start time
            Eigen::Quaterniond current_orientation = Eigen::Quaterniond::Identity();
            for (const auto& state : state_window_) {
                if (std::abs(state.timestamp - start_time) < 0.005) {
                    current_orientation = state.orientation;
                    break;
                }
            }
            
            // Precompute IMU noise matrix
            Eigen::Matrix<double, 6, 6> noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;
            noise_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * imu_acc_noise_ * imu_acc_noise_;
            noise_cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * imu_gyro_noise_ * imu_gyro_noise_;
            
            // Add bias random walk noise (continuous-time)
            Eigen::Matrix<double, 6, 6> bias_noise_cov = Eigen::Matrix<double, 6, 6>::Zero();
            bias_noise_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * imu_acc_bias_noise_ * imu_acc_bias_noise_;
            bias_noise_cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * imu_gyro_bias_noise_ * imu_gyro_bias_noise_;
            
            // Process each IMU measurement
            for (size_t i = 0; i < preint.imu_measurements.size(); ++i) {
                // Get current IMU data
                const auto& imu_msg = preint.imu_measurements[i];
                double timestamp = imu_msg.header.stamp.toSec();
                double dt = timestamp - prev_time;
                
                // Skip invalid time differences
                if (dt <= min_integration_dt_ || dt > max_imu_dt_) {
                    prev_time = timestamp;
                    continue;
                }
                
                // Subdivide large time steps for numerical stability
                int num_steps = 1;
                double step_dt = dt;
                if (dt > max_integration_dt_) {
                    num_steps = std::max(2, static_cast<int>(std::ceil(dt / max_integration_dt_)));
                    step_dt = dt / num_steps;
                }
                
                // Extract current IMU data
                Eigen::Vector3d acc1(
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z
                );
                
                Eigen::Vector3d gyro1(
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z
                );
                
                // Get next IMU data for interpolation (if available)
                Eigen::Vector3d acc2 = acc1;
                Eigen::Vector3d gyro2 = gyro1;
                
                if (i < preint.imu_measurements.size() - 1) {
                    const auto& next_imu = preint.imu_measurements[i + 1];
                    acc2 = Eigen::Vector3d(
                        next_imu.linear_acceleration.x,
                        next_imu.linear_acceleration.y,
                        next_imu.linear_acceleration.z
                    );
                    
                    gyro2 = Eigen::Vector3d(
                        next_imu.angular_velocity.x,
                        next_imu.angular_velocity.y,
                        next_imu.angular_velocity.z
                    );
                }
                
                // Apply bias correction
                acc1 -= clamped_acc_bias;
                acc2 -= clamped_acc_bias;
                gyro1 -= clamped_gyro_bias;
                gyro2 -= clamped_gyro_bias;
                
                // Integrate across subdivided steps
                for (int step = 0; step < num_steps; step++) {
                    // Linear interpolation for IMU data
                    double alpha = static_cast<double>(step) / num_steps;
                    double beta = static_cast<double>(step + 1) / num_steps;
                    
                    Eigen::Vector3d acc_step1 = acc1 * (1.0 - alpha) + acc2 * alpha;
                    Eigen::Vector3d acc_step2 = acc1 * (1.0 - beta) + acc2 * beta;
                    Eigen::Vector3d gyro_step1 = gyro1 * (1.0 - alpha) + gyro2 * alpha;
                    Eigen::Vector3d gyro_step2 = gyro1 * (1.0 - beta) + gyro2 * beta;
                    
                    // Store delta orientation before update
                    Eigen::Quaterniond delta_q_old = preint.delta_orientation;
                    
                    // Integrate delta quaternion (rotation)
                    Eigen::Vector3d integrated_gyro = (gyro_step1 + gyro_step2) * 0.5 * step_dt;
                    Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
                    
                    if (integrated_gyro.norm() > 1e-8) {
                        delta_q = Eigen::Quaterniond(
                            Eigen::AngleAxisd(integrated_gyro.norm(), integrated_gyro.normalized())
                        );
                    }
                    
                    // Update preintegrated delta orientation
                    preint.delta_orientation = preint.delta_orientation * delta_q;
                    preint.delta_orientation.normalize();
                    
                    // Compute half-point rotation for midpoint integration
                    Eigen::Quaterniond delta_q_half = delta_q_old.slerp(0.5, preint.delta_orientation);
                    delta_q_half.normalize();
                    
                    // Get gravity in sensor frame based on current orientation
                    Eigen::Vector3d gravity_sensor = current_orientation.inverse() * gravity_world_;
                    
                    // Remove gravity from accelerometers (gravity is already in sensor frame)
                    Eigen::Vector3d acc_without_gravity1 = acc_step1 - gravity_sensor;
                    Eigen::Vector3d acc_without_gravity2 = acc_step2 - gravity_sensor;
                    
                    // Rotate accelerations to integration frame
                    Eigen::Vector3d acc_int_frame1 = delta_q_half * acc_without_gravity1;
                    Eigen::Vector3d acc_int_frame2 = delta_q_half * acc_without_gravity2;
                    
                    // Integrate velocity using midpoint rule
                    Eigen::Vector3d acc_integrated = (acc_int_frame1 + acc_int_frame2) * 0.5;
                    preint.delta_velocity += acc_integrated * step_dt;
                    
                    // Integrate position using midpoint rule with current velocity
                    Eigen::Vector3d vel_midpoint = preint.delta_velocity - 0.5 * acc_integrated * step_dt;
                    preint.delta_position += vel_midpoint * step_dt;
                    
                    // Update covariance and Jacobians
                    // Calculate Jacobians for noise propagation
                    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
                    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * step_dt;
                    F.block<3, 3>(3, 6) = delta_q_half.toRotationMatrix() * step_dt;
                    F.block<3, 3>(0, 6) = 0.5 * delta_q_half.toRotationMatrix() * step_dt * step_dt;
                    
                    // Noise propagation matrix
                    Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
                    G.block<3, 3>(3, 0) = delta_q_half.toRotationMatrix();
                    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
                    
                    // Bias Jacobians
                    Eigen::Matrix<double, 9, 6> dF_db = Eigen::Matrix<double, 9, 6>::Zero();
                    
                    // Jacobian for accelerometer bias
                    dF_db.block<3, 3>(0, 0) = -0.5 * delta_q_half.toRotationMatrix() * step_dt * step_dt;
                    dF_db.block<3, 3>(3, 0) = -delta_q_half.toRotationMatrix() * step_dt;
                    
                    // Jacobian for gyroscope bias
                    Eigen::Matrix3d dR_dbg_times_a = -step_dt * skewSymmetric(delta_q_half * acc_without_gravity1);
                    dF_db.block<3, 3>(0, 3) = 0.5 * dR_dbg_times_a * step_dt;
                    dF_db.block<3, 3>(3, 3) = dR_dbg_times_a;
                    dF_db.block<3, 3>(6, 3) = -step_dt * Eigen::Matrix3d::Identity();
                    
                    // Update the Jacobian for bias
                    preint.jacobian_bias = F * preint.jacobian_bias + dF_db;
                    
                    // Add bias random walk for the specific time step
                    Eigen::Matrix<double, 6, 6> current_bias_cov = bias_noise_cov * step_dt;
                    
                    // Update the covariance
                    preint.covariance = F * preint.covariance * F.transpose() + 
                      G * noise_cov * G.transpose();
                    
                    // Add bias random walk contribution to the propagated noise
                    Eigen::Matrix<double, 9, 6> bias_noise_mapping = preint.jacobian_bias;
                    preint.covariance += bias_noise_mapping * current_bias_cov * bias_noise_mapping.transpose();
                    
                    // Update sum of dt for diagnostics
                    preint.sum_dt += step_dt;
                }
                
                // Update time tracking
                prev_time = timestamp;
            }
            
            // Check if integration covered the full time range
            double expected_dt = end_time - start_time;
            if (std::abs(preint.sum_dt - expected_dt) > 0.01) {
                ROS_WARN("Integration did not cover full time range. Expected dt=%.3f, got dt=%.3f", 
                      expected_dt, preint.sum_dt);
            }
            
            // Ensure numerical stability of covariance
            for (int i = 0; i < 9; ++i) {
                preint.covariance(i, i) = std::max(preint.covariance(i, i), 1e-8);
            }
            
            // Print diagnostics
            ROS_INFO("Preintegration [%.3f-%.3f] dt=%.3f: dp=[%.3f,%.3f,%.3f], dv=[%.3f,%.3f,%.3f], IMU count=%d",
                    start_time, end_time, preint.sum_dt,
                    preint.delta_position.x(), preint.delta_position.y(), preint.delta_position.z(),
                    preint.delta_velocity.x(), preint.delta_velocity.y(), preint.delta_velocity.z(),
                    (int)preint.imu_measurements.size());
            
            // Store in map
            preintegration_map_[key] = preint;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in performPreintegrationBetweenKeyframes: %s", e.what());
        }
    }

    void optimizationTimerCallback(const ros::TimerEvent& event) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            if (!is_initialized_) {
                return;
            }
            
            // Need at least 2 states for optimization
            if (state_window_.size() < 2) {
                return;
            }
            
            // Reset position if large drift detected
            if (use_gps_instead_of_uwb_) {
                // Check GPS drift
                if (!gps_measurements_.empty() && !state_window_.empty()) {
                    const auto& latest_gps = gps_measurements_.back();
                    auto& latest_state = state_window_.back();
                    
                    double position_error = (latest_state.position - latest_gps.position).norm();
                    
                    // Adjust drift threshold based on velocity
                    double adaptive_drift_threshold = 1.0; // Default threshold
                    double velocity_norm = latest_state.velocity.norm();
                    
                    // Increase allowable drift at higher speeds
                    if (velocity_norm > 10.0) {
                        adaptive_drift_threshold = 1.0 + (velocity_norm - 10.0) * 0.1;
                        adaptive_drift_threshold = std::min(adaptive_drift_threshold, 3.0); // Cap at 3 meters
                    }
                    
                    if (position_error > adaptive_drift_threshold) {
                        ROS_WARN("Position drift detected in GPS mode: %.2f meters. Resetting position.", position_error);
                        resetStateToGps(latest_gps);
                    }
                }
            } else {
                // Check UWB drift
                if (!uwb_measurements_.empty() && !state_window_.empty()) {
                    const auto& latest_uwb = uwb_measurements_.back();
                    auto& latest_state = state_window_.back();
                    
                    double position_error_z = std::abs(latest_state.position.z() - latest_uwb.position.z());
                    double position_error_xy = (latest_state.position.head<2>() - latest_uwb.position.head<2>()).norm();
                    
                    // Adjust position drift threshold for high-speed scenarios
                    double adaptive_drift_threshold = 1.0; // Default threshold
                    double velocity_norm = latest_state.velocity.norm();
                    
                    // Increase allowable drift at higher speeds
                    if (velocity_norm > 10.0) {
                        adaptive_drift_threshold = 1.0 + (velocity_norm - 10.0) * 0.1;
                        adaptive_drift_threshold = std::min(adaptive_drift_threshold, 3.0); // Cap at 3 meters
                    }
                    
                    if (position_error_z > adaptive_drift_threshold || position_error_xy > adaptive_drift_threshold) {
                        ROS_WARN("Position drift detected: Z=%.2f meters, XY=%.2f meters. Resetting to UWB position.", 
                                position_error_z, position_error_xy);
                        resetStateToUwb(latest_uwb);
                    }
                }
            }
            
            // Compute preintegration data between keyframes
            for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                double start_time = state_window_[i].timestamp;
                double end_time = state_window_[i+1].timestamp;
                
                std::pair<double, double> key(start_time, end_time);
                
                if (preintegration_map_.find(key) == preintegration_map_.end()) {
                    performPreintegrationBetweenKeyframes(start_time, end_time, 
                                                         state_window_[i].acc_bias, 
                                                         state_window_[i].gyro_bias);
                }
            }

            // Time the factor graph optimization
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Perform optimization
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
                
                // Output state after optimization
                Eigen::Vector3d euler_angles = quaternionToEulerDegrees(current_state_.orientation);
                
                // Get velocity in km/h for better reporting in high-speed scenario
                double velocity_kmh = current_state_.velocity.norm() * 3.6; // m/s to km/h
                
                ROS_INFO("Optimization time: %.1f ms", duration.count());
                ROS_INFO("State: Pos [%.2f, %.2f, %.2f] | Vel [%.2f, %.2f, %.2f] (%.1f km/h) | Euler [%.1f, %.1f, %.1f] deg | Bias acc [%.3f, %.3f, %.3f] gyro [%.3f, %.3f, %.3f]",
                    current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                    current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z(),
                    velocity_kmh,
                    euler_angles.x(), euler_angles.y(), euler_angles.z(),
                    current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
                    current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
                
                // Publish state
                publishOptimizedPose();

                // After optimization, calculate and visualize errors with GPS
                if (use_gps_instead_of_uwb_) {
                    calculateAndVisualizePositionError();
                    if (use_gps_velocity_) {
                        calculateAndVisualizeVelocityError();
                    }
                }
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in optimizationTimerCallback: %s", e.what());
        }
    }

    // Reset state to UWB if drift is too large
    void resetStateToUwb(const UwbMeasurement& uwb) {
        // Create a reset state that keeps orientation and biases but uses UWB position
        State reset_state = current_state_;
        reset_state.position = uwb.position;
        
        // Keep existing velocity direction but reduce magnitude
        double velocity_norm = reset_state.velocity.norm();
        if (velocity_norm > 0.1) {
            reset_state.velocity.normalize();
            reset_state.velocity *= std::min(min_horizontal_velocity_ * 2.0, velocity_norm * 0.5);
        } else {
            // Initialize with horizontal velocity in direction of orientation if velocity is very small
            double yaw = atan2(2.0 * (reset_state.orientation.w() * reset_state.orientation.z() + 
                           reset_state.orientation.x() * reset_state.orientation.y()),
                    1.0 - 2.0 * (reset_state.orientation.y() * reset_state.orientation.y() + 
                             reset_state.orientation.z() * reset_state.orientation.z()));
            
            reset_state.velocity.x() = min_horizontal_velocity_ * cos(yaw);
            reset_state.velocity.y() = min_horizontal_velocity_ * sin(yaw);
            reset_state.velocity.z() = 0;
        }
        
        // Reset all states in the window
        for (auto& state : state_window_) {
            State new_state = reset_state;
            new_state.timestamp = state.timestamp;
            
            // Keep original biases
            new_state.acc_bias = state.acc_bias;
            new_state.gyro_bias = state.gyro_bias;
            
            // Ensure biases are reasonable
            clampBiases(new_state.acc_bias, new_state.gyro_bias);
            
            state = new_state;
        }
        
        current_state_ = reset_state;
        preintegration_map_.clear();
        
        // Reset marginalization as well
        if (last_marginalization_info_) {
            delete last_marginalization_info_;
            last_marginalization_info_ = nullptr;
        }
        
        // Reset visualization
        resetVisualization();
    }

    // IMPROVED: Reset state to GPS with smoother position transition
    void resetStateToGps(const GpsMeasurement& gps) {
        // Create a reset state that uses GPS data
        State reset_state = current_state_;
        
        // Calculate position difference
        Eigen::Vector3d pos_diff = gps.position - reset_state.position;
        double pos_diff_norm = pos_diff.norm();
        
        // Apply a smooth blending rather than immediate jump
        double blend_factor = 0.7;  // 70% GPS, 30% current position
        
        // For very large jumps, be more conservative
        if (pos_diff_norm > 10.0) {
            blend_factor = 0.5;  // 50% GPS for very large jumps
            ROS_WARN("Very large position jump (%.2f m). Using more conservative blend.", pos_diff_norm);
        }
        
        // Apply blended position
        reset_state.position = reset_state.position * (1.0 - blend_factor) + 
                              gps.position * blend_factor;
        
        // Log the updated position difference
        Eigen::Vector3d new_pos_diff = gps.position - reset_state.position;
        ROS_INFO("Position after blending: diff reduced from %.2f m to %.2f m", 
                 pos_diff_norm, new_pos_diff.norm());
        
        // Update orientation if using GPS orientation
        if (use_gps_orientation_as_initial_) {
            reset_state.orientation = gps.orientation;
        }
        
        // Update velocity if using GPS velocity
        if (use_gps_velocity_) {
            reset_state.velocity = gps.velocity;
        } else {
            // Keep existing velocity direction but reduce magnitude
            double velocity_norm = reset_state.velocity.norm();
            if (velocity_norm > 0.1) {
                reset_state.velocity.normalize();
                reset_state.velocity *= std::min(min_horizontal_velocity_ * 2.0, velocity_norm * 0.5);
            } else {
                // Initialize with horizontal velocity in direction of orientation if velocity is very small
                double yaw = atan2(2.0 * (reset_state.orientation.w() * reset_state.orientation.z() + 
                               reset_state.orientation.x() * reset_state.orientation.y()),
                        1.0 - 2.0 * (reset_state.orientation.y() * reset_state.orientation.y() + 
                                 reset_state.orientation.z() * reset_state.orientation.z()));
                
                reset_state.velocity.x() = min_horizontal_velocity_ * cos(yaw);
                reset_state.velocity.y() = min_horizontal_velocity_ * sin(yaw);
                reset_state.velocity.z() = 0;
            }
        }
        
        // Reset all states in the window gradually
        for (auto& state : state_window_) {
            State new_state = state;  // Keep timestamp and other properties
            
            // Blend position for each state
            new_state.position = state.position * (1.0 - blend_factor) + 
                                gps.position * blend_factor;
            
            // Update orientation if using GPS orientation
            if (use_gps_orientation_as_initial_) {
                new_state.orientation = gps.orientation;
            }
            
            // Update velocity
            if (use_gps_velocity_) {
                new_state.velocity = gps.velocity;
            } else {
                new_state.velocity = reset_state.velocity;
            }
            
            // Keep original biases
            new_state.acc_bias = state.acc_bias;
            new_state.gyro_bias = state.gyro_bias;
            
            // Ensure biases are reasonable
            clampBiases(new_state.acc_bias, new_state.gyro_bias);
            
            state = new_state;
        }
        
        current_state_ = reset_state;
        
        // Recompute preintegration with new bias values
        preintegration_map_.clear();
        
        // Reset marginalization as well
        if (last_marginalization_info_) {
            delete last_marginalization_info_;
            last_marginalization_info_ = nullptr;
        }
        
        // Reset visualization
        resetVisualization();
        
        ROS_INFO("State reset to GPS: position=[%.2f, %.2f, %.2f], using orientation=%s, velocity=%s",
                reset_state.position.x(), reset_state.position.y(), reset_state.position.z(),
                use_gps_orientation_as_initial_ ? "true" : "false", use_gps_velocity_ ? "true" : "false");
    }

    void initializeFromUwb(const UwbMeasurement& uwb) {
        try {
            // Initialize state using UWB position
            current_state_.position = uwb.position;
            current_state_.orientation = Eigen::Quaterniond::Identity();
            
            // Initialize with non-zero velocity but we don't force a particular direction
            current_state_.velocity = Eigen::Vector3d(min_horizontal_velocity_, 0, 0);
            
            // Initialize with proper non-zero biases
            current_state_.acc_bias = initial_acc_bias_;
            current_state_.gyro_bias = initial_gyro_bias_;
            
            current_state_.timestamp = uwb.timestamp;
            
            // Get initial orientation from IMU if available
            sensor_msgs::Imu closest_imu = findClosestImuMeasurement(uwb.timestamp);
            if (closest_imu.header.stamp.toSec() > 0 && 
                closest_imu.orientation_covariance[0] != -1) {
                
                current_state_.orientation = Eigen::Quaterniond(
                    closest_imu.orientation.w,
                    closest_imu.orientation.x,
                    closest_imu.orientation.y,
                    closest_imu.orientation.z
                ).normalized();
                
                ROS_INFO("Using IMU orientation for initialization");
            }
            
            state_window_.clear(); // Ensure clean state window
            state_window_.push_back(current_state_);
            
            // Reset marginalization
            if (last_marginalization_info_) {
                delete last_marginalization_info_;
                last_marginalization_info_ = nullptr;
            }
            
            // Reset optimization count
            optimization_count_ = 0;
            
            // Reset visualization
            resetVisualization();
            
            ROS_INFO("State initialized at position [%.2f, %.2f, %.2f]", 
                    current_state_.position.x(), 
                    current_state_.position.y(), 
                    current_state_.position.z());
            ROS_INFO("Initial velocity [%.2f, %.2f, %.2f]",
                    current_state_.velocity.x(),
                    current_state_.velocity.y(),
                    current_state_.velocity.z());
            ROS_INFO("Initial biases: acc=[%.3f, %.3f, %.3f], gyro=[%.4f, %.4f, %.4f]",
                    current_state_.acc_bias.x(),
                    current_state_.acc_bias.y(),
                    current_state_.acc_bias.z(),
                    current_state_.gyro_bias.x(),
                    current_state_.gyro_bias.y(),
                    current_state_.gyro_bias.z());
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeFromUwb: %s", e.what());
        }
    }

    // Optimization using Ceres solver
    bool optimizeFactorGraph() {
        if (state_window_.size() < 2) {
            return false;
        }

        // Store original feature flags and max iterations
        bool original_enable_horizontal_velocity_incentive = enable_horizontal_velocity_incentive_;
        bool original_enable_orientation_smoothness_factor = enable_orientation_smoothness_factor_;
        int original_max_iterations = max_iterations_;
        
        // Check for initial optimization phase
        bool is_first_optimization = (optimization_count_ < 5);
        if (is_first_optimization) {
            // Disable complex features during initial iterations to improve stability
            enable_horizontal_velocity_incentive_ = false;
            enable_orientation_smoothness_factor_ = false;
            max_iterations_ = 5; // Use fewer iterations during initial phase
            ROS_DEBUG("Using simplified optimization for initial phase (%d/5)", optimization_count_+1);
        }
        
        // Create Ceres problem
        ceres::Problem::Options problem_options;
        problem_options.enable_fast_removal = true;
        ceres::Problem problem(problem_options);
        
        // Create pose parameterization
        ceres::LocalParameterization* pose_parameterization = new PoseParameterization();
        
        try {
            // Structure for storing state variables for Ceres
            struct OptVariables {
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                double pose[7]; // position (3) + quaternion (4)
                double velocity[3];
                double bias[6]; // acc_bias (3) + gyro_bias (3)
            };
            
            // Preallocate with reserve
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
            
            // Add pose parameterization
            for (size_t i = 0; i < state_window_.size(); ++i) {
                problem.AddParameterBlock(variables[i].pose, 7, pose_parameterization);
            }
            
            // If bias estimation is disabled, set biases constant
            if (!enable_bias_estimation_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    problem.SetParameterBlockConstant(variables[i].bias);
                }
            }
            
            // Add position measurements based on fusion mode
            if (use_gps_instead_of_uwb_) {
                // Add GPS position factors
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    double keyframe_time = state_window_[i].timestamp;
                    
                    // Find matching GPS measurement
                    for (const auto& gps : gps_measurements_) {
                        if (std::abs(gps.timestamp - keyframe_time) < 0.01) { // 10ms tolerance
                            // Add position factor
                            ceres::CostFunction* gps_pos_factor = GpsPositionFactor::Create(
                                gps.position, gps_position_noise_);
                            
                            problem.AddResidualBlock(gps_pos_factor, new ceres::HuberLoss(0.1), variables[i].pose);
                            
                            // Add velocity factor if configured to use GPS velocity
                            if (use_gps_velocity_ && enable_velocity_constraint_) {
                                ceres::CostFunction* gps_vel_factor = GpsVelocityFactor::Create(
                                    gps.velocity, gps_velocity_noise_);
                                
                                problem.AddResidualBlock(gps_vel_factor, new ceres::HuberLoss(0.1), variables[i].velocity);
                            }
                            
                            // Add orientation factor if configured to use GPS orientation as constraint
                            if (use_gps_orientation_as_constraint_) {
                                ceres::CostFunction* orientation_factor = GpsOrientationFactor::Create(
                                    gps.orientation, gps_orientation_noise_);
                                
                                problem.AddResidualBlock(orientation_factor, new ceres::HuberLoss(0.2), variables[i].pose);
                            }
                            
                            break;
                        }
                    }
                }
            } else {
                // Original UWB position factors
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    double keyframe_time = state_window_[i].timestamp;
                    
                    // Find matching UWB measurement
                    for (const auto& uwb : uwb_measurements_) {
                        if (std::abs(uwb.timestamp - keyframe_time) < 0.01) { // 10ms tolerance
                            ceres::CostFunction* uwb_factor = UwbPositionFactor::Create(
                                uwb.position, uwb_position_noise_);
                            
                            // Use HuberLoss for robustness
                            ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
                            problem.AddResidualBlock(uwb_factor, loss_function, variables[i].pose);
                            break;
                        }
                    }
                }
            }
            
            // Add roll/pitch constraint to enforce planar motion if enabled
            if (enable_roll_pitch_constraint_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* roll_pitch_prior = RollPitchPriorFactor::Create(roll_pitch_weight_);
                    problem.AddResidualBlock(roll_pitch_prior, nullptr, variables[i].pose);
                }
            }
            
            // Add gravity alignment factors if enabled
            if (enable_gravity_alignment_factor_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    double keyframe_time = state_window_[i].timestamp;
                    
                    // Find IMU measurement closest to this keyframe
                    sensor_msgs::Imu closest_imu = findClosestImuMeasurement(keyframe_time);
                    
                    if (closest_imu.header.stamp.toSec() > 0) {
                        Eigen::Vector3d acc(closest_imu.linear_acceleration.x, 
                                         closest_imu.linear_acceleration.y, 
                                         closest_imu.linear_acceleration.z);
                        
                        // Apply bias correction
                        acc -= state_window_[i].acc_bias;
                        
                        ceres::CostFunction* gravity_factor = 
                            GravityAlignmentFactor::Create(acc, gravity_alignment_weight_);
                        
                        problem.AddResidualBlock(gravity_factor, nullptr, variables[i].pose);
                    }
                }
            }
            
            // Add orientation smoothness constraints between consecutive keyframes if enabled
            if (enable_orientation_smoothness_factor_) {
                for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                    ceres::CostFunction* orientation_smoothness = 
                        OrientationSmoothnessFactor::Create(orientation_smoothness_weight_);
                    
                    problem.AddResidualBlock(orientation_smoothness, nullptr, 
                                           variables[i].pose, variables[i+1].pose);
                }
                
                // Add orientation smoothness constraints between non-adjacent keyframes (i and i+2)
                for (size_t i = 0; i < state_window_.size() - 2; ++i) {
                    ceres::CostFunction* orientation_smoothness = 
                        OrientationSmoothnessFactor::Create(orientation_smoothness_weight_ * 0.5);
                    
                    problem.AddResidualBlock(orientation_smoothness, nullptr, 
                                           variables[i].pose, variables[i+2].pose);
                }
            }
            
            // Add IMU orientation factors if enabled
            if (enable_imu_orientation_factor_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    double keyframe_time = state_window_[i].timestamp;
                    
                    // Find IMU measurement closest to this keyframe
                    sensor_msgs::Imu closest_imu = findClosestImuMeasurement(keyframe_time);
                    
                    // If valid IMU orientation, add orientation factor
                    if (closest_imu.header.stamp.toSec() > 0 && 
                        closest_imu.orientation_covariance[0] != -1) {
                        
                        double time_diff = std::abs(closest_imu.header.stamp.toSec() - keyframe_time);
                        if (time_diff < 0.05) { // 50ms threshold
                            addImuOrientationFactor(problem, variables[i].pose, closest_imu);
                        }
                    }
                }
            }
            
            // CRITICAL: Add hard constraints on bias magnitude if bias estimation is enabled
            if (enable_bias_estimation_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* bias_constraint = BiasMagnitudeConstraint::Create(
                        acc_bias_max_, gyro_bias_max_, bias_constraint_weight_);
                    
                    problem.AddResidualBlock(bias_constraint, nullptr, variables[i].bias);
                }
            }
            
            // IMPROVED: Add adaptive velocity magnitude constraints if enabled
            if (enable_velocity_constraint_) {
                // Estimate max velocity from IMU data for adaptive constraints
                double adaptive_max_velocity = max_velocity_;
                if (imu_buffer_.size() > 10) {
                    adaptive_max_velocity = estimateMaxVelocityFromImu();
                }
                
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* velocity_constraint = VelocityMagnitudeConstraint::Create(
                        adaptive_max_velocity, velocity_constraint_weight_);
                    problem.AddResidualBlock(velocity_constraint, nullptr, variables[i].velocity);
                }
            }
            
            // FIXED: Add horizontal velocity incentive factors that only enforce minimum magnitude
            if (enable_horizontal_velocity_incentive_) {
                for (size_t i = 0; i < state_window_.size(); ++i) {
                    ceres::CostFunction* h_vel_incentive = HorizontalVelocityIncentiveFactor::Create(
                        min_horizontal_velocity_, horizontal_velocity_weight_);
                    problem.AddResidualBlock(h_vel_incentive, nullptr, variables[i].velocity, variables[i].pose);
                }
            }
            
            // Add IMU pre-integration factors between keyframes
            for (size_t i = 0; i < state_window_.size() - 1; ++i) {
                double start_time = state_window_[i].timestamp;
                double end_time = state_window_[i+1].timestamp;
                
                // Skip if the time interval is too short
                if (end_time - start_time < 1e-6) continue;
                
                std::pair<double, double> key(start_time, end_time);
                
                if (preintegration_map_.find(key) != preintegration_map_.end()) {
                    const auto& preint = preintegration_map_[key];
                    
                    // IMPROVED: Pass bias correction threshold to IMU factor
                    ceres::CostFunction* imu_factor = ImuFactor::Create(
                        preint, gravity_world_, bias_correction_threshold_);
                    
                    // IMPROVED: Use HuberLoss for IMU factor
                    problem.AddResidualBlock(imu_factor, new ceres::HuberLoss(1.0),
                                           variables[i].pose, variables[i].velocity, variables[i].bias,
                                           variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                }
            }
            
            // Add marginalization prior if it exists and marginalization is enabled
            if (enable_marginalization_ && last_marginalization_info_ && state_window_.size() >= 2) {
                // Create a new marginalization factor
                MarginalizationFactor* factor = new MarginalizationFactor(last_marginalization_info_);
                
                // CRITICAL: Always use exactly 6 parameter blocks in the exact order expected
                // Adding the residual block with state variables in the correct order, without checks
                if (state_window_.size() >= 2) {
                    problem.AddResidualBlock(factor, nullptr,
                                           variables[1].pose, variables[1].velocity, variables[1].bias,
                                           variables[0].pose, variables[0].velocity, variables[0].bias);
                }
            }
            
            // Configure solver options
            ceres::Solver::Options options;
            options.max_num_iterations = max_iterations_;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 4;
            
            // Solve the optimization problem
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            if (!summary.IsSolutionUsable()) {
                return false;
            }
            
            // Update state with optimized values
            for (size_t i = 0; i < state_window_.size(); ++i) {
                // Update position
                state_window_[i].position = Eigen::Vector3d(
                    variables[i].pose[0], variables[i].pose[1], variables[i].pose[2]);
                
                // Update orientation
                state_window_[i].orientation = Eigen::Quaterniond(
                    variables[i].pose[3], variables[i].pose[4], variables[i].pose[5], variables[i].pose[6]).normalized();
                
                // Get velocity and ensure it's reasonable while preserving direction
                Eigen::Vector3d new_velocity(
                    variables[i].velocity[0], variables[i].velocity[1], variables[i].velocity[2]);
                
                // Use adaptive max velocity for high-speed scenario
                double adaptive_max_velocity = max_velocity_;
                if (imu_buffer_.size() > 10) {
                    adaptive_max_velocity = estimateMaxVelocityFromImu();
                }
                
                clampVelocity(new_velocity, adaptive_max_velocity);
                state_window_[i].velocity = new_velocity;
                
                // Update biases if enabled
                if (enable_bias_estimation_) {
                    // First, get biases from optimization
                    Eigen::Vector3d new_acc_bias(
                        variables[i].bias[0], variables[i].bias[1], variables[i].bias[2]);
                    
                    Eigen::Vector3d new_gyro_bias(
                        variables[i].bias[3], variables[i].bias[4], variables[i].bias[5]);
                    
                    // Ensure biases stay within reasonable limits
                    clampBiases(new_acc_bias, new_gyro_bias);
                    
                    // Update state with clamped biases
                    state_window_[i].acc_bias = new_acc_bias;
                    state_window_[i].gyro_bias = new_gyro_bias;
                }
            }
            
            // Update current state to the latest state in the window
            if (!state_window_.empty()) {
                current_state_ = state_window_.back();
                
                // Keep bias constraints consistent across the system
                clampBiases(current_state_.acc_bias, current_state_.gyro_bias);
                
                // Ensure velocity stays within reasonable limits while preserving direction
                double adaptive_max_velocity = max_velocity_;
                if (imu_buffer_.size() > 10) {
                    adaptive_max_velocity = estimateMaxVelocityFromImu();
                }
                clampVelocity(current_state_.velocity, adaptive_max_velocity);
            }
            
            // Increment optimization count
            optimization_count_++;
            
            // Restore original feature flags and max iterations
            enable_horizontal_velocity_incentive_ = original_enable_horizontal_velocity_incentive;
            enable_orientation_smoothness_factor_ = original_enable_orientation_smoothness_factor;
            max_iterations_ = original_max_iterations;
            
            return true;
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during optimization: %s", e.what());
            
            // Restore original feature flags and max iterations
            enable_horizontal_velocity_incentive_ = original_enable_horizontal_velocity_incentive;
            enable_orientation_smoothness_factor_ = original_enable_orientation_smoothness_factor;
            max_iterations_ = original_max_iterations;
            
            return false;
        }
    }

    // Publish IMU-predicted pose
    void publishImuPose() {
        try {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(current_state_.timestamp);
            odom_msg.header.frame_id = world_frame_id_; // "map"
            // odom_msg.child_frame_id = body_frame_id_;   // "base_link"
            odom_msg.child_frame_id = world_frame_id_;   // "base_link"
            
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
            
            // Publish the message
            imu_pose_pub_.publish(odom_msg);
            
            // Publish the TF transform
            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped.header.stamp = odom_msg.header.stamp;
            transform_stamped.header.frame_id = world_frame_id_; // "map"
            transform_stamped.child_frame_id = body_frame_id_;   // "base_link"
            
            // Set translation
            transform_stamped.transform.translation.x = current_state_.position.x();
            transform_stamped.transform.translation.y = current_state_.position.y();
            transform_stamped.transform.translation.z = current_state_.position.z();
            
            // Set rotation
            transform_stamped.transform.rotation.w = current_state_.orientation.w();
            transform_stamped.transform.rotation.x = current_state_.orientation.x();
            transform_stamped.transform.rotation.y = current_state_.orientation.y();
            transform_stamped.transform.rotation.z = current_state_.orientation.z();
            
            // Publish the transform
            tf_broadcaster_.sendTransform(transform_stamped);
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishImuPose: %s", e.what());
        }
    }

    // Publish optimized pose
    void publishOptimizedPose() {
        try {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time(current_state_.timestamp);
            odom_msg.header.frame_id = world_frame_id_; // "map"
            odom_msg.child_frame_id = body_frame_id_;   // "base_link"
            
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
            
            // Publish the message
            optimized_pose_pub_.publish(odom_msg);
            
            // Publish the TF transform
            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped.header.stamp = odom_msg.header.stamp;
            transform_stamped.header.frame_id = world_frame_id_; // "map"
            transform_stamped.child_frame_id = body_frame_id_;   // "base_link"
            
            // Set translation
            transform_stamped.transform.translation.x = current_state_.position.x();
            transform_stamped.transform.translation.y = current_state_.position.y();
            transform_stamped.transform.translation.z = current_state_.position.z();
            
            // Set rotation
            transform_stamped.transform.rotation.w = current_state_.orientation.w();
            transform_stamped.transform.rotation.x = current_state_.orientation.x();
            transform_stamped.transform.rotation.y = current_state_.orientation.y();
            transform_stamped.transform.rotation.z = current_state_.orientation.z();
            
            // Publish the transform
            tf_broadcaster_.sendTransform(transform_stamped);
            
            // Update and publish the optimized path
            updateOptimizedPath();
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishOptimizedPose: %s", e.what());
        }
    }

    // IMPROVED: Propagate state with RK4 integration and better time handling for high speeds
    State propagateState(const State& reference_state, double target_time) {
        State result = reference_state;
        
        // If target_time is earlier than reference time, just return the reference state
        if (target_time <= reference_state.timestamp) {
            return reference_state;
        }
        
        // Find IMU measurements between reference_state.timestamp and target_time
        std::vector<sensor_msgs::Imu> relevant_imu_msgs;
        relevant_imu_msgs.reserve(100);
        
        for (const auto& imu : imu_buffer_) {
            double timestamp = imu.header.stamp.toSec();
            if (timestamp > reference_state.timestamp && timestamp <= target_time) {
                relevant_imu_msgs.push_back(imu);
            }
        }
        
        // Sort by timestamp
        if (relevant_imu_msgs.size() > 1) {
            std::sort(relevant_imu_msgs.begin(), relevant_imu_msgs.end(), 
                     [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                         return a.header.stamp.toSec() < b.header.stamp.toSec();
                     });
        }
        
        // IMPROVED: Better time interval handling
        double prev_time = reference_state.timestamp;
        size_t imu_idx = 0;
        
        while (prev_time < target_time && imu_idx < relevant_imu_msgs.size()) {
            // Get current IMU data
            const auto& imu_msg = relevant_imu_msgs[imu_idx];
            double timestamp = imu_msg.header.stamp.toSec();
            
            // Calculate time increment with subdivision if needed
            double dt = timestamp - prev_time;
            
            // Skip invalid dt - IMPROVED: more strict checking for tiny time steps
            if (dt <= min_integration_dt_ || dt > max_imu_dt_) {
                prev_time = timestamp;
                imu_idx++;
                continue;
            }
            
            // Subdivide large time steps for better accuracy - use smaller steps for high-speed
            int num_steps = 1;
            double step_dt = dt;
            
            // IMPROVED: If dt is too large, subdivide into smaller steps - more subdivision for high speeds
            if (dt > max_integration_dt_) {
                // For high speeds, use more subdivision steps
                num_steps = std::max(2, static_cast<int>(std::ceil(dt / max_integration_dt_)));
                step_dt = dt / num_steps;
            }
            
            // Extract IMU data
            Eigen::Vector3d acc1(imu_msg.linear_acceleration.x,
                                 imu_msg.linear_acceleration.y,
                                 imu_msg.linear_acceleration.z);
            
            Eigen::Vector3d gyro1(imu_msg.angular_velocity.x,
                                  imu_msg.angular_velocity.y,
                                  imu_msg.angular_velocity.z);
            
            // Get next IMU data for RK4 (use current if last)
            Eigen::Vector3d acc2 = acc1;
            Eigen::Vector3d gyro2 = gyro1;
            
            if (imu_idx < relevant_imu_msgs.size() - 1) {
                const auto& next_imu = relevant_imu_msgs[imu_idx + 1];
                acc2 = Eigen::Vector3d(next_imu.linear_acceleration.x,
                                       next_imu.linear_acceleration.y,
                                       next_imu.linear_acceleration.z);
                
                gyro2 = Eigen::Vector3d(next_imu.angular_velocity.x,
                                        next_imu.angular_velocity.y,
                                        next_imu.angular_velocity.z);
            }
            
            // Apply bias correction
            acc1 -= result.acc_bias;
            acc2 -= result.acc_bias;
            gyro1 -= result.gyro_bias;
            gyro2 -= result.gyro_bias;
            
            // Perform integration using subdivided steps
            for (int step = 0; step < num_steps; step++) {
                // Linear interpolation for IMU data during subdivision
                double alpha = static_cast<double>(step) / num_steps;
                double beta = static_cast<double>(step + 1) / num_steps;
                
                Eigen::Vector3d acc_step1 = acc1 * (1.0 - alpha) + acc2 * alpha;
                Eigen::Vector3d acc_step2 = acc1 * (1.0 - beta) + acc2 * beta;
                Eigen::Vector3d gyro_step1 = gyro1 * (1.0 - alpha) + gyro2 * alpha;
                Eigen::Vector3d gyro_step2 = gyro1 * (1.0 - beta) + gyro2 * beta;
                
                // IMPROVED: Use RK4 integration for orientation
                Eigen::Quaterniond orientation_before = result.orientation;
                rk4IntegrateOrientation(gyro_step1, gyro_step2, step_dt, result.orientation);
                
                // Get gravity in sensor frame before and after orientation update
                Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
                Eigen::Vector3d gravity_sensor2 = result.orientation.inverse() * gravity_world_;
                
                // Remove gravity from accelerometer reading (averaged over rotation change)
                Eigen::Vector3d acc_without_gravity1 = acc_step1 - gravity_sensor1;
                Eigen::Vector3d acc_without_gravity2 = acc_step2 - gravity_sensor2;
                
                // Rotate to world frame using RK4 approach for acceleration
                Eigen::Vector3d acc_world1 = orientation_before * acc_without_gravity1;
                Eigen::Vector3d acc_world2 = result.orientation * acc_without_gravity2;
                
                // IMPROVED: RK4 integration for velocity/position
                Eigen::Vector3d k1v = acc_world1;
                Eigen::Vector3d k2v = 0.5 * (acc_world1 + acc_world2);
                Eigen::Vector3d k3v = 0.5 * (acc_world1 + acc_world2);
                Eigen::Vector3d k4v = acc_world2;
                
                Eigen::Vector3d velocity_before = result.velocity;
                Eigen::Vector3d acc_integrated = (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
                
                // Update velocity with RK4 integration
                result.velocity += acc_integrated * step_dt;
                
                // CRITICAL: Ensure velocity stays within reasonable limits while preserving direction
                // For propagation, we use a high max velocity to avoid artificially limiting
                // the state when using high-accuracy IMU integration
                double adaptive_max_vel = std::max(max_velocity_, 35.0); // Allow higher during propagation
                clampVelocity(result.velocity, adaptive_max_vel);
                
                // RK4 for position
                Eigen::Vector3d k1p = velocity_before;
                Eigen::Vector3d k2p = velocity_before + 0.5 * step_dt * k1v;
                Eigen::Vector3d k3p = velocity_before + 0.5 * step_dt * k2v;
                Eigen::Vector3d k4p = result.velocity;
                
                // Update position with RK4 integration
                Eigen::Vector3d vel_integrated = (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
                result.position += vel_integrated * step_dt;
            }
            
            // Update timestamp for next step
            prev_time = timestamp;
            imu_idx++;
        }
        
        // Final step to target_time if needed
        double dt = target_time - prev_time;
        if (dt > min_integration_dt_ && dt <= max_imu_dt_ && !relevant_imu_msgs.empty()) {
            // Use the last IMU measurement for prediction
            const auto& last_imu = relevant_imu_msgs.back();
            
            Eigen::Vector3d acc(last_imu.linear_acceleration.x,
                               last_imu.linear_acceleration.y,
                               last_imu.linear_acceleration.z);
            
            Eigen::Vector3d gyro(last_imu.angular_velocity.x,
                                last_imu.angular_velocity.y,
                                last_imu.angular_velocity.z);
            
            // Apply bias correction
            Eigen::Vector3d acc_corrected = acc - result.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
            
            // For final small step, use simpler integration to avoid extrapolation errors
            // Update orientation
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            Eigen::Quaterniond orientation_before = result.orientation;
            result.orientation = (result.orientation * dq).normalized();
            
            // Get gravity in sensor frame (average of before and after rotation)
            Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
            Eigen::Vector3d gravity_sensor2 = result.orientation.inverse() * gravity_world_;
            Eigen::Vector3d gravity_sensor = 0.5 * (gravity_sensor1 + gravity_sensor2);
            
            // Remove gravity from accelerometer reading
            Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_sensor;
            
            // Rotate to world frame using average orientation
            Eigen::Quaterniond orientation_mid = orientation_before.slerp(0.5, result.orientation);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity
            Eigen::Vector3d velocity_before = result.velocity;
            result.velocity += acc_world * dt;
            
            // Clamp velocity while preserving direction
            double adaptive_max_vel = std::max(max_velocity_, 35.0); // Allow higher during propagation
            clampVelocity(result.velocity, adaptive_max_vel);
            
            // Update position using trapezoidal integration
            result.position += 0.5 * (velocity_before + result.velocity) * dt;
        }
        
        // Ensure the timestamp is updated correctly
        result.timestamp = target_time;
        
        return result;
    }
    
    // FIXED: Real-time state propagation with IMU that preserves circular motion
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
                
                // Update orientation directly from IMU if available
                if (imu_msg.orientation_covariance[0] != -1) {
                    current_state_.orientation = Eigen::Quaterniond(
                        imu_msg.orientation.w,
                        imu_msg.orientation.x,
                        imu_msg.orientation.y,
                        imu_msg.orientation.z
                    ).normalized();
                }
                
                return;  // Skip integration for the first IMU message
            }
            
            // Skip integration for invalid dt
            if (dt <= min_integration_dt_ || dt > max_imu_dt_) {
                current_state_.timestamp = timestamp;
                return;
            }
            
            // Use IMU orientation if available
            if (imu_msg.orientation_covariance[0] != -1) {
                // Get orientation from IMU
                Eigen::Quaterniond imu_orientation(
                    imu_msg.orientation.w,
                    imu_msg.orientation.x,
                    imu_msg.orientation.y,
                    imu_msg.orientation.z
                );
                
                // Update orientation directly - but use a weighted average to smooth transitions
                Eigen::Quaterniond blended_orientation = current_state_.orientation.slerp(0.3, imu_orientation);
                current_state_.orientation = blended_orientation.normalized();
            } else {
                // Apply bias correction
                Eigen::Vector3d acc_corrected = acc - current_state_.acc_bias;
                Eigen::Vector3d gyro_corrected = gyro - current_state_.gyro_bias;
                
                // Store orientation before update
                Eigen::Quaterniond orientation_before = current_state_.orientation;
                
                // Update orientation with simple integration for real-time
                Eigen::Vector3d angle_axis = gyro_corrected * dt;
                Eigen::Quaterniond dq = deltaQ(angle_axis);
                
                // Update orientation
                current_state_.orientation = (current_state_.orientation * dq).normalized();
                
                // Get gravity in sensor frame (average of before and after rotation)
                Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
                Eigen::Vector3d gravity_sensor2 = current_state_.orientation.inverse() * gravity_world_;
                Eigen::Vector3d gravity_sensor = 0.5 * (gravity_sensor1 + gravity_sensor2);
                
                // Remove gravity from accelerometer reading
                Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_sensor;
                
                // Rotate to world frame using midpoint rotation
                Eigen::Quaterniond orientation_mid = orientation_before.slerp(0.5, current_state_.orientation);
                Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
                
                // Store velocity before update for trapezoidal integration
                Eigen::Vector3d velocity_before = current_state_.velocity;
                
                // Update velocity
                current_state_.velocity += acc_world * dt;
                
                // IMPROVED: For high-speed scenarios - remove vertical damping
                // Only apply slight damping if we have a large spurious vertical velocity
                double v_vel_abs = std::abs(current_state_.velocity.z());
                if (v_vel_abs > 5.0) {  // Only dampen extreme vertical velocities
                    current_state_.velocity.z() *= 0.95;  // Mild damping only on extreme values
                }
                
                // Adaptive max velocity based on IMU data for real-time propagation
                double adaptive_max_vel = max_velocity_;
                if (imu_buffer_.size() > 10) {
                    adaptive_max_vel = std::max(max_velocity_, estimateMaxVelocityFromImu());
                }
                
                // Ensure velocity stays within reasonable limits while preserving direction
                clampVelocity(current_state_.velocity, adaptive_max_vel);
                
                // Update position using trapezoidal integration
                current_state_.position += 0.5 * (velocity_before + current_state_.velocity) * dt;
            }
            
            // Update timestamp
            current_state_.timestamp = timestamp;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in propagateStateWithImu: %s", e.what());
        }
    }
};

int main(int argc, char **argv) {
    try {
        ros::init(argc, argv, "uwb_imu_batch_node");
        
        {
            UwbImuFusion fusion; 
            ros::spin();
        }
        
        return 0;
    } catch (const std::exception& e) {
        ROS_ERROR("Fatal exception in main: %s", e.what());
        return 1;
    } catch (...) {
        ROS_ERROR("Unknown fatal exception in main");
        return 1;
    }
}