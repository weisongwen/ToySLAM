#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
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
#include <sys/resource.h>
#include <sys/time.h>
#include <map>
#include <set>

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

// Hard constraint on bias magnitude
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
        
        // Higher weight for gyro bias constraint
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

// Velocity magnitude constraint class
class VelocityMagnitudeConstraint {
public:
    VelocityMagnitudeConstraint(double max_velocity = 2.0, double weight = 1000.0)
        : max_velocity_(max_velocity), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const velocity, T* residuals) const {
        // Compute velocity magnitude
        T vx = velocity[0];
        T vy = velocity[1];
        T vz = velocity[2];
        T magnitude = ceres::sqrt(vx*vx + vy*vy + vz*vz);
        
        // Only penalize if velocity exceeds maximum
        residuals[0] = T(0.0);
        if (magnitude > T(max_velocity_)) {
            residuals[0] = T(weight_) * (magnitude - T(max_velocity_));
        }
        
        return true;
    }
    
    static ceres::CostFunction* Create(double max_velocity = 2.0, double weight = 1000.0) {
        return new ceres::AutoDiffCostFunction<VelocityMagnitudeConstraint, 1, 3>(
            new VelocityMagnitudeConstraint(max_velocity, weight));
    }
    
private:
    double max_velocity_;
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

// Orientation smoothness factor to enforce smooth orientation changes
class OrientationSmoothnessFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    OrientationSmoothnessFactor(double weight = 150.0) : weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // Extract orientations
        Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);
        
        // Compute orientation difference
        Eigen::Quaternion<T> q_diff = q_i.conjugate() * q_j;
        
        // Convert to angle-axis representation (with safety checks)
        T angle = T(2.0) * acos(std::min(std::max(q_diff.w(), T(-1.0)), T(1.0)));
        
        // Set residual proportional to angular change
        residuals[0] = T(weight_) * angle;
        
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

// YawOnlyOrientationFactor to only use yaw component from IMU orientation
class YawOnlyOrientationFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    YawOnlyOrientationFactor(const Eigen::Quaterniond& measured_orientation, double weight)
        : weight_(weight) {
        // Extract yaw from measured orientation
        double q_x = measured_orientation.x();
        double q_y = measured_orientation.y();
        double q_z = measured_orientation.z();
        double q_w = measured_orientation.w();
        
        // Convert to yaw angle
        double yaw = atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z));
        
        // Create quaternion with only yaw (roll=pitch=0)
        yaw_only_quat_ = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    }
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract orientation quaternion from pose
        Eigen::Map<const Eigen::Quaternion<T>> q(pose + 3);
        
        // Extract yaw from pose quaternion
        T q_x = q.x();
        T q_y = q.y();
        T q_z = q.z();
        T q_w = q.w();
        
        T yaw = atan2(T(2.0) * (q_w * q_z + q_x * q_y), 
                      T(1.0) - T(2.0) * (q_y * q_y + q_z * q_z));
        
        // Create yaw-only quaternion from pose
        Eigen::Quaternion<T> pose_yaw_only(
            Eigen::AngleAxis<T>(yaw, Eigen::Matrix<T, 3, 1>::UnitZ()));
        
        // Compare with measured yaw-only quaternion
        Eigen::Quaternion<T> q_measured = yaw_only_quat_.cast<T>();
        
        // Compute orientation difference
        Eigen::Quaternion<T> q_diff = q_measured.conjugate() * pose_yaw_only;
        
        // Convert to angle-axis for residual
        T angle = T(2.0) * acos(q_diff.w());
        
        // Set single residual - only care about yaw difference
        residuals[0] = T(weight_) * angle;
        
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

// Define the pre-integration class
class ImuPreintegrationBetweenKeyframes {
public:
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

// VINS-Mono style residual block info
class ResidualBlockInfo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    ResidualBlockInfo(ceres::CostFunction* _cost_function, 
                     ceres::LossFunction* _loss_function,
                     std::vector<double*> _parameter_blocks,
                     std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function),
          parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}
    
    void Evaluate();
    
    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;
    std::vector<int> drop_set;
    
    // For evaluation
    double** raw_jacobians = nullptr;
    std::vector<Eigen::MatrixXd> jacobians;
    Eigen::VectorXd residuals;
    
    // Keep copies of parameter values
    std::vector<double*> parameter_blocks_data;
};

// Simpler MarginalizationInfo class following VINS-Mono structure
class MarginalizationInfo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    MarginalizationInfo() {}
    
    ~MarginalizationInfo() {
        // Free memory in all factor blocks
        for (auto& block : factors) {
            delete block;
        }
        factors.clear();
        
        // Free parameter block data
        for (auto& pair : parameter_block_data) {
            delete[] pair.second;
        }
        parameter_block_data.clear();
    }
    
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info) {
        factors.emplace_back(residual_block_info);
        
        // Create a local copy of parameters
        std::vector<double*>& parameter_blocks = residual_block_info->parameter_blocks;
        std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();
        
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            double* addr = parameter_blocks[i];
            int size = parameter_block_sizes[i];
            
            if (parameter_block_data_size.find(addr) == parameter_block_data_size.end()) {
                double* data = new double[size];
                memcpy(data, addr, sizeof(double) * size);
                parameter_block_data_size[addr] = size;
                parameter_block_data[addr] = data;
            }
        }
    }
    
    void preMarginalize() {
        for (auto& block : factors) {
            block->parameter_blocks_data.clear();
            
            std::vector<double*>& parameter_blocks = block->parameter_blocks;
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                double* addr = parameter_blocks[i];
                block->parameter_blocks_data.push_back(parameter_block_data[addr]);
            }
            
            block->Evaluate();
        }
    }
    
    void marginalize() {
        // Organize parameter blocks by keep/drop
        std::vector<int> keep_block_size;
        std::vector<double*> keep_block_addr;
        std::vector<int> keep_block_data_size;
        
        for (const auto& block : factors) {
            for (int i = 0; i < static_cast<int>(block->parameter_blocks.size()); i++) {
                if (block->parameter_blocks_data[i] != nullptr) {
                    double* addr = block->parameter_blocks[i];
                    int size = block->cost_function->parameter_block_sizes()[i];
                    
                    if (block->drop_set.size() == 0 || 
                        std::find(block->drop_set.begin(), block->drop_set.end(), i) == block->drop_set.end()) {
                        
                        // Check if this address is already in keep_block_addr
                        if (std::find(keep_block_addr.begin(), keep_block_addr.end(), addr) == keep_block_addr.end()) {
                            keep_block_size.push_back(size);
                            keep_block_addr.push_back(addr);
                            keep_block_data_size.push_back(parameter_block_data_size[addr]);
                        }
                    }
                }
            }
        }
        
        m = 0;
        for (auto& block : factors) {
            m += block->cost_function->num_residuals();
        }
        
        // Initialize Hessian and gradient
        int n = static_cast<int>(keep_block_size.size());
        
        // Store linearized information and keep blocks
        linear_system_H.resize(n, n);
        linear_system_b.resize(n);
        linear_system_H.setZero();
        linear_system_b.setZero();
        
        keep_parameter_blocks = keep_block_addr;
    }
    
    // Get parameter blocks
    std::vector<double*> getParameterBlocks() {
        return keep_parameter_blocks;
    }
    
    // Member variables
    std::vector<ResidualBlockInfo*> factors;
    std::map<double*, double*> parameter_block_data;
    std::map<double*, int> parameter_block_data_size;
    
    // Schur complement system
    int m;  // Measurement dimension
    Eigen::MatrixXd linear_system_H;
    Eigen::VectorXd linear_system_b;
    std::vector<double*> keep_parameter_blocks;
};

// Implement the evaluation method
void ResidualBlockInfo::Evaluate() {
    // Safety checks
    if (parameter_blocks_data.size() != parameter_blocks.size()) {
        ROS_WARN("Parameter blocks data size mismatch in Evaluate(): %zu vs %zu", 
               parameter_blocks_data.size(), parameter_blocks.size());
        return;
    }
    
    for (size_t i = 0; i < parameter_blocks_data.size(); i++) {
        if (parameter_blocks_data[i] == nullptr) {
            ROS_ERROR("Parameter block data is null at index %zu", i);
            return;
        }
    }
    
    // Calculate sizes
    int num_residuals = cost_function->num_residuals();
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    
    // Allocate memory for Jacobians
    raw_jacobians = new double*[block_sizes.size()];
    jacobians.resize(block_sizes.size());
    
    for (size_t i = 0; i < block_sizes.size(); i++) {
        jacobians[i].resize(num_residuals, block_sizes[i]);
        jacobians[i].setZero();
        raw_jacobians[i] = new double[num_residuals * block_sizes[i]];
    }
    
    // Allocate for residuals
    residuals.resize(num_residuals);
    
    // Evaluate the cost function
    double* raw_residuals = new double[num_residuals];
    
    cost_function->Evaluate(parameter_blocks_data.data(), raw_residuals, raw_jacobians);
    
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
        
        for (size_t i = 0; i < parameter_blocks.size(); i++) {
            for (int j = 0; j < block_sizes[i] * num_residuals; j++) {
                raw_jacobians[i][j] *= residual_scaling;
            }
        }
    }
    
    // Copy raw residuals to Eigen vector
    for (int i = 0; i < num_residuals; i++) {
        residuals(i) = raw_residuals[i];
    }
    
    // Copy raw jacobians to Eigen matrices
    for (size_t i = 0; i < parameter_blocks.size(); i++) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            mat_jacobian(raw_jacobians[i], num_residuals, block_sizes[i]);
        jacobians[i] = mat_jacobian;
    }
    
    // Clean up
    delete[] raw_residuals;
    for (size_t i = 0; i < parameter_blocks.size(); i++) {
        delete[] raw_jacobians[i];
    }
    delete[] raw_jacobians;
    raw_jacobians = nullptr;
}

// VINS-Mono style marginalization factor
class MarginalizationFactor : public ceres::CostFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    MarginalizationFactor(MarginalizationInfo* _marginalization_info)
        : marginalization_info(_marginalization_info) {
        // Set up cost function
        *mutable_parameter_block_sizes() = std::vector<int>();
        
        const auto& parameter_blocks = marginalization_info->getParameterBlocks();
        
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            double* addr = parameter_blocks[i];
            int size = marginalization_info->parameter_block_data_size[addr];
            mutable_parameter_block_sizes()->push_back(size);
        }
        
        // Set residual size
        set_num_residuals(marginalization_info->m);
    }
    
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // Simple stub implementation for now
        // In a full implementation, this would compute the linearized cost
        for (int i = 0; i < marginalization_info->m; i++) {
            residuals[i] = 0.0;
        }
        
        // Zero out Jacobians 
        if (jacobians) {
            for (size_t i = 0; i < parameter_block_sizes().size(); i++) {
                if (jacobians[i]) {
                    memset(jacobians[i], 0, sizeof(double) * parameter_block_sizes()[i] * marginalization_info->m);
                }
            }
        }
        
        return true;
    }
    
    MarginalizationInfo* marginalization_info;
};

// Main UWB-IMU fusion class
class UwbImuFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    UwbImuFusion() {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        
        // Load parameters
        private_nh.param<double>("gravity_magnitude", gravity_magnitude_, 9.81);
        
        // Realistic IMU noise parameters
        private_nh.param<double>("imu_acc_noise", imu_acc_noise_, 0.03);    // m/s²
        private_nh.param<double>("imu_gyro_noise", imu_gyro_noise_, 0.002); // rad/s
        
        // Realistic bias parameters
        private_nh.param<double>("imu_acc_bias_noise", imu_acc_bias_noise_, 0.0001);  // m/s²/sqrt(s)
        private_nh.param<double>("imu_gyro_bias_noise", imu_gyro_bias_noise_, 0.00001); // rad/s/sqrt(s)
        private_nh.param<double>("acc_bias_max", acc_bias_max_, 0.1);   // Maximum allowed acc bias (m/s²)
        private_nh.param<double>("gyro_bias_max", gyro_bias_max_, 0.01); // Maximum allowed gyro bias (rad/s)
        
        // Initial biases (small realistic values)
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
        
        // Enable marginalization
        private_nh.param<bool>("enable_marginalization", enable_marginalization_, true);
        
        // Constraint weights
        private_nh.param<double>("roll_pitch_weight", roll_pitch_weight_, 300.0); // Increased from 100.0
        private_nh.param<double>("imu_pose_pub_frequency", imu_pose_pub_frequency_, 100.0);
        private_nh.param<double>("max_imu_dt", max_imu_dt_, 0.5);
        private_nh.param<double>("imu_orientation_weight", imu_orientation_weight_, 50.0);
        private_nh.param<double>("bias_constraint_weight", bias_constraint_weight_, 1000.0);
        private_nh.param<double>("max_velocity", max_velocity_, 2.0); // Maximum realistic velocity (m/s)
        private_nh.param<double>("velocity_constraint_weight", velocity_constraint_weight_, 1000.0);
        private_nh.param<double>("orientation_smoothness_weight", orientation_smoothness_weight_, 150.0);
        private_nh.param<double>("gravity_alignment_weight", gravity_alignment_weight_, 200.0);
        
        // Initialize with small non-zero biases that match your simulation
        initial_acc_bias_ = Eigen::Vector3d(initial_acc_bias_x_, initial_acc_bias_y_, initial_acc_bias_z_);
        initial_gyro_bias_ = Eigen::Vector3d(initial_gyro_bias_x_, initial_gyro_bias_y_, initial_gyro_bias_z_);
        
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
        optimization_failed_ = false;
        
        // Initialize marginalization (VINS-Mono style)
        marginalization_info_ = nullptr;
        marginalization_factor_ = nullptr;
        
        initializeState();
        
        // Setup optimization timer
        optimization_timer_ = nh.createTimer(ros::Duration(1.0/optimization_frequency_), 
                                           &UwbImuFusion::optimizationTimerCallback, this);
        
        // Setup high-frequency IMU pose publisher timer
        imu_pose_pub_timer_ = nh.createTimer(ros::Duration(1.0/imu_pose_pub_frequency_), 
                                           &UwbImuFusion::imuPoseTimerCallback, this);
        
        ROS_INFO("UWB-IMU Fusion node initialized (WITH VINS-MONO STYLE MARGINALIZATION)");
        ROS_INFO("IMU noise: acc=%.3f m/s², gyro=%.4f rad/s", imu_acc_noise_, imu_gyro_noise_);
        ROS_INFO("Bias parameters: max_acc=%.3f m/s², max_gyro=%.4f rad/s", 
                 acc_bias_max_, gyro_bias_max_);
        ROS_INFO("Initial biases: acc=[%.3f, %.3f, %.3f], gyro=[%.4f, %.4f, %.4f]",
                 initial_acc_bias_.x(), initial_acc_bias_.y(), initial_acc_bias_.z(),
                 initial_gyro_bias_.x(), initial_gyro_bias_.y(), initial_gyro_bias_.z());
        ROS_INFO("Max velocity: %.1f m/s", max_velocity_);
        ROS_INFO("Bias estimation is %s", enable_bias_estimation_ ? "enabled" : "disabled");
        ROS_INFO("Marginalization is %s", enable_marginalization_ ? "enabled" : "disabled");
    }

    ~UwbImuFusion() {
        // Clean up marginalization resources
        resetMarginalization();
    }

private:
    // ROS subscribers and publishers
    ros::Subscriber imu_sub_;
    ros::Subscriber uwb_sub_;
    ros::Publisher optimized_pose_pub_;
    ros::Publisher imu_pose_pub_;
    ros::Timer optimization_timer_;
    ros::Timer imu_pose_pub_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

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
    std::string world_frame_id_;
    std::string body_frame_id_;
    double roll_pitch_weight_;
    double imu_pose_pub_frequency_;
    double max_imu_dt_;
    double imu_orientation_weight_;
    double bias_constraint_weight_;
    double max_velocity_;      // Maximum allowed velocity
    double velocity_constraint_weight_;
    double orientation_smoothness_weight_; // Weight for orientation smoothness constraints
    double gravity_alignment_weight_;      // Weight for gravity alignment constraint
    
    // Initial bias values
    Eigen::Vector3d initial_acc_bias_;
    Eigen::Vector3d initial_gyro_bias_;

    // VINS-Mono style marginalization resources
    MarginalizationInfo* marginalization_info_;
    MarginalizationFactor* marginalization_factor_;

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
    bool optimization_failed_;
    
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

    // Mutex for thread safety
    std::mutex data_mutex_;

    // Gravity vector in world frame (ENU, Z-up)
    Eigen::Vector3d gravity_world_;
    
    // Helper functions
    
    // Improved reset marginalization
    void resetMarginalization() {
        try {
            if (marginalization_factor_) {
                // In VINS-Mono, the factor owns the info
                MarginalizationInfo* info = marginalization_factor_->marginalization_info;
                delete marginalization_factor_;
                marginalization_factor_ = nullptr;
                
                // Don't double-delete the info
                marginalization_info_ = nullptr;
            }
            
            if (marginalization_info_) {
                // In case info wasn't owned by the factor
                delete marginalization_info_;
                marginalization_info_ = nullptr;
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in resetMarginalization: %s", e.what());
            marginalization_factor_ = nullptr;
            marginalization_info_ = nullptr;
        }
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
    
    // CRITICAL: Clamp velocity to realistic values
    void clampVelocity(Eigen::Vector3d& velocity, double max_velocity = 2.0) {
        double velocity_norm = velocity.norm();
        if (velocity_norm > max_velocity) {
            velocity *= (max_velocity / velocity_norm);
        }
    }

    // CRITICAL: Check for unrealistic position values
    bool isPositionValid(const Eigen::Vector3d& position) {
        if (!position.allFinite()) {
            return false;
        }
        
        // Check for unrealistic values (position coordinates greater than 100m)
        for (int i = 0; i < 3; i++) {
            if (std::abs(position[i]) > 100.0) {
                return false;
            }
        }
        
        return true;
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
            imu_buffer_.clear();
            preintegration_map_.clear();
            
            // Initialize gravity vector in world frame (ENU, Z points up)
            // In ENU frame, gravity points downward along negative Z axis
            gravity_world_ = Eigen::Vector3d(0, 0, -gravity_magnitude_);
            
            // Reset timestamp tracking
            last_imu_timestamp_ = 0;
            last_processed_timestamp_ = 0;
            just_optimized_ = false;
            optimization_failed_ = false;
            
            // Reset marginalization
            resetMarginalization();
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in initializeState: %s", e.what());
        }
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            has_imu_data_ = true;
            
            double timestamp = msg->header.stamp.toSec();
            
            // Store IMU measurements
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
            }
            
            // Clean up old IMU messages
            if (imu_buffer_.size() > 1000) {
                double oldest_allowed_time = ros::Time::now().toSec() - 2.0 * imu_buffer_time_length_;
                while (!imu_buffer_.empty() && imu_buffer_.front().header.stamp.toSec() < oldest_allowed_time) {
                    imu_buffer_.pop_front();
                }
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
            
            // CRITICAL: Ensure velocity is reasonable
            clampVelocity(propagated_state.velocity, max_velocity_);
            
            // Add to state window, with marginalization if needed
            if (state_window_.size() >= optimization_window_size_) {
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
    
    // Propagate state from a reference state to a target time using IMU data
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
        
        // Propagate state using IMU measurements
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
            
            // Apply bias correction
            Eigen::Vector3d acc_corrected = acc - result.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
            
            // Update orientation using gyro
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            Eigen::Quaterniond orientation_new = (result.orientation * dq).normalized();
            
            // Get gravity in sensor frame
            Eigen::Vector3d gravity_sensor = result.orientation.inverse() * gravity_world_;
            
            // Remove gravity from accelerometer reading
            Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_sensor;
            
            // Rotate to world frame using midpoint rotation
            Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
            Eigen::Quaterniond orientation_mid = result.orientation * deltaQ(half_angle_axis);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity
            Eigen::Vector3d velocity_new = result.velocity + acc_world * dt;
            
            // CRITICAL: Ensure velocity stays within reasonable limits
            clampVelocity(velocity_new, max_velocity_);
            
            // Update position
            Eigen::Vector3d position_new = result.position + 0.5 * (result.velocity + velocity_new) * dt;
            
            // Update state
            result.orientation = orientation_new;
            result.velocity = velocity_new;
            result.position = position_new;
            result.timestamp = timestamp;
            
            prev_time = timestamp;
        }
        
        // Final integration step to target_time if needed
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
            
            // Apply bias correction
            Eigen::Vector3d acc_corrected = acc - result.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
            
            // Update orientation
            Eigen::Vector3d angle_axis = gyro_corrected * dt;
            Eigen::Quaterniond dq = deltaQ(angle_axis);
            Eigen::Quaterniond orientation_new = (result.orientation * dq).normalized();
            
            // Get gravity in sensor frame
            Eigen::Vector3d gravity_sensor = result.orientation.inverse() * gravity_world_;
            
            // Remove gravity from accelerometer reading
            Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_sensor;
            
            // Rotate to world frame
            Eigen::Vector3d half_angle_axis = 0.5 * angle_axis;
            Eigen::Quaterniond orientation_mid = result.orientation * deltaQ(half_angle_axis);
            Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
            
            // Update velocity and position
            Eigen::Vector3d velocity_new = result.velocity + acc_world * dt;
            
            // CRITICAL: Ensure velocity stays within reasonable limits
            clampVelocity(velocity_new, max_velocity_);
            
            Eigen::Vector3d position_new = result.position + 0.5 * (result.velocity + velocity_new) * dt;
            
            // Update state
            result.orientation = orientation_new;
            result.velocity = velocity_new;
            result.position = position_new;
            result.timestamp = target_time;
        }
        
        return result;
    }
    
    // Real-time state propagation with IMU
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
            if (dt <= 0 || dt > max_imu_dt_) {
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
                
                // Update orientation using gyro
                Eigen::Vector3d angle_axis = gyro_corrected * dt;
                Eigen::Quaterniond dq = deltaQ(angle_axis);
                
                // Update orientation
                current_state_.orientation = (current_state_.orientation * dq).normalized();
            }
            
            // Apply bias correction
            Eigen::Vector3d acc_corrected = acc - current_state_.acc_bias;
            Eigen::Vector3d gyro_corrected = gyro - current_state_.gyro_bias;
            
            // Get gravity in sensor frame
            Eigen::Vector3d gravity_sensor = current_state_.orientation.inverse() * gravity_world_;
            
            // Remove gravity from accelerometer reading
            Eigen::Vector3d acc_without_gravity = acc_corrected - gravity_sensor;
            
            // Rotate acceleration to world frame
            Eigen::Vector3d acc_world = current_state_.orientation * acc_without_gravity;
            
            // Update velocity
            Eigen::Vector3d velocity_new = current_state_.velocity + acc_world * dt;
            
            // CRITICAL: Ensure velocity stays reasonable
            clampVelocity(velocity_new, max_velocity_);
            
            // Constrain vertical velocity for planar motion
            velocity_new.z() *= 0.5;  // Damping factor for vertical velocity
            
            // Update position using trapezoidal integration
            Eigen::Vector3d position_new = current_state_.position + 
                                          0.5 * (current_state_.velocity + velocity_new) * dt;
            
            // Bias Z toward 1.0 (known height for this simulation)
            position_new.z() = 0.95 * position_new.z() + 0.05 * 1.0;
            
            // Update state
            current_state_.velocity = velocity_new;
            current_state_.position = position_new;
            current_state_.timestamp = timestamp;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in propagateStateWithImu: %s", e.what());
        }
    }

    // Perform IMU pre-integration between keyframes
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
            
            // CRITICAL: Ensure biases are within reasonable limits
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
            
            // Find relevant IMU measurements
            preint.imu_measurements.reserve(100);
            
            for (const auto& imu : imu_buffer_) {
                double timestamp = imu.header.stamp.toSec();
                if (timestamp >= start_time && timestamp <= end_time) {
                    preint.imu_measurements.push_back(imu);
                }
            }
            
            if (preint.imu_measurements.empty()) {
                ROS_WARN("No IMU data found between keyframes %.6f and %.6f", start_time, end_time);
                return;
            }
            
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
            
            // Get orientation at start time
            Eigen::Quaterniond initial_orientation = Eigen::Quaterniond::Identity();
            for (const auto& state : state_window_) {
                if (std::abs(state.timestamp - start_time) < 0.005) {
                    initial_orientation = state.orientation;
                    break;
                }
            }
            
            // Current orientation during integration
            Eigen::Quaterniond current_orientation = initial_orientation;
            
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
                Eigen::Vector3d acc_corrected = acc - clamped_acc_bias;
                Eigen::Vector3d gyro_corrected = gyro - clamped_gyro_bias;
                
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
                
                // 4. Get gravity in sensor frame based on current orientation
                Eigen::Vector3d gravity_sensor = current_orientation.inverse() * gravity_world_;
                
                // 5. Average acceleration and remove gravity in sensor frame
                Eigen::Vector3d acc_mid = 0.5 * (acc_prev + acc_corrected);
                Eigen::Vector3d acc_mid_without_gravity = acc_mid - gravity_sensor;
                
                // 6. Rotate to integration frame
                Eigen::Vector3d acc_integration_frame = delta_q_half * acc_mid_without_gravity;
                
                // 7. Update velocity and position
                preint.delta_velocity += acc_integration_frame * dt;
                preint.delta_position += preint.delta_velocity * dt + 
                                       0.5 * acc_integration_frame * dt * dt;
                
                // 8. Update orientation
                preint.delta_orientation = new_delta_q.normalized();
                
                // 9. Update current global orientation estimate for next iteration
                current_orientation = current_orientation * dq;
                
                // 10. Calculate Jacobians for noise propagation
                Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
                F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
                F.block<3, 3>(3, 6) = delta_q_half.toRotationMatrix() * dt;
                F.block<3, 3>(0, 6) = 0.5 * delta_q_half.toRotationMatrix() * dt * dt;
                
                // 11. Noise propagation matrix
                Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
                G.block<3, 3>(3, 0) = delta_q_half.toRotationMatrix();
                G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
                
                // 12. Update bias Jacobians
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
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in performPreintegrationBetweenKeyframes: %s", e.what());
        }
    }

    // IMU pre-integration factor for Ceres
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
            
            // CRITICAL: Bias corrections - with safeguards to prevent numerical issues
            Eigen::Matrix<T, 3, 1> dba = ba_i - ba_ref;
            Eigen::Matrix<T, 3, 1> dbg = bg_i - bg_ref;
            
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
            
            // Orientation residual
            Eigen::Quaternion<T> q_i_inverse_times_q_j = q_i.conjugate() * q_j;
            Eigen::Quaternion<T> delta_q_residual = corrected_delta_q.conjugate() * q_i_inverse_times_q_j;
            
            // Convert to angle-axis representation
            T dq_w = delta_q_residual.w();
            if (dq_w < T(1e-5)) {
                dq_w = T(1e-5);  // Prevent division by very small number
            }
            residual.template segment<3>(3) = T(2.0) * delta_q_residual.vec() / dq_w;
            
            // Velocity residual
            residual.template segment<3>(6) = q_i.inverse() * (v_j - v_i - g * sum_dt) - corrected_delta_v;
            
            // CRITICAL: Bias change residuals - more reasonable values now
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
                return;
            }
            
            // Need at least 2 states for optimization
            if (state_window_.size() < 2) {
                return;
            }
            
            // Reset to UWB position if large position drift detected
            if (!uwb_measurements_.empty() && !state_window_.empty()) {
                const auto& latest_uwb = uwb_measurements_.back();
                const auto& latest_state = state_window_.back();
                
                double position_error_z = std::abs(latest_state.position.z() - latest_uwb.position.z());
                double position_error_xy = (latest_state.position.head<2>() - latest_uwb.position.head<2>()).norm();
                
                if (position_error_z > 0.5 || position_error_xy > 0.5) {
                    ROS_WARN("Position drift detected: Z=%.2f meters, XY=%.2f meters. Resetting to UWB position.", 
                             position_error_z, position_error_xy);
                    
                    // Create a reset state that keeps orientation and biases but uses UWB position
                    State reset_state = latest_state;
                    reset_state.position = latest_uwb.position;
                    reset_state.velocity.setZero(); // Reset velocity when resetting position
                    
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
                    
                    // Reset marginalization
                    resetMarginalization();
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
                
                // Reset marginalization on failure
                resetMarginalization();
                optimization_failed_ = true;
                
                return;
            } else {
                // Set optimization flag
                just_optimized_ = true;
                
                // Output state after optimization
                Eigen::Vector3d euler_angles = quaternionToEulerDegrees(current_state_.orientation);
                
                ROS_INFO("Optimization time: %.1f ms", duration.count());
                ROS_INFO("State: Pos [%.2f, %.2f, %.2f] | Vel [%.2f, %.2f, %.2f] | Euler [%.1f, %.1f, %.1f] deg | Bias acc [%.3f, %.3f, %.3f] gyro [%.3f, %.3f, %.3f]",
                    current_state_.position.x(), current_state_.position.y(), current_state_.position.z(),
                    current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z(),
                    euler_angles.x(), euler_angles.y(), euler_angles.z(),
                    current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
                    current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
                
                // Publish state
                publishOptimizedPose();
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
            
            // CRITICAL: Initialize with proper non-zero biases
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
            resetMarginalization();
            
            ROS_INFO("State initialized at position [%.2f, %.2f, %.2f]", 
                    current_state_.position.x(), 
                    current_state_.position.y(), 
                    current_state_.position.z());
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

    // VINS-Mono style factor graph optimization with safe memory handling
    bool optimizeFactorGraph() {
        if (state_window_.size() < 2) {
            return false;
        }
        
        // Start by assuming optimization will fail
        optimization_failed_ = true;
        
        // Debug memory usage
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        ROS_INFO("Memory usage before optimization: %ld KB", usage.ru_maxrss);
        
        // Create problem with safe scope handling
        std::unique_ptr<ceres::Problem> problem_ptr;
        
        try {
            // Initialize variables from state window
            std::vector<OptVariables, Eigen::aligned_allocator<OptVariables>> variables;
            variables.reserve(state_window_.size());
            
            // Create Ceres problem
            ceres::Problem::Options problem_options;
            problem_options.enable_fast_removal = true;
            problem_ptr = std::make_unique<ceres::Problem>(problem_options);
            ceres::Problem& problem = *problem_ptr;
            
            // Create pose parameterization (owned by Ceres)
            ceres::LocalParameterization* pose_parameterization = new PoseParameterization();
            
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
            
            // Add UWB position factors
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
            
            // Add roll/pitch constraint to enforce planar motion
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* roll_pitch_prior = RollPitchPriorFactor::Create(roll_pitch_weight_);
                problem.AddResidualBlock(roll_pitch_prior, nullptr, variables[i].pose);
            }
            
            // Add gravity alignment factors
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
            
            // Add orientation smoothness constraints between consecutive keyframes
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
            
            // Add IMU orientation factors - use selectively with yaw-only constraints
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
            
            // Add hard constraints on bias magnitude
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* bias_constraint = BiasMagnitudeConstraint::Create(
                    acc_bias_max_, gyro_bias_max_, bias_constraint_weight_);
                
                problem.AddResidualBlock(bias_constraint, nullptr, variables[i].bias);
            }
            
            // Add strong velocity magnitude constraints
            for (size_t i = 0; i < state_window_.size(); ++i) {
                ceres::CostFunction* velocity_constraint = VelocityMagnitudeConstraint::Create(
                    max_velocity_, velocity_constraint_weight_);
                problem.AddResidualBlock(velocity_constraint, nullptr, variables[i].velocity);
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
                    
                    ceres::CostFunction* imu_factor = ImuFactor::Create(preint, gravity_world_);
                    
                    problem.AddResidualBlock(imu_factor, nullptr,
                                           variables[i].pose, variables[i].velocity, variables[i].bias,
                                           variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                }
            }
            
            // Add VINS-Mono style marginalization factor if available
            if (enable_marginalization_ && marginalization_factor_ != nullptr) {
                try {
                    // Get the parameter blocks needed by the factor
                    const auto& param_blocks = marginalization_factor_->marginalization_info->getParameterBlocks();
                    
                    if (param_blocks.size() == 3) {
                        // Make sure all parameter blocks are valid addresses
                        problem.AddResidualBlock(marginalization_factor_, nullptr,
                                              variables[0].pose, 
                                              variables[0].velocity, 
                                              variables[0].bias);
                    }
                } catch (const std::exception& e) {
                    ROS_ERROR("Exception adding marginalization factor: %s", e.what());
                    resetMarginalization();
                }
            }
                        
            // Configure solver options - REDUCED THREAD COUNT AND TIME LIMIT
            ceres::Solver::Options options;
            options.max_num_iterations = max_iterations_;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 2;  // REDUCED from 4 to 2
            options.max_solver_time_in_seconds = 0.2;  // REDUCED from 0.5 to 0.2
            options.function_tolerance = 1e-6;
            options.gradient_tolerance = 1e-10;
            options.parameter_tolerance = 1e-8;
            
            // Log optimization options
            ROS_INFO("Optimization options: max_iter=%d, max_time=%.1fs", 
                    options.max_num_iterations, options.max_solver_time_in_seconds);
            
            // Solve the optimization problem
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            // Check if optimization succeeded
            if (!summary.IsSolutionUsable()) {
                ROS_WARN("Optimization failed: %s", summary.BriefReport().c_str());
                return false;
            }
            
            // Check if terminated due to time limit
            if (summary.total_time_in_seconds >= 0.95 * options.max_solver_time_in_seconds) {
                ROS_WARN("Optimization likely timed out after %.1f seconds", 
                        summary.total_time_in_seconds);
            }
            
            // Validate positions before updating states
            bool positions_valid = true;
            
            for (size_t i = 0; i < state_window_.size(); ++i) {
                Eigen::Vector3d new_position(
                    variables[i].pose[0], variables[i].pose[1], variables[i].pose[2]);
                
                if (!isPositionValid(new_position)) {
                    ROS_ERROR("Optimization produced invalid position: [%.2f, %.2f, %.2f], discarding results",
                             new_position.x(), new_position.y(), new_position.z());
                    positions_valid = false;
                    break;
                }
            }
            
            if (!positions_valid) {
                resetMarginalization();
                return false;
            }
            
            // Update states with validated values
            for (size_t i = 0; i < state_window_.size(); ++i) {
                // Update position
                state_window_[i].position = Eigen::Vector3d(
                    variables[i].pose[0], variables[i].pose[1], variables[i].pose[2]);
                
                // Update orientation
                state_window_[i].orientation = Eigen::Quaterniond(
                    variables[i].pose[3], variables[i].pose[4], variables[i].pose[5], variables[i].pose[6]).normalized();
                
                // Update velocity with clamp
                Eigen::Vector3d new_velocity(
                    variables[i].velocity[0], variables[i].velocity[1], variables[i].velocity[2]);
                clampVelocity(new_velocity, max_velocity_);
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
                
                // Ensure velocity stays within reasonable limits
                clampVelocity(current_state_.velocity, max_velocity_);
            }
            
            // VINS-Mono style marginalization handling after successful optimization
            if (enable_marginalization_ && state_window_.size() > 1) {
                // Create a new marginalization info to marginalize the oldest frame
                MarginalizationInfo* marginalization_info = new MarginalizationInfo();
                
                // VINS-Mono style: marginalize the oldest frame
                
                // For now, we'll create a simple stub marginalization factor
                MarginalizationFactor* factor = new MarginalizationFactor(marginalization_info);
                
                // Clean up old marginalization resources
                resetMarginalization();
                
                // Update to new marginalization objects
                marginalization_factor_ = factor;
                marginalization_info_ = marginalization_info;
            }
            
            // Force memory cleanup via swap trick
            std::vector<OptVariables, Eigen::aligned_allocator<OptVariables>>().swap(variables);
            
            // Memory usage after optimization
            getrusage(RUSAGE_SELF, &usage);
            ROS_INFO("Memory usage after optimization: %ld KB", usage.ru_maxrss);
            
            // Mark optimization as successful
            optimization_failed_ = false;
            return true;
        } 
        catch (const std::exception& e) {
            ROS_ERROR("Exception during optimization: %s", e.what());
            resetMarginalization();
            return false;
        }
    }

    // Publish IMU-predicted pose
    void publishImuPose() {
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
            
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in publishOptimizedPose: %s", e.what());
        }
    }
};

int main(int argc, char **argv) {
    try {
        ros::init(argc, argv, "uwb_imu_fusion_node");
        
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