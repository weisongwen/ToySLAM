#include "factors.h"
#include "../include/MarginalizationInfo.h"
#include "utility.h"             

// ==================== RollPitchPriorFactor ====================

RollPitchPriorFactor::RollPitchPriorFactor(double weight) : weight_(weight) {}

template <typename T>
bool RollPitchPriorFactor::operator()(const T* const pose, T* residuals) const {
    // Extract quaternion from pose block (p, q), where q starts at index 3
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

ceres::CostFunction* RollPitchPriorFactor::Create(double weight) {
    return new ceres::AutoDiffCostFunction<RollPitchPriorFactor, 2, 7>(
        new RollPitchPriorFactor(weight));
}


// ==================== OrientationSmoothnessFactor ====================

OrientationSmoothnessFactor::OrientationSmoothnessFactor(double weight) : weight_(weight) {}

template <typename T>
bool OrientationSmoothnessFactor::operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
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

ceres::CostFunction* OrientationSmoothnessFactor::Create(double weight) {
    return new ceres::AutoDiffCostFunction<OrientationSmoothnessFactor, 1, 7, 7>(
        new OrientationSmoothnessFactor(weight));
}


// ==================== GpsPositionFactor ====================

GpsPositionFactor::GpsPositionFactor(const Eigen::Vector3d& measured_position, const Eigen::Matrix3d& covariance)
    : measured_position_(measured_position) {
    
    // Compute the square root of the information matrix (L^T)
    // 1. Information = Covariance⁻¹
    Eigen::Matrix3d information_matrix = covariance.inverse();
    
    // 2. Decompose Information = L * Lᵀ (Cholesky decomposition)
    Eigen::LLT<Eigen::Matrix3d> llt(information_matrix);

    // Handle cases where covariance is not positive definite (e.g., due to numerical issues)
    if (llt.info() == Eigen::NumericalIssue) {
        // Fallback to a matrix with very low weight (high uncertainty)
        sqrt_information_transpose_ = Eigen::Matrix3d::Identity() * 1e-6; 
    } else {
        // 3. We need Lᵀ to pre-multiply the error vector.
        // llt.matrixL() returns L, so we store its transpose.
        sqrt_information_transpose_ = llt.matrixL().transpose();
    }
}

template <typename T>
bool GpsPositionFactor::operator()(const T* const pose, T* residuals) const {
    // Extract position from the pose block (assuming first 3 elements are position x, y, z)
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(pose);
    
    // Compute the error vector e = (predicted - measured)
    Eigen::Matrix<T, 3, 1> error = position - measured_position_.template cast<T>();
    
    // Map the residuals array to an Eigen vector
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
    
    // MODIFIED: Compute the whitened residuals: r = Lᵀ * e
    // This correctly incorporates the full covariance, including off-diagonal terms.
    residuals_map = sqrt_information_transpose_.template cast<T>() * error;
    
    return true;
}

ceres::CostFunction* GpsPositionFactor::Create(const Eigen::Vector3d& measured_position, const Eigen::Matrix3d& covariance) {
    return new ceres::AutoDiffCostFunction<GpsPositionFactor, 3, 7>(
        new GpsPositionFactor(measured_position, covariance));
}


// ==================== GpsVelocityFactor ====================

GpsVelocityFactor::GpsVelocityFactor(const Eigen::Vector3d& measured_velocity, const Eigen::Matrix3d& covariance)
    : measured_velocity_(measured_velocity) {

    Eigen::Matrix3d information_matrix = covariance.inverse();
    Eigen::LLT<Eigen::Matrix3d> llt(information_matrix);

    if (llt.info() == Eigen::NumericalIssue) {
        sqrt_information_transpose_ = Eigen::Matrix3d::Identity() * 1e-6;
    } else {
        sqrt_information_transpose_ = llt.matrixL().transpose();
    }
}

template <typename T>
bool GpsVelocityFactor::operator()(const T* const velocity, T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> vel(velocity);
        
    Eigen::Matrix<T, 3, 1> error = vel - measured_velocity_.template cast<T>();
    
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
    
    // MODIFIED: Compute the whitened residuals: r = Lᵀ * e
    residuals_map = sqrt_information_transpose_.template cast<T>() * error;
    
    return true;
}

ceres::CostFunction* GpsVelocityFactor::Create(const Eigen::Vector3d& measured_velocity, const Eigen::Matrix3d& covariance) {
    return new ceres::AutoDiffCostFunction<GpsVelocityFactor, 3, 3>(
        new GpsVelocityFactor(measured_velocity, covariance));
}


// ==================== MarginalizationFactor ====================

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* marginalization_info)
    : marginalization_info_(marginalization_info) {
const Eigen::VectorXd& r = marginalization_info_->getLinearizedResiduals();
        
    // Set residual size
    set_num_residuals(r.size());
    
    // CRITICAL: Always expect exactly 6 parameter blocks with fixed sizes
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(7); // pose1 (position + quaternion)
    mutable_parameter_block_sizes()->push_back(3); // velocity1
    mutable_parameter_block_sizes()->push_back(6); // bias1 (acc + gyro)
}

bool MarginalizationFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    const Eigen::VectorXd& linearized_residuals = marginalization_info_->getLinearizedResiduals();
    const Eigen::MatrixXd& linearized_jacobians = marginalization_info_->getLinearizedJacobians();

    int n = marginalization_info_->getKeepBlockSize();
    Eigen::VectorXd dx(n);

    for(int i = 0; i<static_cast<int>(marginalization_info_->getKeepBlockSizes().size()); i++){
        int size = marginalization_info_->getKeepBlockSizes()[i];
        int idx = marginalization_info_->getKeepBlockIdx()[i];

        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info_->getKeepBlockData()[i], size);
        // ROS_INFO_STREAM("MarginalizationFactor: size: " << size << ", idx: " << idx << ", x: " << x.transpose() << ", x0: " << x0.transpose());
        if (size != 7) 
            dx.segment(idx, size) = x - x0;
        else {
            dx.segment<3>(idx) = x.head<3>() - x0.head<3>(); // position
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info_->getLinearizedResiduals() + marginalization_info_->getLinearizedJacobians() * dx;
    // ROS_INFO_STREAM("Residualas: "<<marginalization_info->getLinearizedResiduals() + marginalization_info->getLinearizedJacobians() * dx);


    // Fill residuals
    // for (int i = 0; i < num_residuals() && i < linearized_residuals.size(); i++) {
    //     residuals[i] = linearized_residuals(i);
    // }

    // Check if jacobians are requested
    if (!jacobians) {
        return true; // No jacobians requested, just return
    }

    for(int i=0; i<static_cast<int>(marginalization_info_->getKeepBlockSizes().size()); i++){
        // ROS_INFO_STREAM("MarginalizationFactor: Jacobian for block " << i << ", size: " << marginalization_info->getKeepBlockSizes()[i]);
        if (!jacobians[i]) {
            continue; // Skip null jacobians
        }
        int size = marginalization_info_->getKeepBlockSizes()[i], local_size = marginalization_info_->localSize(size);
        int idx = marginalization_info_->getKeepBlockIdx()[i];
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) = marginalization_info_->getLinearizedJacobians().middleCols(idx, local_size);

        // ROS_INFO_STREAM("Jacobian for block " << i << " value:"  << jacobian);
    }
    
    return true;
}
