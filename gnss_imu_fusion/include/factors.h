#ifndef FACTORS_H
#define FACTORS_H

#include <Eigen/Dense>
#include "ceres/ceres.h"

// Forward-declare MarginalizationInfo to avoid including its full header here.
class MarginalizationInfo;

/**
 * @class RollPitchPriorFactor
 * @brief Adds a cost to penalize non-zero roll and pitch, enforcing planar motion.
 * Assumes the sensor operates in an ENU (East-North-Up) frame where gravity is aligned with the z-axis.
 */
class RollPitchPriorFactor {
public:
    RollPitchPriorFactor(double weight);

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const;

    static ceres::CostFunction* Create(double weight = 300.0);

private:
    double weight_;
};

/**
 * @class OrientationSmoothnessFactor
 * @brief Penalizes large, abrupt changes in orientation between two consecutive poses.
 */
class OrientationSmoothnessFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OrientationSmoothnessFactor(double weight);

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const;

    static ceres::CostFunction* Create(double weight = 150.0);

private:
    double weight_;
};


// ==================== GPS-RELATED FACTORS ====================

/**
 * @class GpsPositionFactor
 * @brief Adds a cost based on the difference between a measured GPS position and the state's position.
 * The factor correctly handles multi-dimensional uncertainty by using the Cholesky decomposition
 * of the information matrix (inverse covariance).
 */
class GpsPositionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GpsPositionFactor(const Eigen::Vector3d& measured_position, const Eigen::Matrix3d& covariance);

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const;

    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_position, const Eigen::Matrix3d& covariance);

private:
    const Eigen::Vector3d measured_position_;
    Eigen::Matrix3d sqrt_information_transpose_; // Stores Lᵀ from Information = L*Lᵀ
};

/**
 * @class GpsVelocityFactor
 * @brief Adds a cost based on the difference between a measured GPS velocity and the state's velocity.
 * Handles uncertainty using the Cholesky decomposition of the information matrix.
 */
class GpsVelocityFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GpsVelocityFactor(const Eigen::Vector3d& measured_velocity, const Eigen::Matrix3d& covariance);

    template <typename T>
    bool operator()(const T* const velocity, T* residuals) const;

    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_velocity, const Eigen::Matrix3d& covariance);

private:
    const Eigen::Vector3d measured_velocity_;
    Eigen::Matrix3d sqrt_information_transpose_;
};


// ==================== MARGINALIZATION FACTOR ====================

/**
 * @class MarginalizationFactor
 * @brief A custom Ceres factor that applies the constraint derived from the Schur complement
 * during the marginalization process. This acts as a prior on the kept states.
 */
class MarginalizationFactor : public ceres::CostFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit MarginalizationFactor(MarginalizationInfo* marginalization_info);

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

private:
    MarginalizationInfo* marginalization_info_;
};



/**
 * @class UwbPositionFactor
 * @brief Adds a cost based on the difference between a UWB position measurement and the state's position.
 * Assumes isotropic noise (the same uncertainty in all directions).
 */
class UwbPositionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    UwbPositionFactor(const Eigen::Vector3d& measured_position, double noise_std)
        : measured_position_(measured_position), noise_std_(noise_std) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Extract position from pose block (first 3 elements)
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> position(pose);
        
        // Compute the whitened residual: (predicted - measured) / noise_std
        residuals[0] = (position[0] - T(measured_position_[0])) / T(noise_std_);
        residuals[1] = (position[1] - T(measured_position_[1])) / T(noise_std_);
        residuals[2] = (position[2] - T(measured_position_[2])) / T(noise_std_);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& measured_position, double noise_std){
            return new ceres::AutoDiffCostFunction<UwbPositionFactor, 3, 7>(
            new UwbPositionFactor(measured_position, noise_std));
    }
    
private:
    const Eigen::Vector3d measured_position_;
    const double noise_std_;
};


#endif // FACTORS_H
