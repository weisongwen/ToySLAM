#include "math_helpers.h"
#include <cmath>
#include <algorithm>


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
        q = q * Utility::deltaQ(omega_integrated);
    }
    q.normalize();  // Ensure quaternion stays normalized
}

// Helper for RK4 integration - compute omega_dot (angular acceleration)
Eigen::Vector3d omegaDot(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2) {
    // Simple linear approximation of angular acceleration
    return (omega2 - omega1);
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

// Compute quaternion for small angle rotation
template <typename T>
Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta) {
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
Eigen::Quaterniond deltaQ(const Eigen::Vector3d& theta) {
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

