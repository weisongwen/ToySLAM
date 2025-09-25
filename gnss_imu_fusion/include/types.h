#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <Eigen/Geometry>


struct State {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyro_bias;
    double timestamp;
    
    bool has_gps_pos_factor = false;
    bool has_gps_vel_factor = false;
    Eigen::Matrix3d final_gps_pos_cov;
    Eigen::Matrix3d final_gps_vel_cov;
};

// struct ErrorStats {
//     double position_error_e = 0.0;
//     double position_error_n = 0.0;
//     double position_error_u = 0.0;
//     double position_error_norm = 0.0;
//     double velocity_error_e = 0.0;
//     double velocity_error_n = 0.0;
//     double velocity_error_u = 0.0;
//     double velocity_error_norm = 0.0;
//     double timestamp = 0.0;
// };

struct OptVariables {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double pose[7]; // position (3) + quaternion (4)
    double velocity[3];
    double bias[6]; // acc_bias (3) + gyro_bias (3)
};

// UWB measurements
struct UwbMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d position;
    double timestamp;
};

// Define colors and directions for each component
struct ComponentInfo {
    Eigen::Vector3d direction;
    std::array<float, 3> color;
    int index;
};

#endif // TYPES_H