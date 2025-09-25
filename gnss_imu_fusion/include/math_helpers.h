#ifndef MATH_HELPERS_H
#define MATH_HELPERS_H

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <limits>
#include "../include/utility.h" // For deltaQ


/**
 * @brief 使用四阶龙格-库塔方法积分更新姿态四元数。
 * @param omega1 起始时刻的角速度。
 * @param omega2 结束时刻的角速度。
 * @param dt 时间间隔。
 * @param q 被更新的四元数（输入/输出）。
 */
void rk4IntegrateOrientation(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2,
                             const double dt, Eigen::Quaterniond& q);

/**
 * @brief RK4积分的辅助函数，计算角加速度（omega_dot）的简单线性近似。
 * @param omega1 起始时刻的角速度。
 * @param omega2 结束时刻的角速度。
 * @return (omega2 - omega1)。
 */
Eigen::Vector3d omegaDot(const Eigen::Vector3d& omega1, const Eigen::Vector3d& omega2);

/**
 * @brief 将四元数转换为欧拉角（单位：度）。
 * @param q 输入的四元数。
 * @return 包含[roll, pitch, yaw]的Eigen::Vector3d，单位为度。
 */
Eigen::Vector3d quaternionToEulerDegrees(const Eigen::Quaterniond& q);

Eigen::Quaterniond deltaQ(const Eigen::Vector3d& theta);

template <typename T>
Eigen::Quaternion<T> deltaQ(const Eigen::Matrix<T, 3, 1>& theta);

#endif // MATH_HELPERS_H