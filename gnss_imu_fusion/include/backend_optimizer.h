#ifndef BACKEND_OPTIMIZER_H
#define BACKEND_OPTIMIZER_H

#include <ros/ros.h>
#include <ceres/ceres.h>
#include <deque>
#include <map>
#include <vector>
#include <sensor_msgs/Imu.h>

#include "../include/parameter_server.h"
#include "../include/types.h"           // For State struct
#include "../include/imu_preint.h"
#include "../include/ResidualBlockInfo.h"
#include "../include/MarginalizationInfo.h"
#include "../include/ceres_logger.h"
#include "../include/factors.h"
#include "../include/imu_factor.h"
#include "../include/utility.h"
#include "../include/gnss_parser.h"
#include "../include/math_helpers.h"

// Custom parameterization for pose (position + quaternion)
class PoseParameterization : public ceres::LocalParameterization {
public:
    virtual ~PoseParameterization() {}

    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }

    bool Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        
        Eigen::Map<const Eigen::Vector3d> _p(x);
        // ROS_INFO_STREAM("p: " << _p.transpose());
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        Eigen::Map<const Eigen::Vector3d> dp(delta);

        Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();

        return true;
    }

    bool ComputeJacobian(const double *x, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }
};

class BackendOptimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 构造函数，接收参数服务器的引用
    BackendOptimizer(ParameterServer& params);
    
    // 析构函数，用于清理资源
    ~BackendOptimizer() = default;

    /**
     * @brief 对滑动窗口中的状态执行因子图优化
     * @param state_window 滑动窗口中的状态队列 (将被就地修改)
     * @param gps_measurements 当前窗口中所有有效的GPS测量
     * @param preintegration_map 关键帧之间的IMU预积分数据
     * @param last_marginalization_info 指向上一次边缘化信息的指针 (将被修改)
     * @param optimization_count 优化执行次数的计数器 (将被修改)
     * @return 如果优化成功且解可用，则返回 true
     */
    bool optimize(
        std::deque<State, Eigen::aligned_allocator<State>>& state_window,
        const std::vector<GnssMeasurement>& gps_measurements,
        std::map<std::pair<double, double>, imu_preint>& preintegration_map,
        MarginalizationInfo*& last_marginalization_info,
        int& optimization_count,
        const std::deque<sensor_msgs::Imu>& imu_buffer 
    );


    void prepareMarginalization(
        const std::deque<State, Eigen::aligned_allocator<State>>& state_window,
        const std::vector<GnssMeasurement>& gps_measurements,
        const std::map<std::pair<double, double>, imu_preint>& preintegration_map,
        MarginalizationInfo*& last_marginalization_info
    );

    State propagateState(
        const State& reference_state, 
        double target_time, 
        const std::deque<sensor_msgs::Imu>& imu_buffer
    );

    void clampBiases(Eigen::Vector3d& acc_bias, Eigen::Vector3d& gyro_bias);
    void clampVelocity(Eigen::Vector3d& velocity, double max_velocity);
    double estimateMaxVelocityFromImu(const std::deque<sensor_msgs::Imu>& imu_buffer);

private:
    /**
     * @brief 准备边缘化，为下一次优化创建先验因子
     * @param state_window 当前的滑动窗口状态
     * @param gps_measurements GPS测量数据
     * @param preintegration_map IMU预积分数据
     * @param last_marginalization_info 边缘化信息的指针
     */


    
    ParameterServer& params_;
    CeresLogger logger_;
    Eigen::Vector3d gravity_world_;
};

#endif // BACKEND_OPTIMIZER_H