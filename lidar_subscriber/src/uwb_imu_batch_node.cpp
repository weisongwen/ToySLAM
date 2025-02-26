#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <mutex>

using namespace Eigen;

// Constants
constexpr double GRAVITY = 9.81;
constexpr int STATE_DIM = 16; // [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]

struct State {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector3d p = Vector3d::Zero();
    Vector3d v = Vector3d::Zero();
    Quaterniond q = Quaterniond::Identity();
    Vector3d ba = Vector3d::Zero();
    Vector3d bg = Vector3d::Zero();
    double timestamp = 0.0;

    double* data() { return reinterpret_cast<double*>(this); }
    const double* data() const { return reinterpret_cast<const double*>(this); }
};

class IMUPreintegrator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IMUPreintegrator(const Vector3d& acc_noise, const Vector3d& gyro_noise) 
        : acc_noise_sigma(acc_noise), gyro_noise_sigma(gyro_noise) {
        Reset();
    }

    void Reset() {
        delta_p.setZero();
        delta_v.setZero();
        delta_q.setIdentity();
        covariance.setZero();
        Jacobian.setIdentity();
        J_p_ba.setZero();
        J_p_bg.setZero();
        J_v_ba.setZero();
        J_v_bg.setZero();
        J_q_bg.setZero();
        dt = 0.0;
    }

    void Integrate(const Vector3d& acc, const Vector3d& gyro, double delta_t) {
        const Matrix3d R = delta_q.toRotationMatrix();
        const Vector3d acc_unbias = acc - ba;
        const Vector3d gyro_unbias = gyro - bg;
        const Vector3d delta_angle = gyro_unbias * delta_t;

        // State update
        delta_p += delta_v * delta_t + 0.5 * R * acc_unbias * delta_t * delta_t;
        delta_v += R * acc_unbias * delta_t;
        delta_q *= Quaterniond(AngleAxisd(delta_angle.norm(), delta_angle.normalized()));
        delta_q.normalize();

        // Jacobian calculations
        J_p_ba = -R * delta_t * delta_t;
        J_p_bg = 0.5 * R * delta_t * delta_t * skew(acc_unbias);
        J_v_ba = -R * delta_t;
        J_v_bg = R * delta_t * skew(acc_unbias);
        J_q_bg = -Matrix3d::Identity() * delta_t;

        // Covariance propagation
        Matrix<double, 15, 15> F = Matrix<double, 15, 15>::Identity();
        F.block<3, 3>(0, 3) = Matrix3d::Identity() * delta_t;
        F.block<3, 3>(3, 6) = -R * skew(acc_unbias) * delta_t;
        F.block<3, 3>(3, 9) = -R * delta_t;
        F.block<3, 3>(6, 6) = AngleAxisd(delta_angle.norm(), delta_angle.normalized()).toRotationMatrix().transpose();
        F.block<3, 3>(6, 12) = -Matrix3d::Identity() * delta_t;

        Matrix<double, 15, 12> G = Matrix<double, 15, 12>::Zero();
        G.block<3, 3>(3, 0) = R * delta_t;
        G.block<3, 3>(6, 3) = Matrix3d::Identity() * delta_t;

        covariance = F * covariance * F.transpose() + G * noise_cov * G.transpose();
        Jacobian = F * Jacobian;
        dt += delta_t;
    }

    static Matrix3d skew(const Vector3d& v) {
        Matrix3d m;
        m << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return m;
    }

    // Pre-integrated measurements
    Vector3d delta_p;
    Vector3d delta_v;
    Quaterniond delta_q;
    Matrix<double, 15, 15> covariance;
    Matrix<double, 15, 15> Jacobian;
    double dt;
    
    // Jacobians
    Matrix3d J_p_ba, J_p_bg, J_v_ba, J_v_bg, J_q_bg;

    // Biases
    Vector3d ba = Vector3d::Zero();
    Vector3d bg = Vector3d::Zero();

private:
    Vector3d acc_noise_sigma;
    Vector3d gyro_noise_sigma;
    Matrix<double, 12, 12> noise_cov;
};

struct IMUFactor {
    IMUFactor(const IMUPreintegrator& preint, const Matrix<double, 15, 15>& info)
        : preint_(preint), sqrt_info_(info.llt().matrixL().transpose()) {}

    template <typename T>
    bool operator()(const T* const state_i, const T* const state_j, T* residuals) const {
        // State parameters
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pi(state_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vi(state_i + 3);
        Eigen::Map<const Eigen::Quaternion<T>> qi(state_i + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bai(state_i + 10);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bgi(state_i + 13);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pj(state_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vj(state_j + 3);
        Eigen::Map<const Eigen::Quaternion<T>> qj(state_j + 6);

        // Bias updates
        Eigen::Matrix<T, 3, 1> delta_ba = bai - preint_.ba.cast<T>();
        Eigen::Matrix<T, 3, 1> delta_bg = bgi - preint_.bg.cast<T>();

        // Correct pre-integration
        Eigen::Matrix<T, 3, 1> corrected_delta_p = preint_.delta_p.cast<T>() + 
            preint_.J_p_ba.cast<T>() * delta_ba + preint_.J_p_bg.cast<T>() * delta_bg;
        Eigen::Matrix<T, 3, 1> corrected_delta_v = preint_.delta_v.cast<T>() + 
            preint_.J_v_ba.cast<T>() * delta_ba + preint_.J_v_bg.cast<T>() * delta_bg;
        
        Eigen::Quaternion<T> corrected_delta_q = preint_.delta_q.cast<T>() * 
            Eigen::Quaternion<T>(Eigen::AngleAxis<T>(
                (preint_.J_q_bg * delta_bg.template cast<double>()).norm(),
                (preint_.J_q_bg * delta_bg.template cast<double>()).normalized().cast<T>()
            ));

        // Compute residuals
        Eigen::Matrix<T, 3, 1> r_p = qi.conjugate() * (pj - pi - vi * T(preint_.dt) + 
            T(0.5) * Eigen::Matrix<T, 3, 1>(T(0), T(0), T(GRAVITY)) * T(preint_.dt * preint_.dt)) - corrected_delta_p;
        
        Eigen::Matrix<T, 3, 1> r_v = qi.conjugate() * (vj - vi + 
            Eigen::Matrix<T, 3, 1>(T(0), T(0), T(GRAVITY)) * T(preint_.dt)) - corrected_delta_v;
        
        Eigen::Quaternion<T> r_q = corrected_delta_q.conjugate() * (qi.conjugate() * qj);
        Eigen::Matrix<T, 3, 1> r_ba = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(state_j + 10) - bai;
        Eigen::Matrix<T, 3, 1> r_bg = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(state_j + 13) - bgi;

        // Combine residuals
        Eigen::Map<Eigen::Matrix<T, 15, 1>> r(residuals);
        r << r_p, r_v, T(2) * r_q.vec(), r_ba, r_bg;
        r = sqrt_info_.cast<T>() * r;

        return true;
    }

private:
    const IMUPreintegrator& preint_;
    const Matrix<double, 15, 15> sqrt_info_;
};

struct UWBFactor {
    UWBFactor(const Vector3d& z, const Matrix3d& info)
        : z_(z), sqrt_info_(info.llt().matrixL().transpose()) {}

    template <typename T>
    bool operator()(const T* const state, T* residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(state);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> r(residuals);
        r = p - z_.cast<T>();
        r = sqrt_info_.cast<T>() * r;
        return true;
    }

private:
    Vector3d z_;
    Matrix3d sqrt_info_;
};

struct PriorFactor {
    PriorFactor(const Vector3d& prior, const Matrix3d& info)
        : prior_(prior), sqrt_info_(info.llt().matrixL().transpose()) {}

    template <typename T>
    bool operator()(const T* const state, T* res) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(res);
        residuals = Eigen::Matrix<T, 3, 1>(state[0], state[1], state[2]) - prior_.cast<T>();
        residuals = sqrt_info_.cast<T>() * residuals;
        return true;
    }

private:
    Vector3d prior_;
    Matrix3d sqrt_info_;
};

class FusionNode {
public:
    FusionNode() : preintegrator(Vector3d(0.1, 0.1, 0.1), Vector3d(0.01, 0.01, 0.01)) {
        options.max_num_iterations = 10;
        options.linear_solver_type = ceres::DENSE_QR;
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

        if (!imu_buffer_.empty()) {
            double dt = msg->header.stamp.toSec() - imu_buffer_.back()->header.stamp.toSec();
            preintegrator.Integrate(acc, gyro, dt);
        }
        imu_buffer_.push_back(msg);
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        if (states_.empty()) {
            // Initialize first state
            State initial_state;
            initial_state.p << msg->point.x, msg->point.y, msg->point.z;
            initial_state.timestamp = msg->header.stamp.toSec();
            states_.push_back(initial_state);

            // Add prior factor
            problem_.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorFactor, 3, STATE_DIM>(
                    new PriorFactor(initial_state.p, Matrix3d::Identity() * 1e-6)),
                nullptr,
                states_.back().data()
            );
            return;
        }

        // Create new state
        State new_state;
        new_state.timestamp = msg->header.stamp.toSec();

        // Add IMU factor
        if (states_.size() >= 1) {
            Matrix<double, 15, 15> info = preintegrator.covariance.inverse();
            problem_.AddResidualBlock(
                new ceres::AutoDiffCostFunction<IMUFactor, 15, STATE_DIM, STATE_DIM>(
                    new IMUFactor(preintegrator, info)),
                nullptr,
                states_.back().data(),
                new_state.data()
            );
        }

        // Add UWB factor
        problem_.AddResidualBlock(
            new ceres::AutoDiffCostFunction<UWBFactor, 3, STATE_DIM>(
                new UWBFactor(Vector3d(msg->point.x, msg->point.y, msg->point.z), Matrix3d::Identity()/0.1)),
            nullptr,
            new_state.data()
        );

        // Set quaternion parameterization
        ceres::LocalParameterization* quaternion_param = 
            new ceres::EigenQuaternionParameterization;
        problem_.SetParameterization(new_state.data() + 6, quaternion_param);

        // Solve optimization
        ceres::Solve(options_, &problem_, &summary_);

        // Update states
        states_.push_back(new_state);
        preintegrator.Reset();
    }

private:
    std::vector<State, Eigen::aligned_allocator<State>> states_;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
    IMUPreintegrator preintegrator;
    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;
    std::mutex mtx_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_imu_fusion");
    ros::NodeHandle nh;
    
    FusionNode node;
    
    ros::Subscriber imu_sub = nh.subscribe("/imu", 1000, &FusionNode::imuCallback, &node);
    ros::Subscriber uwb_sub = nh.subscribe("/uwb", 1000, &FusionNode::uwbCallback, &node);

    ros::spin();
    return 0;
}