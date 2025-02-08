#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <queue>
#include <mutex>

using namespace Eigen;

typedef Matrix<double, 9, 9> Matrix9d;
typedef Matrix<double, 9, 1> Vector9d;

// IMU Noise parameters
const double ACCEL_NOISE_SIGMA = 0.1;
const double GYRO_NOISE_SIGMA = 0.01;
const double ACCEL_BIAS_SIGMA = 0.001;
const double GYRO_BIAS_SIGMA = 0.001;

double GRAVITY = 9.81;

struct State {
    double timestamp;
    Vector3d p;  // Position
    Vector3d v;  // Velocity
    Quaterniond q;  // Orientation
    Vector3d accel_bias;
    Vector3d gyro_bias;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class ImuPreintegrator {
public:
    struct PreintResult {
        Matrix3d delta_R;
        Vector3d delta_v;
        Vector3d delta_p;
        double delta_t;
        
        // Jacobian matrices
        Matrix3d dR_dbg;  // ∂ΔR/∂Δbg
        Matrix3d dV_dba;  // ∂Δv/∂Δba
        Matrix3d dV_dbg;  // ∂Δv/∂Δbg
        Matrix3d dP_dba;  // ∂Δp/∂Δba
        Matrix3d dP_dbg;  // ∂Δp/∂Δbg
        
        // Covariance matrix
        Matrix<double, 9, 9> covariance;
    };

    ImuPreintegrator() : acc_noise(1e-3), gyro_noise(1e-4) {
        Reset(Vector3d::Zero(), Vector3d::Zero());
    }

    void Reset(const Vector3d& ba, const Vector3d& bg) {
        result.delta_R.setIdentity();
        result.delta_v.setZero();
        result.delta_p.setZero();
        result.delta_t = 0.0;
        result.dR_dbg.setZero();
        result.dV_dba.setZero();
        result.dV_dbg.setZero();
        result.dP_dba.setZero();
        result.dP_dbg.setZero();
        result.covariance.setZero();
        acc_bias = ba;
        gyro_bias = bg;
    }

    // void Integrate(const sensor_msgs::Imu::ConstPtr& imu, double dt) {
    void integrate(const Vector3d& acc, const Vector3d& omega, double dt) {
        // const Vector3d omega(imu->angular_velocity.x - gyro_bias.x(),
        //                     imu->angular_velocity.y - gyro_bias.y(),
        //                     imu->angular_velocity.z - gyro_bias.z());
        
        // const Vector3d acc(imu->linear_acceleration.x - acc_bias.x(),
        //                   imu->linear_acceleration.y - acc_bias.y(),
        //                   imu->linear_acceleration.z - acc_bias.z());

        // Pre-integration terms
        const Matrix3d delta_R = AngleAxisd(omega.norm() * dt, omega.normalized()).toRotationMatrix();
        const Vector3d delta_v = result.delta_R * acc * dt;
        const Vector3d delta_p = result.delta_v * dt + 0.5 * result.delta_R * acc * dt * dt;

        // Jacobian updates
        const Matrix3d I = Matrix3d::Identity();
        const Matrix3d R = result.delta_R;
        const Matrix3d acc_skew = skew(acc);
        
        result.dR_dbg = delta_R.transpose() * result.dR_dbg - R * dt * acc_skew;
        result.dV_dba += R * dt;
        result.dV_dbg += -R * acc_skew * dt * dt;
        result.dP_dba += 0.5 * R * dt * dt;
        result.dP_dbg += -0.5 * R * acc_skew * dt * dt * dt;

        // Update pre-integration results
        result.delta_R *= delta_R;
        result.delta_v += delta_v;
        result.delta_p += delta_p;
        result.delta_t += dt;

        // Covariance propagation (simplified)
        Matrix<double, 9, 9> F = Matrix<double, 9, 9>::Identity();
        F.block<3, 3>(0, 0) = delta_R.transpose();
        F.block<3, 3>(3, 0) = -R * acc_skew * dt;
        F.block<3, 3>(6, 0) = -0.5 * R * acc_skew * dt * dt;
        F.block<3, 3>(6, 3) = I * dt;

        Matrix<double, 9, 6> G = Matrix<double, 9, 6>::Zero();
        G.block<3, 3>(0, 0) = I * dt;
        G.block<3, 3>(3, 3) = R * dt;
        G.block<3, 3>(6, 3) = 0.5 * R * dt * dt;

        Matrix<double, 6, 6> W = Matrix<double, 6, 6>::Identity();
        W.block<3, 3>(0, 0) *= gyro_noise * dt * dt;
        W.block<3, 3>(3, 3) *= acc_noise * dt * dt;

        result.covariance = F * result.covariance * F.transpose() + G * W * G.transpose();
    }

    PreintResult result;

    Vector3d acc_bias;
    Vector3d gyro_bias;
    double acc_noise;
    double gyro_noise;

private:
    

    Matrix3d skew(const Vector3d& v) {
        Matrix3d S;
        S << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return S;
    }
};

class ImuFactor : public ceres::SizedCostFunction<15, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3> {
public:
    ImuFactor(const ImuPreintegrator::PreintResult& preint) : preint(preint) {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        // Previous state
        Vector3d p_i(parameters[0]);
        Vector3d v_i(parameters[1]);
        Quaterniond q_i(parameters[2][3], parameters[2][0], parameters[2][1], parameters[2][2]);
        Vector3d ba_i(parameters[3]);
        Vector3d bg_i(parameters[4]);

        // Current state
        Vector3d p_j(parameters[5]);
        Vector3d v_j(parameters[6]);
        Quaterniond q_j(parameters[7][3], parameters[7][0], parameters[7][1], parameters[7][2]);
        Vector3d ba_j(parameters[8]);
        Vector3d bg_j(parameters[9]);

        // Compute residuals
        Matrix3d R_i = q_i.toRotationMatrix();
        Matrix3d R_j = q_j.toRotationMatrix();

        // Rotation residual
        Matrix3d R_res = preint.delta_R.transpose() * R_i.transpose() * R_j;
        AngleAxisd aa(R_res);
        Vector3d r_R = aa.angle() * aa.axis();

        // Velocity residual
        Vector3d r_v = v_j - (v_i + R_i * preint.delta_v + Vector3d(0, 0, GRAVITY) * preint.delta_t);

        // Position residual
        Vector3d r_p = p_j - (p_i + v_i * preint.delta_t + 
                            0.5 * Vector3d(0, 0, GRAVITY) * preint.delta_t * preint.delta_t +
                            R_i * preint.delta_p);

        // Bias residuals
        Vector3d r_ba = ba_j - ba_i;
        Vector3d r_bg = bg_j - bg_i;

        // Assign residuals
        // Map<Vector3d>(residuals) = r_R;
        residuals[0] = r_R(0);
        residuals[1] = r_R(1);
        residuals[2] = r_R(2);

        // Map<Vector3d>(residuals+3) = r_v;
        residuals[3] = r_v(3);
        residuals[4] = r_v(4);
        residuals[5] = r_v(5);

        // Map<Vector3d>(residuals+6) = r_p;
        residuals[6] = r_p(6);
        residuals[7] = r_p(7);
        residuals[8] = r_p(8);

        // Map<Vector3d>(residuals+9) = r_ba;
        residuals[9] = r_ba(9);
        residuals[10] = r_ba(10);
        residuals[11] = r_ba(11);

        // Map<Vector3d>(residuals+12) = r_bg;
        residuals[12] = r_bg(12);
        residuals[13] = r_bg(13);
        residuals[14] = r_bg(14);

        // Jacobian calculations
        if (jacobians) {
            const Matrix3d I = Matrix3d::Identity();
            const Matrix3d O = Matrix3d::Zero();

            // Previous state Jacobians
            if (jacobians[0]) { // ∂r/∂p_i
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[0]);
                J.setZero();
                J.block<3, 3>(6, 0) = -I;
            }
            if (jacobians[1]) { // ∂r/∂v_i
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3, 3>(3, 0) = -I;
                J.block<3, 3>(6, 0) = -I * preint.delta_t;
            }
            if (jacobians[2]) { // ∂r/∂q_i
                Map<Matrix<double, 15, 4, RowMajor>> J(jacobians[2]);
                J.setZero();
                Matrix3d dr_dq = -R_i * skew(preint.delta_v);
                J.block<3, 3>(3, 0) = dr_dq;
                J.block<3, 3>(6, 0) = -R_i * skew(preint.delta_p);
                J.block<3, 3>(0, 0) = preint.dR_dbg;
            }
            if (jacobians[3]) { // ∂r/∂ba_i
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[3]);
                J.setZero();
                J.block<3, 3>(3, 0) = -preint.dV_dba;
                J.block<3, 3>(6, 0) = -preint.dP_dba;
                J.block<3, 3>(9, 0) = -I;
            }
            if (jacobians[4]) { // ∂r/∂bg_i
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3, 3>(0, 0) = -preint.dR_dbg;
                J.block<3, 3>(3, 0) = -preint.dV_dbg;
                J.block<3, 3>(6, 0) = -preint.dP_dbg;
                J.block<3, 3>(12, 0) = -I;
            }

            // Current state Jacobians
            if (jacobians[5]) { // ∂r/∂p_j
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[5]);
                J.setZero();
                J.block<3, 3>(6, 0) = I;
            }
            if (jacobians[6]) { // ∂r/∂v_j
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[6]);
                J.setZero();
                J.block<3, 3>(3, 0) = I;
            }
            if (jacobians[7]) { // ∂r/∂q_j
                Map<Matrix<double, 15, 4, RowMajor>> J(jacobians[7]);
                J.setZero();
                J.block<3, 3>(0, 0) = I;
            }
            if (jacobians[8]) { // ∂r/∂ba_j
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[8]);
                J.setZero();
                J.block<3, 3>(9, 0) = I;
            }
            if (jacobians[9]) { // ∂r/∂bg_j
                Map<Matrix<double, 15, 3, RowMajor>> J(jacobians[9]);
                J.setZero();
                J.block<3, 3>(12, 0) = I;
            }
        }

        return true;
    }

private:
    const ImuPreintegrator::PreintResult preint;

    Matrix3d skew(const Vector3d& v) const {
        Matrix3d S;
        S << 0, -v.z(), v.y(),
             v.z(), 0, -v.x(),
             -v.y(), v.x(), 0;
        return S;
    }
};

struct UWBFactor : public ceres::SizedCostFunction<1, 7> {
    UWBFactor(const Vector3d& anchor, double measurement) 
        : anchor_(anchor), measurement_(measurement) {}

    virtual bool Evaluate(double const* const* parameters, 
                         double* residuals, 
                         double** jacobians) const {
        const Vector3d p(parameters[0][0], parameters[0][1], parameters[0][2]);
        double distance = (p - anchor_).norm();
        residuals[0] = distance - measurement_;

        if (jacobians) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();
            Vector3d direction = (p - anchor_).normalized();
            J.block<1,3>(0,0) = direction.transpose();
        }
        return true;
    }

    Vector3d anchor_;
    double measurement_;
};

class UWBFactorLoose : public ceres::CostFunction {
public:
    UWBFactorLoose(const Vector3d& z) : z_(z) {
        set_num_residuals(3);
        mutable_parameter_block_sizes()->push_back(3);  
    }

    virtual bool Evaluate(double const*const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Vector3d p(parameters[0]);
        residuals[0] = p.x() - z_.x();
        residuals[1] = p.y() - z_.y();
        residuals[2] = p.z() - z_.z(); 
        return true;
    }
    
private:
    Vector3d z_;
};

class FusionNode {
public:
    FusionNode() : nh_("~") {
        imu_sub_ = nh_.subscribe("/sensor_simulator/imu_data", 1000, &FusionNode::imuCallback, this);
        uwb_sub_ = nh_.subscribe("/sensor_simulator/UWBPoistion", 10, &FusionNode::uwbCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/fused_pose", 10);

        // Initialize states
        current_state_.timestamp = 0;
        current_state_.p = Vector3d::Zero();
        current_state_.v = Vector3d::Zero();
        current_state_.q.setIdentity();
        current_state_.accel_bias = Vector3d::Zero();
        current_state_.gyro_bias = Vector3d::Zero();
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        imu_queue_.push(*msg);
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        processUWBMeasurement(*msg);
    }

    void processUWBMeasurement(const geometry_msgs::PointStamped& uwb_msg) {
        // Get all IMU measurements since last UWB measurement
        std::vector<sensor_msgs::Imu> imu_measurements;
        while (!imu_queue_.empty() && 
               imu_queue_.front().header.stamp <= uwb_msg.header.stamp) {
            imu_measurements.push_back(imu_queue_.front());
            imu_queue_.pop();
        }

        // Perform IMU pre-integration
        ImuPreintegrator preint; // ImuPreintegrator

        preint.acc_bias = current_state_.accel_bias;
        preint.gyro_bias = current_state_.gyro_bias;

        // Vector3d acc_bias;
        // Vector3d gyro_bias;
        // double acc_noise;
        // double gyro_noise;

        for (const auto& imu : imu_measurements) {
            double delta_t = (imu.header.stamp.toSec() - current_state_.timestamp);
            Vector3d accel(imu.linear_acceleration.x, 
                          imu.linear_acceleration.y, 
                          imu.linear_acceleration.z);
            Vector3d gyro(imu.angular_velocity.x,
                         imu.angular_velocity.y,
                         imu.angular_velocity.z);
            preint.integrate(accel, gyro, delta_t);
            // preint.Integrate(imu, delta_t);
        }

        // Create new state
        State new_state;
        new_state.timestamp = uwb_msg.header.stamp.toSec();
        new_state.p = current_state_.p;
        new_state.v = current_state_.v;
        new_state.q = current_state_.q;
        new_state.accel_bias = current_state_.accel_bias;
        new_state.gyro_bias = current_state_.gyro_bias;

        // Add to optimization window
        states_.push_back(new_state);
        if (states_.size() > 5) {  // Keep window size 5
            states_.pop_front();
        }

        // Build factor graph
        ceres::Problem problem;
        std::vector<double*> parameter_blocks;

        for (auto& state : states_) {
            double* pose = new double[7];
            pose[0] = state.p.x();
            pose[1] = state.p.y();
            pose[2] = state.p.z();
            pose[3] = state.q.x();
            pose[4] = state.q.y();
            pose[5] = state.q.z();
            pose[6] = state.q.w();
            
            double* speed_bias = new double[9];
            speed_bias[0] = state.v.x();
            speed_bias[1] = state.v.y();
            speed_bias[2] = state.v.z();
            speed_bias[3] = state.accel_bias.x();
            speed_bias[4] = state.accel_bias.y();
            speed_bias[5] = state.accel_bias.z();
            speed_bias[6] = state.gyro_bias.x();
            speed_bias[7] = state.gyro_bias.y();
            speed_bias[8] = state.gyro_bias.z();

            parameter_blocks.push_back(pose);
            parameter_blocks.push_back(speed_bias);
        }

        // Add IMU factors
        for (size_t i = 1; i < states_.size(); ++i) {
            ceres::CostFunction* factor = new ImuFactor(preint.result);
            problem.AddResidualBlock(factor, NULL, 
                                   parameter_blocks[2*(i-1)], 
                                   parameter_blocks[2*(i-1)+1],
                                   parameter_blocks[2*i],
                                   parameter_blocks[2*i+1]);
        }

        // Add UWB factors (tightly coupled integration)
        for (size_t i = 0; i < states_.size(); ++i) {
            for (const auto& anchor : anchors_) {
                ceres::CostFunction* factor = new UWBFactor(anchor, uwb_msg.point.x);
                problem.AddResidualBlock(factor, NULL, parameter_blocks[2*i]);
            }
        }

        // Solve optimization problem
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 50;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Update current state
        current_state_ = states_.back();
        publishPose(current_state_);
    }

    void publishPose(const State& state) {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = ros::Time(state.timestamp);
        msg.header.frame_id = "world";
        msg.pose.position.x = state.p.x();
        msg.pose.position.y = state.p.y();
        msg.pose.position.z = state.p.z();
        msg.pose.orientation.x = state.q.x();
        msg.pose.orientation.y = state.q.y();
        msg.pose.orientation.z = state.q.z();
        msg.pose.orientation.w = state.q.w();
        pose_pub_.publish(msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_, uwb_sub_;
    ros::Publisher pose_pub_;
    std::queue<sensor_msgs::Imu> imu_queue_;
    std::deque<State> states_;
    State current_state_;
    std::vector<Vector3d> anchors_;
    std::mutex mutex_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_imu_fusion");
    FusionNode node;
    ros::spin();
    return 0;
}