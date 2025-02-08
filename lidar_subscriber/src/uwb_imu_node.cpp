#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <memory>

using namespace Eigen;

// EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// struct State {
//     double timestamp;
//     Vector3d p;        // Position
//     Vector3d v;        // Velocity
//     Quaterniond q;     // Orientation
//     Vector3d ba;       // Accelerometer bias
//     Vector3d bg;       // Gyro bias
//     double* param_ptr; // Parameter block pointer
    
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
//     void updateParamBlock() {
//         param_ptr = new double[16];
//         Eigen::Map<Eigen::Vector3d>(param_ptr) = p;

//         // Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

//         Map<Vector3d>(param_ptr+3) = v;
//         Map<Quaterniond>(param_ptr+6) = q;
//         Map<Vector3d>(param_ptr+10) = ba;
//         Map<Vector3d>(param_ptr+13) = bg;
//     }
// };


// struct State {
//     double timestamp;
//     Eigen::Vector3d p;        // Position
//     Eigen::Vector3d v;        // Velocity
//     Eigen::Quaterniond q;     // Orientation
//     Eigen::Vector3d ba;       // Accelerometer bias
//     Eigen::Vector3d bg;       // Gyro bias
//     std::unique_ptr<double[]> param_ptr; // Parameter block pointer
    
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
//     void updateParamBlock() {
//         param_ptr = std::make_unique<double[]>(16);
//         Eigen::Map<Eigen::Vector3d>(param_ptr.get()) = p;
//         Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 3) = v;
//         Eigen::Map<Eigen::Quaterniond>(param_ptr.get() + 6) = q;
//         Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 10) = ba;
//         Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 13) = bg;
//     }
// };


struct State {
    double timestamp;
    Eigen::Vector3d p;        // Position
    Eigen::Vector3d v;        // Velocity
    Eigen::Quaterniond q;     // Orientation
    Eigen::Vector3d ba;       // Accelerometer bias
    Eigen::Vector3d bg;       // Gyro bias
    std::unique_ptr<double[]> param_ptr; // Parameter block pointer
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Default constructor
    State() : param_ptr(nullptr) {}

    // Custom copy constructor
    State(const State& other)
        : timestamp(other.timestamp), p(other.p), v(other.v), q(other.q),
          ba(other.ba), bg(other.bg), param_ptr(std::make_unique<double[]>(16)) {
        std::copy(other.param_ptr.get(), other.param_ptr.get() + 16, param_ptr.get());
    }

    // Custom copy assignment operator
    State& operator=(const State& other) {
        if (this != &other) {
            timestamp = other.timestamp;
            p = other.p;
            v = other.v;
            q = other.q;
            ba = other.ba;
            bg = other.bg;
            param_ptr = std::make_unique<double[]>(16);
            std::copy(other.param_ptr.get(), other.param_ptr.get() + 16, param_ptr.get());
        }
        return *this;
    }

    void updateParamBlock() {
        param_ptr = std::make_unique<double[]>(16);
        Eigen::Map<Eigen::Vector3d>(param_ptr.get()) = p;
        Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 3) = v;
        Eigen::Map<Eigen::Quaterniond>(param_ptr.get() + 6) = q;
        Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 10) = ba;
        Eigen::Map<Eigen::Vector3d>(param_ptr.get() + 13) = bg;
    }
};

class IMUFactor : public ceres::SizedCostFunction<15, 16, 16> {
public:
    IMUFactor(const Vector3d& acc, const Vector3d& gyro, double dt, const Matrix3d& noise)
        : acc_(acc), gyro_(gyro), dt_(dt), noise_inv_(noise.inverse()) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        // Simplified IMU error model (replace with proper preintegration)
        const double* state_i = parameters[0];
        const double* state_j = parameters[1];

        Map<const Vector3d> p_i(state_i);
        Map<const Vector3d> v_i(state_i+3);
        Map<const Quaterniond> q_i(state_i+6);
        Map<const Vector3d> ba_i(state_i+10);
        Map<const Vector3d> bg_i(state_i+13);

        Map<const Vector3d> p_j(state_j);
        Map<const Vector3d> v_j(state_j+3);
        Map<const Quaterniond> q_j(state_j+6);

        // Compute residuals
        Vector3d residual_p = p_j - (p_i + v_i*dt_ + 0.5*q_i.toRotationMatrix()*acc_*dt_*dt_);
        Vector3d residual_v = v_j - (v_i + q_i*acc_*dt_);
        Quaterniond dq = q_i.conjugate() * q_j;
        Vector3d residual_q = 2.0 * dq.vec(); // log map
        
        Map<Vector3d> res_p(residuals);
        Map<Vector3d> res_v(residuals+3);
        Map<Vector3d> res_q(residuals+6);
        res_p = residual_p;
        res_v = residual_v;
        res_q = residual_q;

        if(jacobians) {
            // Simplified Jacobians (implement proper derivatives)
            if(jacobians[0]) {
                Map<Matrix<double, 15, 16, RowMajor>> J0(jacobians[0]);
                J0.setZero();
                J0.block<3,3>(0,0) = -Matrix3d::Identity();
                J0.block<3,3>(0,3) = -dt_ * Matrix3d::Identity();
                J0.block<3,3>(3,3) = -Matrix3d::Identity();
                J0.block<3,3>(6,6) = -Matrix3d::Identity();
            }
            if(jacobians[1]) {
                Map<Matrix<double, 15, 16, RowMajor>> J1(jacobians[1]);
                J1.setZero();
                J1.block<3,3>(0,0) = Matrix3d::Identity();
                J1.block<3,3>(3,3) = Matrix3d::Identity();
                J1.block<3,3>(6,6) = Matrix3d::Identity();
            }
        }
        return true;
    }

private:
    Vector3d acc_, gyro_;
    double dt_;
    Matrix3d noise_inv_;
};

class UWBFactor : public ceres::SizedCostFunction<3, 16> {
public:
    UWBFactor(const Vector3d& z, const Matrix3d& noise) 
        : z_(z), noise_inv_(noise.inverse()) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Map<const Vector3d> p(parameters[0]);
        Map<Vector3d> res(residuals);
        res = p - z_;

        if(jacobians && jacobians[0]) {
            Map<Matrix<double, 3, 16, RowMajor>> J(jacobians[0]);
            J.setZero();
            J.block<3,3>(0,0) = Matrix3d::Identity();
        }
        return true;
    }

private:
    Vector3d z_;
    Matrix3d noise_inv_;
};

class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(MatrixXd&& H, VectorXd&& b) 
        : H_(H), b_(b) {
        set_num_residuals(b_.rows());
        mutable_parameter_block_sizes()->push_back(H_.cols());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Map<const VectorXd> x(parameters[0], H_.cols());
        Map<VectorXd> res(residuals, b_.rows());
        res = H_ * x - b_;

        if(jacobians && jacobians[0]) {
            Map<MatrixXd> J(jacobians[0], H_.rows(), H_.cols());
            J = H_;
        }
        return true;
    }

private:
    MatrixXd H_;
    VectorXd b_;
};

class SlidingWindowFilter {
public:
    SlidingWindowFilter() : nh_("~"), window_size_(10) {
        imu_sub_ = nh_.subscribe("/imu/data", 100, &SlidingWindowFilter::imuCallback, this);
        uwb_sub_ = nh_.subscribe("/vins_estimator/UWBPoistionPS", 10, &SlidingWindowFilter::uwbCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/fused_pose", 10);
        
        // Initialize noise parameters
        imu_acc_noise_ = Matrix3d::Identity() * 0.01;
        imu_gyro_noise_ = Matrix3d::Identity() * 0.001;
        uwb_noise_ = Matrix3d::Identity() * 0.1;
        
        // problem_options_.local_parameterization = new ceres::EigenQuaternionParameterization;
        // problem_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

        // Correct Ceres problem options
        problem_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem_options_.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        
        resetProblem();
    }

private:
    void resetProblem() {
        problem_.reset(new ceres::Problem(problem_options_));
        if(prior_factor_) {
            problem_->AddResidualBlock(prior_factor_, nullptr, states_[1].param_ptr.get());
        }
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        current_acc_ = Vector3d(msg->linear_acceleration.x,
                               msg->linear_acceleration.y,
                               msg->linear_acceleration.z);
        current_gyro_ = Vector3d(msg->angular_velocity.x,
                                msg->angular_velocity.y,
                                msg->angular_velocity.z);

        if(states_.empty()) {
            initializeFirstState(msg->header.stamp);
            return;
        }

        integrateIMU(msg->header.stamp);
    }

    void uwbCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        if(states_.size() < 2) return;

        addUWBMeasurement(msg);
        optimizeGraph();
        manageWindow();
        publishPose(msg->header.stamp);
    }

    void initializeFirstState(const ros::Time& stamp) {
        State new_state;
        new_state.timestamp = stamp.toSec();
        new_state.p.setZero();
        new_state.v.setZero();
        new_state.q.setIdentity();
        new_state.ba.setZero();
        new_state.bg.setZero();
        new_state.updateParamBlock();
        states_.push_back(new_state);
        problem_->AddParameterBlock(states_.back().param_ptr.get(), 16);
    }

    void integrateIMU(const ros::Time& stamp) {
        double dt = stamp.toSec() - states_.back().timestamp;
        
        State new_state = states_.back();
        new_state.timestamp = stamp.toSec();
        new_state.updateParamBlock();
        
        // Add IMU factor between previous and new state
        ceres::CostFunction* imu_factor = new IMUFactor(current_acc_, current_gyro_, dt, imu_acc_noise_);
        
        problem_->AddParameterBlock(new_state.param_ptr.get(), 16);
        
        problem_->SetParameterization(new_state.param_ptr.get() + 6, 
                                     new ceres::EigenQuaternionParameterization());
        problem_->AddResidualBlock(imu_factor, nullptr, 
                                        states_.back().param_ptr.get(), 
                                        new_state.param_ptr.get());
        states_.push_back(new_state);
    }

    void addUWBMeasurement(const geometry_msgs::PointStamped::ConstPtr& msg) {
        Vector3d z(msg->point.x, msg->point.y, msg->point.z);
        ceres::CostFunction* uwb_factor = new UWBFactor(z, uwb_noise_);
        problem_->AddResidualBlock(uwb_factor, nullptr, states_.back().param_ptr.get());
    }

    void optimizeGraph() {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = 10;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
    }

    void manageWindow() {
        if(states_.size() <= window_size_) return;
        marginalizeOldestState();
    }

    void marginalizeOldestState() {
        // 1. Linearize all factors connected to oldest state
        MatrixXd H = MatrixXd::Zero(32, 32); // Old + new state
        VectorXd b = VectorXd::Zero(32);

        // Get all residual blocks connected to oldest state
        std::vector<ceres::ResidualBlockId> old_residuals;
        problem_->GetResidualBlocksForParameterBlock(states_[0].param_ptr.get(), &old_residuals);

        // Linearize each residual block
        for (auto& residual_id : old_residuals) {
            // Get the cost function and parameter blocks
            const ceres::CostFunction* cf = problem_->GetCostFunctionForResidualBlock(residual_id);
            std::vector<double*> parameter_blocks;
            problem_->GetParameterBlocksForResidualBlock(residual_id, &parameter_blocks);

            // Prepare storage for residuals and Jacobians
            VectorXd residuals(cf->num_residuals());
            std::vector<double*> jacobians(parameter_blocks.size());
            std::vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> jacobian_matrices(parameter_blocks.size());

            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                jacobian_matrices[i].resize(cf->num_residuals(), problem_->ParameterBlockSize(parameter_blocks[i]));
                jacobians[i] = jacobian_matrices[i].data();
            }

            // Evaluate the cost function
            cf->Evaluate(parameter_blocks.data(), residuals.data(), jacobians.data());

            // Accumulate the Jacobians and residuals into H and b
            for (size_t i = 0; i < parameter_blocks.size(); ++i) {
                Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J = jacobian_matrices[i];
                H += J.transpose() * J;
                b += J.transpose() * residuals;
            }
        }

        // 2. Perform Schur complement
        MatrixXd H_mm = H.block<16,16>(0,0);
        MatrixXd H_mr = H.block<0,16>(16,16);
        MatrixXd H_rm = H.block<16,0>(0,16);
        MatrixXd H_rr = H.block<16,16>(16,16);
        VectorXd b_m = b.segment<16>(0);
        VectorXd b_r = b.segment<16>(16);

        MatrixXd H_mm_inv = H_mm.ldlt().solve(MatrixXd::Identity(16,16));
        MatrixXd H_schur = H_rr - H_rm * H_mm_inv * H_mr;
        VectorXd b_schur = b_r - H_rm * H_mm_inv * b_m;

        // 3. Add prior factor
        prior_factor_ = new MarginalizationFactor(std::move(H_schur), std::move(b_schur));
        
        // 4. Remove old state and reset problem
        problem_->RemoveParameterBlock(states_[0].param_ptr.get());
        states_.pop_front();
        resetProblem();
    }

    void publishPose(const ros::Time& stamp) {
        geometry_msgs::PointStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = "world";
        pose_msg.point.x = states_.back().p.x();
        pose_msg.point.y = states_.back().p.y();
        pose_msg.point.z = states_.back().p.z();
        pose_pub_.publish(pose_msg);
    }

    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_, uwb_sub_;
    ros::Publisher pose_pub_;
    
    std::deque<State> states_;
    std::mutex data_mutex_;
    const size_t window_size_;
    
    Vector3d current_acc_, current_gyro_;
    Matrix3d imu_acc_noise_, imu_gyro_noise_, uwb_noise_;
    
    ceres::Problem::Options problem_options_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::CostFunction* prior_factor_ = nullptr;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_imu_fusion");
    SlidingWindowFilter filter;
    ros::spin();
    return 0;
}