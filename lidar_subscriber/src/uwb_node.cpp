#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <vector>
#include <xmlrpcpp/XmlRpcException.h>

// roslaunch ToySLAM fusion.launch 

// simulate the UWB ranging measurements and do the positioning

class UWBPositionEstimator {
public:
    UWBPositionEstimator() : nh_("~"), gen_(std::random_device{}()) {
        loadParameters(); // load the anchor position, etc
        initializePublishers();
        initializeMarkers();
        initializeTimer();
    }

private:
    // ROS components
    ros::NodeHandle nh_;
    ros::Publisher est_pub_, gt_pub_, anchor_pub_, trajectory_pub_;
    ros::Timer timer_;
    
    // Simulation parameters
    std::vector<Eigen::Vector3d> anchors_;
    double noise_std_;
    double motion_radius_;
    double motion_speed_;
    std::string motion_type_;
    
    // Optimization parameters
    int max_iterations_;
    double solver_tolerance_;
    std::string linear_solver_type_;
    bool use_huber_loss_;
    double huber_loss_threshold_;
    
    // State variables
    Eigen::Vector3d current_gt_position_;
    Eigen::Vector3d current_velocity_;
    std::mt19937 gen_;
    std::vector<Eigen::Vector3d> estimated_trajectory_;
    ros::Time last_update_time_;
    size_t max_trajectory_size_ = 1000; // Limit trajectory memory usage

    struct RangeResidual {
        RangeResidual(const Eigen::Vector3d& anchor, double measurement)
            : anchor_(anchor), measurement_(measurement) {}

        template <typename T>
        bool operator()(const T* const position, T* residual) const {
            T dx = position[0] - T(anchor_.x());
            T dy = position[1] - T(anchor_.y());
            T dz = position[2] - T(anchor_.z());
            T distance = ceres::sqrt(dx*dx + dy*dy + dz*dz);
            residual[0] = distance - T(measurement_);
            return true;
        }

        Eigen::Vector3d anchor_;
        double measurement_;
    };

    void loadParameters() {
        double scale = 10;
        // Load anchor positions
        XmlRpc::XmlRpcValue anchor_list;
        if (!nh_.getParam("anchors", anchor_list)) {
            ROS_WARN("No anchor positions defined in parameter server, using defaults");
            // Default anchor positions forming a cube
            anchors_.push_back(Eigen::Vector3d(0, 0, 0) * scale);
            anchors_.push_back(Eigen::Vector3d(5, 0, 0) * scale);
            anchors_.push_back(Eigen::Vector3d(0, 5, 0) * scale);
            anchors_.push_back(Eigen::Vector3d(5, 5, 0) * scale);
            anchors_.push_back(Eigen::Vector3d(0, 0, 5) * scale);
            anchors_.push_back(Eigen::Vector3d(5, 0, 5) * scale);
            anchors_.push_back(Eigen::Vector3d(0, 5, 5) * scale);
            anchors_.push_back(Eigen::Vector3d(5, 5, 5) * scale);
        } else {
            for (int i = 0; i < anchor_list.size(); ++i) {
                try {
                    Eigen::Vector3d anchor(
                        anchor_list[i]["x"],
                        anchor_list[i]["y"],
                        anchor_list[i]["z"]
                    );
                    anchors_.push_back(anchor * scale);
                } catch (const XmlRpc::XmlRpcException& e) {
                    ROS_ERROR("Error parsing anchor %d: %s", i, e.getMessage().c_str());
                }
            }
        }
        
        // Check if we have enough anchors for 3D positioning
        if (anchors_.size() < 4) {
            ROS_WARN("At least 4 anchors are needed for reliable 3D positioning");
        }

        // Load other parameters
        nh_.param<double>("noise_std", noise_std_, 0.1);
        nh_.param<double>("motion_radius", motion_radius_, 1.5);
        nh_.param<double>("motion_speed", motion_speed_, 0.5);
        nh_.param<std::string>("motion_type", motion_type_, "helical");
        nh_.param<int>("max_iterations", max_iterations_, 100);
        nh_.param<double>("solver_tolerance", solver_tolerance_, 1e-6);
        nh_.param<std::string>("linear_solver_type", linear_solver_type_, "DENSE_QR");
        nh_.param<bool>("use_huber_loss", use_huber_loss_, false);
        nh_.param<double>("huber_loss_threshold", huber_loss_threshold_, 1.0);
    }

    void initializePublishers() {
        est_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/uwb/estimated_position", 10);
        gt_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/uwb/ground_truth", 10);
        anchor_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/uwb/anchors", 1);
        trajectory_pub_ = nh_.advertise<visualization_msgs::Marker>("/uwb/trajectory", 1);
    }

    void initializeMarkers() {
        visualization_msgs::MarkerArray anchor_markers;
        for (size_t i = 0; i < anchors_.size(); ++i) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "anchors";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = anchors_[i].x();
            marker.pose.position.y = anchors_[i].y();
            marker.pose.position.z = anchors_[i].z();
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3;
            marker.color.r = 1.0;
            marker.color.a = 1.0;
            anchor_markers.markers.push_back(marker);
        }
        anchor_pub_.publish(anchor_markers);
    }

    void initializeTimer() {
        last_update_time_ = ros::Time::now();
        timer_ = nh_.createTimer(ros::Duration(0.1), &UWBPositionEstimator::estimationLoop, this);
    }

    void estimationLoop(const ros::TimerEvent& event) {
        updateGroundTruthPosition();
        std::vector<double> measurements = simulateMeasurements();
        Eigen::Vector3d estimated_position = solvePosition(measurements);
        publishResults(estimated_position);
        updateTrajectory(estimated_position);
    }

    void updateGroundTruthPosition() {
        // Use actual time difference for more accurate simulation
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_update_time_).toSec();
        last_update_time_ = current_time;
        
        // Use accumulating time for simulation
        static double t = 0.0;
        t += dt;
        
        if (motion_type_ == "circular") {
            current_gt_position_ = Eigen::Vector3d(
                2.0 + motion_radius_ * cos(motion_speed_ * t),
                2.0 + motion_radius_ * sin(motion_speed_ * t),
                0.5
            );
        }
        else if (motion_type_ == "helical") {
            current_gt_position_ = Eigen::Vector3d(
                2.0 + motion_radius_ * cos(motion_speed_ * t),
                2.0 + motion_radius_ * sin(motion_speed_ * t),
                0.5 + 0.2 * t
            );
        }
        else { // Linear motion
            current_gt_position_ = Eigen::Vector3d(
                2.0 + motion_speed_ * t,
                2.0,
                0.5
            );
        }
    }

    std::vector<double> simulateMeasurements() {
        std::vector<double> measurements;
        std::normal_distribution<double> noise(0.0, noise_std_);
        
        for (const auto& anchor : anchors_) {
            double true_distance = (current_gt_position_ - anchor).norm();
            measurements.push_back(true_distance + noise(gen_));
        }
        return measurements;
    }

    Eigen::Vector3d solvePosition(const std::vector<double>& measurements) {
        // Verify we have enough measurements
        if (measurements.size() < 4) {
            ROS_WARN("Too few measurements (%zu) for reliable 3D positioning", measurements.size());
            // Return last position or default if no history
            return estimated_trajectory_.empty() ? Eigen::Vector3d(0, 0, 0) : estimated_trajectory_.back();
        }
        
        ceres::Problem problem;
        
        // More realistic initial guess:
        // If we have previous estimates, use the last one
        // Otherwise use centroid of anchors as starting point
        Eigen::Vector3d position;
        if (!estimated_trajectory_.empty()) {
            position = estimated_trajectory_.back();
        } else {
            // Calculate centroid of anchors as starting point
            Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
            for (const auto& anchor : anchors_) {
                centroid += anchor;
            }
            if (!anchors_.empty()) {
                centroid /= anchors_.size();
            }
            position = centroid;
        }

        for (size_t i = 0; i < anchors_.size(); ++i) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<RangeResidual, 1, 3>(
                    new RangeResidual(anchors_[i], measurements[i]));
            
            ceres::LossFunction* loss_function = nullptr;
            if (use_huber_loss_) {
                loss_function = new ceres::HuberLoss(huber_loss_threshold_);
            }

            problem.AddResidualBlock(cost_function, loss_function, position.data());
        }

        ceres::Solver::Options options;
        options.max_num_iterations = max_iterations_;
        options.linear_solver_type = getLinearSolverType();
        options.minimizer_progress_to_stdout = false;
        options.function_tolerance = solver_tolerance_;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Better error handling
        if (summary.termination_type != ceres::CONVERGENCE) {
            ROS_WARN("Optimization failed to converge: %s", summary.BriefReport().c_str());
            // For debug builds, log full report
            ROS_DEBUG("%s", summary.FullReport().c_str());
            
            // If this is a serious failure, return the previous position instead
            if (summary.termination_type == ceres::FAILURE || 
                summary.final_cost > 10.0 * measurements.size()) {
                if (!estimated_trajectory_.empty()) {
                    ROS_WARN("Using previous position estimate due to optimization failure");
                    return estimated_trajectory_.back();
                }
            }
        }

        return position;
    }

    ceres::LinearSolverType getLinearSolverType() {
        if (linear_solver_type_ == "DENSE_QR") return ceres::DENSE_QR;
        if (linear_solver_type_ == "SPARSE_NORMAL_CHOLESKY") return ceres::SPARSE_NORMAL_CHOLESKY;
        if (linear_solver_type_ == "DENSE_SCHUR") return ceres::DENSE_SCHUR;
        return ceres::DENSE_QR;
    }

    void publishResults(const Eigen::Vector3d& estimated_position) {
        geometry_msgs::PointStamped est_msg, gt_msg;
        est_msg.header.stamp = gt_msg.header.stamp = ros::Time::now();
        est_msg.header.frame_id = gt_msg.header.frame_id = "world";
        
        est_msg.point.x = estimated_position.x();
        est_msg.point.y = estimated_position.y();
        est_msg.point.z = estimated_position.z();
        
        gt_msg.point.x = current_gt_position_.x();
        gt_msg.point.y = current_gt_position_.y();
        gt_msg.point.z = current_gt_position_.z();
        
        est_pub_.publish(est_msg);
        gt_pub_.publish(gt_msg);
    }

    void updateTrajectory(const Eigen::Vector3d& position) {
        estimated_trajectory_.push_back(position);
        
        // Limit trajectory size to prevent memory issues
        if (estimated_trajectory_.size() > max_trajectory_size_) {
            estimated_trajectory_.erase(estimated_trajectory_.begin());
        }
        
        visualization_msgs::Marker trajectory;
        trajectory.header.frame_id = "world";
        trajectory.header.stamp = ros::Time::now();
        trajectory.ns = "trajectory";
        trajectory.id = 0;
        trajectory.type = visualization_msgs::Marker::LINE_STRIP;
        trajectory.action = visualization_msgs::Marker::ADD;
        trajectory.scale.x = 0.05;
        trajectory.color.b = 1.0;
        trajectory.color.a = 1.0;
        
        for (const auto& p : estimated_trajectory_) {
            geometry_msgs::Point point;
            point.x = p.x();
            point.y = p.y();
            point.z = p.z();
            trajectory.points.push_back(point);
        }
        
        trajectory_pub_.publish(trajectory);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_position_estimator");
    UWBPositionEstimator estimator;
    ros::spin();
    return 0;
}