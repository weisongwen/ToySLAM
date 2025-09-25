#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <vector>
//#include <deque>

#include "parameter_server.h" // For params
#include "gnss_parser.h"      // For GnssMeasurement
#include "types.h"            // For State struct

class Visualizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor
     * @param nh ROS node handle
     * @param params Reference to parameter server object
     */
    Visualizer(ros::NodeHandle& nh, const ParameterServer& params);
    
    /**
     * @brief Destructor
     */
    ~Visualizer() = default;
    
    /**
     * @brief Publish the optimized trajectory
     * @param latest_state Latest state in the sliding window
     */
    void publishOptimizedPath(const State& latest_state);

    /**
     * @brief Publish the final trajectory points that are marginalized out of the window
     * @param old_state Old state removed from the window
     */
    void publishFinalPath(const State& old_state);

    /**
     * @brief Publish GPS trajectory
     * @param measurement GPS measurement data
     */
    void publishGpsPath(const GnssMeasurement& measurement);

    /**
     * @brief Publish ground-truth trajectory
     * @param measurement Ground-truth measurement data
     */
    void publishGroundTruthPath(const GnssMeasurement& measurement);

    /**
     * @brief Publish trajectory from pure IMU integration
     * @param odom_msg Odometry message produced by IMU integration
     */
    void publishImuPath(const nav_msgs::Odometry& odom_msg);

    void publishImuPose(const State& current_state);
    void publishOptimizedPose(const State& latest_state);

     /**
     * @brief Publish visualization markers for position and velocity errors
     * @param current_state Current state
     * @param gps_measurements Queue of GPS measurements for matching
     */
    void publishErrorVisualizations(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements);


    /**
     * @brief Compute and visualize position error
     */
    void calculateAndVisualizePositionError(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements);
    
    /**
     * @brief Compute and visualize velocity error
     */
    void calculateAndVisualizeVelocityError(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements);


    /**
     * @brief Reset all visualization data
     */
    void reset();

private:


    const ParameterServer& params_;

    // ROS Publishers
    ros::Publisher gps_path_pub_;
    ros::Publisher optimized_path_pub_;
    ros::Publisher optimized_pose_pub_;
    ros::Publisher final_path_pub_;
    ros::Publisher gt_path_pub_;
    ros::Publisher imu_path_pub_;
    ros::Publisher imu_pose_pub_;
    ros::Publisher position_error_pub_;
    ros::Publisher velocity_error_pub_;

    // Path messages
    nav_msgs::Path gps_path_msg_;
    nav_msgs::Path gt_path_msg_;
    nav_msgs::Path optimized_path_msg_;
    nav_msgs::Path imu_path_msg_;
    nav_msgs::Path final_path_msg_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;
};

#endif // VISUALIZER_H