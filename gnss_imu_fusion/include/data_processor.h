#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <novatel_msgs/INSPVAX.h>
#include <gnss_comm/GnssPVTSolnMsg.h>

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <mutex>
#include <optional>
#include <fstream>
#include <random>

#include "../include/parameter_server.h"
#include "../include/gnss_parser.h"
#include "../include/visualizer.h"
#include "../include/backend_optimizer.h"
#include "../include/imu_preint.h"
#include "../include/math_helpers.h"


class DataProcessor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DataProcessor(ros::NodeHandle& nh, const ParameterServer& params, 
                  BackendOptimizer* optimizer, Visualizer* visualizer);

    ~DataProcessor();
    
    // FIXED: Corrected signature to accept timer event
    void optimizationTimerCallback(const ros::TimerEvent& event);

private:
    // ROS Callbacks
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void groundTruthCallback(const novatel_msgs::INSPVAX::ConstPtr& msg);
    void inspvaxCallback(const novatel_msgs::INSPVAX::ConstPtr& msg);
    void gnssCommCallback(const gnss_comm::GnssPVTSolnMsg::ConstPtr& msg);
    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);

    // Core Algorithm Logic
    void processGnssMeasurement(const GnssMeasurement& measurement);
    void initializeFromGps(const GnssMeasurement& gps);
    void createKeyframeFromGps(const GnssMeasurement& gps);
    void performPreintegrationBetweenKeyframes(const double &start_time, const double &end_time);
    void propagateStateWithImu(const sensor_msgs::Imu& imu_msg);

    // Helper methods
    void setupRosCommunications(ros::NodeHandle& nh);
    void syncEnuReference();
    void initializeState();
    sensor_msgs::Imu findClosestImuMeasurement(double timestamp);
    void logGpsData(const GnssMeasurement& meas);
    void logGroundTruthData(const GnssMeasurement& meas);
    void logOptimizedState(const State& state);
    void logKeyframeBias(const State& keyframe_state_to_log);


    // === Member Variables ===
    const ParameterServer& params_;
    BackendOptimizer* backend_optimizer_;
    Visualizer* visualizer_;

    ros::Subscriber imu_sub_;
    ros::Subscriber gnss_sub_;
    ros::Subscriber ground_truth_sub_;

    // Data Buffers
    std::deque<sensor_msgs::Imu> imu_buffer_;
    // This buffer is for optimization history, not for new measurements.
    std::vector<GnssMeasurement> gps_measurements_; 
    
    // Parsers
    GnssCommParser gnss_comm_parser_;
    InspvaxParser inspvax_parser_;
    OdometryParser odom_parser_;
    std::vector<GnssParser*> all_parsers_;
    
    // Logging
    std::ofstream gps_log_file_;
    std::ofstream gt_log_file_;
    std::ofstream optimized_log_file_;
    std::ofstream bias_fs;
    
    // Core State and Window
    State current_state_;
    std::deque<State, Eigen::aligned_allocator<State>> state_window_;
    // ADDED: Missing member variables
    bool is_initialized_;
    bool has_imu_data_;
    double last_imu_timestamp_;
    double last_processed_timestamp_;
    bool just_optimized_;
    int optimization_count_;
    int gps_measurement_count_; 
    
    // IMU Preintegration
    std::map<std::pair<double, double>, imu_preint> preintegration_map_test;
    imu_preint current_preint_test;

    // Marginalization
    MarginalizationInfo* last_marginalization_info_;
    
    // Utilities
    std::mutex data_mutex_;
    std::default_random_engine random_generator_;
    Eigen::Vector3d gravity_world_;
    Eigen::Vector3d initial_acc_bias_;
    Eigen::Vector3d initial_gyro_bias_;
};

#endif // DATA_PROCESSOR_H