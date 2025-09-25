#ifndef PARAMETER_SERVER_H
#define PARAMETER_SERVER_H

#include <ros/ros.h>
#include <string>
#include <sstream>
#include <ctime>
#include "../include/ceres_logger.h"

class ParameterServer {
public:

    std::string imu_topic, uwb_topic, ground_truth_topic, gps_log_path,
                gt_log_path, optimized_log_path, world_frame_id,
                body_frame_id, gnss_topic, gnss_message_type;

    bool subscribe_to_ground_truth;
    bool use_gps_instead_of_uwb;
    bool use_gps_orientation_as_initial;
    bool use_gps_velocity;
    bool enable_consistency_check;
    bool enable_bias_estimation;
    bool enable_marginalization;
    bool enable_roll_pitch_constraint;
    bool enable_orientation_smoothness_factor;
    bool enable_velocity_constraint;
    bool enable_horizontal_velocity_incentive;
    bool enable_imu_orientation_factor;
    bool use_gps_orientation_as_constraint = false;

    int imu_queue_size, uwb_queue_size, gnss_queue_size;
    int optimized_pose_queue_size, imu_pose_queue_size;
    int optimization_window_size, initial_grace_epochs, max_iterations;

    double artificial_pos_noise_std, artificial_vel_noise_std;
    double gravity_magnitude, artificial_gps_noise;
    double imu_acc_noise, imu_gyro_noise;
    double imu_acc_bias_noise, imu_gyro_bias_noise;
    double acc_bias_max, gyro_bias_max;
    double initial_acc_bias_x, initial_acc_bias_y, initial_acc_bias_z;
    double initial_gyro_bias_x, initial_gyro_bias_y, initial_gyro_bias_z;
    double uwb_position_noise;
    double optimization_frequency;
    double imu_buffer_time_length;
    double roll_pitch_weight, max_imu_dt;
    double imu_orientation_weight, bias_constraint_weight;
    double max_velocity, velocity_constraint_weight;
    double min_horizontal_velocity, horizontal_velocity_weight;
    double orientation_smoothness_weight, gravity_alignment_weight;
    double max_integration_dt, min_integration_dt, bias_correction_threshold;
    double gps_position_noise, gps_velocity_noise;
    double nis_threshold_position;
    double nis_threshold_velocity;
    double max_covariance_scale_factor;

    std::string optimized_pose_topic, imu_pose_topic;
    std::string bias_path;
    std::string results_log_param;
    std::string metrics_log_param;
    // std::string gps_topic_;
    // std::string gt_topic_;

    std::ofstream gps_log_file;
    std::ofstream gt_log_file;
    std::ofstream optimized_log_file;
    std::ofstream bias_fs;

    

    // Logger
    //CeresLogger logger;

    ParameterServer() = default;

    void loadParameters(ros::NodeHandle& nh) {

        // Load topic name parameters
        nh.param<std::string>("imu_topic", imu_topic, "/imu/data");
        nh.param<std::string>("uwb_topic", uwb_topic, "/sensor_simulator/UWBPoistionPS");

        nh.param<bool>("subscribe_to_ground_truth", subscribe_to_ground_truth, false);
        nh.param<std::string>("ground_truth_topic", ground_truth_topic, "/novatel_data/inspvax_gt");
        nh.param<double>("artificial_pos_noise_std", artificial_pos_noise_std, 0.5);
        nh.param<double>("artificial_vel_noise_std", artificial_vel_noise_std, 0.1);

        nh.param<std::string>("gps_log_path", gps_log_path, "gps_log.csv");
        nh.param<std::string>("gt_log_path", gt_log_path, "gt_log.csv");
        nh.param<std::string>("optimized_log_path", optimized_log_path, "optimized_log.csv");

        nh.param<int>("imu_queue_size", imu_queue_size, 1000);
        nh.param<int>("uwb_queue_size", uwb_queue_size, 100);
        
        // Add GPS/UWB mode selection
        nh.param<bool>("use_gps_instead_of_uwb", use_gps_instead_of_uwb, false);
        nh.param<std::string>("gps_topic", gnss_topic, "/novatel_data/inspvax");
        nh.param<int>("gps_queue_size", gnss_queue_size, 100);
        nh.param<std::string>("gps_message_type", gnss_message_type, "inspvax");
        nh.param<double>("gps_position_noise", gps_position_noise, 0.01);
        nh.param<double>("gps_velocity_noise", gps_velocity_noise, 0.01);
        
        // GPS data usage configuration
        nh.param<bool>("use_gps_orientation_as_initial", use_gps_orientation_as_initial, false);
        nh.param<bool>("use_gps_velocity", use_gps_velocity, true);
        
        // Load output topic parameters
        nh.param<std::string>("optimized_pose_topic", optimized_pose_topic, "/uwb_imu_fusion/optimized_pose");
        nh.param<std::string>("imu_pose_topic", imu_pose_topic, "/uwb_imu_fusion/imu_pose");
        nh.param<int>("optimized_pose_queue_size", optimized_pose_queue_size, 10);
        nh.param<int>("imu_pose_queue_size", imu_pose_queue_size, 20000);
        
        // Load parameters
        nh.param<double>("gravity_magnitude", gravity_magnitude, 9.81);
        nh.param<double>("artificial_gps_noise", artificial_gps_noise, 0.01);// 1cm - for testing
        
        // Realistic IMU noise parameters
        nh.param<double>("imu_acc_noise", imu_acc_noise, 0.03);    // m/s²
        nh.param<double>("imu_gyro_noise", imu_gyro_noise, 0.002); // rad/s
        
        // CRITICAL: Realistic bias parameters
        nh.param<double>("imu_acc_bias_noise", imu_acc_bias_noise, 0.0001);  // m/s²/sqrt(s)
        nh.param<double>("imu_gyro_bias_noise", imu_gyro_bias_noise, 0.00001); // rad/s/sqrt(s)
        nh.param<double>("acc_bias_max", acc_bias_max, 1.1);   // Maximum allowed acc bias (m/s²) 0.1 
        nh.param<double>("gyro_bias_max", gyro_bias_max, 1.01); // Maximum allowed gyro bias (rad/s) 0.01
        
        // CRITICAL: Initial biases (small realistic values)
        nh.param<double>("initial_acc_bias_x", initial_acc_bias_x, 0.05);
        nh.param<double>("initial_acc_bias_y", initial_acc_bias_y, -0.05);
        nh.param<double>("initial_acc_bias_z", initial_acc_bias_z, 0.05);
        nh.param<double>("initial_gyro_bias_x", initial_gyro_bias_x, 0.001);
        nh.param<double>("initial_gyro_bias_y", initial_gyro_bias_y, -0.001);
        nh.param<double>("initial_gyro_bias_z", initial_gyro_bias_z, 0.001);
        
        nh.param<double>("uwb_position_noise", uwb_position_noise, 0.05);  // m
        nh.param<int>("optimization_window_size", optimization_window_size, 20); // Reduced for stability
        
        // Frame IDs
        nh.param<std::string>("world_frame_id", world_frame_id, "map");
        nh.param<std::string>("body_frame_id", body_frame_id, "base_link");

        nh.param<bool>("enable_consistency_check", enable_consistency_check, false);
        nh.param<double>("nis_threshold_position", nis_threshold_position, 11.345); // 卡方, 3 DoF, 95%
        nh.param<double>("nis_threshold_velocity", nis_threshold_velocity, 11.345); // 卡方, 3 DoF, 95%
        nh.param<double>("max_covariance_scale_factor", max_covariance_scale_factor, 100000.0);
        nh.param<int>("initial_grace_epochs", initial_grace_epochs, 50);
        
        nh.param<double>("optimization_frequency", optimization_frequency, 10.0);
        nh.param<double>("imu_buffer_time_length", imu_buffer_time_length, 10.0);
        nh.param<int>("max_iterations", max_iterations, 10); // Lower iterations
        
        nh.param<bool>("enable_bias_estimation", enable_bias_estimation, true);
        
        // NEW: Enable marginalization
        nh.param<bool>("enable_marginalization", enable_marginalization, true);
        
        // NEW: Add feature configuration parameters
        nh.param<bool>("enable_roll_pitch_constraint", enable_roll_pitch_constraint, false);

        nh.param<bool>("enable_orientation_smoothness_factor", enable_orientation_smoothness_factor, false);
        nh.param<bool>("enable_velocity_constraint", enable_velocity_constraint, false);
        nh.param<bool>("enable_horizontal_velocity_incentive", enable_horizontal_velocity_incentive, false);
        nh.param<bool>("enable_imu_orientation_factor", enable_imu_orientation_factor, false);
        
        // Constraint weights
        nh.param<double>("roll_pitch_weight", roll_pitch_weight, 300.0); // Increased from 100.0
        nh.param<double>("max_imu_dt", max_imu_dt, 0.5);
        nh.param<double>("imu_orientation_weight", imu_orientation_weight, 50.0);
        nh.param<double>("bias_constraint_weight", bias_constraint_weight, 1000.0);
        
        // IMPROVED: Velocity parameters for high-speed scenarios (0-70 km/h)
        nh.param<double>("max_velocity", max_velocity, 25.0); // Maximum velocity (m/s) = 90 km/h
        nh.param<double>("velocity_constraint_weight", velocity_constraint_weight, 150.0);
        nh.param<double>("min_horizontal_velocity", min_horizontal_velocity, 0.5); // Minimum desired velocity
        nh.param<double>("horizontal_velocity_weight", horizontal_velocity_weight, 10.0);
        
        nh.param<double>("orientation_smoothness_weight", orientation_smoothness_weight, 100.0);
        nh.param<double>("gravity_alignment_weight", gravity_alignment_weight, 150.0);
        
        // IMPROVED: RK4 integration parameters
        nh.param<double>("max_integration_dt", max_integration_dt, 0.005); // Reduced for high-speed scenarios
        nh.param<double>("min_integration_dt", min_integration_dt, 1e-8); // Minimum step size
        nh.param<double>("bias_correction_threshold", bias_correction_threshold, 0.05); // Threshold for bias validity check

        nh.param<std::string>("bias_log_path", bias_path, "");

        nh.param<std::string>("results_log_path", results_log_param, "");
        nh.param<std::string>("metrics_log_path", metrics_log_param, "");

        ROS_INFO("ParameterServer loaded");
    }


    void openBiasLogFile() {
            bias_fs.open(bias_path, std::ios_base::out); // Open in write mode, overwriting if exists
            if (!bias_fs.is_open()) {
                ROS_ERROR("Failed to open bias log file for writing: %s", bias_path.c_str());
                return;
            }
            // Write header
            bias_fs << "timestamp,acc_bias_x,acc_bias_y,acc_bias_z,gyro_bias_x,gyro_bias_y,gyro_bias_z\n";
            bias_fs << std::fixed << std::setprecision(9); // Set precision for the file stream
            ROS_INFO("Bias log file opened and header written: %s", bias_path.c_str());
        }

    void openLogFiles() {
        // GPS Log
        gps_log_file.open(gps_log_path);
        if (gps_log_file.is_open()) {
            gps_log_file << "timestamp,px,py,pz,vx,vy,vz\n";
            ROS_INFO("Logging FGO GPS input to: %s", gps_log_path.c_str());
        } else {
            ROS_ERROR("Failed to open GPS log file: %s", gps_log_path.c_str());
        }

        // Ground Truth Log
        gt_log_file.open(gt_log_path);
        if (gt_log_file.is_open()) {
            gt_log_file << "timestamp,px,py,pz,vx,vy,vz,roll,pitch,yaw\n";
            ROS_INFO("Logging Ground Truth data to: %s", gt_log_path.c_str());
        } else {
            ROS_ERROR("Failed to open Ground Truth log file: %s", gt_log_path.c_str());
        }
        
        // optimized state log
        optimized_log_file.open(optimized_log_path);
        if (optimized_log_file.is_open()) {
            optimized_log_file << "timestamp,px,py,pz,qx,qy,qz,qw,vx,vy,vz,bax,bay,baz,bgx,bgy,bgz\n";
            ROS_INFO("Logging optimized state data to: %s", optimized_log_path.c_str());
        } else {
            ROS_ERROR("Failed to open optimized state log file: %s", optimized_log_path.c_str());
        }
    }

    void initLogging(CeresLogger &logger)
    {
        openBiasLogFile();
        // ceres logger path initialization
            std::string final_results_log_path = results_log_param;
            std::string final_metrics_log_path = metrics_log_param;
            // generate file name based on current time, if the param is empty
            if (final_results_log_path.empty()) {
                std::time_t now = std::time(nullptr);
                std::stringstream ss_results;
                char mbstr[100];
                if (std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S", std::localtime(&now))) {
                    ss_results << "./src/data/default_fusion_results_" << mbstr << ".txt";
                } else { // Fallback if strftime fails
                    ss_results << "./src/data/default_fusion_results_" << now << ".txt";
                }
                final_results_log_path = ss_results.str();
                ROS_INFO("Parameter 'results_log_path' is empty. Using default filename: %s", final_results_log_path.c_str());
            }

            if (final_metrics_log_path.empty()) {
                std::time_t now = std::time(nullptr);
                std::stringstream ss_metrics;
                char mbstr[100];
                if (std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S", std::localtime(&now))) {
                    ss_metrics << "./src/data/default_fusion_metrics_" << mbstr << ".txt";
                } else { // Fallback if strftime fails
                    ss_metrics << "./src/data/default_fusion_metrics_" << now << ".txt";
                }
                final_metrics_log_path = ss_metrics.str();
                ROS_INFO("Parameter 'metrics_log_path' is empty. Using default filename: %s", final_metrics_log_path.c_str());
            }
            logger.initialize(final_results_log_path, final_metrics_log_path);

            // --- ★ Log Static Configuration ONCE Here ★ ---
            ROS_INFO("Logging static configuration parameters...");
            logger.addMetadata("Config: Gravity Magnitude", std::to_string(gravity_magnitude));
            logger.addMetadata("Config: Use GPS Instead of UWB", use_gps_instead_of_uwb ? "True" : "False");
            logger.addMetadata("Config: Use GPS Orientation as Initial", use_gps_orientation_as_initial ? "True" : "False");
            logger.addMetadata("Config: Use GPS Orientation as Constraint", use_gps_orientation_as_constraint ? "True" : "False");
            logger.addMetadata("Config: Use GPS Velocity", use_gps_velocity ? "True" : "False");
            logger.addMetadata("Config: IMU Topic", imu_topic);
            logger.addMetadata("Config: GPS Topic", gnss_topic);
            logger.addMetadata("Config: Ground Truth Topic", ground_truth_topic);
            logger.addMetadata("Config: UWB Topic", uwb_topic);
            logger.addMetadata("Config: World Frame ID", world_frame_id);
            logger.addMetadata("Config: Body Frame ID", body_frame_id);
            logger.addMetadata("Config: IMU Acc Noise", std::to_string(imu_acc_noise));
            logger.addMetadata("Config: IMU Gyro Noise", std::to_string(imu_gyro_noise));
            logger.addMetadata("Config: IMU Acc Bias Noise", std::to_string(imu_acc_bias_noise));
            logger.addMetadata("Config: IMU Gyro Bias Noise", std::to_string(imu_gyro_bias_noise));
            logger.addMetadata("Config: GPS Pos Noise", std::to_string(gps_position_noise));
            logger.addMetadata("Config: GPS Vel Noise", std::to_string(gps_velocity_noise));
            //logger.addMetadata("Config: GPS Orient Noise", std::to_string(gps_orientation_noise));
            logger.addMetadata("Config: UWB Pos Noise", std::to_string(uwb_position_noise));
            logger.addMetadata("Config: Initial Acc Bias X", std::to_string(initial_acc_bias_x));
            logger.addMetadata("Config: Initial Acc Bias Y", std::to_string(initial_acc_bias_y));
            logger.addMetadata("Config: Initial Acc Bias Z", std::to_string(initial_acc_bias_z));
            logger.addMetadata("Config: Initial Gyro Bias X", std::to_string(initial_gyro_bias_x));
            logger.addMetadata("Config: Initial Gyro Bias Y", std::to_string(initial_gyro_bias_y));
            logger.addMetadata("Config: Initial Gyro Bias Z", std::to_string(initial_gyro_bias_z));
            logger.addMetadata("Config: Optimization Window Size", std::to_string(optimization_window_size));
            logger.addMetadata("Config: Optimization Frequency (Hz)", std::to_string(optimization_frequency));
            logger.addMetadata("Config: Max Iterations (Param)", std::to_string(max_iterations)); // Log the configured value
            logger.addMetadata("Config: Enable Bias Estimation", enable_bias_estimation ? "True" : "False");
            logger.addMetadata("Config: Enable Marginalization", enable_marginalization ? "True" : "False");
            logger.addMetadata("Config: Enable Roll/Pitch Constraint", enable_roll_pitch_constraint ? "True" : "False");

            logger.addMetadata("Config: Enable Orientation Smoothness Factor", enable_orientation_smoothness_factor ? "True" : "False");
            logger.addMetadata("Config: Enable Velocity Constraint", enable_velocity_constraint ? "True" : "False");
            logger.addMetadata("Config: Enable Horizontal Velocity Incentive", enable_horizontal_velocity_incentive ? "True" : "False");
            logger.addMetadata("Config: Enable IMU Orientation Factor", enable_imu_orientation_factor ? "True" : "False");
            logger.addMetadata("Config: Bias Max Acc", std::to_string(acc_bias_max));
            logger.addMetadata("Config: Bias Max Gyro", std::to_string(gyro_bias_max));
            logger.addMetadata("Config: Roll/Pitch Weight", std::to_string(roll_pitch_weight));
            logger.addMetadata("Config: IMU Orientation Weight", std::to_string(imu_orientation_weight));
            logger.addMetadata("Config: Bias Constraint Weight", std::to_string(bias_constraint_weight));
            logger.addMetadata("Config: Max Velocity Setting", std::to_string(max_velocity));
            logger.addMetadata("Config: Velocity Constraint Weight", std::to_string(velocity_constraint_weight));
            logger.addMetadata("Config: Min Horizontal Velocity", std::to_string(min_horizontal_velocity));
            logger.addMetadata("Config: Horizontal Velocity Weight", std::to_string(horizontal_velocity_weight));
            logger.addMetadata("Config: Orientation Smoothness Weight", std::to_string(orientation_smoothness_weight));
            logger.addMetadata("Config: Gravity Alignment Weight", std::to_string(gravity_alignment_weight));
            logger.addMetadata("Config: Max Integration dt", std::to_string(max_integration_dt));
            logger.addMetadata("Config: Bias Correction Threshold", std::to_string(bias_correction_threshold));

            // ★★★ Call log() immediately after adding static metadata ★★★
            // The modified log() method in CeresLogger handles this initial call correctly.
            bool static_log_success = logger.log();
            if (!static_log_success) {
                ROS_ERROR("Failed to write initial static configuration to log files!");
            } else {
                ROS_INFO("Static configuration logged successfully.");
            }
        // logger state is now reset automatically by log().
        // end of logger initialization
        openLogFiles();
        
    }
};


#endif // PARAMETER_SERVER_H