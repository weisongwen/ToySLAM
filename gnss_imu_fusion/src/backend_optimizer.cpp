#include "backend_optimizer.h"
#include <sensor_msgs/Imu.h> // for estimateMaxVelocityFromImu

BackendOptimizer::BackendOptimizer(ParameterServer& params) : params_(params) {
    // 初始化日志记录器
    params_.initLogging(logger_);
    
    // 初始化世界坐标系下的重力向量 (ENU, Z轴向上)
    gravity_world_ = Eigen::Vector3d(0, 0, -params_.gravity_magnitude);
}

void BackendOptimizer::clampBiases(Eigen::Vector3d& acc_bias, Eigen::Vector3d& gyro_bias) {
    double acc_bias_norm = acc_bias.norm();
    double gyro_bias_norm = gyro_bias.norm();
    
    if (acc_bias_norm > params_.acc_bias_max) {
        acc_bias *= (params_.acc_bias_max / acc_bias_norm);
    }
    if (gyro_bias_norm > params_.gyro_bias_max) {
        gyro_bias *= (params_.gyro_bias_max / gyro_bias_norm);
    }
}

void BackendOptimizer::clampVelocity(Eigen::Vector3d& velocity, double max_velocity = 55.0) {
    double velocity_norm = velocity.norm();
        
    if (velocity_norm > max_velocity) {
        // Scale velocity proportionally to keep direction but limit magnitude
        velocity *= (max_velocity / velocity_norm);
        ROS_DEBUG("Velocity clamped from %.2f to %.2f m/s (%.1f km/h)", 
                    velocity_norm, max_velocity, max_velocity * 3.6);
    }
    
    // If horizontal velocity incentive is disabled, don't enforce minimum
    if (!params_.enable_horizontal_velocity_incentive) {
        return;
    }
    
    // Only enforce a minimum horizontal velocity if it's very small and we're not moving vertically
    double h_vel_norm = std::sqrt(velocity.x()*velocity.x() + velocity.y()*velocity.y());
    double v_vel_abs = std::abs(velocity.z());
    
    // If horizontal velocity is small but vertical velocity is significant, don't enforce minimum
    if (h_vel_norm < 0.05 && v_vel_abs < 0.5) {
        // Set a minimum velocity in the current horizontal direction or along x-axis if zero
        if (h_vel_norm > 1e-6) {
            // Scale up existing direction to minimum
            double scale = params_.min_horizontal_velocity * 0.2 / h_vel_norm;
            velocity.x() *= scale;
            velocity.y() *= scale;
        } else {
            // Add a small default velocity along x-axis
            velocity.x() = params_.min_horizontal_velocity * 0.2;
            velocity.y() = 0.0;
        }
        
        // Re-verify total velocity is within bounds
        velocity_norm = velocity.norm();
        if (velocity_norm > max_velocity) {
            velocity *= (max_velocity / velocity_norm);
        }
    }
}

// Estimate velocity magnitude from IMU data
double BackendOptimizer::estimateMaxVelocityFromImu(const std::deque<sensor_msgs::Imu>& imu_buffer) {
    // Default value if we don't have enough data
    double estimated_max_velocity = params_.max_velocity;
    
    // Look at recent IMU data to estimate reasonable velocity bound
    if (imu_buffer.size() >= 10) {
        double max_acc = 0.0;
        
        // Find maximum acceleration magnitude in recent data
        for (size_t i = imu_buffer.size() - 10; i < imu_buffer.size(); i++) {
            const auto& imu = imu_buffer[i];
            double acc_mag = std::sqrt(
                imu.linear_acceleration.x * imu.linear_acceleration.x +
                imu.linear_acceleration.y * imu.linear_acceleration.y +
                imu.linear_acceleration.z * imu.linear_acceleration.z
            );
            max_acc = std::max(max_acc, acc_mag - params_.gravity_magnitude);
        }
        
        // Use a reasonable time period for acceleration (1-5 seconds)
        // v = a * t, assuming constant acceleration
        double assumed_acc_time = 3.0;
        double potential_max_vel = max_acc * assumed_acc_time;
        
        // Ensure a reasonable range: between 5 m/s and 35 m/s (18-126 km/h)
        estimated_max_velocity = std::max(5.0, std::min(35.0, potential_max_vel));
    }
    
    return estimated_max_velocity;
}

// Prepare marginalization by adding factors connected to the oldest state
void BackendOptimizer::prepareMarginalization(
    const std::deque<State, Eigen::aligned_allocator<State>>& state_window,
    const std::vector<GnssMeasurement>& gps_measurements,
    const std::map<std::pair<double, double>, imu_preint>& preintegration_map,
    MarginalizationInfo*& last_marginalization_info)
{
    try {
        if (!params_.enable_marginalization || state_window.size() < 2) {
            return;
        }
        
        // Create a new marginalization info object
        MarginalizationInfo* marginalization_info = new MarginalizationInfo();
        
        // Local tracking of allocated memory for cleanup in case of exception
        std::vector<double*> local_allocations;
        
        try {
            // The oldest state is being marginalized
            const State& oldest_state = state_window.front();
            const State& next_state = state_window[1];
            
            // Create parameter blocks for the two states involved
            double* pose_param1 = new double[7];
            double* vel_param1 = new double[3];
            double* bias_param1 = new double[6];
            double* pose_param2 = new double[7];
            double* vel_param2 = new double[3];
            double* bias_param2 = new double[6];
            
            // Add to local tracking for cleanup in case of exception
            local_allocations.push_back(pose_param1);
            local_allocations.push_back(vel_param1);
            local_allocations.push_back(bias_param1);
            local_allocations.push_back(pose_param2);
            local_allocations.push_back(vel_param2);
            local_allocations.push_back(bias_param2);
            
            // Copy the state data to the parameters
            // Oldest state
            pose_param1[0] = oldest_state.position.x();
            pose_param1[1] = oldest_state.position.y();
            pose_param1[2] = oldest_state.position.z();
            pose_param1[3] = oldest_state.orientation.x();
            pose_param1[4] = oldest_state.orientation.y();
            pose_param1[5] = oldest_state.orientation.z();
            pose_param1[6] = oldest_state.orientation.w();
            
            vel_param1[0] = oldest_state.velocity.x();
            vel_param1[1] = oldest_state.velocity.y();
            vel_param1[2] = oldest_state.velocity.z();
            
            bias_param1[0] = oldest_state.acc_bias.x();
            bias_param1[1] = oldest_state.acc_bias.y();
            bias_param1[2] = oldest_state.acc_bias.z();
            bias_param1[3] = oldest_state.gyro_bias.x();
            bias_param1[4] = oldest_state.gyro_bias.y();
            bias_param1[5] = oldest_state.gyro_bias.z();
            
            // Next state
            pose_param2[0] = next_state.position.x();
            pose_param2[1] = next_state.position.y();
            pose_param2[2] = next_state.position.z();
            pose_param2[3] = next_state.orientation.x();
            pose_param2[4] = next_state.orientation.y();
            pose_param2[5] = next_state.orientation.z();
            pose_param2[6] = next_state.orientation.w();
            
            vel_param2[0] = next_state.velocity.x();
            vel_param2[1] = next_state.velocity.y();
            vel_param2[2] = next_state.velocity.z();
            
            bias_param2[0] = next_state.acc_bias.x();
            bias_param2[1] = next_state.acc_bias.y();
            bias_param2[2] = next_state.acc_bias.z();
            bias_param2[3] = next_state.gyro_bias.x();
            bias_param2[4] = next_state.gyro_bias.y();
            bias_param2[5] = next_state.gyro_bias.z();
            
            // Add position factor for the oldest state based on fusion mode
            if (params_.use_gps_instead_of_uwb) {
                double keyframe_time = oldest_state.timestamp;
                    std::optional<GnssMeasurement> matching_gps_meas;
                for (const auto& gps : gps_measurements) {
                    if (std::abs(gps.timestamp - keyframe_time) < 0.05) {
                        matching_gps_meas = gps;
                        break;
                    }
                }
                // Add GPS position factor
                if(matching_gps_meas) {
                    // --- A. 为边缘化问题添加GPS位置因子 ---
                    // 检查标志位，并使用之前在 optimizeFactorGraph 中存储的、最终使用的协方差
                    if (oldest_state.has_gps_pos_factor) {
                        ROS_INFO("Marginalizing GPS position factor with stored covariance.");
                        ceres::CostFunction* gps_factor = GpsPositionFactor::Create(
                            matching_gps_meas->position,
                            oldest_state.final_gps_pos_cov // ★ 使用存储的协方差
                        );
                        
                        std::vector<double*> parameter_blocks = {pose_param1};
                        std::vector<int> drop_set = {0};
                        auto* residual_info = new ResidualBlockInfo(gps_factor, nullptr, parameter_blocks, drop_set);
                        marginalization_info->addResidualBlockInfo(residual_info);
                    }
                    
                    // --- B. 为边缘化问题添加GPS速度因子 ---
                    // 检查标志位，并使用存储的协方差
                    if (params_.use_gps_velocity && oldest_state.has_gps_vel_factor) {
                        ROS_INFO("Marginalizing GPS velocity factor with stored covariance.");
                        ceres::CostFunction* gps_vel_factor = GpsVelocityFactor::Create(
                            matching_gps_meas->velocity,
                            oldest_state.final_gps_vel_cov // ★ 使用存储的协方差
                        );
                        
                        std::vector<double*> vel_parameter_blocks = {vel_param1};
                        std::vector<int> vel_drop_set = {0};
                        auto* vel_residual_info = new ResidualBlockInfo(gps_vel_factor, nullptr, vel_parameter_blocks, vel_drop_set);
                        marginalization_info->addResidualBlockInfo(vel_residual_info);
                    }
                }
            } 
            // else {
            //     // Add UWB position factor
            //     for (const auto& uwb : uwb_measurements_) {
            //         if (std::abs(uwb.timestamp - oldest_state.timestamp) < 0.01) {
            //             // Create UWB factor
            //             ceres::CostFunction* uwb_factor = UwbPositionFactor::Create(
            //                 uwb.position, params.uwb_position_noise);
                        
            //             std::vector<double*> parameter_blocks = {pose_param1};
            //             std::vector<int> drop_set = {0}; // Drop the pose parameter
                        
            //             auto* residual_info = new ResidualBlockInfo(
            //                 uwb_factor, nullptr, parameter_blocks, drop_set);
            //             marginalization_info->addResidualBlockInfo(residual_info);
            //             break;
            //         }
            //     }
            // }
            
            // Add IMU factor between oldest state and second oldest state
            double start_time = oldest_state.timestamp;
            double end_time = next_state.timestamp;
            std::pair<double, double> key(start_time, end_time);
            


            // adopt to new imu factor
            if (preintegration_map.find(key) != preintegration_map.end()) {
                //auto& preint = preintegration_map[key];
                const auto& preint = preintegration_map.at(key);
                ceres::CostFunction* imu_factor_ = new imu_factor(&preint);
                
                std::vector<double*> parameter_blocks = {
                    pose_param1, vel_param1, bias_param1,
                    pose_param2, vel_param2, bias_param2
                };
                
                // Drop only parameters from oldest state
                std::vector<int> drop_set = {0, 1, 2}; // Pose, velocity, bias of oldest state
                
                auto* residual_info = new ResidualBlockInfo(
                    imu_factor_, nullptr, parameter_blocks, drop_set);
                marginalization_info->addResidualBlockInfo(residual_info);
                
            }
            

            // add last marginalization factor 
            if(last_marginalization_info) {
                // Add last marginalization info to the new marginalization info
                MarginalizationFactor* last_margin_factor = new MarginalizationFactor(last_marginalization_info);
                std::vector<double*> last_margin_params = {pose_param1, vel_param1, bias_param1};
                std::vector<int> last_margin_drop_set = {0, 1, 2}; // Drop pose, velocity, bias of oldest state

                auto* last_margin_residual = new ResidualBlockInfo(
                    last_margin_factor, nullptr, last_margin_params, last_margin_drop_set);
                marginalization_info->addResidualBlockInfo(last_margin_residual);
                    
            }
            
            // Perform pre-marginalization
            marginalization_info->preMarginalize();
            
            // Perform marginalization
            marginalization_info->marginalize();
            
            // Clean up previous marginalization info
            if (last_marginalization_info) {
                delete last_marginalization_info;
                last_marginalization_info = nullptr;
            }
            
            // Store new marginalization info
            last_marginalization_info = marginalization_info;
            
            // Clear local allocations since they're now owned by marginalization_info
            local_allocations.clear();
            
        } catch (const std::exception& e) {
            // Clean up locally allocated memory if exception occurs
            for (auto ptr : local_allocations) {
                delete[] ptr;
            }
            delete marginalization_info;
            throw;
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in prepareMarginalization: %s", e.what());
    }
}


bool BackendOptimizer::optimize(
    std::deque<State, Eigen::aligned_allocator<State>>& state_window,
    const std::vector<GnssMeasurement>& gps_measurements,
    std::map<std::pair<double, double>, imu_preint>& preintegration_map,
    MarginalizationInfo*& last_marginalization_info,
    int& optimization_count,
    const std::deque<sensor_msgs::Imu>& imu_buffer)
{
    if (state_window.size() < 2) {
        return false;
    }
 
    // Store original feature flags and max iterations
    bool original_enable_horizontal_velocity_incentive = params_.enable_horizontal_velocity_incentive;
    bool original_enable_orientation_smoothness_factor = params_.enable_orientation_smoothness_factor;
    int original_max_iterations = params_.max_iterations;
    State current_state;
    
    // Check for initial optimization phase
    bool is_first_optimization = (optimization_count < 5);
    if (is_first_optimization) {
        // Disable complex features during initial iterations to improve stability
        params_.enable_horizontal_velocity_incentive = false;
        params_.enable_orientation_smoothness_factor = false;
        // max_iterations_ = 5; // Use fewer iterations during initial phase
        ROS_DEBUG("Using simplified optimization for initial phase (%d/5)", optimization_count+1);
    }
    
    // Create Ceres problem
    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;
    ceres::Problem problem(problem_options);
    
    // Create pose parameterization
    ceres::LocalParameterization* pose_parameterization = new PoseParameterization();
    // ROS_INFO("Try to add state variables for ceres");
    
    try {
                    
        // Preallocate with reserve
        // ROS_INFO("Preallocate with reserve");
        std::vector<OptVariables, Eigen::aligned_allocator<OptVariables>> variables;
        variables.reserve(state_window.size());
        
        // Initialize variables from state window
        for (size_t i = 0; i < state_window.size(); ++i) {
            OptVariables var;
            const auto& state = state_window[i];
            
            // Position
            var.pose[0] = state.position.x();
            var.pose[1] = state.position.y();
            var.pose[2] = state.position.z();

            
            // Orientation (quaternion): in x y z w order
            var.pose[3] = state.orientation.x();
            var.pose[4] = state.orientation.y();
            var.pose[5] = state.orientation.z();
            var.pose[6] = state.orientation.w();
            
            // Velocity
            var.velocity[0] = state.velocity.x();
            var.velocity[1] = state.velocity.y();
            var.velocity[2] = state.velocity.z();
            
            // Biases
            var.bias[0] = state.acc_bias.x();
            var.bias[1] = state.acc_bias.y();
            var.bias[2] = state.acc_bias.z();
            var.bias[3] = state.gyro_bias.x();
            var.bias[4] = state.gyro_bias.y();
            var.bias[5] = state.gyro_bias.z();
            
            variables.push_back(var);
        }


        
        // Add pose parameterization
        for (size_t i = 0; i < state_window.size(); ++i) {
            problem.AddParameterBlock(variables[i].pose, 7, pose_parameterization);
            problem.AddParameterBlock(variables[i].velocity, 3);
            problem.AddParameterBlock(variables[i].bias, 6);
        }
        
        // If bias estimation is disabled, set biases constant
        // set bias as constant to test the result
        if (!params_.enable_bias_estimation) {
            for (size_t i = 0; i < state_window.size(); ++i) {
                problem.SetParameterBlockConstant(variables[i].bias);
            }
        }
        
        // Add position measurements based on fusion mode
        if (params_.use_gps_instead_of_uwb) {
            for (size_t i = 0; i < state_window.size(); ++i) {
                double keyframe_time = state_window[i].timestamp;
                
                // 找到匹配的GPS测量数据
                std::optional<GnssMeasurement> matching_gps_meas;
                for (const auto& gps : gps_measurements) {
                    if (std::abs(gps.timestamp - keyframe_time) < 0.05) {
                        matching_gps_meas = gps;
                        break;
                    }
                }

                if (!matching_gps_meas) {
                    continue;
                }

                // 重置当前帧的标志位和协方差
                state_window[i].has_gps_pos_factor = false;
                state_window[i].has_gps_vel_factor = false;

                double pos_covariance_scale = 1.0;
                double vel_covariance_scale = 1.0;

                // --- 卡方一致性检验 (从第二个关键帧开始) ---
                if (params_.enable_consistency_check && i > 0 && optimization_count > params_.initial_grace_epochs) {
                    const auto& prev_state = state_window[i-1];
                    const auto& current_meas = *matching_gps_meas;

                    std::pair<double, double> key(prev_state.timestamp, keyframe_time);
                    if (preintegration_map.count(key)) {
                        const auto& preint = preintegration_map.at(key);
                        double dt = preint.get_sum_dt();

                        State propagated_state = propagateState(prev_state, current_meas.timestamp, imu_buffer);

                        // --- 位置检验 ---
                        if (current_meas.position_valid) {
                            Eigen::Vector3d predicted_pos = propagated_state.position;
                            Eigen::Vector3d innovation_pos = current_meas.position - predicted_pos;
                            Eigen::Matrix3d S_pos = preint.getCovariance().block<3, 3>(0, 0) + current_meas.position_covariance;
                            double nis_pos = innovation_pos.transpose() * S_pos.inverse() * innovation_pos;

                            if (nis_pos > params_.nis_threshold_position) {
                                pos_covariance_scale = std::min(params_.max_covariance_scale_factor, nis_pos / 3.0);
                                pos_covariance_scale = std::max(1.0, pos_covariance_scale);
                                ROS_WARN("GPS position didn't pass consistency check! NIS=%.2f > threshold=%.2f. covariance will be scale %.2f times (frame %zu)",
                                        nis_pos, params_.nis_threshold_position, pos_covariance_scale, i);
                            }
                        }
                        
                        // --- 速度检验 ---
                        if (params_.use_gps_velocity && current_meas.velocity_valid) {
                            Eigen::Vector3d predicted_vel = propagated_state.velocity;
                            Eigen::Vector3d innovation_vel = current_meas.velocity - predicted_vel;
                            Eigen::Matrix3d S_vel = preint.getCovariance().block<3, 3>(6, 6) + current_meas.velocity_covariance;
                            double nis_vel = innovation_vel.transpose() * S_vel.inverse() * innovation_vel;

                            if (nis_vel > params_.nis_threshold_velocity) {
                                vel_covariance_scale = std::min(params_.max_covariance_scale_factor, nis_vel / 3.0);
                                vel_covariance_scale = std::max(1.0, vel_covariance_scale);
                                ROS_WARN("GPS velocity didn't pass consistency check! NIS=%.2f > threshold=%.2f. covariance will be scale %.2f times (frame %zu)",
                                        nis_vel, params_.nis_threshold_velocity, vel_covariance_scale, i);
                            }
                        }
                    }
                }

                // --- 添加GPS位置因子 ---
                if (matching_gps_meas->position_valid) {
                    Eigen::Matrix3d position_noise_cov = matching_gps_meas->position_covariance.norm() > 1e-8 ?
                                                        matching_gps_meas->position_covariance :
                                                        Eigen::Matrix3d::Identity() * params_.gps_position_noise * params_.gps_position_noise;
                    
                    position_noise_cov *= pos_covariance_scale; // 应用缩放因子

                    const double min_pos_variance = 0.4; // 对应标准差为 0.2米 (20cm)
                    if (position_noise_cov.trace() < min_pos_variance * 3) {
                        // 如果协方差矩阵太小，就用设定的下限值来覆盖它
                        position_noise_cov = Eigen::Matrix3d::Identity() * min_pos_variance;
                        // ROS_WARN("GPS position covariance is too optimistic. Using floor value (std=%.2f m).", std::sqrt(min_pos_variance));
                    }
                    // 【重要】将最终使用的协方差存入State对象，供边缘化使用
                    state_window[i].final_gps_pos_cov = position_noise_cov;
                    state_window[i].has_gps_pos_factor = true;

                    ceres::CostFunction* gps_pos_factor = GpsPositionFactor::Create(matching_gps_meas->position, position_noise_cov);
                    problem.AddResidualBlock(gps_pos_factor, new ceres::HuberLoss(1.0), variables[i].pose);
                }

                // --- 添加GPS速度因子 ---
                if (params_.use_gps_velocity && matching_gps_meas->velocity_valid) {
                    Eigen::Matrix3d velocity_noise_cov = matching_gps_meas->velocity_covariance.norm() > 1e-8 ?
                                                        matching_gps_meas->velocity_covariance :
                                                        Eigen::Matrix3d::Identity() * params_.gps_velocity_noise * params_.gps_velocity_noise;
                    
                    velocity_noise_cov *= vel_covariance_scale; // 应用缩放因子

                    const double min_vel_variance = 0.1; // 对应标准差为 0.1m/s
                    if (velocity_noise_cov.trace() < min_vel_variance * 3) {
                        velocity_noise_cov = Eigen::Matrix3d::Identity() * min_vel_variance;
                        // ROS_WARN("GPS velocity covariance is too optimistic. Using floor value (std=%.2f m/s).", std::sqrt(min_vel_variance));
                    }
                    // 【重要】将最终使用的协方差存入State对象，供边缘化使用
                    state_window[i].final_gps_vel_cov = velocity_noise_cov;
                    state_window[i].has_gps_vel_factor = true;
                    
                    ceres::CostFunction* gps_vel_factor = GpsVelocityFactor::Create(matching_gps_meas->velocity, velocity_noise_cov);
                    problem.AddResidualBlock(gps_vel_factor, new ceres::HuberLoss(1.0), variables[i].velocity);
                }
            }
        }
        
        // Add roll/pitch constraint to enforce planar motion if enabled
        if (params_.enable_roll_pitch_constraint) {
            ROS_INFO("Added roll/pitch constraints");
            for (size_t i = 0; i < state_window.size(); ++i) {
                ceres::CostFunction* roll_pitch_prior = RollPitchPriorFactor::Create(params_.roll_pitch_weight);
                problem.AddResidualBlock(roll_pitch_prior, nullptr, variables[i].pose);
            }
        }
        
        
        // Add orientation smoothness constraints between consecutive keyframes if enabled
        if (params_.enable_orientation_smoothness_factor) {
            ROS_INFO("Added orientation smoothness factors");
            for (size_t i = 0; i < state_window.size() - 1; ++i) {
                ceres::CostFunction* orientation_smoothness = 
                    OrientationSmoothnessFactor::Create(params_.orientation_smoothness_weight);
                
                problem.AddResidualBlock(orientation_smoothness, nullptr, 
                                        variables[i].pose, variables[i+1].pose);
            }
            
            // Add orientation smoothness constraints between non-adjacent keyframes (i and i+2)
            for (size_t i = 0; i < state_window.size() - 2; ++i) {
                ceres::CostFunction* orientation_smoothness = 
                    OrientationSmoothnessFactor::Create(params_.orientation_smoothness_weight * 0.5);
                
                problem.AddResidualBlock(orientation_smoothness, nullptr, 
                                        variables[i].pose, variables[i+2].pose);
            }
        }
        

        
        // Add IMU pre-integration factors between keyframes
        for (size_t i = 0; i < state_window.size() - 1; ++i) {
            double start_time = state_window[i].timestamp;
            double end_time = state_window[i+1].timestamp;
            
            // Skip if the time interval is too short
            if (end_time - start_time < 1e-6) continue;
            
            std::pair<double, double> key(start_time, end_time);
            
            if (preintegration_map.find(key) != preintegration_map.end()) {
                // const auto& preint = preintegration_map_[key];
                
                // // IMPROVED: Pass bias correction threshold to IMU factor
                // ceres::CostFunction* imu_factor = ImuFactor::Create(
                //     preint, gravity_world_, bias_correction_threshold_);
                
                // // IMPROVED: Use HuberLoss for IMU factor
                // problem.AddResidualBlock(imu_factor, NULL,
                //                        variables[i].pose, variables[i].velocity, variables[i].bias,
                //                        variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);
                //auto& preint = preintegration_map[key];
                const auto& preint = preintegration_map.at(key);
                ceres::CostFunction* imu_factor_ = new imu_factor(&preint);

                // ROS_INFO("IMU PREINT NOISE LEVEL: acc, gyro, acc_bias, gyro_bias = %.3f, %.3f, %.3f, %.3f",
                //         preint.getAccNoiseSigma(), preint.getGyroNoiseSigma(),
                //         preint.getAccBiasWalkSigma(), preint.getGyroBiasWalkSigma());

                problem.AddResidualBlock(imu_factor_, NULL,
                                        variables[i].pose, variables[i].velocity, variables[i].bias,
                                        variables[i+1].pose, variables[i+1].velocity, variables[i+1].bias);


            }
        }
        
        // Add marginalization prior if it exists and marginalization is enabled
        if (params_.enable_marginalization && last_marginalization_info && state_window.size() >= 2) {
            // Create a new marginalization factor
            MarginalizationFactor* factor = new MarginalizationFactor(last_marginalization_info);
            
            // CRITICAL: Always use exactly 6 parameter blocks in the exact order expected
            // Adding the residual block with state variables in the correct order, without checks
            if (state_window.size() >= 2) {
                // problem.AddResidualBlock(factor, nullptr,
                //                        variables[1].pose, variables[1].velocity, variables[1].bias,
                //                        variables[0].pose, variables[0].velocity, variables[0].bias);
                // problem.AddResidualBlock(factor, nullptr,
                //                        variables[0].pose, variables[0].velocity, variables[0].bias,
                //                        variables[1].pose, variables[1].velocity, variables[1].bias);
                problem.AddResidualBlock(factor, nullptr,
                    variables[0].pose, variables[0].velocity, variables[0].bias);
            }
        }
        
        logger_.addMetadata("Problem: Num Residuals", std::to_string(problem.NumResiduals()));
        logger_.addMetadata("Problem: Num Parameter Blocks", std::to_string(problem.NumParameterBlocks()));

        // Configure solver options
        ceres::Solver::Options options;
        options.max_num_iterations = params_.max_iterations;
        // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        // options.trust_region_strategy_type = ceres::DOGLEG;
        // options.line_search_direction_type = ceres::LBFGS;

        // ROS_INFO("Optimization problem has %zu residuals and %zu parameter blocks",
        //         problem.NumResiduals(), problem.NumParameterBlocks());
        // add output for debug use
        // options.minimizer_progress_to_stdout = false;
        // options.minimizer_progress_to_stdout = true;
        options.num_threads = 16;

        
        // Solve the optimization problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        logger_.setSummary(summary);
        
        if (!summary.IsSolutionUsable()) {
            // ROS_WARN("Optimization failed or solution not usable. Report:\n%s", summary.FullReport().c_str());
                logger_.addMetadata("Optimization Status", "Failed - Solution Unusable");
                logger_.log(); // ★ Attempt to log partial dynamic info + summary on failure ★
                // Restore original settings before returning
                params_.enable_horizontal_velocity_incentive = original_enable_horizontal_velocity_incentive;
                params_.enable_orientation_smoothness_factor = original_enable_orientation_smoothness_factor;
                params_.max_iterations = original_max_iterations;
                return false;
        }

            
        
        // Update state with optimized values
        for (size_t i = 0; i < state_window.size(); ++i) {

            Eigen::Vector3d old_pos = state_window[i].position;
            // Update position
            state_window[i].position = Eigen::Vector3d(
                variables[i].pose[0], variables[i].pose[1], variables[i].pose[2]);
            
            // Update orientation
            state_window[i].orientation = Eigen::Quaterniond(
                variables[i].pose[6], variables[i].pose[3], variables[i].pose[4], variables[i].pose[5]).normalized();
            
            // Get velocity and ensure it's reasonable while preserving direction
            Eigen::Vector3d new_velocity(
                variables[i].velocity[0], variables[i].velocity[1], variables[i].velocity[2]);
            
            // Use adaptive max velocity for high-speed scenario
            double adaptive_max_velocity = params_.max_velocity;
            if (imu_buffer.size() > 10) {
                adaptive_max_velocity = estimateMaxVelocityFromImu(imu_buffer);
            }
            
            clampVelocity(new_velocity, adaptive_max_velocity);
            state_window[i].velocity = new_velocity;
            
            // Update biases if enabled
            if (params_.enable_bias_estimation) {
                // First, get biases from optimization
                Eigen::Vector3d new_acc_bias(
                    variables[i].bias[0], variables[i].bias[1], variables[i].bias[2]);
                
                Eigen::Vector3d new_gyro_bias(
                    variables[i].bias[3], variables[i].bias[4], variables[i].bias[5]);
                
                // Ensure biases stay within reasonable limits
                clampBiases(new_acc_bias, new_gyro_bias);

                // check the difference of new and old biases
                double acc_bias_diff = (new_acc_bias - state_window[i].acc_bias).norm();
                double gyro_bias_diff = (new_gyro_bias - state_window[i].gyro_bias).norm();

                // If the difference is too large, repropagate the preintegration
                if (acc_bias_diff > 0.0005 || gyro_bias_diff > 0.005) {
                    // ROS_WARN("Large bias difference detected: acc [%.3f, %.3f, %.3f], gyro [%.3f, %.3f, %.3f]",
                    //     acc_bias_diff, gyro_bias_diff);

                    std::pair<double, double> key(state_window[i].timestamp, state_window[i+1].timestamp);
                    // repropagate calculating the preintegration
                    if (preintegration_map.find(key) != preintegration_map.end()) {
                        auto& preint = preintegration_map[key];
                        preint.repropagate(new_acc_bias, new_gyro_bias);
                    }
                }
                // Clamp biases to ensure they are within reasonable limits


                // Update state with clamped biases
                state_window[i].acc_bias = new_acc_bias;
                state_window[i].gyro_bias = new_gyro_bias;

                // 打印零偏
                ROS_INFO("Epoch %zu: Acc Bias [%.6f, %.6f, %.6f], Gyro Bias [%.6f, %.6f, %.6f]",
                    i,
                    new_acc_bias.x(), new_acc_bias.y(), new_acc_bias.z(),
                    new_gyro_bias.x(), new_gyro_bias.y(), new_gyro_bias.z());

            }
        }
        
        // get the newest state in the state window
        const State& newest_state = state_window.back();
        //logKeyframeBias(newest_state);
        
        // Update current state to the latest state in the window
        if (!state_window.empty()) {
            
            current_state = state_window.back();
            
            // Keep bias constraints consistent across the system
            clampBiases(current_state.acc_bias, current_state.gyro_bias);
            
            // Ensure velocity stays within reasonable limits while preserving direction
            double adaptive_max_velocity = params_.max_velocity;
            if (imu_buffer.size() > 10) {
                adaptive_max_velocity = estimateMaxVelocityFromImu(imu_buffer);
            }
            clampVelocity(current_state.velocity, adaptive_max_velocity);

            //logOptimizedState(current_state);
        }

        // Log detailed velocity information
        if (!state_window.empty()) {
            double vel_norm = current_state.velocity.norm();
            // ROS_INFO("After optimization: velocity [%.2f, %.2f, %.2f] m/s, magnitude: %.2f m/s (%.1f km/h)",
            //         current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z(),
            //         vel_norm, vel_norm * 3.6);
            
            // If we have GPS data, compare with GPS velocity
            if (params_.use_gps_instead_of_uwb && !gps_measurements.empty()) {
                // Find closest GPS measurement to current time
                double min_time_diff = std::numeric_limits<double>::max();
                GnssMeasurement closest_gps;
                bool found_gps = false;
                
                for (const auto& gps : gps_measurements) {
                    double time_diff = std::abs(gps.timestamp - current_state.timestamp);
                    if (time_diff < min_time_diff) {
                        min_time_diff = time_diff;
                        closest_gps = gps;
                        found_gps = true;
                    }
                }
                
                if (found_gps && min_time_diff < 1.0) {  // Within 1 second
                    double gps_vel_norm = closest_gps.velocity.norm();
                    double vel_diff = (current_state.velocity - closest_gps.velocity).norm();
                    // ROS_INFO("velocity comparison: current_state_ [%.2f, %.2f, %.2f] m/s , closest_gps: [%.2f, %.2f, %.2f] m/s",
                    //          current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z(),
                    //          closest_gps.velocity.x(), closest_gps.velocity.y(), closest_gps.velocity.z());
                    //ROS_INFO("GPS velocity comparison: GPS [%.2f, %.2f, %.2f] m/s (%.1f km/h), diff: %.2f m/s (%.1f km/h)",
                    //         closest_gps.velocity.x(), closest_gps.velocity.y(), closest_gps.velocity.z(),
                    //         gps_vel_norm * 3.6, vel_diff, vel_diff * 3.6);
                }
            }
        }

        // --- ★ 7. Add Optimized Parameters to Logger ★ ---
        // This should happen AFTER the state_window has been updated
        // with the final, potentially clamped/normalized values from the 'variables' array.
        // ROS_INFO("Logging optimized parameters...");
        for (size_t i = 0; i < state_window.size(); ++i) {
            // Get the finalized state from the window
            const auto& final_state = state_window[i];
            // Create a unique prefix for this state's parameters in the log
            std::string prefix = "State_" + std::to_string(i) + "_"; // e.g., "State_0_", "State_1_"

            // --- Log Pose (Position + Orientation Quaternion) ---
            std::vector<double> pose_data(7);
            // Position (x, y, z)
            pose_data[0] = final_state.position.x();
            pose_data[1] = final_state.position.y();
            pose_data[2] = final_state.position.z();
            // Orientation Quaternion (w, x, y, z) - Ensure correct order w,x,y,z
            pose_data[3] = final_state.orientation.x();
            pose_data[4] = final_state.orientation.y();
            pose_data[5] = final_state.orientation.z();
            pose_data[6] = final_state.orientation.w();
            logger_.addParameterBlock(prefix + "Pose", pose_data);

            // --- Log Velocity ---
            std::vector<double> vel_data(3);
            vel_data[0] = final_state.velocity.x();
            vel_data[1] = final_state.velocity.y();
            vel_data[2] = final_state.velocity.z();
            logger_.addParameterBlock(prefix + "Velocity", vel_data);

            // --- Log Biases (Accelerometer + Gyroscope) ---
            std::vector<double> bias_data(6);
            // Accelerometer Bias (ax, ay, az)
            bias_data[0] = final_state.acc_bias.x();
            bias_data[1] = final_state.acc_bias.y();
            bias_data[2] = final_state.acc_bias.z();
            // Gyroscope Bias (gx, gy, gz)
            bias_data[3] = final_state.gyro_bias.x();
            bias_data[4] = final_state.gyro_bias.y();
            bias_data[5] = final_state.gyro_bias.z();
            logger_.addParameterBlock(prefix + "Bias", bias_data);

        } // End of loop through state_window
        // ROS_INFO("Optimized parameters added to logger.");


        // --- ★ 8. Write This Run's Log Entries to Files ★ ---
        // This call triggers writing the run separator, the dynamic metadata added earlier,
        // the summary (to the metrics file), and the parameters added above (to the results file).
        // It also resets the logger's internal state for the next run.
        // ROS_INFO("Writing log entries to files...");
        bool log_success = logger_.log(); // Call the unified log method

        // Check if logging was successful and report
        if (!log_success) {
            // Use the getter methods to report which files failed
            ROS_WARN("Failed to write optimization logs! Results File: %s, Metrics File: %s",
                        logger_.getResultsFilename().c_str(), logger_.getMetricsFilename().c_str());
        } else {
            // ROS_INFO("Optimization logs appended successfully. Results: %s, Metrics: %s",
            //          logger_.getResultsFilename().c_str(), logger_.getMetricsFilename().c_str());
        }
        
        // Increment optimization count
        optimization_count++;
        
        // Restore original feature flags and max iterations
        params_.enable_horizontal_velocity_incentive = original_enable_horizontal_velocity_incentive;
        params_.enable_orientation_smoothness_factor = original_enable_orientation_smoothness_factor;
        params_.max_iterations = original_max_iterations;
        
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Exception during optimization: %s", e.what());
        
        // Restore original feature flags and max iterations
        params_.enable_horizontal_velocity_incentive = original_enable_horizontal_velocity_incentive;
        params_.enable_orientation_smoothness_factor = original_enable_orientation_smoothness_factor;
        params_.max_iterations = original_max_iterations;
        
        return false;
    }
}


// IMPROVED: Propagate state with RK4 integration and better time handling for high speeds
State BackendOptimizer::propagateState(
    const State& reference_state, 
    double target_time, 
    const std::deque<sensor_msgs::Imu>& imu_buffer) {
    
    State result = reference_state;
    
    // If target_time is earlier than reference time, just return the reference state
    if (target_time <= reference_state.timestamp) {
        return reference_state;
    }
    
    // Find IMU measurements between reference_state.timestamp and target_time
    std::vector<sensor_msgs::Imu> relevant_imu_msgs;
    relevant_imu_msgs.reserve(100);
    
    for (const auto& imu : imu_buffer) {
        double timestamp = imu.header.stamp.toSec();
        if (timestamp > reference_state.timestamp && timestamp <= target_time) {
            relevant_imu_msgs.push_back(imu);
        }
    }
    
    // Sort by timestamp
    if (relevant_imu_msgs.size() > 1) {
        std::sort(relevant_imu_msgs.begin(), relevant_imu_msgs.end(), 
                    [](const sensor_msgs::Imu& a, const sensor_msgs::Imu& b) {
                        return a.header.stamp.toSec() < b.header.stamp.toSec();
                    });
    }
    
    // IMPROVED: Better time interval handling
    double prev_time = reference_state.timestamp;
    size_t imu_idx = 0;
    
    while (prev_time < target_time && imu_idx < relevant_imu_msgs.size()) {
        // Get current IMU data
        const auto& imu_msg = relevant_imu_msgs[imu_idx];
        double timestamp = imu_msg.header.stamp.toSec();
        
        // Calculate time increment with subdivision if needed
        double dt = timestamp - prev_time;
        
        // Skip invalid dt - IMPROVED: more strict checking for tiny time steps
        if (dt <= params_.min_integration_dt || dt > params_.max_imu_dt) {
            prev_time = timestamp;
            imu_idx++;
            continue;
        }
        
        // Subdivide large time steps for better accuracy - use smaller steps for high-speed
        int num_steps = 1;
        double step_dt = dt;
        
        // IMPROVED: If dt is too large, subdivide into smaller steps - more subdivision for high speeds
        if (dt > params_.max_integration_dt) {
            // For high speeds, use more subdivision steps
            num_steps = std::max(2, static_cast<int>(std::ceil(dt / params_.max_integration_dt)));
            step_dt = dt / num_steps;
        }
        
        // Extract IMU data
        Eigen::Vector3d acc1(imu_msg.linear_acceleration.x,
                                imu_msg.linear_acceleration.y,
                                imu_msg.linear_acceleration.z);
        
        Eigen::Vector3d gyro1(imu_msg.angular_velocity.x,
                                imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z);
        
        // Get next IMU data for RK4 (use current if last)
        Eigen::Vector3d acc2 = acc1;
        Eigen::Vector3d gyro2 = gyro1;
        
        if (imu_idx < relevant_imu_msgs.size() - 1) {
            const auto& next_imu = relevant_imu_msgs[imu_idx + 1];
            acc2 = Eigen::Vector3d(next_imu.linear_acceleration.x,
                                    next_imu.linear_acceleration.y,
                                    next_imu.linear_acceleration.z);
            
            gyro2 = Eigen::Vector3d(next_imu.angular_velocity.x,
                                    next_imu.angular_velocity.y,
                                    next_imu.angular_velocity.z);
        }
        
        // Apply bias correction
        acc1 -= result.acc_bias;
        acc2 -= result.acc_bias;
        gyro1 -= result.gyro_bias;
        gyro2 -= result.gyro_bias;
        
        // Perform integration using subdivided steps
        for (int step = 0; step < num_steps; step++) {
            // Linear interpolation for IMU data during subdivision
            double alpha = static_cast<double>(step) / num_steps;
            double beta = static_cast<double>(step + 1) / num_steps;
            
            Eigen::Vector3d acc_step1 = acc1 * (1.0 - alpha) + acc2 * alpha;
            Eigen::Vector3d acc_step2 = acc1 * (1.0 - beta) + acc2 * beta;
            Eigen::Vector3d gyro_step1 = gyro1 * (1.0 - alpha) + gyro2 * alpha;
            Eigen::Vector3d gyro_step2 = gyro1 * (1.0 - beta) + gyro2 * beta;
            
            // IMPROVED: Use RK4 integration for orientation
            Eigen::Quaterniond orientation_before = result.orientation;
            rk4IntegrateOrientation(gyro_step1, gyro_step2, step_dt, result.orientation);
            
            // Get gravity in sensor frame before and after orientation update
            Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
            Eigen::Vector3d gravity_sensor2 = result.orientation.inverse() * gravity_world_;
            
            // Remove gravity from accelerometer reading (averaged over rotation change)
            Eigen::Vector3d acc_without_gravity1 = acc_step1 + gravity_sensor1;
            Eigen::Vector3d acc_without_gravity2 = acc_step2 + gravity_sensor2;
            
            // Rotate to world frame using RK4 approach for acceleration
            Eigen::Vector3d acc_world1 = orientation_before * acc_without_gravity1;
            Eigen::Vector3d acc_world2 = result.orientation * acc_without_gravity2;
            
            // IMPROVED: RK4 integration for velocity/position
            Eigen::Vector3d k1v = acc_world1;
            Eigen::Vector3d k2v = 0.5 * (acc_world1 + acc_world2);
            Eigen::Vector3d k3v = 0.5 * (acc_world1 + acc_world2);
            Eigen::Vector3d k4v = acc_world2;
            
            Eigen::Vector3d velocity_before = result.velocity;
            Eigen::Vector3d acc_integrated = (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
            
            // Update velocity with RK4 integration
            result.velocity += acc_integrated * step_dt;
            
            // CRITICAL: Ensure velocity stays within reasonable limits while preserving direction
            // For propagation, we use a high max velocity to avoid artificially limiting
            // the state when using high-accuracy IMU integration
            double adaptive_max_vel = std::max(params_.max_velocity, 35.0); // Allow higher during propagation
            clampVelocity(result.velocity, adaptive_max_vel);
            
            // RK4 for position
            Eigen::Vector3d k1p = velocity_before;
            Eigen::Vector3d k2p = velocity_before + 0.5 * step_dt * k1v;
            Eigen::Vector3d k3p = velocity_before + 0.5 * step_dt * k2v;
            Eigen::Vector3d k4p = result.velocity;
            
            // Update position with RK4 integration
            Eigen::Vector3d vel_integrated = (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
            result.position += vel_integrated * step_dt;
        }
        
        // Update timestamp for next step
        prev_time = timestamp;
        imu_idx++;
    }
    
    // Final step to target_time if needed
    double dt = target_time - prev_time;
    if (dt > params_.min_integration_dt && dt <= params_.max_imu_dt && !relevant_imu_msgs.empty()) {
        // Use the last IMU measurement for prediction
        const auto& last_imu = relevant_imu_msgs.back();
        
        Eigen::Vector3d acc(last_imu.linear_acceleration.x,
                            last_imu.linear_acceleration.y,
                            last_imu.linear_acceleration.z);
        
        Eigen::Vector3d gyro(last_imu.angular_velocity.x,
                            last_imu.angular_velocity.y,
                            last_imu.angular_velocity.z);
        
        // Apply bias correction
        Eigen::Vector3d acc_corrected = acc - result.acc_bias;
        Eigen::Vector3d gyro_corrected = gyro - result.gyro_bias;
        
        // For final small step, use simpler integration to avoid extrapolation errors
        // Update orientation
        Eigen::Vector3d angle_axis = gyro_corrected * dt;
        Eigen::Quaterniond dq = deltaQ(angle_axis);
        Eigen::Quaterniond orientation_before = result.orientation;
        result.orientation = (result.orientation * dq).normalized();
        
        // Get gravity in sensor frame (average of before and after rotation)
        Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
        Eigen::Vector3d gravity_sensor2 = result.orientation.inverse() * gravity_world_;
        Eigen::Vector3d gravity_sensor = 0.5 * (gravity_sensor1 + gravity_sensor2);
        
        // Remove gravity from accelerometer reading
        Eigen::Vector3d acc_without_gravity = acc_corrected + gravity_sensor;

        // Add improved debug logging
        static int debug_counter = 0;
        if (debug_counter++ % 100 == 0) {  // Only log every 100th message
            ROS_DEBUG("IMU gravity handling: raw=[%.2f, %.2f, %.2f], gravity=[%.2f, %.2f, %.2f], corrected=[%.2f, %.2f, %.2f]",
                    acc_corrected.x(), acc_corrected.y(), acc_corrected.z(),
                    gravity_sensor.x(), gravity_sensor.y(), gravity_sensor.z(),
                    acc_without_gravity.x(), acc_without_gravity.y(), acc_without_gravity.z());
        }
        
        // Rotate to world frame using average orientation
        Eigen::Quaterniond orientation_mid = orientation_before.slerp(0.5, result.orientation);
        Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
        
        // Update velocity
        Eigen::Vector3d velocity_before = result.velocity;
        result.velocity += acc_world * dt;
        
        // Clamp velocity while preserving direction
        double adaptive_max_vel = std::max(params_.max_velocity, 35.0); // Allow higher during propagation
        clampVelocity(result.velocity, adaptive_max_vel);
        
        // Update position using trapezoidal integration
        result.position += 0.5 * (velocity_before + result.velocity) * dt;
    }
    
    // Ensure the timestamp is updated correctly
    result.timestamp = target_time;
    
    return result;
}
    