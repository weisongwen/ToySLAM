#include "data_processor.h"
#include <iomanip> // For std::setprecision

DataProcessor::DataProcessor(ros::NodeHandle& nh, const ParameterServer& params, 
                             BackendOptimizer* optimizer, Visualizer* visualizer)
    : params_(params), backend_optimizer_(optimizer), visualizer_(visualizer), last_marginalization_info_(nullptr) {
    
    // Setup logging
    gps_log_file_.open(params_.gps_log_path);
    gt_log_file_.open(params_.gt_log_path);
    //optimized_log_file_.open("optimized_path.csv"); // Example path
    //bias_fs.open("bias_log.csv"); // Example path

    // Initialize parsers
    all_parsers_.push_back(&gnss_comm_parser_);
    all_parsers_.push_back(&inspvax_parser_);
    all_parsers_.push_back(&odom_parser_);

    // Initialize the system state and preintegrator
    initializeState();

    // Start listening to ROS topics
    setupRosCommunications(nh);
}

DataProcessor::~DataProcessor() {
    if (gps_log_file_.is_open()) gps_log_file_.close();
    if (gt_log_file_.is_open()) gt_log_file_.close();
    if (optimized_log_file_.is_open()) optimized_log_file_.close();
    if (last_marginalization_info_) delete last_marginalization_info_;
}

void DataProcessor::setupRosCommunications(ros::NodeHandle& nh) {
    // 订阅者
    imu_sub_ = nh.subscribe(params_.imu_topic, params_.imu_queue_size, &DataProcessor::imuCallback, this);

    // 根据参数选择订阅哪个GNSS话题
    if (params_.use_gps_instead_of_uwb) {
        if (params_.gnss_message_type == "inspvax") {
            gnss_sub_ = nh.subscribe(params_.gnss_topic, params_.gnss_queue_size, &DataProcessor::inspvaxCallback, this);
            ROS_INFO("Subscribing to GNSS topic [novatel_msgs/INSPVAX]: %s", params_.gnss_topic.c_str());
        } else if (params_.gnss_message_type == "gnss_comm") {
            gnss_sub_ = nh.subscribe(params_.gnss_topic, params_.gnss_queue_size, &DataProcessor::gnssCommCallback, this);
            ROS_INFO("Subscribing to GNSS topic [gnss_comm/GnssPVTSolnMsg]: %s", params_.gnss_topic.c_str());
        } else if (params_.gnss_message_type == "odometry") {
            gnss_sub_ = nh.subscribe(params_.gnss_topic, params_.gnss_queue_size, &DataProcessor::odometryCallback, this);
            ROS_INFO("Subscribing to GNSS topic [nav_msgs/Odometry]: %s", params_.gnss_topic.c_str());
        } else {
            ROS_ERROR("Unsupported gps_message_type: %s. GPS fusion disabled.", params_.gnss_message_type.c_str());
            ROS_ERROR("Unsupported gps_message_type: %s. GPS fusion disabled.", params_.gnss_message_type.c_str());
            //params_.use_gps_instead_of_uwb = false;
        }
    } else {
        // UWB 订阅逻辑 (如果需要可以放在这里)
        ROS_INFO("Using UWB+IMU fusion mode (subscriber not implemented in this refactor).");
    }

    if (params_.subscribe_to_ground_truth) {

        ground_truth_sub_ = nh.subscribe(params_.ground_truth_topic, params_.gnss_queue_size, &DataProcessor::groundTruthCallback, this);
        ROS_INFO("Subscribing to Ground Truth topic [novatel_msgs/INSPVAX]: %s", params_.ground_truth_topic.c_str());
    }

    
    initializeState();

    all_parsers_.push_back(&gnss_comm_parser_);
    all_parsers_.push_back(&inspvax_parser_);
    all_parsers_.push_back(&odom_parser_);
    
    // Setup optimization timer
    // optimization_timer_ = nh.createTimer(ros::Duration(1.0/params_.optimization_frequency), 
    //                                     &DataProcessor::optimizationTimerCallback, this);
}

// ================================================================
//                       ROS CALLBACKS
// ================================================================

void DataProcessor::imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    try {

        std::lock_guard<std::mutex> lock(data_mutex_);
        // add IMU measurement to the preintegration map
        // extract the data first, including the stamp in seconds, the acceleration, and the angular velocity
        double unix_timestamp = msg->header.stamp.toSec();
        Eigen::Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        // add the IMU measurement to the preintegration map
        current_preint_test.push_back(unix_timestamp, acc, gyro);
        // ROS_INFO("[IMU] %f, %f, %f, %f, %f, %f", unix_timestamp, acc.x(), acc.y(), acc.z(), gyro.x(), gyro.y(), gyro.z());
        

        static int imu_count = 0;
        static double last_report_time = 0;
        
        // std::lock_guard<std::mutex> lock(data_mutex_);
        
        double timestamp = msg->header.stamp.toSec();
        has_imu_data_ = true;
        imu_count++;
        
        // For the first IMU message, print the timestamp for debugging
        static bool first_imu = true;
        if (first_imu) {
            ROS_INFO("First IMU timestamp: %.3f", timestamp);
            first_imu = false;
            last_report_time = timestamp;
        }
        
        // Store IMU measurements with original bag timestamps
        imu_buffer_.push_back(*msg);
        
        // Skip messages with duplicate or old timestamps 
        if (timestamp <= last_processed_timestamp_) {
            return;
        }
        
        // Update tracking timestamps
        last_imu_timestamp_ = timestamp;
        last_processed_timestamp_ = timestamp;
        
        // Process IMU data for real-time state propagation
        if (is_initialized_) {
            propagateStateWithImu(*msg);
            // publishImuPose();
            visualizer_->publishImuPose(current_state_);
        }
        
        // Report IMU statistics periodically based on message timestamps, not system time
        if (timestamp - last_report_time > 5.0) {  // Every 5 seconds in bag time
            double rate = imu_count / (timestamp - last_report_time);
            
            if (!imu_buffer_.empty()) {
                double buffer_start = imu_buffer_.front().header.stamp.toSec();
                double buffer_end = imu_buffer_.back().header.stamp.toSec();
                
                ROS_INFO("IMU stats: %.1f Hz, buffer: %zu msgs spanning %.3f sec [%.3f to %.3f]", 
                        rate, imu_buffer_.size(), buffer_end - buffer_start, buffer_start, buffer_end);
            }
            
            imu_count = 0;
            last_report_time = timestamp;
        }
        
        // Modified IMU buffer cleanup based on time difference from latest timestamp
        if (imu_buffer_.size() > 6000) {  // Larger buffer for 400Hz IMU
            double latest_time = imu_buffer_.back().header.stamp.toSec();
            double oldest_allowed_time = latest_time - 15.0;  // Keep 15 seconds of data
            
            int count_before = imu_buffer_.size();
            while (imu_buffer_.size() > 1000 && imu_buffer_.front().header.stamp.toSec() < oldest_allowed_time) {
                imu_buffer_.pop_front();
            }
            
            int count_after = imu_buffer_.size();
            if (count_before - count_after > 100) {
                ROS_INFO("Cleaned %d old IMU messages, remaining: %d", count_before - count_after, count_after);
            }
        }



    } catch (const std::exception& e) {
        ROS_ERROR("Exception in imuCallback: %s", e.what());
    }
}

void DataProcessor::inspvaxCallback(const novatel_msgs::INSPVAX::ConstPtr& msg) {
    // 1. 解析消息
    std::optional<GnssMeasurement> meas_opt = inspvax_parser_.parse(msg);
    
    if (meas_opt) {
        GnssMeasurement meas = *meas_opt;

        // 2. 条件性地添加噪声
        if (params_.subscribe_to_ground_truth && (params_.gnss_topic == params_.ground_truth_topic) &&
            (params_.artificial_pos_noise_std > 0.0 || params_.artificial_vel_noise_std > 0.0))
        {
            ROS_INFO_ONCE("Adding artificial noise because gps_topic and ground_truth_topic are the same.");
            
            std::normal_distribution<double> pos_noise(0.0, params_.artificial_pos_noise_std);
            std::normal_distribution<double> vel_noise(0.0, params_.artificial_vel_noise_std);

            meas.position.x() += pos_noise(random_generator_);
            meas.position.y() += pos_noise(random_generator_);
            meas.position.z() += pos_noise(random_generator_);

            meas.velocity.x() += vel_noise(random_generator_);
            meas.velocity.y() += vel_noise(random_generator_);
            meas.velocity.z() += vel_noise(random_generator_);
        }
        
        // 3. 记录FGO的GPS输入 (可能是原始的，也可能是加噪后的)
        logGpsData(meas);
        syncEnuReference();

        // 4. 使用该测量值进行融合处理
        processGnssMeasurement(meas);
    }
}

void DataProcessor::gnssCommCallback(const gnss_comm::GnssPVTSolnMsg::ConstPtr& msg) {
        // 调用对应的解析器
        std::optional<GnssMeasurement> meas_opt = gnss_comm_parser_.parse(msg);
        // 如果解析成功，则传递给统一的处理函数
        if (meas_opt) {
            gps_measurement_count_++;
            if (gps_measurement_count_ % 10 != 0) {
                return; // 每10个测量只处理一次
            }
            GnssMeasurement meas = *meas_opt;
            processGnssMeasurement(*meas_opt);
            syncEnuReference();
            logGpsData(meas);
        }
    }

void DataProcessor::odometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 调用 Odometry 解析器
    std::optional<GnssMeasurement> meas_opt = odom_parser_.parse(msg);
    
    // 如果解析成功，则传递给统一的处理函数
    if (meas_opt) {
        GnssMeasurement meas = *meas_opt;
        processGnssMeasurement(*meas_opt);
        syncEnuReference();
        logGpsData(meas);
    }
}


void DataProcessor::groundTruthCallback(const novatel_msgs::INSPVAX::ConstPtr& msg) {
    std::optional<GnssMeasurement> meas_opt = inspvax_parser_.parse(msg);
    if (meas_opt) {
        logGroundTruthData(*meas_opt);
        
        visualizer_->publishGroundTruthPath(*meas_opt); 
    }
}


void DataProcessor::optimizationTimerCallback(const ros::TimerEvent& event) {
    try {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        if (!is_initialized_) {
            return;
        }
        
        // Need at least 2 states for optimization
        if (state_window_.size() < 2) {
            return;
        }
        
        // Reset position if large drift detected
        if (params_.use_gps_instead_of_uwb) {
            // Check GPS drift
            if (!gps_measurements_.empty() && !state_window_.empty()) {
                const auto& latest_gps = gps_measurements_.back();
                auto& latest_state = state_window_.back();
                
                double position_error = (latest_state.position - latest_gps.position).norm();
                
                // Adjust drift threshold based on velocity
                double adaptive_drift_threshold = 5.0; // Default threshold
                double velocity_norm = latest_state.velocity.norm();
                
                // Increase allowable drift at higher speeds
                if (velocity_norm > 10.0) {
                    adaptive_drift_threshold = 1.0 + (velocity_norm - 10.0) * 0.1;
                    adaptive_drift_threshold = std::min(adaptive_drift_threshold, 15.0); // Cap at 3 meters
                }
                
                // if (position_error > adaptive_drift_threshold) {
                //     ROS_WARN("Position drift detected in GPS mode: %.2f meters. Resetting position.", position_error);
                //     resetStateToGps(latest_gps);
                // }
            }
        } 

        // Time the factor graph optimization
        auto start_time = std::chrono::high_resolution_clock::now();
        // ROS_INFO("Recorde Time before optimization");
        
        // Perform optimization
        bool success = false;
        
        try {
            // ROS_INFO("Start to optimize factor graph");
            // success = optimizeFactorGraph();
            success = backend_optimizer_->optimize(
                            state_window_, 
                            gps_measurements_, 
                            preintegration_map_test, 
                            last_marginalization_info_, 
                            optimization_count_,
                            imu_buffer_);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during factor graph optimization: %s", e.what());
            success = false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        
        if (!success) {
            ROS_WARN("Factor graph optimization failed (took %.1f ms)", duration.count());
        } else {
            // Set optimization flag
            just_optimized_ = true;
            
            // Output state after optimization
            Eigen::Vector3d euler_angles = quaternionToEulerDegrees(current_state_.orientation);
            
            // Get velocity in km/h for better reporting in high-speed scenario
            double velocity_kmh = current_state_.velocity.norm() * 3.6; // m/s to km/h
            
            // 更新当前状态为优化后的最新状态
            if (!state_window_.empty()) {
                current_state_ = state_window_.back();
                logOptimizedState(current_state_);
            }
            
            // Publish state
            //publishOptimizedPose();
            visualizer_->publishOptimizedPose(state_window_.back());

            // After optimization, calculate and visualize errors with GPS
            if (params_.use_gps_instead_of_uwb) {
                //calculateAndVisualizePositionError();
                visualizer_->calculateAndVisualizePositionError(current_state_, gps_measurements_);
                if (params_.use_gps_velocity) {
                    //calculateAndVisualizeVelocityError();
                    visualizer_->calculateAndVisualizeVelocityError(current_state_, gps_measurements_);
                }
            }
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in optimizationTimerCallback: %s", e.what());
    }
}

// ================================================================
//                       ALGORITHM LOGIC
// ================================================================

void DataProcessor::initializeState() {
    try{
        current_state_.position = Eigen::Vector3d::Zero();
        current_state_.orientation = Eigen::Quaterniond::Identity();
        current_state_.velocity = Eigen::Vector3d::Zero();
        
        // CRITICAL: Initialize with sane non-zero biases
        initial_acc_bias_ = Eigen::Vector3d(params_.initial_acc_bias_x, params_.initial_acc_bias_y, params_.initial_acc_bias_z);
        initial_gyro_bias_ = Eigen::Vector3d(params_.initial_gyro_bias_x, params_.initial_gyro_bias_y, params_.initial_gyro_bias_z);
        current_state_.acc_bias = initial_acc_bias_;
        current_state_.gyro_bias = initial_gyro_bias_;

        // ROS_INFO("Initialized Acc Bias [%.6f, %.6f, %.6f], Gyro Bias [%.6f, %.6f, %.6f]",
        //     current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
        //     current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
        
        current_state_.timestamp = 0;
        
        state_window_.clear();
        gps_measurements_.clear();  // Clear GPS measurements
        imu_buffer_.clear();
        // preintegration_map_.clear();
        preintegration_map_test.clear();
        
        gnss_comm_parser_.reset();
        inspvax_parser_.reset();
        odom_parser_.reset();
        
        // Initialize gravity vector in world frame (ENU, Z points up)
        // In ENU frame, gravity points downward along negative Z axis
        gravity_world_ = Eigen::Vector3d(0, 0, -params_.gravity_magnitude);
        
        // Reset timestamp tracking
        last_imu_timestamp_ = 0;
        last_processed_timestamp_ = 0;
        just_optimized_ = false;
        is_initialized_ = false;
        has_imu_data_ = false;
        
        // Reset optimization count
        optimization_count_ = 0;
        
        // Reset marginalization
        if (last_marginalization_info_){
            delete last_marginalization_info_;
            last_marginalization_info_ = nullptr;
        }
        
        // Reset visualization
        visualizer_->reset();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in initializeState: %s", e.what());
    }
}

void DataProcessor::processGnssMeasurement(const GnssMeasurement& measurement) {
     try {
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            if (measurement.position_valid) {
                visualizer_->publishGpsPath(measurement);
            }
            
            gps_measurements_.push_back(measurement);

            if (!is_initialized_) {
                if (imu_buffer_.size() >= 5) {
                    ROS_INFO("Initializing system with GNSS measurement at timestamp: %.3f", measurement.timestamp);
                    initializeFromGps(measurement);
                    is_initialized_ = true;
                }
                return;
            }

            if (is_initialized_ && has_imu_data_) {
                bool has_surrounding_imu_data = false;
                double closest_time_diff = std::numeric_limits<double>::max();
                for (const auto& imu : imu_buffer_) {
                    double time_diff = std::abs(imu.header.stamp.toSec() - measurement.timestamp);
                    if (time_diff < 0.3) { // 50ms tolerance
                        has_surrounding_imu_data = true;
                        break;
                    }
                }
                if (has_surrounding_imu_data) {
                    ROS_INFO("Creating keyframe from GNSS measurement at timestamp: %.3f", measurement.timestamp);
                    createKeyframeFromGps(measurement);
                }
                else{
                    ROS_WARN("No surrounding IMU data found for GNSS measurement at timestamp: %.3f", measurement.timestamp);
                }
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in generic gnssCallback: %s", e.what());
        }
}

void DataProcessor::initializeFromGps(const GnssMeasurement& gps) {
    try {
        // --- Pre-condition Check ---
        if (!gps.position_valid) {
            ROS_ERROR("Failed to initialize: The provided GNSS measurement does not have a valid position.");
            return;
        }

        // --- Set State from Measurement ---
        current_state_.position = gps.position;
        current_state_.timestamp = gps.timestamp;

        // --- Initialize Orientation ---
        // Priority: Use GPS orientation if available and enabled. Otherwise, fallback to a default.
        if (params_.use_gps_orientation_as_initial && gps.orientation_valid) {
            current_state_.orientation = gps.orientation.normalized();
            ROS_INFO("Initializer: Orientation set from GNSS data.");
        } else {
            // Fallback: use IMU orientation if available, otherwise identity.
            sensor_msgs::Imu closest_imu = findClosestImuMeasurement(gps.timestamp);
            // check if the imu orientation value 
            Eigen::Quaterniond imu_orientation;
            imu_orientation = Eigen::Quaterniond(
                closest_imu.orientation.w, closest_imu.orientation.x,
                closest_imu.orientation.y, closest_imu.orientation.z);
            if (closest_imu.header.stamp.toSec() > 0 && closest_imu.orientation_covariance[0] != -1 && imu_orientation.coeffs().norm() > 1e-4) {
                current_state_.orientation = imu_orientation.normalized();
                ROS_INFO("Initializer: Orientation set from IMU data as a fallback.");
            } else {
                current_state_.orientation = Eigen::Quaterniond::Identity();
                ROS_INFO("Initializer: Orientation set to Identity as a fallback.");
            }
            // current_state_.orientation = Eigen::Quaterniond::Identity();
        }

        // --- Initialize Velocity ---
        // Priority: Use GPS velocity if available and enabled. Otherwise, fallback to zero.
        if (params_.use_gps_velocity && gps.velocity_valid) {
            current_state_.velocity = gps.velocity;
            ROS_INFO("Initializer: Velocity set from GNSS data.");
        } else {
            current_state_.velocity = Eigen::Vector3d::Zero();
            ROS_INFO("Initializer: Velocity set to Zero as a fallback.");
        }

        // --- Initialize Biases ---
        initial_acc_bias_ = Eigen::Vector3d(params_.initial_acc_bias_x, params_.initial_acc_bias_y, params_.initial_acc_bias_z);
        initial_gyro_bias_ = Eigen::Vector3d(params_.initial_gyro_bias_x, params_.initial_gyro_bias_y, params_.initial_gyro_bias_z);
        current_state_.acc_bias = initial_acc_bias_;
        current_state_.gyro_bias = initial_gyro_bias_;

        // --- Reset System Components ---
        state_window_.clear();
        state_window_.push_back(current_state_);
        
        preintegration_map_test.clear(); // Or your equivalent pre-integration map
        current_preint_test.reset();     // Reset the current pre-integrator
        
        if (last_marginalization_info_) {
            delete last_marginalization_info_;
            last_marginalization_info_ = nullptr;
        }
        
        optimization_count_ = 0;
        visualizer_->reset();

        ROS_INFO("System initialized at position [%.2f, %.2f, %.2f]",
                current_state_.position.x(), current_state_.position.y(), current_state_.position.z());
        ROS_INFO("Initial velocity: [%.2f, %.2f, %.2f] m/s",
                current_state_.velocity.x(), current_state_.velocity.y(), current_state_.velocity.z());

    } catch (const std::exception& e) {
        ROS_ERROR("Exception in initializeFromGps: %s", e.what());
    }
}

void DataProcessor::createKeyframeFromGps(const GnssMeasurement& gps) {
    try {
        // --- Pre-condition Checks ---
        if (state_window_.empty()) {
            ROS_ERROR("Cannot create keyframe: state window is empty. Should have been initialized first.");
            return;
        }
        if (std::abs(state_window_.back().timestamp - gps.timestamp) < 0.01) {
            ROS_WARN("Skipping keyframe creation: time difference too small (%.4f s).", 
                    std::abs(state_window_.back().timestamp - gps.timestamp));
            return;
        }

        performPreintegrationBetweenKeyframes(state_window_.back().timestamp, gps.timestamp);
        

        // --- Propagate and Update State ---
        // 1. Propagate the last keyframe's state forward to the new timestamp using IMU data.
        State propagated_state = backend_optimizer_->propagateState(state_window_.back(), gps.timestamp,imu_buffer_);

        // 2. Update the propagated state with the new, more accurate GNSS data.
        // This effectively "corrects" the IMU-only prediction.
        if (gps.position_valid) {
            propagated_state.position = (gps.position + propagated_state.position) / 2.0; // Average with IMU prediction   
        }
        if (params_.use_gps_orientation_as_initial && gps.orientation_valid) {
            propagated_state.orientation = gps.orientation.normalized();
        } else {
            // If GPS orientation is not used, we can still use the IMU orientation as a fallback.
            sensor_msgs::Imu closest_imu = findClosestImuMeasurement(gps.timestamp);
            Eigen::Quaterniond imu_orientation;
            imu_orientation = Eigen::Quaterniond(
                closest_imu.orientation.w, closest_imu.orientation.x,
                closest_imu.orientation.y, closest_imu.orientation.z);
            if (closest_imu.header.stamp.toSec() > 0 && imu_orientation.coeffs().norm() > 1e-4) {
                // check the IMU orientation is valid
                // and use it to update the propagated state orientation
                propagated_state.orientation = imu_orientation.normalized();
            }
            else{
                //find out the orientation change from last keyframe to this new one
                std::pair<double, double> key(state_window_.back().timestamp, gps.timestamp);
                if (preintegration_map_test.find(key) != preintegration_map_test.end()) {
                    Eigen::Quaterniond delta_orientation = preintegration_map_test[key].getDeltaGamma();
                    ROS_INFO("Delta Gamma found for keyframe propagation: %.4f, %.4f", 
                                state_window_.back().timestamp, gps.timestamp);
                    ROS_INFO("Delta Orientation: %.4f, %.4f, %.4f, %.4f", 
                                delta_orientation.x(), delta_orientation.y(), delta_orientation.z(), delta_orientation.w());
                    propagated_state.orientation = propagated_state.orientation * delta_orientation;
                    propagated_state.orientation.normalize();
                } else {
                    propagated_state.orientation = state_window_.back().orientation.normalized();
                }
                
            }
        }
        if (params_.use_gps_velocity && gps.velocity_valid) {
            propagated_state.velocity = (gps.velocity + propagated_state.velocity)/2; // Average with IMU prediction
        }

        // The biases are carried over from the previous state. The optimizer will adjust them.
        propagated_state.acc_bias = state_window_.back().acc_bias;
        propagated_state.gyro_bias = state_window_.back().gyro_bias;

        // --- Manage Pre-integration and State Window ---
        // Finalize the pre-integration measurement between the last keyframe and this new one.
        // performPreintegrationBetweenKeyframes_(state_window_.back().timestamp, gps.timestamp);
        
        // If the window is full, marginalize the oldest state.
        if (state_window_.size() >= params_.optimization_window_size) {
            if (params_.enable_marginalization) {
                //prepareMarginalization();
                backend_optimizer_->prepareMarginalization(
                    state_window_, 
                    gps_measurements_, 
                    preintegration_map_test, 
                    last_marginalization_info_
                );
            }
            State oldest_state = state_window_.front();
            //updateFinalPath(oldest_state);
            visualizer_->publishFinalPath(oldest_state);
            state_window_.pop_front();
        }

        // Add the newly created keyframe to the window.
        state_window_.push_back(propagated_state);

        // Update the global "current state" for high-frequency IMU propagation.
        current_state_ = propagated_state;

    } catch (const std::exception& e) {
        ROS_ERROR("Exception in createKeyframeFromGps: %s", e.what());
    }
}


void DataProcessor::propagateStateWithImu(const sensor_msgs::Imu& imu_msg) {
    try {
        double timestamp = imu_msg.header.stamp.toSec();
        
        // Special handling if we just ran optimization
        if (just_optimized_) {
            ROS_INFO("Just optimized, skipping IMU integration");
            // Just update timestamp without integration
            current_state_.timestamp = timestamp;
            just_optimized_ = false;
            return;
        }
        // ROS_INFO("IMU integration: timestamp %.3f", timestamp);
        
        // Extract IMU measurements
        Eigen::Vector3d acc(imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z);
        
        Eigen::Vector3d gyro(imu_msg.angular_velocity.x,
                            imu_msg.angular_velocity.y,
                            imu_msg.angular_velocity.z);
        
        // Check for NaN/Inf values
        if (!acc.allFinite() || !gyro.allFinite()) {
            ROS_WARN_THROTTLE(1.0, "Non-finite IMU values detected");
            return;
        }
        
        // Calculate time difference
        double dt = 0;
        if (current_state_.timestamp > 0) {
            dt = timestamp - current_state_.timestamp;
        } else {
            // First IMU measurement after initialization
            current_state_.timestamp = timestamp;
            
            // Update orientation directly from IMU if available
            if (imu_msg.orientation_covariance[0] != -1) {
                current_state_.orientation = Eigen::Quaterniond(
                    imu_msg.orientation.w,
                    imu_msg.orientation.x,
                    imu_msg.orientation.y,
                    imu_msg.orientation.z
                ).normalized();
            }
            
            return;  // Skip integration for the first IMU message
        }
        
        // Skip integration for invalid dt
        if (dt <= params_.min_integration_dt || dt > params_.max_imu_dt) {
            current_state_.timestamp = timestamp;
            return;
        }
        
        // Use IMU orientation if available
        if (imu_msg.orientation_covariance[0] != -1) {
            // Get orientation from IMU
            Eigen::Quaterniond imu_orientation(
                imu_msg.orientation.w,
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z
            );
            
            // Update orientation directly - but use a weighted average to smooth transitions 
            // The new attitude is derived 70% from previous calculations and 30% from direct measurements by the IMU.
            Eigen::Quaterniond blended_orientation = current_state_.orientation.slerp(0.3, imu_orientation);
            current_state_.orientation = imu_orientation.normalized();
        } else {
        // Apply bias correction
        Eigen::Vector3d acc_corrected = acc - current_state_.acc_bias;
        Eigen::Vector3d gyro_corrected = gyro - current_state_.gyro_bias;

        // // 打印当前零偏
        // ROS_INFO("Timestamp %.3f: Acc Bias [%.6f, %.6f, %.6f], Gyro Bias [%.6f, %.6f, %.6f]",
        //     timestamp,
        //     current_state_.acc_bias.x(), current_state_.acc_bias.y(), current_state_.acc_bias.z(),
        //     current_state_.gyro_bias.x(), current_state_.gyro_bias.y(), current_state_.gyro_bias.z());
        
        // Store orientation before update
        Eigen::Quaterniond orientation_before = current_state_.orientation;
        
        // Update orientation with simple integration for real-time
        Eigen::Vector3d angle_axis = gyro_corrected * dt;
        Eigen::Quaterniond dq = deltaQ(angle_axis);
        
        // Update orientation
        current_state_.orientation = (current_state_.orientation * dq).normalized();
        
        // Get gravity in sensor frame (average of before and after rotation)
        Eigen::Vector3d gravity_sensor1 = orientation_before.inverse() * gravity_world_;
        Eigen::Vector3d gravity_sensor2 = current_state_.orientation.inverse() * gravity_world_;
        Eigen::Vector3d gravity_sensor = 0.5 * (gravity_sensor1 + gravity_sensor2);
        
        // Remove gravity from accelerometer reading (accelerometer measures gravity + acceleration)
        // In ENU frame with Z-up, gravity is [0, 0, -9.81], so we add the gravity_sensor vector
        Eigen::Vector3d acc_without_gravity = acc_corrected + gravity_sensor;
        // ROS_INFO("IMU gravity compensation: raw=[%.2f, %.2f, %.2f], gravity_sensor=[%.2f, %.2f, %.2f], corrected=[%.2f, %.2f, %.2f]",
        //         acc_corrected.x(), acc_corrected.y(), acc_corrected.z(),
        //         gravity_sensor.x(), gravity_sensor.y(), gravity_sensor.z(),
        //         acc_without_gravity.x(), acc_without_gravity.y(), acc_without_gravity.z());
        
        // Rotate to world frame using midpoint rotation
        Eigen::Quaterniond orientation_mid = orientation_before.slerp(0.5, current_state_.orientation);
        Eigen::Vector3d acc_world = orientation_mid * acc_without_gravity;
        
        // Store velocity before update for trapezoidal integration
        Eigen::Vector3d velocity_before = current_state_.velocity;
        
        // Update velocity
        current_state_.velocity += acc_world * dt;
        
        // IMPROVED: For high-speed scenarios - remove vertical damping
        // Only apply slight damping if we have a large spurious vertical velocity
        // double v_vel_abs = std::abs(current_state_.velocity.z());
        // if (v_vel_abs > 5.0) {  // Only dampen extreme vertical velocities
        //     current_state_.velocity.z() *= 0.95;  // Mild damping only on extreme values
        // }
        
        // Adaptive max velocity based on IMU data for real-time propagation
        double adaptive_max_vel = params_.max_velocity;
        if (imu_buffer_.size() > 10) {
            adaptive_max_vel = std::max(params_.max_velocity, backend_optimizer_->estimateMaxVelocityFromImu(imu_buffer_));
        }
        
        // Ensure velocity stays within reasonable limits while preserving direction
        backend_optimizer_->clampVelocity(current_state_.velocity, adaptive_max_vel);
        
        // Update position using trapezoidal integration
        current_state_.position += 0.5 * (velocity_before + current_state_.velocity) * dt;
        }
        
        // Update timestamp
        current_state_.timestamp = timestamp;
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in propagateStateWithImu: %s", e.what());
    }
}


void DataProcessor::performPreintegrationBetweenKeyframes(const double &start_time, const double &end_time) {
    // save the preint data to the map
    preintegration_map_test[std::make_pair(start_time, end_time)] = current_preint_test;

    // ROS_INFO("Current preint data:");
    // ROS_INFO("delta_p: %f, %f, %f", current_preint_test.getDeltaAlpha().x(), current_preint_test.getDeltaAlpha().y(), current_preint_test.getDeltaAlpha().z());

    // // print the preint data
    // ROS_INFO("Preintegration data saved for keyframes: [%.3f, %.3f]", start_time, end_time);

    // reset the current preint data to accept imu messages for next keyframe
    current_preint_test.reset();
    current_preint_test.set_gravity(params_.gravity_magnitude);
    // current_preint_test.setBias(initial_acc_bias_, initial_gyro_bias_);
    current_preint_test.set_noise(
        params_.imu_acc_noise, params_.imu_gyro_noise,
        params_.imu_acc_bias_noise, params_.imu_gyro_bias_noise
    );
}


// *** Modified function to log a single keyframe's bias ***
void DataProcessor::logKeyframeBias(const State& keyframe_state_to_log) {
    if (!bias_fs.is_open()) {
        ROS_WARN_THROTTLE(5.0, "Bias log file is not open. Cannot log bias for timestamp %.6f.", keyframe_state_to_log.timestamp);
        return;
    }

    bias_fs << keyframe_state_to_log.timestamp << ","
                    << keyframe_state_to_log.acc_bias.x() << "," << keyframe_state_to_log.acc_bias.y() << "," << keyframe_state_to_log.acc_bias.z() << ","
                    << keyframe_state_to_log.gyro_bias.x() << "," << keyframe_state_to_log.gyro_bias.y() << "," << keyframe_state_to_log.gyro_bias.z() << "\n";
    
    // Flush periodically or rely on fstream's buffering. For "real-time" visibility, occasional flush is good.
    // Consider flushing less frequently if performance is an issue.
    static int log_counter = 0;
    if (++log_counter % 10 == 0) { // Flush every 10 entries, for example
        bias_fs.flush();
    }
}

// Helper to find closest IMU measurement to a given timestamp
sensor_msgs::Imu DataProcessor::findClosestImuMeasurement(double timestamp) {
    sensor_msgs::Imu closest_imu;
    double min_time_diff = std::numeric_limits<double>::max();
    
    for (const auto& imu : imu_buffer_) {
        double imu_time = imu.header.stamp.toSec();
        double time_diff = std::abs(imu_time - timestamp);
        
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            closest_imu = imu;
        }
    }
    
    return closest_imu;
}

void DataProcessor::syncEnuReference() {
    GnssParser* source_parser = nullptr;


    for (GnssParser* parser : all_parsers_) {
        if (parser->hasEnuReference()) {
            source_parser = parser;
            break; 
        }
    }

    if (source_parser != nullptr) {
        double ref_lat, ref_lon, ref_alt;

        if (source_parser->getEnuReference(ref_lat, ref_lon, ref_alt)) {
            for (GnssParser* parser : all_parsers_) {
                if (!parser->hasEnuReference()) {
                    parser->setEnuReference(ref_lat, ref_lon, ref_alt);
                }
            }
        }
    }
}


void DataProcessor::logGpsData(const GnssMeasurement& meas) {
    if (gps_log_file_.is_open()) {
        gps_log_file_ << std::fixed << std::setprecision(6) << meas.timestamp << ","
                        << meas.position.x() << "," << meas.position.y() << "," << meas.position.z() << ","
                        << meas.velocity.x() << "," << meas.velocity.y() << "," << meas.velocity.z() << "\n";
    }
}

void DataProcessor::logGroundTruthData(const GnssMeasurement& meas) {
    if (gt_log_file_.is_open()) {
        Eigen::Vector3d rpy = meas.orientation.toRotationMatrix().eulerAngles(2, 1, 0);
        gt_log_file_ << std::fixed << std::setprecision(6) << meas.timestamp << ","
                            << meas.position.x() << "," << meas.position.y() << "," << meas.position.z() << ","
                            << meas.velocity.x() << "," << meas.velocity.y() << "," << meas.velocity.z() << ","
                            << rpy.z() << "," << rpy.y() << "," << rpy.x() << "\n";
    }
}

void DataProcessor::logOptimizedState(const State& state) {
    if (optimized_log_file_.is_open()) {
        optimized_log_file_ << std::fixed << std::setprecision(6) << state.timestamp << ","
                                << state.position.x() << "," << state.position.y() << "," << state.position.z() << ","
                                << state.orientation.x() << "," << state.orientation.y() << "," << state.orientation.z() << "," << state.orientation.w() << ","
                                << state.velocity.x() << "," << state.velocity.y() << "," << state.velocity.z() << ","
                                << state.acc_bias.x() << "," << state.acc_bias.y() << "," << state.acc_bias.z() << ","
                                << state.gyro_bias.x() << "," << state.gyro_bias.y() << "," << state.gyro_bias.z() << "\n";
    }
}