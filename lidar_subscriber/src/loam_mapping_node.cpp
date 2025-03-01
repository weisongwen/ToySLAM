#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>

// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

// NDT cell structure to store statistics
struct NDTCell {
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix3d inverse_covariance;
    double det_covariance;
    int point_count;

    NDTCell() : mean(Eigen::Vector3d::Zero()), 
                covariance(Eigen::Matrix3d::Identity()),
                inverse_covariance(Eigen::Matrix3d::Identity()),
                det_covariance(1.0), 
                point_count(0) {}
};

// Ceres cost function for NDT registration
class NDTCostFunction : public ceres::SizedCostFunction<1, 6> {
public:
    NDTCostFunction(const Eigen::Vector3d& point, const NDTCell& cell, double weight = 1.0)
        : point_(point), cell_(cell), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const {
        
        // Extract transformation parameters (3 for translation, 3 for rotation in angle-axis form)
        Eigen::Vector3d translation(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d rotation_aa(parameters[0][3], parameters[0][4], parameters[0][5]);
        
        // Handle zero rotation case to avoid divide-by-zero
        double angle = rotation_aa.norm();
        Eigen::Matrix3d rotation_matrix;
        
        if (angle < 1e-10) {
            rotation_matrix = Eigen::Matrix3d::Identity();
        } else {
            Eigen::AngleAxisd rotation(angle, rotation_aa / angle);
            rotation_matrix = rotation.toRotationMatrix();
        }
        
        // Transform the point
        Eigen::Vector3d transformed_point = rotation_matrix * point_ + translation;
        
        // Compute the difference from the mean
        Eigen::Vector3d d = transformed_point - cell_.mean;
        
        // Compute the NDT score (negative log-likelihood)
        double exp_term = -0.5 * d.transpose() * cell_.inverse_covariance * d;
        
        // Residual is negative to maximize the likelihood, weighted by confidence
        residuals[0] = -weight_ * exp(exp_term);
        
        // Compute jacobians if needed
        if (jacobians && jacobians[0]) {
            Eigen::Matrix<double, 1, 3> gradient_translation = 
                -residuals[0] * d.transpose() * cell_.inverse_covariance;
            
            // Jacobian for translation part (first 3 elements)
            jacobians[0][0] = gradient_translation(0);
            jacobians[0][1] = gradient_translation(1);
            jacobians[0][2] = gradient_translation(2);
            
            // Jacobian for rotation part (last 3 elements)
            Eigen::Matrix3d skew_point;
            skew_point << 0, -point_(2), point_(1),
                          point_(2), 0, -point_(0),
                          -point_(1), point_(0), 0;
            
            Eigen::Matrix<double, 3, 3> jacobian_rotation = -rotation_matrix * skew_point;
            Eigen::Matrix<double, 1, 3> gradient_rotation = 
                gradient_translation * jacobian_rotation;
            
            jacobians[0][3] = gradient_rotation(0);
            jacobians[0][4] = gradient_rotation(1);
            jacobians[0][5] = gradient_rotation(2);
        }
        
        return true;
    }

private:
    Eigen::Vector3d point_;
    NDTCell cell_;
    double weight_; // Weight for this point-to-cell correspondence
};

// Regularization cost function to limit transformation magnitude
class TransformationRegularizationCostFunction : public ceres::SizedCostFunction<6, 6> {
public:
    TransformationRegularizationCostFunction(double translation_weight = 0.1, double rotation_weight = 0.1)
        : translation_weight_(translation_weight), rotation_weight_(rotation_weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const {
        
        // Extract parameters
        Eigen::Vector3d translation(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d rotation_aa(parameters[0][3], parameters[0][4], parameters[0][5]);
        
        // Regularize translation
        residuals[0] = translation_weight_ * translation.x();
        residuals[1] = translation_weight_ * translation.y();
        residuals[2] = translation_weight_ * translation.z();
        
        // Regularize rotation
        residuals[3] = rotation_weight_ * rotation_aa.x();
        residuals[4] = rotation_weight_ * rotation_aa.y();
        residuals[5] = rotation_weight_ * rotation_aa.z();
        
        if (jacobians && jacobians[0]) {
            // Set Jacobian to identity matrix with weights
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    jacobians[0][i * 6 + j] = 0.0;
                }
                
                double weight = (i < 3) ? translation_weight_ : rotation_weight_;
                jacobians[0][i * 6 + i] = weight;
            }
        }
        
        return true;
    }

private:
    double translation_weight_;
    double rotation_weight_;
};

class HighAccuracyNDTRegistration {
public:
    HighAccuracyNDTRegistration(ros::NodeHandle& nh) : 
        nh_(nh), 
        has_target_cloud_(false), 
        processing_thread_active_(true),
        frames_processed_(0),
        accumulated_error_(0.0),
        use_motion_undistortion_(true),
        use_keyframe_strategy_(true),
        use_icp_refinement_(true),
        use_adaptive_parameters_(true) {
            
        // Load parameters (hardcoded for higher accuracy)
        // NDT parameters - refined for higher accuracy
        voxel_size_ = 0.3;                   // Decreased for more detailed matching
        downsample_resolution_ = 0.05;       // Finer resolution to preserve details
        max_queue_size_ = 1000;              // Large queue for offline processing
        fixed_frame_ = "map";
        publish_tf_ = true;
        outlier_radius_ = 0.3;               // Decreased to remove fewer points
        outlier_min_neighbors_ = 3;          // Lower threshold to keep more points
        max_iterations_ = 300;               // More iterations for better convergence
        reg_translation_weight_ = 0.001;     // Lower regularization to allow more freedom
        reg_rotation_weight_ = 0.001;        // Lower regularization for rotation
        multi_res_size_1_ = 0.3;             // Fine resolution
        multi_res_size_2_ = 0.6;             // Medium resolution
        multi_res_size_3_ = 1.2;             // Coarse resolution
        
        // Segmentation parameters
        use_ground_segmentation_ = true;     // Keep ground segmentation on
        ground_z_threshold_ = -1.5;          // For VLP-32 on car, ground is below
        max_z_value_ = 5.0;                  // Max height for points (remove sky points)
        distance_weight_factor_ = 0.2;       // Slightly increased weight factor
        
        // Motion compensation
        scan_period_ = 0.1;                  // 100ms for a typical Velodyne scan
        min_scan_range_ = 1.0;               // Min range to filter out near points
        max_scan_range_ = 100.0;             // Max range for point filtering
        
        // Keyframe-based strategy
        keyframe_trans_threshold_ = 0.5;     // meters
        keyframe_rot_threshold_ = 0.1;       // radians (about 5.7 degrees)
        keyframe_overlap_factor_ = 0.7;      // Percentage of overlap required
        
        // ICP refinement
        icp_max_iterations_ = 50;
        icp_max_correspondence_distance_ = 0.1;
        icp_transformation_epsilon_ = 1e-8;
        
        // Adaptive parameters
        velocity_threshold_slow_ = 0.5;      // m/s
        velocity_threshold_fast_ = 2.0;      // m/s
        
        // Status reporting
        status_publish_rate_ = 1;            // Hz
        
        // Initialize publishers and subscribers
        cloud_sub_ = nh_.subscribe("/velodyne_points", 100, &HighAccuracyNDTRegistration::cloudCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("ndt_pose", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("ndt_odom", 10);
        registered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("registered_cloud", 10);
        queue_size_pub_ = nh_.advertise<std_msgs::Int32>("queue_size", 10);
        frames_processed_pub_ = nh_.advertise<std_msgs::Int32>("frames_processed", 10);
        error_pub_ = nh_.advertise<std_msgs::Float64>("registration_error", 10);
        
        // Initialize position
        current_pose_ = Eigen::Matrix4d::Identity();
        last_odom_pose_ = Eigen::Matrix4d::Identity();
        
        // Initialize point clouds
        global_map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        keyframe_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Start processing thread
        processing_thread_ = std::thread(&HighAccuracyNDTRegistration::processQueue, this);
        
        // Start status publisher timer
        status_timer_ = nh_.createTimer(ros::Duration(1.0/status_publish_rate_), 
                                      &HighAccuracyNDTRegistration::publishStatus, this);
        
        ROS_INFO("High Accuracy NDT Registration node initialized");
        ROS_INFO("Parameters: voxel_size=%.2f, downsample_resolution=%.3f", voxel_size_, downsample_resolution_);
        ROS_INFO("Using multi-resolution NDT with sizes: %.2f, %.2f, %.2f", 
                multi_res_size_1_, multi_res_size_2_, multi_res_size_3_);
        ROS_INFO("Keyframe-based strategy: %s", use_keyframe_strategy_ ? "enabled" : "disabled");
        ROS_INFO("ICP refinement: %s", use_icp_refinement_ ? "enabled" : "disabled");
        ROS_INFO("Motion undistortion: %s", use_motion_undistortion_ ? "enabled" : "disabled");
    }
    
    ~HighAccuracyNDTRegistration() {
        processing_thread_active_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    // ROS related
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher odom_pub_;
    ros::Publisher registered_cloud_pub_;
    ros::Publisher queue_size_pub_;
    ros::Publisher frames_processed_pub_;
    ros::Publisher error_pub_;
    ros::Timer status_timer_;
    tf::TransformBroadcaster tf_broadcaster_;
    
    // Parameters
    double voxel_size_;
    double downsample_resolution_;
    int max_queue_size_;
    std::string fixed_frame_;
    bool publish_tf_;
    double outlier_radius_;
    int outlier_min_neighbors_;
    int max_iterations_;
    double reg_translation_weight_;
    double reg_rotation_weight_;
    double multi_res_size_1_;
    double multi_res_size_2_;
    double multi_res_size_3_;
    bool use_ground_segmentation_;
    double ground_z_threshold_;
    double max_z_value_;
    double distance_weight_factor_;
    int status_publish_rate_;
    
    // Motion compensation parameters
    bool use_motion_undistortion_;
    double scan_period_;
    double min_scan_range_;
    double max_scan_range_;
    
    // Keyframe strategy parameters
    bool use_keyframe_strategy_;
    double keyframe_trans_threshold_;
    double keyframe_rot_threshold_;
    double keyframe_overlap_factor_;
    
    // ICP refinement parameters
    bool use_icp_refinement_;
    int icp_max_iterations_;
    double icp_max_correspondence_distance_;
    double icp_transformation_epsilon_;
    
    // Adaptive parameters
    bool use_adaptive_parameters_;
    double velocity_threshold_slow_;
    double velocity_threshold_fast_;
    
    // Statistics
    int frames_processed_;
    std::map<int, double> frame_times_; // frame_id -> processing time
    double accumulated_error_;
    
    // Point cloud processing
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_cloud_;
    bool has_target_cloud_;
    Eigen::Matrix4d current_pose_;
    Eigen::Matrix4d last_odom_pose_;
    ros::Time last_cloud_time_;
    Eigen::Vector3d current_velocity_;
    
    // Thread-related
    std::queue<sensor_msgs::PointCloud2::ConstPtr> cloud_queue_;
    std::mutex queue_mutex_;
    std::thread processing_thread_;
    bool processing_thread_active_;
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Limit queue size but allow large queues for offline processing
        if (cloud_queue_.size() >= max_queue_size_) {
            ROS_WARN_THROTTLE(5, "NDT registration queue has %zu messages, dropping oldest", cloud_queue_.size());
            cloud_queue_.pop();
        }
        
        cloud_queue_.push(cloud_msg);
        
        // Output queue size for monitoring
        ROS_INFO_THROTTLE(2, "Queue status: %zu frames waiting to be processed", cloud_queue_.size());
    }
    
    void publishStatus(const ros::TimerEvent&) {
        std_msgs::Int32 queue_size_msg;
        std_msgs::Int32 frames_processed_msg;
        std_msgs::Float64 error_msg;
        
        // Get current queue size
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_size_msg.data = cloud_queue_.size();
        }
        
        frames_processed_msg.data = frames_processed_;
        error_msg.data = accumulated_error_ / std::max(1, frames_processed_);
        
        // Publish statistics
        queue_size_pub_.publish(queue_size_msg);
        frames_processed_pub_.publish(frames_processed_msg);
        error_pub_.publish(error_msg);
        
        // Log status information
        ROS_INFO("Status: %d frames processed, %d frames in queue, avg error: %.6f", 
                frames_processed_, queue_size_msg.data, error_msg.data);
    }
    
    void processQueue() {
        while (processing_thread_active_ && ros::ok()) {
            sensor_msgs::PointCloud2::ConstPtr cloud_msg;
            
            // Get next cloud from queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (!cloud_queue_.empty()) {
                    cloud_msg = cloud_queue_.front();
                    cloud_queue_.pop();
                }
            }
            
            // Process cloud if available
            if (cloud_msg) {
                // Start tracking processing time for this frame
                auto process_start_time = std::chrono::steady_clock::now();
                
                // Convert from ROS to PCL
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::fromROSMsg(*cloud_msg, *cloud);
                
                ROS_INFO("Processing frame %d with %ld points", frames_processed_ + 1, cloud->size());
                
                // Apply motion undistortion if enabled
                if (use_motion_undistortion_ && has_target_cloud_) {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr undistorted_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    undistortPointCloud(cloud, undistorted_cloud, current_velocity_, cloud_msg->header.stamp);
                    cloud = undistorted_cloud;
                }
                
                // Basic pre-filtering: remove NaN points and range filtering
                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                preFilterCloud(cloud, filtered_cloud);
                
                // Process filtered point cloud
                processPointCloud(filtered_cloud, cloud_msg->header);
                
                // Update velocity estimate
                if (last_cloud_time_.toSec() > 0 && cloud_msg->header.stamp > last_cloud_time_) {
                    double dt = (cloud_msg->header.stamp - last_cloud_time_).toSec();
                    if (dt > 0) {
                        // Simple velocity estimate from position change
                        Eigen::Vector3d position_change = current_pose_.block<3, 1>(0, 3) - 
                                                         last_odom_pose_.block<3, 1>(0, 3);
                        current_velocity_ = position_change / dt;
                        
                        // Adjust parameters based on velocity if adaptive mode is enabled
                        if (use_adaptive_parameters_) {
                            adjustParametersByVelocity(current_velocity_.norm());
                        }
                    }
                }
                
                // Update time stamp
                last_cloud_time_ = cloud_msg->header.stamp;
                last_odom_pose_ = current_pose_;
                
                // Calculate total processing time
                auto process_end_time = std::chrono::steady_clock::now();
                double total_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    process_end_time - process_start_time).count() / 1000.0;
                
                // Update statistics
                frames_processed_++;
                frame_times_[frames_processed_] = total_processing_time;
                
                // Output timing information
                ROS_INFO("Frame %d processed in %.3f seconds, velocity: %.2f m/s",
                        frames_processed_, total_processing_time, current_velocity_.norm());
                
                // Output queue size after processing
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    ROS_INFO("Queue status: %zu frames remaining to be processed", cloud_queue_.size());
                }
            }
            
            // Sleep a bit to avoid consuming all CPU and to allow status updates
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    // New function to adjust parameters based on vehicle velocity
    void adjustParametersByVelocity(double velocity) {
        if (velocity < velocity_threshold_slow_) {
            // Slow motion - more accurate registration
            multi_res_size_1_ = 0.2;  // Finer resolution
            max_iterations_ = 300;    // More iterations
            outlier_radius_ = 0.2;    // Stricter outlier rejection
            reg_translation_weight_ = 0.0005; // Lower regularization
            reg_rotation_weight_ = 0.0005;    // Lower regularization
        }
        else if (velocity > velocity_threshold_fast_) {
            // Fast motion - more robust registration
            multi_res_size_1_ = 0.4;  // Coarser resolution
            max_iterations_ = 200;    // Fewer iterations for speed
            outlier_radius_ = 0.4;    // More relaxed outlier rejection
            reg_translation_weight_ = 0.002; // Higher regularization
            reg_rotation_weight_ = 0.002;    // Higher regularization
        }
        else {
            // Normal motion - balanced parameters
            multi_res_size_1_ = 0.3;
            max_iterations_ = 250;
            outlier_radius_ = 0.3;
            reg_translation_weight_ = 0.001;
            reg_rotation_weight_ = 0.001;
        }
        
        // Adjust multi-resolution parameters accordingly
        multi_res_size_2_ = multi_res_size_1_ * 2.0;
        multi_res_size_3_ = multi_res_size_1_ * 4.0;
    }
    
    // Pre-filtering for point clouds
    void preFilterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, 
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*input, *output, indices);
        
        // Apply range filtering
        pcl::PointCloud<pcl::PointXYZ>::Ptr range_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        range_filtered->reserve(output->size());
        
        for (const auto& point : output->points) {
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range >= min_scan_range_ && range <= max_scan_range_) {
                range_filtered->push_back(point);
            }
        }
        
        output = range_filtered;
    }
    
    // Motion undistortion for point clouds
    void undistortPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, 
                           pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                           const Eigen::Vector3d& velocity,
                           const ros::Time& cloud_time) {
        output->clear();
        output->reserve(input->size());
        
        // If no motion or this is the first cloud, return input unchanged
        if (!has_target_cloud_ || velocity.norm() < 0.1 || scan_period_ <= 0) {
            *output = *input;
            return;
        }
        
        // Calculate time since last frame
        double dt = 0;
        if (last_cloud_time_.toSec() > 0) {
            dt = (cloud_time - last_cloud_time_).toSec();
        }
        
        if (dt <= 0) {
            *output = *input;
            return;
        }
        
        // Extract rotation from current pose
        Eigen::Matrix3d current_rotation = current_pose_.block<3, 3>(0, 0);
        
        // For each point, apply motion compensation based on estimated time within scan
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& point = input->points[i];
            
            // Estimate relative time within scan (0.0 to 1.0) based on point index
            // This assumes points are roughly in order of scan time
            double point_time_ratio = static_cast<double>(i) / input->size();
            
            // Extrapolate position at point's scan time
            Eigen::Vector3d compensation = velocity * (point_time_ratio * scan_period_);
            
            // Transform compensation to world frame
            Eigen::Vector3d world_compensation = current_rotation * compensation;
            
            // Apply compensation
            pcl::PointXYZ corrected_point;
            corrected_point.x = point.x + world_compensation.x();
            corrected_point.y = point.y + world_compensation.y();
            corrected_point.z = point.z + world_compensation.z();
            
            output->push_back(corrected_point);
        }
        
        output->width = output->size();
        output->height = 1;
        output->is_dense = false;
    }
    
    // Process a point cloud for registration
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                         const std_msgs::Header& header) {
        // Filter points above max height (sky/noise) and optionally segment ground
        pcl::PointCloud<pcl::PointXYZ>::Ptr height_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Height-based filtering
        filterCloudByHeight(input_cloud, height_filtered_cloud, ground_z_threshold_, max_z_value_);
        
        // Ground segmentation if enabled
        if (use_ground_segmentation_) {
            segmentGroundPoints(height_filtered_cloud, non_ground_cloud);
            height_filtered_cloud = non_ground_cloud;
        }
        
        // Remove outliers for more robust registration
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_outliers(new pcl::PointCloud<pcl::PointXYZ>);
        removeOutliers(height_filtered_cloud, cloud_no_outliers);
        
        // Downsample the filtered point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        downsampleCloud(cloud_no_outliers, downsampled_cloud, downsample_resolution_);
        
        ROS_INFO("After preprocessing: %ld points", downsampled_cloud->size());
        
        // If this is the first cloud, use it as target and skip alignment
        if (!has_target_cloud_) {
            target_cloud_ = downsampled_cloud;
            keyframe_cloud_ = downsampled_cloud;
            
            // Initialize global map
            *global_map_cloud_ = *downsampled_cloud;
            
            has_target_cloud_ = true;
            ROS_INFO("Received first point cloud, using as reference");
            return;
        }
        
        // Determine whether to use keyframe or last frame as reference
        pcl::PointCloud<pcl::PointXYZ>::Ptr reference_cloud;
        
        if (use_keyframe_strategy_) {
            reference_cloud = keyframe_cloud_;
        } else {
            reference_cloud = target_cloud_;
        }
        
        // Perform multi-resolution NDT registration
        auto ndt_start_time = std::chrono::steady_clock::now();
        
        Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity();
        
        // Multi-resolution alignment, from coarse to fine
        ROS_INFO("Starting coarse-to-fine NDT registration");
        
        auto coarse_start = std::chrono::steady_clock::now();
        Eigen::Matrix4d transform_1 = ndtRegistrationCeres(downsampled_cloud, reference_cloud, initial_guess, multi_res_size_3_);
        auto coarse_end = std::chrono::steady_clock::now();
        
        auto medium_start = std::chrono::steady_clock::now();
        Eigen::Matrix4d transform_2 = ndtRegistrationCeres(downsampled_cloud, reference_cloud, transform_1, multi_res_size_2_);
        auto medium_end = std::chrono::steady_clock::now();
        
        auto fine_start = std::chrono::steady_clock::now();
        Eigen::Matrix4d transform_3 = ndtRegistrationCeres(downsampled_cloud, reference_cloud, transform_2, multi_res_size_1_);
        auto fine_end = std::chrono::steady_clock::now();
        
        // Final transform
        Eigen::Matrix4d final_transform = transform_3;
        
        // Refine with ICP if enabled
        if (use_icp_refinement_) {
            auto icp_start = std::chrono::steady_clock::now();
            
            // Prepare ICP
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setMaximumIterations(icp_max_iterations_);
            icp.setMaxCorrespondenceDistance(icp_max_correspondence_distance_);
            icp.setTransformationEpsilon(icp_transformation_epsilon_);
            
            // Apply NDT transform to create initial alignment for ICP
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*downsampled_cloud, *transformed_cloud, final_transform);
            
            // Set up ICP
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(reference_cloud);
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr icp_aligned(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*icp_aligned);
            
            if (icp.hasConverged()) {
                // Convert ICP result to Eigen matrix
                Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
                Eigen::Matrix4d icp_transform_d = icp_transform.cast<double>();
                
                // Combine NDT and ICP transformations
                final_transform = icp_transform_d * final_transform;
                
                ROS_INFO("ICP refinement converged with fitness score: %f", icp.getFitnessScore());
            } else {
                ROS_WARN("ICP refinement did not converge, using NDT result only");
            }
            
            auto icp_end = std::chrono::steady_clock::now();
            double icp_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                icp_end - icp_start).count() / 1000.0;
            
            ROS_INFO("ICP refinement took %.3f seconds", icp_time);
        }
        
        auto ndt_end_time = std::chrono::steady_clock::now();
        
        // Calculate timing for each stage
        double coarse_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            coarse_end - coarse_start).count() / 1000.0;
        double medium_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            medium_end - medium_start).count() / 1000.0;
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            fine_end - fine_start).count() / 1000.0;
        double total_ndt_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            ndt_end_time - ndt_start_time).count() / 1000.0;
        
        ROS_INFO("NDT Registration times - Coarse: %.3fs, Medium: %.3fs, Fine: %.3fs, Total: %.3fs",
                coarse_time, medium_time, fine_time, total_ndt_time);
        
        // Calculate registration error metric
        double registration_error = calculateRegistrationError(downsampled_cloud, reference_cloud, final_transform);
        accumulated_error_ += registration_error;
        
        ROS_INFO("Registration error metric: %.6f", registration_error);
        
        // Update global pose
        current_pose_ = final_transform * current_pose_;
        
        // Check whether to update keyframe
        if (use_keyframe_strategy_) {
            bool update_keyframe = shouldUpdateKeyframe(final_transform);
            if (update_keyframe) {
                keyframe_cloud_ = downsampled_cloud;
                ROS_INFO("Updated keyframe at frame %d", frames_processed_ + 1);
            }
        }
        
        // Update target cloud for next registration
        target_cloud_ = downsampled_cloud;
        
        // Transform current cloud to global coordinates and add to map
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*downsampled_cloud, *transformed_cloud, current_pose_);
        
        // Optionally add to global map (can be used for visualization or loop closure)
        *global_map_cloud_ += *transformed_cloud;
        
        // Downsample global map periodically to manage memory
        if (global_map_cloud_->size() > 100000) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            downsampleCloud(global_map_cloud_, temp_cloud, downsample_resolution_ * 2.0);
            global_map_cloud_ = temp_cloud;
        }
        
        // Publish registered cloud
        publishRegisteredCloud(transformed_cloud, header);
        
        // Publish results
        publishResults(header);
        
        // Print pose for debugging
        printCurrentPose();
    }
    
    // Calculate a registration error metric
    double calculateRegistrationError(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
                                     const Eigen::Matrix4d& transform) {
        // Transform source cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*source, *transformed_source, transform);
        
        // Build KD-tree for target
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(target);
        
        // Calculate mean squared error
        double total_error = 0.0;
        int count = 0;
        
        for (const auto& point : transformed_source->points) {
            std::vector<int> indices(1);
            std::vector<float> distances(1);
            
            if (kdtree.nearestKSearch(point, 1, indices, distances)) {
                total_error += distances[0];
                count++;
            }
        }
        
        if (count > 0) {
            return std::sqrt(total_error / count);
        } else {
            return 0.0;
        }
    }
    
    // Determine if we should update the keyframe
    bool shouldUpdateKeyframe(const Eigen::Matrix4d& relative_transform) {
        // Extract translation
        Eigen::Vector3d translation = relative_transform.block<3, 1>(0, 3);
        
        // Extract rotation in angle-axis form
        Eigen::AngleAxisd angle_axis(relative_transform.block<3, 3>(0, 0));
        double rotation_angle = std::abs(angle_axis.angle());
        
        // Check if translation or rotation exceeds thresholds
        if (translation.norm() > keyframe_trans_threshold_ || 
            rotation_angle > keyframe_rot_threshold_) {
            return true;
        }
        
        return false;
    }
    
    void printCurrentPose() {
        Eigen::Vector3d translation = current_pose_.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation_matrix = current_pose_.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);
        
        ROS_INFO_THROTTLE(1, "Current Pose - Position: [%.3f, %.3f, %.3f], Orientation: [%.3f, %.3f, %.3f, %.3f]",
                        translation.x(), translation.y(), translation.z(),
                        quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
    }
    
    void publishRegisteredCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                               const std_msgs::Header& header) {
        if (registered_cloud_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*cloud, cloud_msg);
            cloud_msg.header = header;
            cloud_msg.header.frame_id = fixed_frame_;
            registered_cloud_pub_.publish(cloud_msg);
        }
    }
    
    void publishResults(const std_msgs::Header& header) {
        // Extract translation and rotation from the 4x4 matrix
        Eigen::Vector3d translation = current_pose_.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation_matrix = current_pose_.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);
        
        // Create and publish pose message
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header = header;
        pose_msg.header.frame_id = fixed_frame_;
        
        pose_msg.pose.position.x = translation.x();
        pose_msg.pose.position.y = translation.y();
        pose_msg.pose.position.z = translation.z();
        
        pose_msg.pose.orientation.w = quaternion.w();
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        
        pose_pub_.publish(pose_msg);
        
        // Create and publish odometry message
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = fixed_frame_;
        odom_msg.child_frame_id = "lidar_link";
        
        odom_msg.pose.pose = pose_msg.pose;
        
        // Estimate velocity from position changes
        if (current_velocity_.norm() > 0) {
            odom_msg.twist.twist.linear.x = current_velocity_.x();
            odom_msg.twist.twist.linear.y = current_velocity_.y();
            odom_msg.twist.twist.linear.z = current_velocity_.z();
        } else {
            odom_msg.twist.twist.linear.x = 0.0;
            odom_msg.twist.twist.linear.y = 0.0;
            odom_msg.twist.twist.linear.z = 0.0;
        }
        
        odom_msg.twist.twist.angular.x = 0.0;
        odom_msg.twist.twist.angular.y = 0.0;
        odom_msg.twist.twist.angular.z = 0.0;
        
        odom_pub_.publish(odom_msg);
        
        // Publish TF transform if enabled
        if (publish_tf_) {
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
            tf::Quaternion tf_quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
            transform.setRotation(tf_quaternion);
            
            tf_broadcaster_.sendTransform(tf::StampedTransform(
                transform, header.stamp, fixed_frame_, "lidar_link"));
        }
    }
    
    // Filter points by height - useful for Velodyne on car
    void filterCloudByHeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                           double min_z = -std::numeric_limits<double>::max(),
                           double max_z = std::numeric_limits<double>::max()) {
        // Use PCL's PassThrough filter for more efficient filtering
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(input);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(min_z, max_z);
        pass.filter(*output);
    }
    
    // Improved ground segmentation based on height and normals
    void segmentGroundPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground) {
        // Compute normals
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        
        ne.setInputCloud(input);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.5);  // 50cm radius for normal estimation
        ne.compute(*normals);
        
        non_ground->clear();
        non_ground->reserve(input->size());
        
        // Points with normals not pointing up are not ground
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& point = input->points[i];
            const auto& normal = normals->points[i];
            
            // Check if normal points up (z component close to 1)
            if (std::abs(normal.normal_z) < 0.8 || normal.normal_z < 0) {  // Not ground if normal doesn't point up
                non_ground->push_back(point);
            }
        }
        
        // Add additional check - non-ground must be above a certain height
        pcl::PointCloud<pcl::PointXYZ>::Ptr height_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        height_filtered->reserve(non_ground->size());
        
        for (const auto& point : non_ground->points) {
            if (point.z > ground_z_threshold_ + 0.2) {  // Points significantly above ground threshold
                height_filtered->push_back(point);
            }
        }
        
        *non_ground = *height_filtered;
        non_ground->width = non_ground->size();
        non_ground->height = 1;
        non_ground->is_dense = false;
    }
    
    // Remove outliers for better registration
    void removeOutliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
        // Use radius outlier removal
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> rad_rem;
        rad_rem.setInputCloud(input);
        rad_rem.setRadiusSearch(outlier_radius_);
        rad_rem.setMinNeighborsInRadius(outlier_min_neighbors_);
        
        // Apply filter
        rad_rem.filter(*output);
        
        // If too many points were removed, use statistical outlier removal instead
        if (output->size() < input->size() * 0.5) {
            ROS_WARN("Radius outlier removal filtered too many points, using statistical filter instead");
            
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(input);
            sor.setMeanK(20);
            sor.setStddevMulThresh(1.0);
            sor.filter(*output);
        }
        
        // If still too few points, use original cloud
        if (output->size() < 100) {
            ROS_WARN("Outlier removal resulted in too few points, using original cloud");
            *output = *input;
        }
    }
    
    // Downsample point cloud
    void downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, 
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                       double resolution) {
        try {
            pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
            voxel_grid.setInputCloud(input);
            voxel_grid.setLeafSize(resolution, resolution, resolution);
            voxel_grid.filter(*output);
            
            if (output->empty()) {
                ROS_WARN("Downsampling resulted in empty cloud, using original");
                *output = *input;
            }
        }
        catch (const std::exception& e) {
            ROS_ERROR("Error in downsampling: %s", e.what());
            *output = *input;  // Use original if downsampling fails
        }
    }
    
    // Function to build NDT grid from target point cloud
    std::vector<NDTCell> buildNDTGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                                   double voxel_size = 1.0) {
        // Grid dimensions
        Eigen::Vector3d min_pt(std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max());
        Eigen::Vector3d max_pt(std::numeric_limits<double>::lowest(),
                              std::numeric_limits<double>::lowest(),
                              std::numeric_limits<double>::lowest());
        
        // Find bounding box
        for (const auto& pt : cloud->points) {
            min_pt.x() = std::min(min_pt.x(), static_cast<double>(pt.x));
            min_pt.y() = std::min(min_pt.y(), static_cast<double>(pt.y));
            min_pt.z() = std::min(min_pt.z(), static_cast<double>(pt.z));
            
            max_pt.x() = std::max(max_pt.x(), static_cast<double>(pt.x));
            max_pt.y() = std::max(max_pt.y(), static_cast<double>(pt.y));
            max_pt.z() = std::max(max_pt.z(), static_cast<double>(pt.z));
        }
        
        // Add margin to avoid boundary issues
        min_pt -= Eigen::Vector3d::Constant(voxel_size * 0.5);
        max_pt += Eigen::Vector3d::Constant(voxel_size * 0.5);
        
        // Use a map for sparse storage of cells
        std::unordered_map<size_t, NDTCell> cell_map;
        
        // Helper function to compute hash key
        auto computeHashKey = [&](const Eigen::Vector3i& idx) -> size_t {
            // Simple hash function for 3D grid coordinates
            return ((idx.x() * 73856093) ^ (idx.y() * 19349663) ^ (idx.z() * 83492791)) % 2147483647;
        };
        
        // First pass: Accumulate points and compute cell means
        for (const auto& pt : cloud->points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            Eigen::Vector3i idx = ((p - min_pt) / voxel_size).cast<int>();
            
            size_t key = computeHashKey(idx);
            auto& cell = cell_map[key];
            cell.mean += p;
            cell.point_count++;
        }
        
        // Finalize means
        for (auto& pair : cell_map) {
            auto& cell = pair.second;
            if (cell.point_count > 0) {
                cell.mean /= cell.point_count;
            }
        }
        
        // Second pass: Compute covariances
        for (const auto& pt : cloud->points) {
            Eigen::Vector3d p(pt.x, pt.y, pt.z);
            Eigen::Vector3i idx = ((p - min_pt) / voxel_size).cast<int>();
            
            size_t key = computeHashKey(idx);
            if (cell_map.count(key) > 0) {
                auto& cell = cell_map[key];
                if (cell.point_count > 0) {
                    Eigen::Vector3d diff = p - cell.mean;
                    cell.covariance += diff * diff.transpose();
                }
            }
        }
        
        // Finalize covariances and compute inverses
        // Finalize covariances and compute inverses
std::vector<NDTCell> active_cells;
const double min_eigenvalue = 0.005;  // Reduced to allow more flexibility

for (auto& pair : cell_map) {
    auto& cell = pair.second;
    if (cell.point_count > 3) {  // Need at least 3 points for 3D covariance
        cell.covariance /= (cell.point_count - 1);
        
        // Add regularization to avoid singularity
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cell.covariance);
        Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
        Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
        
        // Apply minimum eigenvalue constraint
        for (int l = 0; l < 3; l++) {
            if (eigenvalues(l) < min_eigenvalue) {
                eigenvalues(l) = min_eigenvalue;
            }
        }
        
        // Reconstruct regularized covariance
        cell.covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        
        // Check determinant before inversion to avoid singular matrices
        double det = cell.covariance.determinant();
        bool is_invertible = std::abs(det) > 1e-12;
        
        if (is_invertible) {
            // Matrix can be inverted normally
            cell.inverse_covariance = cell.covariance.inverse();
            cell.det_covariance = det;
            
            if (!std::isnan(cell.inverse_covariance.sum()) && !std::isinf(cell.inverse_covariance.sum())) {
                active_cells.push_back(cell);
            }
        } else {
            // Add more regularization and try again
            for (int l = 0; l < 3; l++) {
                eigenvalues(l) += 0.01;
            }
            cell.covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            cell.inverse_covariance = cell.covariance.inverse();
            cell.det_covariance = cell.covariance.determinant();
            
            if (!std::isnan(cell.inverse_covariance.sum()) && !std::isinf(cell.inverse_covariance.sum())) {
                active_cells.push_back(cell);
            }
        }
    }
}
        
        return active_cells;
    }

    // Main NDT registration function with Ceres optimization
    Eigen::Matrix4d ndtRegistrationCeres(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
        const Eigen::Matrix4d& initial_guess = Eigen::Matrix4d::Identity(),
        double voxel_size = 1.0) {
        
        auto grid_start = std::chrono::steady_clock::now();
        
        // Build NDT grid for target cloud
        std::vector<NDTCell> target_cells = buildNDTGrid(target_cloud, voxel_size);
        
        auto grid_end = std::chrono::steady_clock::now();
        double grid_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            grid_end - grid_start).count() / 1000.0;
            
        ROS_DEBUG("NDT grid building took %.3f seconds, created %zu cells", 
                grid_time, target_cells.size());
        
        if (target_cells.empty()) {
            ROS_WARN("No valid NDT cells created from target cloud");
            return initial_guess;
        }
        
        auto kdtree_start = std::chrono::steady_clock::now();
        
        // Build KD-tree for nearest cell search
        pcl::PointCloud<pcl::PointXYZ>::Ptr cell_centers(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& cell : target_cells) {
            pcl::PointXYZ pt;
            pt.x = cell.mean.x();
            pt.y = cell.mean.y();
            pt.z = cell.mean.z();
            cell_centers->push_back(pt);
        }
        
        pcl::search::KdTree<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cell_centers);
        
        auto kdtree_end = std::chrono::steady_clock::now();
        double kdtree_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            kdtree_end - kdtree_start).count() / 1000.0;
            
        ROS_DEBUG("KD-tree building took %.3f seconds", kdtree_time);
        
        auto problem_start = std::chrono::steady_clock::now();
        
        // Setup Ceres optimization problem
        ceres::Problem problem;
        
        // Extract initial guess parameters
        Eigen::Matrix3d init_rotation = initial_guess.block<3, 3>(0, 0);
        Eigen::Vector3d init_translation = initial_guess.block<3, 1>(0, 3);
        
        // Ensure rotation matrix is valid (orthogonal)
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(init_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
        init_rotation = svd.matrixU() * svd.matrixV().transpose();
        
        // Convert rotation matrix to angle-axis representation
        Eigen::AngleAxisd angle_axis(init_rotation);
        Eigen::Vector3d rotation_aa = angle_axis.angle() * angle_axis.axis();
        
        // Handle edge case where angle is almost zero
        if (angle_axis.angle() < 1e-10) {
            rotation_aa = Eigen::Vector3d::Zero();
        }
        
        // Optimization parameters: [tx, ty, tz, rx, ry, rz]
        double transformation[6] = {
            init_translation.x(), init_translation.y(), init_translation.z(),
            rotation_aa.x(), rotation_aa.y(), rotation_aa.z()
        };
        
        // Add regularization term to stabilize optimization
        ceres::CostFunction* regularization_cost = 
            new TransformationRegularizationCostFunction(reg_translation_weight_, reg_rotation_weight_);
        problem.AddResidualBlock(
            regularization_cost,
            nullptr,
            transformation
        );
        
        // Add cost functions for each source point
        int added_residuals = 0;
        int max_residuals = 2000;  // Limit number of residuals for performance
        
        // Randomly select points if source cloud is too large
        std::vector<int> indices;
        if (source_cloud->size() > max_residuals) {
            indices.resize(max_residuals);
            // Generate random indices without replacement
            std::vector<int> all_indices(source_cloud->size());
            for (size_t i = 0; i < source_cloud->size(); ++i) {
                all_indices[i] = i;
            }
            std::random_shuffle(all_indices.begin(), all_indices.end());
            for (int i = 0; i < max_residuals; ++i) {
                indices[i] = all_indices[i];
            }
        } else {
            indices.resize(source_cloud->size());
            for (size_t i = 0; i < source_cloud->size(); ++i) {
                indices[i] = i;
            }
        }
        
        // Add residuals for selected points
        for (int idx : indices) {
            const auto& pt = source_cloud->points[idx];
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            
            // Find the nearest NDT cell
            std::vector<int> nn_indices(1);
            std::vector<float> nn_distances(1);
            pcl::PointXYZ search_pt;
            search_pt.x = point.x();
            search_pt.y = point.y();
            search_pt.z = point.z();
            
            if (kdtree.nearestKSearch(search_pt, 1, nn_indices, nn_distances)) {
                const NDTCell& nearest_cell = target_cells[nn_indices[0]];
                
                // Only use cells with reasonable distance
                if (nn_distances[0] <= voxel_size * 2.0) {
                    // Calculate weight based on distance to cell center
                    double distance_weight = 1.0;
                    if (distance_weight_factor_ > 0) {
                        // Points closer to cell center get higher weight
                        distance_weight = std::exp(-distance_weight_factor_ * nn_distances[0]);
                    }
                    
                    // Add cost function with weight
                    ceres::CostFunction* cost_function = 
                        new NDTCostFunction(point, nearest_cell, distance_weight);
                        
                    problem.AddResidualBlock(
                        cost_function,
                        new ceres::HuberLoss(0.1),  // Robust loss function
                        transformation
                    );
                    added_residuals++;
                }
            }
        }
        
        if (added_residuals == 0) {
            ROS_WARN("No valid residuals added to optimization problem");
            return initial_guess;
        }
        
        auto problem_end = std::chrono::steady_clock::now();
        double problem_setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            problem_end - problem_start).count() / 1000.0;
            
        ROS_DEBUG("Problem setup took %.3f seconds, added %d residuals", 
                problem_setup_time, added_residuals);
        
        // Set solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = max_iterations_;
        options.function_tolerance = 1e-8;  // Tighter tolerances for better accuracy
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-10;
        options.num_threads = 4;  // Use multiple threads
        
        auto solve_start = std::chrono::steady_clock::now();
        
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        auto solve_end = std::chrono::steady_clock::now();
        double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            solve_end - solve_start).count() / 1000.0;
            
        ROS_DEBUG("Ceres solver took %.3f seconds, %d iterations",
                 solve_time, summary.iterations.size());
        
        // Check if optimization was successful
        if (!summary.IsSolutionUsable()) {
            ROS_WARN("NDT optimization failed: %s", summary.BriefReport().c_str());
            return initial_guess;
        }
        
        // Convert solution back to transformation matrix
        Eigen::Vector3d final_translation(
            transformation[0], transformation[1], transformation[2]);
            
        Eigen::Vector3d final_rotation_aa(
            transformation[3], transformation[4], transformation[5]);
        
        Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
        
        // Handle case where rotation angle is very small
        double rotation_norm = final_rotation_aa.norm();
        if (rotation_norm < 1e-10) {
            // Identity rotation if angle is too small
            result.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        } else {
            Eigen::AngleAxisd final_rotation(rotation_norm, final_rotation_aa / rotation_norm);
            result.block<3, 3>(0, 0) = final_rotation.toRotationMatrix();
        }
        
        result.block<3, 1>(0, 3) = final_translation;
        
        // Log the final transformation parameters
        ROS_DEBUG("NDT Registration result - Translation: [%.3f, %.3f, %.3f], Rotation magnitude: %.6f rad",
                final_translation.x(), final_translation.y(), final_translation.z(), rotation_norm);
        
        double initial_cost = summary.initial_cost;
        double final_cost = summary.final_cost;
        double cost_reduction = 100.0 * (initial_cost - final_cost) / initial_cost;
        
        ROS_DEBUG("NDT optimization reduced cost by %.2f%% (from %.6f to %.6f)",
                cost_reduction, initial_cost, final_cost);
        
        return result;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "high_accuracy_ndt_registration");
    ros::NodeHandle nh("~");
    
    HighAccuracyNDTRegistration ndt_registration(nh);
    
    ros::spin();
    
    return 0;
}