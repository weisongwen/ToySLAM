#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>

// TASLO: Optimized for Velodyne HDL-32E LiDAR
// Based on FLOAM/A-LOAM implementation strategies

class TASLO {
public:
    TASLO(ros::NodeHandle& nh) : nh_(nh) {
        // Load parameters
        nh_.param<int>("scan_line", scan_line_, 32);  // Updated for HDL-32E
        nh_.param<double>("edge_threshold", edge_threshold_, 0.25);  // Adjusted for HDL-32E
        nh_.param<double>("surf_threshold", surf_threshold_, 0.08);  // Adjusted for HDL-32E
        nh_.param<double>("map_resolution", map_resolution_, 0.3);   // Adjusted for HDL-32E
        nh_.param<double>("min_scan_range", min_scan_range_, 1.0);
        nh_.param<double>("max_scan_range", max_range_, 80.0);
        nh_.param<bool>("mapping_flag", mapping_flag_, true);
        nh_.param<bool>("publish_debug_clouds", publish_debug_clouds_, true);
        nh_.param<bool>("use_laser_scan_lines", use_laser_scan_lines_, true);
        nh_.param<bool>("use_sub_maps", use_sub_maps_, true);
        nh_.param<int>("optimization_iterations", optimization_iterations_, 10);  // Reduced for faster processing
        nh_.param<double>("max_angular_velocity", max_angular_velocity_, 0.5); // rad/s
        nh_.param<double>("max_linear_velocity", max_linear_velocity_, 1.0); // m/s
        nh_.param<double>("distance_threshold_downsample", distance_threshold_downsample_, 5.0);
        nh_.param<bool>("enable_motion_compensation", enable_motion_compensation_, true);
        nh_.param<double>("mapping_frequency", mapping_frequency_, 10.0);
        nh_.param<double>("scan_period", scan_period_, 0.05);  // 20Hz for HDL-32E
        nh_.param<bool>("save_trajectory", save_trajectory_, true);
        nh_.param<std::string>("trajectory_filename", trajectory_filename_, "taslo_trajectory.txt");
        nh_.param<bool>("use_ring_field", use_ring_field_, true);  // Use ring field if available
        nh_.param<double>("keyframe_angle_threshold", keyframe_angle_threshold_, 0.05);  // More sensitive
        nh_.param<double>("keyframe_distance_threshold", keyframe_distance_threshold_, 0.2);  // More sensitive
        nh_.param<int>("keyframe_time_interval", keyframe_time_interval_, 10);  // Create keyframe every N frames
        nh_.param<std::string>("map_frame", map_frame_, "map");
        nh_.param<std::string>("odom_frame", odom_frame_, "odom");
        nh_.param<std::string>("lidar_frame", lidar_frame_, "velodyne");
        nh_.param<double>("icp_fitness_threshold", icp_fitness_threshold_, 0.3);  // Threshold for ICP convergence
        
        // Enhanced motion detection parameters
        nh_.param<bool>("use_aggressive_motion_detection", use_aggressive_motion_detection_, true);
        nh_.param<double>("min_motion_threshold", min_motion_threshold_, 0.05);  // 5cm minimum motion detection
        nh_.param<int>("forced_motion_interval", forced_motion_interval_, 20);  // Force motion detection every N frames
        nh_.param<bool>("use_constant_velocity", use_constant_velocity_, true);  // Use constant velocity motion model
        nh_.param<bool>("use_aloam_factors", use_aloam_factors_, true);  // Use A-LOAM correspondence factors
        nh_.param<double>("feature_min_distance", feature_min_distance_, 0.15);  // Minimum distance between features
        nh_.param<double>("system_noise", system_noise_, 0.001);  // System noise for regularization
        
        // Initialize clouds
        edge_points_sharp_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        edge_points_less_sharp_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        surf_points_flat_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        surf_points_less_flat_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        curr_cloud_raw_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        prev_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        
        // For mapping
        edge_points_map_local_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        surf_points_map_local_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        edge_points_map_global_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        surf_points_map_global_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        
        // KD-Trees
        kdtree_edge_map_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtree_surf_map_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        
        // Initial pose
        system_initialized_ = false;
        first_frame_processed_ = false;
        frame_count_ = 0;
        frames_without_motion_ = 0;
        q_w_curr_ = Eigen::Quaterniond(1, 0, 0, 0);
        t_w_curr_ = Eigen::Vector3d(0, 0, 0);
        last_keyframe_t_ = t_w_curr_;
        last_keyframe_q_ = q_w_curr_;
        
        // Motion tracking
        prev_to_curr_transform_ = Eigen::Matrix4f::Identity();
        
        // Motion prediction
        last_frame_time_ = ros::Time(0);
        linear_velocity_ = Eigen::Vector3d::Zero();
        angular_velocity_ = Eigen::Vector3d::Zero();
        
        // Subscribers
        sub_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, &TASLO::cloudHandler, this);
        
        // Publishers
        pub_odom_ = nh_.advertise<nav_msgs::Odometry>("odometry", 10);
        pub_path_ = nh_.advertise<nav_msgs::Path>("path", 10);
        pub_edge_points_ = nh_.advertise<sensor_msgs::PointCloud2>("edge_points", 10);
        pub_surf_points_ = nh_.advertise<sensor_msgs::PointCloud2>("surf_points", 10);
        pub_local_map_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map", 10);
        pub_global_map_ = nh_.advertise<sensor_msgs::PointCloud2>("global_map", 1);
        
        // TF broadcaster
        tf_broadcaster_ = new tf::TransformBroadcaster();
        
        // Path
        path_.header.frame_id = map_frame_;
        
        // Start processing thread
        processing_thread_ = std::thread(&TASLO::processQueue, this);
        mapping_thread_ = std::thread(&TASLO::mappingThread, this);
        thread_running_ = true;
        
        if (save_trajectory_) {
            trajectory_file_.open(trajectory_filename_);
            if (!trajectory_file_.is_open()) {
                ROS_ERROR("Failed to open trajectory file: %s", trajectory_filename_.c_str());
                save_trajectory_ = false;
            } else {
                // Write header
                trajectory_file_ << "# timestamp tx ty tz qx qy qz qw" << std::endl;
            }
        }
        
        ROS_INFO("TASLO initialized with scan_line = %d (HDL-32E), map_resolution = %.2f", scan_line_, map_resolution_);
        ROS_INFO("Using A-LOAM factors: %s, Constant velocity model: %s", 
                use_aloam_factors_ ? "true" : "false", 
                use_constant_velocity_ ? "true" : "false");
        ROS_INFO("Frame IDs: map=%s, odom=%s, lidar=%s", 
                map_frame_.c_str(), odom_frame_.c_str(), lidar_frame_.c_str());
    }
    
    ~TASLO() {
        thread_running_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        if (mapping_thread_.joinable()) {
            mapping_thread_.join();
        }
        
        if (tf_broadcaster_ != nullptr) {
            delete tf_broadcaster_;
        }
        
        if (trajectory_file_.is_open()) {
            trajectory_file_.close();
        }
    }

private:
    // ROS
    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_edge_points_;
    ros::Publisher pub_surf_points_;
    ros::Publisher pub_local_map_;
    ros::Publisher pub_global_map_;
    tf::TransformBroadcaster* tf_broadcaster_;
    
    // Parameters
    int scan_line_;
    double edge_threshold_;
    double surf_threshold_;
    double map_resolution_;
    double min_scan_range_;
    double max_range_;
    bool mapping_flag_;
    bool publish_debug_clouds_;
    bool use_laser_scan_lines_;
    bool use_sub_maps_;
    int optimization_iterations_;
    double max_angular_velocity_;
    double max_linear_velocity_;
    double distance_threshold_downsample_;
    bool enable_motion_compensation_;
    double mapping_frequency_;
    double scan_period_;
    bool save_trajectory_;
    std::string trajectory_filename_;
    std::ofstream trajectory_file_;
    bool use_ring_field_;
    double keyframe_angle_threshold_;
    double keyframe_distance_threshold_;
    int keyframe_time_interval_;
    double icp_fitness_threshold_;
    std::string map_frame_;
    std::string odom_frame_;
    std::string lidar_frame_;
    
    // Enhanced motion detection parameters
    bool use_aggressive_motion_detection_;
    double min_motion_threshold_;
    int forced_motion_interval_;
    int frames_without_motion_;
    bool use_constant_velocity_;
    bool use_aloam_factors_;
    double feature_min_distance_;
    double system_noise_;
    
    // Point clouds
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points_sharp_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points_less_sharp_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_flat_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_less_flat_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_cloud_raw_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr prev_cloud_; // For frame-to-frame matching
    
    // Map point clouds
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points_map_local_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_map_local_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points_map_global_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_map_global_;
    
    // KD-Trees
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_edge_map_;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surf_map_;
    
    // Pose
    bool system_initialized_;
    bool first_frame_processed_;
    int frame_count_;
    Eigen::Quaterniond q_w_curr_;
    Eigen::Vector3d t_w_curr_;
    
    // For transform tracking
    Eigen::Matrix4f prev_to_curr_transform_;
    
    // For keyframe
    Eigen::Vector3d last_keyframe_t_;
    Eigen::Quaterniond last_keyframe_q_;
    
    // For motion prediction and compensation
    ros::Time last_frame_time_;
    Eigen::Vector3d linear_velocity_;
    Eigen::Vector3d angular_velocity_;
    
    // For path visualization
    nav_msgs::Path path_;
    
    // Thread for processing
    std::thread processing_thread_;
    std::thread mapping_thread_;
    bool thread_running_;
    
    // Mutexes
    std::mutex cloud_mutex_;
    std::mutex map_mutex_;
    std::mutex pose_mutex_;
    
    // Data queues
    std::queue<sensor_msgs::PointCloud2ConstPtr> cloud_queue_;
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> edge_map_update_queue_;
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> surf_map_update_queue_;
    
    // For transformation representation
    struct TransformationParameters {
        Eigen::Vector3d translation;
        Eigen::Vector3d rotation; // Euler angles or axis-angle representation
    };
    
    // For curvature calculation
    struct PointInfo {
        int index;
        float curvature;
        int label; // 0-normal, 1-edge, 2-surface, 3-flat, 4-less-flat
        Eigen::Vector3f raw_point;
        Eigen::Vector3f point;
        
        bool operator<(const PointInfo& other) const {
            return curvature < other.curvature;
        }
    };
    
    struct ScanLine {
        std::vector<PointInfo> point_infos;
        float start_orientation;
        float end_orientation;
    };
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        cloud_queue_.push(cloud_msg);
    }
    
    void processQueue() {
        while(thread_running_) {
            sensor_msgs::PointCloud2ConstPtr cloud_msg = nullptr;
            
            {
                std::lock_guard<std::mutex> lock(cloud_mutex_);
                if (!cloud_queue_.empty()) {
                    cloud_msg = cloud_queue_.front();
                    cloud_queue_.pop();
                }
            }
            
            if (cloud_msg) {
                processCloud(cloud_msg);
            }
            
            // Sleep to avoid CPU thrashing
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    
    void mappingThread() {
        while(thread_running_) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud = nullptr;
            pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud = nullptr;
            
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                if (!edge_map_update_queue_.empty() && !surf_map_update_queue_.empty()) {
                    edge_cloud = edge_map_update_queue_.front();
                    surf_cloud = surf_map_update_queue_.front();
                    edge_map_update_queue_.pop();
                    surf_map_update_queue_.pop();
                }
            }
            
            if (edge_cloud && surf_cloud) {
                // Transform to world frame
                transformToWorld(edge_cloud);
                transformToWorld(surf_cloud);
                
                // Add to global map
                *edge_points_map_global_ += *edge_cloud;
                *surf_points_map_global_ += *surf_cloud;
                
                // Downsample global map
                downsampleGlobalMap();
                
                // Publish global map
                publishGlobalMap();
            }
            
            // Sleep to maintain mapping frequency
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / mapping_frequency_)));
        }
    }
    
    void transformToWorld(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            q = q_w_curr_;
            t = t_w_curr_;
        }
        
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3,3>(0,0) = q.toRotationMatrix().cast<float>();
        transform.block<3,1>(0,3) = t.cast<float>();
        
        pcl::transformPointCloud(*cloud, *cloud, transform);
    }
    
    void downsampleGlobalMap() {
        if (edge_points_map_global_->size() > 10000) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr edge_map_ds(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> edge_downsize_filter;
            edge_downsize_filter.setLeafSize(map_resolution_ * 0.75, map_resolution_ * 0.75, map_resolution_ * 0.75);
            edge_downsize_filter.setInputCloud(edge_points_map_global_);
            edge_downsize_filter.filter(*edge_map_ds);
            edge_points_map_global_ = edge_map_ds;
        }
        
        if (surf_points_map_global_->size() > 20000) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr surf_map_ds(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> surf_downsize_filter;
            surf_downsize_filter.setLeafSize(map_resolution_ * 1.5, map_resolution_ * 1.5, map_resolution_ * 1.5);
            surf_downsize_filter.setInputCloud(surf_points_map_global_);
            surf_downsize_filter.filter(*surf_map_ds);
            surf_points_map_global_ = surf_map_ds;
        }
    }
    
    void publishGlobalMap() {
        if (pub_global_map_.getNumSubscribers() > 0) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZI>());
            *global_map = *edge_points_map_global_;
            *global_map += *surf_points_map_global_;
            
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*global_map, map_msg);
            map_msg.header.frame_id = map_frame_;
            map_msg.header.stamp = ros::Time::now();
            pub_global_map_.publish(map_msg);
        }
    }
    
    // Transform conversion utilities
    Eigen::Matrix4f transformToMatrix(const TransformationParameters& params) {
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        
        // Convert rotation to matrix form
        Eigen::AngleAxisf roll(params.rotation.x(), Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitch(params.rotation.y(), Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yaw(params.rotation.z(), Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f rotation_matrix = (yaw * pitch * roll).matrix();
        
        // Set rotation part
        matrix.block<3, 3>(0, 0) = rotation_matrix;
        
        // Set translation part
        matrix.block<3, 1>(0, 3) = params.translation.cast<float>();
        
        return matrix;
    }
    
    TransformationParameters matrixToTransform(const Eigen::Matrix4f& matrix) {
        TransformationParameters params;
        
        // Extract translation
        params.translation = matrix.block<3, 1>(0, 3).cast<double>();
        
        // Extract rotation (convert to Euler angles)
        Eigen::Matrix3f rotation_matrix = matrix.block<3, 3>(0, 0);
        params.rotation.x() = atan2(rotation_matrix(2, 1), rotation_matrix(2, 2));
        params.rotation.y() = -asin(rotation_matrix(2, 0));
        params.rotation.z() = atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
        
        return params;
    }
    
    void processCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert to PCL point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*cloud_msg, *current_cloud);
        
        if (current_cloud->empty()) {
            ROS_WARN("Empty cloud received!");
            return;
        }
        
        ROS_INFO("Processing cloud with %ld points", current_cloud->size());
        curr_cloud_raw_ = current_cloud;
        
        // Clear feature clouds
        edge_points_sharp_->clear();
        edge_points_less_sharp_->clear();
        surf_points_flat_->clear();
        surf_points_less_flat_->clear();
        
        // Extract features using adaptive thresholding
        extractFeatures(curr_cloud_raw_, cloud_msg);
        
        // Frame-to-frame odometry: calculate initial transformation from previous to current frame
        Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
        
        if (!first_frame_processed_) {
            // First frame - just initialize
            first_frame_processed_ = true;
            system_initialized_ = true;
            
            // Copy current cloud as previous for next iteration
            *prev_cloud_ = *curr_cloud_raw_;
            
            // Set initial map
            edge_points_map_local_->clear();
            surf_points_map_local_->clear();
            *edge_points_map_local_ = *edge_points_less_sharp_;
            *surf_points_map_local_ = *surf_points_less_flat_;
            
            kdtree_edge_map_->setInputCloud(edge_points_map_local_);
            kdtree_surf_map_->setInputCloud(surf_points_map_local_);
            
            // Add to map update queue
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                edge_map_update_queue_.push(edge_points_less_sharp_);
                surf_map_update_queue_.push(surf_points_less_flat_);
            }
            
            // Set initial pose
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                q_w_curr_ = Eigen::Quaterniond(1, 0, 0, 0);
                t_w_curr_ = Eigen::Vector3d(0, 0, 0);
            }
            
            ROS_INFO("System initialized!");
            
            // Publish initial results
            publishResults(cloud_msg->header.stamp);
            
            // Update time
            last_frame_time_ = cloud_msg->header.stamp;
            
            return;
        }
        
        // Frame count for debugging
        frame_count_++;
        
        // Use motion prediction based on previous transforms
        if (use_constant_velocity_ && frame_count_ >= 2) {
            initial_guess = predictMotion();
            
            Eigen::Vector3f translation = initial_guess.block<3,1>(0,3);
            double motion_magnitude = translation.norm();
            
            ROS_INFO("Used motion prediction for initial guess: tx=%.3f, ty=%.3f, tz=%.3f, magnitude=%.3f", 
                    translation.x(), translation.y(), translation.z(), motion_magnitude);
        } else {
            // For first frames or if constant velocity is disabled, use identity
            initial_guess.setIdentity();
            
            // If we haven't had motion in a while, inject a small forward motion
            if (frames_without_motion_ > forced_motion_interval_ / 2) {
                initial_guess(0, 3) = 0.05; // Small forward motion
                initial_guess(1, 3) = 0.01 * (rand() % 3 - 1); // Small random lateral motion
                ROS_WARN("Injecting small forward motion after %d static frames", frames_without_motion_);
            }
        }
        
        // Convert initial guess to quaternion and translation
        Eigen::Quaterniond q_init;
        Eigen::Vector3d t_init;
        
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            // Get current global pose
            Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
            current_pose.block<3,3>(0,0) = q_w_curr_.toRotationMatrix().cast<float>();
            current_pose.block<3,1>(0,3) = t_w_curr_.cast<float>();
            
            // Apply initial guess (local transform) to global pose
            Eigen::Matrix4f new_pose = current_pose * initial_guess;
            
            // Extract rotation and translation
            Eigen::Matrix3f rotation = new_pose.block<3,3>(0,0);
            q_init = Eigen::Quaterniond(rotation.cast<double>());
            t_init = new_pose.block<3,1>(0,3).cast<double>();
            
            q_init.normalize();
        }
        
        // Optimize the initial guess using point-based alignment (A-LOAM style)
        optimizeOdometry(q_init, t_init);
        
        // Check if this is a keyframe
        bool is_keyframe = isKeyframe();
        if (is_keyframe && mapping_flag_) {
            updateLocalMap();
            
            // Save keyframe info
            last_keyframe_q_ = q_w_curr_;
            last_keyframe_t_ = t_w_curr_;
        }
        
        // Calculate the transform from prev to current for motion prediction
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity();
            current_pose.block<3,3>(0,0) = q_w_curr_.toRotationMatrix().cast<float>();
            current_pose.block<3,1>(0,3) = t_w_curr_.cast<float>();
            
            // Store last transform for next frame
            if (frame_count_ >= 2) {
                Eigen::Matrix4f prev_pose = current_pose * initial_guess.inverse();
                prev_to_curr_transform_ = prev_pose.inverse() * current_pose;
            } else {
                prev_to_curr_transform_ = initial_guess;
            }
            
            // Log the transform for debugging
            Eigen::Vector3f translation = prev_to_curr_transform_.block<3,1>(0,3);
            double motion_magnitude = translation.norm();
            ROS_INFO("Frame-to-frame transform: [%.3f, %.3f, %.3f], magnitude=%.3f", 
                    translation.x(), translation.y(), translation.z(), motion_magnitude);
                    
            // Check if this is significant motion
            if (motion_magnitude > min_motion_threshold_) {
                frames_without_motion_ = 0;
            } else {
                frames_without_motion_++;
                if (frames_without_motion_ > 5) {
                    ROS_WARN("Minimal motion detected for %d consecutive frames", frames_without_motion_);
                }
            }
        }
        
        // Store current cloud as previous for next iteration
        *prev_cloud_ = *curr_cloud_raw_;
        
        // Publish results
        publishResults(cloud_msg->header.stamp);
        
        // Save trajectory if enabled
        if (save_trajectory_) {
            saveTrajectory(cloud_msg->header.stamp);
        }
        
        // Update time
        last_frame_time_ = cloud_msg->header.stamp;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        ROS_INFO("Processing time: %ld ms", duration);
    }
    
    // A-LOAM/FLOAM-inspired motion prediction
    Eigen::Matrix4f predictMotion() {
        Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
        
        // If we've processed at least 2 frames, use constant velocity model
        if (frame_count_ >= 2) {
            // Use previous transform directly
            result = prev_to_curr_transform_;
            
            // Add a bit more forward motion if we've had consecutive frames with little motion
            if (frames_without_motion_ > 5) {
                Eigen::Vector3f translation = result.block<3,1>(0,3);
                double motion_magnitude = translation.norm();
                
                // If previous motion was very small, add a minimum forward motion
                if (motion_magnitude < 0.02) {
                    // Assume forward is in x direction (adjust as needed for your coordinate system)
                    result(0, 3) += 0.05; // 5cm forward
                    result(1, 3) += 0.01 * ((frame_count_ % 3) - 1); // Small random lateral motion
                    ROS_WARN("Adding minimum forward motion after %d static frames", frames_without_motion_);
                }
            }
        }
        
        return result;
    }
    
    // Enhanced feature extraction with adaptive thresholds (FLOAM/A-LOAM style)
    void extractFeatures(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
                        const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        std::vector<ScanLine> scan_lines;
        
        // Organize point cloud by ring or scan line
        if (use_ring_field_) {
            // Find ring field
            int ring_field_idx = -1;
            for (size_t i = 0; i < cloud_msg->fields.size(); ++i) {
                if (cloud_msg->fields[i].name == "ring") {
                    ring_field_idx = i;
                    break;
                }
            }
            
            if (ring_field_idx >= 0) {
                organizePointCloudByRing(cloud, scan_lines, cloud_msg, ring_field_idx);
            } else {
                organizeByScanAngles(cloud, scan_lines);
            }
        } else if (use_laser_scan_lines_) {
            organizePointCloudByScanLines(cloud, scan_lines);
        } else {
            organizeByScanAngles(cloud, scan_lines);
        }
        
        // Extract features from each scan line with adaptive thresholds
        int total_points = 0;
        for (const auto& scan_line : scan_lines) {
            total_points += scan_line.point_infos.size();
        }
        
        // Clear feature clouds
        edge_points_sharp_->clear();
        edge_points_less_sharp_->clear();
        surf_points_flat_->clear();
        surf_points_less_flat_->clear();
        
        // Target numbers of features - these should be proportional to the scan density
        int target_sharp_points = std::min(2000, total_points / 100); 
        int target_less_sharp_points = std::min(4000, total_points / 50);
        int target_flat_points = std::min(4000, total_points / 50);
        int target_less_flat_points = std::min(8000, total_points / 20);
        
        // Process each scan line
        for (auto& scan_line : scan_lines) {
            if (scan_line.point_infos.size() < 20) {
                continue;  // Skip scan lines with too few points
            }
            
            // Calculate curvature for each point
            calculateCurvatureForScanLine(scan_line);
            
            // Sort points by curvature
            std::sort(scan_line.point_infos.begin(), scan_line.point_infos.end());
            
            // Calculate adaptive thresholds per scan line
            float edge_threshold_for_line = calculateAdaptiveEdgeThreshold(scan_line);
            float surf_threshold_for_line = calculateAdaptiveSurfThreshold(scan_line);
            
            // Compute counts based on proportion of points in this line vs. total
            float ratio = static_cast<float>(scan_line.point_infos.size()) / static_cast<float>(total_points);
            int line_sharp_count = std::max(2, static_cast<int>(target_sharp_points * ratio));
            int line_less_sharp_count = std::max(4, static_cast<int>(target_less_sharp_points * ratio));
            int line_flat_count = std::max(4, static_cast<int>(target_flat_points * ratio));
            int line_less_flat_count = std::max(8, static_cast<int>(target_less_flat_points * ratio));
            
            // Extract sharp edge points (high curvature)
            extractSharpPointsFromLine(scan_line, line_sharp_count, line_less_sharp_count, edge_threshold_for_line);
            
            // Extract flat surface points (low curvature)
            extractFlatPointsFromLine(scan_line, line_flat_count, line_less_flat_count, surf_threshold_for_line);
        }
        
        // Downsample to ensure manageable feature counts while maintaining distribution
        downsampleFeatures();
        
        ROS_INFO("Extracted: %ld sharp corners, %ld less-sharp corners, %ld flat surfaces, %ld less-flat surfaces",
                edge_points_sharp_->size(), edge_points_less_sharp_->size(),
                surf_points_flat_->size(), surf_points_less_flat_->size());
                
        // Publish feature clouds for debugging if requested
        if (publish_debug_clouds_) {
            publishFeatureClouds();
        }
    }
    
    // Adaptive threshold calculations (FLOAM-inspired)
    float calculateAdaptiveEdgeThreshold(const ScanLine& scan_line) {
        if (scan_line.point_infos.size() < 20) return static_cast<float>(edge_threshold_);
        
        // Use percentile-based approach for more robust threshold estimation
        int high_curvature_idx = std::max(0, static_cast<int>(scan_line.point_infos.size() * 0.9));
        float high_curvature = scan_line.point_infos[high_curvature_idx].curvature;
        
        // Apply adaptive scaling but ensure minimum threshold
        return std::max(static_cast<float>(edge_threshold_), high_curvature * 0.5f);
    }

    float calculateAdaptiveSurfThreshold(const ScanLine& scan_line) {
        if (scan_line.point_infos.size() < 20) return static_cast<float>(surf_threshold_);
        
        // Use percentile-based approach for more robust threshold estimation
        int low_curvature_idx = std::min(static_cast<int>(scan_line.point_infos.size() * 0.1), 
                                        static_cast<int>(scan_line.point_infos.size()) - 1);
        float low_curvature = scan_line.point_infos[low_curvature_idx].curvature;
        
        // Apply adaptive scaling but ensure minimum threshold
        return std::max(static_cast<float>(surf_threshold_), low_curvature * 2.0f);
    }
    
    void calculateCurvatureForScanLine(ScanLine& scan_line) {
        int point_count = scan_line.point_infos.size();
        if (point_count < 10) return;
        
        // Ensure points are sorted by orientation
        std::sort(scan_line.point_infos.begin(), scan_line.point_infos.end(),
                [](const PointInfo& a, const PointInfo& b) {
                    return std::atan2(a.raw_point.y(), a.raw_point.x()) < std::atan2(b.raw_point.y(), b.raw_point.x());
                });
        
        // Calculate curvature for each point using a window of points
        for (int i = 5; i < point_count - 5; i++) {
            float diff_x = scan_line.point_infos[i - 5].point.x() + scan_line.point_infos[i - 4].point.x() +
                          scan_line.point_infos[i - 3].point.x() + scan_line.point_infos[i - 2].point.x() +
                          scan_line.point_infos[i - 1].point.x() - 10 * scan_line.point_infos[i].point.x() +
                          scan_line.point_infos[i + 1].point.x() + scan_line.point_infos[i + 2].point.x() +
                          scan_line.point_infos[i + 3].point.x() + scan_line.point_infos[i + 4].point.x() +
                          scan_line.point_infos[i + 5].point.x();
            float diff_y = scan_line.point_infos[i - 5].point.y() + scan_line.point_infos[i - 4].point.y() +
                          scan_line.point_infos[i - 3].point.y() + scan_line.point_infos[i - 2].point.y() +
                          scan_line.point_infos[i - 1].point.y() - 10 * scan_line.point_infos[i].point.y() +
                          scan_line.point_infos[i + 1].point.y() + scan_line.point_infos[i + 2].point.y() +
                          scan_line.point_infos[i + 3].point.y() + scan_line.point_infos[i + 4].point.y() +
                          scan_line.point_infos[i + 5].point.y();
            float diff_z = scan_line.point_infos[i - 5].point.z() + scan_line.point_infos[i - 4].point.z() +
                          scan_line.point_infos[i - 3].point.z() + scan_line.point_infos[i - 2].point.z() +
                          scan_line.point_infos[i - 1].point.z() - 10 * scan_line.point_infos[i].point.z() +
                          scan_line.point_infos[i + 1].point.z() + scan_line.point_infos[i + 2].point.z() +
                          scan_line.point_infos[i + 3].point.z() + scan_line.point_infos[i + 4].point.z() +
                          scan_line.point_infos[i + 5].point.z();
            
            scan_line.point_infos[i].curvature = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        }
    }
    
    void extractSharpPointsFromLine(ScanLine& scan_line, int num_sharp, int num_less_sharp, float threshold) {
        int point_count = scan_line.point_infos.size();
        if (point_count < 10) return;
        
        // Points are already sorted by curvature (ascending)
        int selected_sharp = 0;
        int selected_less_sharp = 0;
        
        // Select sharp points from highest curvature
        for (int i = point_count - 1; i >= 0 && (selected_sharp < num_sharp || selected_less_sharp < num_less_sharp); i--) {
            if (scan_line.point_infos[i].curvature < threshold * 0.8) {
                break; // Not sharp enough
            }
            
            if (scan_line.point_infos[i].label != 0) {
                continue; // Already labeled
            }
            
            // Check if the point is far enough from already selected sharp points
            bool is_far_enough = true;
            if (selected_sharp > 0) {
                for (int j = 0; j < point_count; j++) {
                    if (scan_line.point_infos[j].label == 1) { // Check against already selected sharp points
                        float dist = (scan_line.point_infos[i].raw_point - scan_line.point_infos[j].raw_point).norm();
                        if (dist < feature_min_distance_) {
                            is_far_enough = false;
                            break;
                        }
                    }
                }
            }
            
            // Select as sharp if far enough and we need more sharp points
            if (is_far_enough && selected_sharp < num_sharp) {
                scan_line.point_infos[i].label = 1; // Mark as sharp
                selected_sharp++;
                
                // Also add to clouds
                int idx = scan_line.point_infos[i].index;
                pcl::PointXYZI point = curr_cloud_raw_->points[idx];
                edge_points_sharp_->push_back(point);
                edge_points_less_sharp_->push_back(point);
            } 
            // Select as less-sharp if we need more less-sharp points
            else if (selected_less_sharp < num_less_sharp) {
                scan_line.point_infos[i].label = 2; // Mark as less-sharp
                selected_less_sharp++;
                
                // Add to less-sharp cloud
                int idx = scan_line.point_infos[i].index;
                pcl::PointXYZI point = curr_cloud_raw_->points[idx];
                edge_points_less_sharp_->push_back(point);
            }
        }
    }
    
    void extractFlatPointsFromLine(ScanLine& scan_line, int num_flat, int num_less_flat, float threshold) {
        int point_count = scan_line.point_infos.size();
        if (point_count < 10) return;
        
        int selected_flat = 0;
        int selected_less_flat = 0;
        
        // Select flat points from lowest curvature
        for (int i = 0; i < point_count && (selected_flat < num_flat || selected_less_flat < num_less_flat); i++) {
            if (scan_line.point_infos[i].curvature > threshold * 1.5) {
                break; // Not flat enough
            }
            
            if (scan_line.point_infos[i].label != 0) {
                continue; // Already labeled
            }
            
            // Check distance from existing flat points (similar to sharp points)
            bool is_far_enough = true;
            if (selected_flat > 0) {
                for (int j = 0; j < point_count; j++) {
                    if (scan_line.point_infos[j].label == 3) { // Check against already selected flat points
                        float dist = (scan_line.point_infos[i].raw_point - scan_line.point_infos[j].raw_point).norm();
                        if (dist < feature_min_distance_ * 2.0) { // Flat features can be closer
                            is_far_enough = false;
                            break;
                        }
                    }
                }
            }
            
            // Select as flat if far enough and we need more flat points
            if (is_far_enough && selected_flat < num_flat) {
                scan_line.point_infos[i].label = 3; // Mark as flat
                selected_flat++;
                
                // Add to clouds
                int idx = scan_line.point_infos[i].index;
                pcl::PointXYZI point = curr_cloud_raw_->points[idx];
                surf_points_flat_->push_back(point);
                surf_points_less_flat_->push_back(point);
            } 
            // Select as less-flat if we need more less-flat points
            else if (selected_less_flat < num_less_flat) {
                scan_line.point_infos[i].label = 4; // Mark as less-flat
                selected_less_flat++;
                
                // Add to less-flat cloud
                int idx = scan_line.point_infos[i].index;
                pcl::PointXYZI point = curr_cloud_raw_->points[idx];
                surf_points_less_flat_->push_back(point);
            }
        }
    }
    
    void organizePointCloudByRing(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                std::vector<ScanLine>& scan_lines,
                                const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                int ring_field_idx) {
        // Initialize scan lines
        scan_lines.resize(scan_line_);
        for (int i = 0; i < scan_line_; i++) {
            scan_lines[i].point_infos.clear();
            scan_lines[i].start_orientation = 0;
            scan_lines[i].end_orientation = 0;
        }
        
        // Extract ring information from point cloud data
        for (size_t i = 0; i < cloud->size(); i++) {
            const auto& point = cloud->points[i];
            
            // Skip invalid points
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range < min_scan_range_ || range > max_range_) {
                continue;
            }
            
            // Extract ring value from cloud_msg
            uint8_t ring = 0;
            memcpy(&ring, &cloud_msg->data[i * cloud_msg->point_step + cloud_msg->fields[ring_field_idx].offset], sizeof(uint8_t));
            
            // Make sure ring is within valid range
            if (ring >= scan_line_) {
                continue;
            }
            
            // Calculate horizontal orientation for ordering
            float ori = std::atan2(point.y, point.x) * 180.0 / M_PI;
            
            // Add point info to corresponding scan line
            PointInfo point_info;
            point_info.index = i;
            point_info.curvature = 0; // Will be calculated later
            point_info.label = 0;
            point_info.raw_point = Eigen::Vector3f(point.x, point.y, point.z);
            point_info.point = point_info.raw_point;
            
            // Track orientation range for each scan line
            if (scan_lines[ring].point_infos.empty()) {
                scan_lines[ring].start_orientation = ori;
            }
            scan_lines[ring].end_orientation = ori;
            
            scan_lines[ring].point_infos.push_back(point_info);
        }
    }
    
    void organizePointCloudByScanLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                     std::vector<ScanLine>& scan_lines) {
        // Initialize scan lines
        scan_lines.resize(scan_line_);
        for (int i = 0; i < scan_line_; i++) {
            scan_lines[i].point_infos.clear();
            scan_lines[i].start_orientation = 0;
            scan_lines[i].end_orientation = 0;
        }
        
        // For each point, determine which scan line it belongs to
        for (size_t i = 0; i < cloud->size(); i++) {
            const auto& point = cloud->points[i];
            
            // Skip invalid points
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range < min_scan_range_ || range > max_range_) {
                continue;
            }
            
            // Calculate vertical angle
            float vert_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI;
            
            // Calculate horizontal orientation
            float ori = std::atan2(point.y, point.x) * 180.0 / M_PI;
            
            // Map vertical angle to scan line index for HDL-32E
            // HDL-32E has vertical FOV from around -30.67° to +10.67°
            int scan_id = static_cast<int>((vert_angle + 30.67) / 41.34 * scan_line_);
            if (scan_id < 0 || scan_id >= scan_line_) {
                continue;
            }
            
            // Add point info to corresponding scan line
            PointInfo point_info;
            point_info.index = i;
            point_info.curvature = 0; // Will be calculated later
            point_info.label = 0;
            point_info.raw_point = Eigen::Vector3f(point.x, point.y, point.z);
            point_info.point = point_info.raw_point;
            
            // Track orientation range for each scan line
            if (scan_lines[scan_id].point_infos.empty()) {
                scan_lines[scan_id].start_orientation = ori;
            }
            scan_lines[scan_id].end_orientation = ori;
            
            scan_lines[scan_id].point_infos.push_back(point_info);
        }
    }
    
    void organizeByScanAngles(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                            std::vector<ScanLine>& scan_lines) {
        // This method organizes points based on vertical angles when ring information is not available
        
        // First pass: compute min/max vertical angle
        float min_vert_angle = M_PI;
        float max_vert_angle = -M_PI;
        
        for (const auto& point : cloud->points) {
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range < min_scan_range_ || range > max_range_) {
                continue;
            }
            
            float vert_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));
            min_vert_angle = std::min(min_vert_angle, vert_angle);
            max_vert_angle = std::max(max_vert_angle, vert_angle);
        }
        
        // For HDL-32E, vertical FOV is approximately -30.67° to +10.67° in radians
        float min_expected_angle = -30.67 * M_PI / 180.0;
        float max_expected_angle = 10.67 * M_PI / 180.0;
        
        // Use expected range if detected range is too small
        if (max_vert_angle - min_vert_angle < 0.5) {
            min_vert_angle = min_expected_angle;
            max_vert_angle = max_expected_angle;
        }
        
        float vert_angle_range = max_vert_angle - min_vert_angle;
        float vert_angle_step = vert_angle_range / scan_line_;
        
        // Initialize scan lines
        scan_lines.resize(scan_line_);
        for (int i = 0; i < scan_line_; i++) {
            scan_lines[i].point_infos.clear();
            scan_lines[i].start_orientation = 0;
            scan_lines[i].end_orientation = 0;
        }
        
        // Second pass: assign points to scan lines
        for (size_t i = 0; i < cloud->size(); i++) {
            const auto& point = cloud->points[i];
            
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range < min_scan_range_ || range > max_range_) {
                continue;
            }
            
            float vert_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y));
            int scan_id = static_cast<int>((vert_angle - min_vert_angle) / vert_angle_step);
            scan_id = std::min(std::max(scan_id, 0), scan_line_ - 1);
            
            float ori = std::atan2(point.y, point.x) * 180.0 / M_PI;
            
            PointInfo point_info;
            point_info.index = i;
            point_info.curvature = 0;
            point_info.label = 0;
            point_info.raw_point = Eigen::Vector3f(point.x, point.y, point.z);
            point_info.point = point_info.raw_point;
            
            // Track orientation range for each scan line
            if (scan_lines[scan_id].point_infos.empty()) {
                scan_lines[scan_id].start_orientation = ori;
            }
            scan_lines[scan_id].end_orientation = ori;
            
            scan_lines[scan_id].point_infos.push_back(point_info);
        }
    }
    
    void downsampleFeatures() {
        // Only downsample if we have too many features
        if (edge_points_less_sharp_->size() > 2000) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr edge_less_sharp_ds(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> edge_downsize_filter;
            edge_downsize_filter.setLeafSize(0.2, 0.2, 0.2);  // A-LOAM uses larger voxels for efficiency
            edge_downsize_filter.setInputCloud(edge_points_less_sharp_);
            edge_downsize_filter.filter(*edge_less_sharp_ds);
            edge_points_less_sharp_ = edge_less_sharp_ds;
        }
        
        if (surf_points_less_flat_->size() > 4000) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr surf_less_flat_ds(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::VoxelGrid<pcl::PointXYZI> surf_downsize_filter;
            surf_downsize_filter.setLeafSize(0.4, 0.4, 0.4);  // A-LOAM uses larger voxels for surfaces
            surf_downsize_filter.setInputCloud(surf_points_less_flat_);
            surf_downsize_filter.filter(*surf_less_flat_ds);
            surf_points_less_flat_ = surf_less_flat_ds;
        }
    }
    
    // A-LOAM style optimization
    void optimizeOdometry(const Eigen::Quaterniond& q_init, const Eigen::Vector3d& t_init) {
        // Initialize pose with initial guess
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            q_w_curr_ = q_init;
            t_w_curr_ = t_init;
        }
        
        // Multiple iterations of Gauss-Newton optimization
        int valid_iterations = 0;
        for (int iter = 0; iter < optimization_iterations_; iter++) {
            // Transform feature points to world frame for matching
            pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points_world(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_world(new pcl::PointCloud<pcl::PointXYZI>());
            
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                transformPointCloud(edge_points_sharp_, edge_points_world, q_w_curr_, t_w_curr_);
                transformPointCloud(surf_points_flat_, surf_points_world, q_w_curr_, t_w_curr_);
            }
            
            // Setup system matrices
            Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            
            // Find edge and surface correspondences and accumulate into system
            int edge_factors = 0;
            int surf_factors = 0;
            
            if (use_aloam_factors_) {
                edge_factors = findEdgeFactorsALOAM(edge_points_world, A, b);
                surf_factors = findSurfFactorsALOAM(surf_points_world, A, b);
            } else {
                edge_factors = findEdgeFeatureCorrespondences(edge_points_world, A, b);
                surf_factors = findSurfFeatureCorrespondences(surf_points_world, A, b);
            }
            
            int total_factors = edge_factors + surf_factors;
            
            // Skip iteration if not enough correspondences
            if (total_factors < 50) {
                ROS_WARN("Not enough correspondences: edge=%d, surf=%d", edge_factors, surf_factors);
                continue;
            }
            
            valid_iterations++;
            
            // Add regularization - Levenberg-Marquardt style
            for (int i = 0; i < 6; i++) {
                A(i, i) += system_noise_ * 1000;  // Higher regularization for stability
            }
            
            // Solve system Ax = b 
            Eigen::Matrix<double, 6, 1> dx = A.ldlt().solve(-b);
            
            // Check for convergence or invalid solution
            if (!std::isfinite(dx.sum()) || !std::isfinite(dx.norm())) {
                ROS_WARN("Invalid optimization solution");
                continue;
            }
            
            // Update pose
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                
                // Update translation
                t_w_curr_[0] += dx(0);
                t_w_curr_[1] += dx(1);
                t_w_curr_[2] += dx(2);
                
                // Update rotation using axis-angle
                double angle = std::sqrt(dx(3)*dx(3) + dx(4)*dx(4) + dx(5)*dx(5));
                Eigen::Vector3d axis;
                
                if (angle < 1e-10) {
                    axis = Eigen::Vector3d(1, 0, 0); // Default axis if angle is too small
                } else {
                    axis = Eigen::Vector3d(dx(3), dx(4), dx(5)) / angle;
                }
                
                Eigen::AngleAxisd rot(angle, axis);
                q_w_curr_ = q_w_curr_ * Eigen::Quaterniond(rot);
                q_w_curr_.normalize();
            }
            
            // Report progress less frequently
            if (iter % 4 == 0) {
                double delta_norm = dx.norm();
                
                {
                    std::lock_guard<std::mutex> lock(pose_mutex_);
                    ROS_INFO("Opt iter %d: pos=[%.3f, %.3f, %.3f], delta=%.6f, factors: edge=%d, surf=%d", 
                            iter, t_w_curr_[0], t_w_curr_[1], t_w_curr_[2], 
                            delta_norm, edge_factors, surf_factors);
                }
                
                // Check for convergence
                if (delta_norm < 1e-6) {
                    ROS_INFO("Optimization converged at iteration %d", iter);
                    break;
                }
            }
        }
        
        // If no valid iterations, keep the initial guess
        if (valid_iterations == 0) {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            q_w_curr_ = q_init;
            t_w_curr_ = t_init;
            ROS_WARN("No valid optimization iterations, keeping initial guess");
        }
    }
    
    // A-LOAM style edge factors
    int findEdgeFactorsALOAM(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_points,
                           Eigen::Matrix<double, 6, 6>& A, 
                           Eigen::Matrix<double, 6, 1>& b) {
        int num_factors = 0;
        
        for (const auto& point : edge_points->points) {
            std::vector<int> point_search_idx;
            std::vector<float> point_search_sq_dist;
            
            // Find closest points in the map - ALOAM uses 5
            kdtree_edge_map_->nearestKSearch(point, 5, point_search_idx, point_search_sq_dist);
            
            if (point_search_idx.size() < 5) continue;
            
            if (point_search_sq_dist[4] < 0.01) continue; // Too close
            
            // Calculate centroid of the five points
            Eigen::Vector3d centroid(0, 0, 0);
            for (int i = 0; i < 5; i++) {
                centroid += Eigen::Vector3d(
                    edge_points_map_local_->points[point_search_idx[i]].x,
                    edge_points_map_local_->points[point_search_idx[i]].y,
                    edge_points_map_local_->points[point_search_idx[i]].z
                );
            }
            centroid /= 5.0;
            
            // Calculate covariance
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            
            for (int i = 0; i < 5; i++) {
                Eigen::Vector3d point_i(
                    edge_points_map_local_->points[point_search_idx[i]].x,
                    edge_points_map_local_->points[point_search_idx[i]].y,
                    edge_points_map_local_->points[point_search_idx[i]].z
                );
                
                Eigen::Vector3d zero_mean = point_i - centroid;
                covariance += zero_mean * zero_mean.transpose();
            }
            
            // Compute eigenvectors
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
            
            if (eigen_solver.eigenvalues()[2] < 3 * eigen_solver.eigenvalues()[0]) {
                continue; // Not a good line correspondence if eigenvalues are too similar
            }
            
            // Get the dominant direction (eigenvector of largest eigenvalue)
            Eigen::Vector3d line_direction = eigen_solver.eigenvectors().col(2);
            
            // Current point
            Eigen::Vector3d curr_point(point.x, point.y, point.z);
            
            // Project the current point onto the line
            Eigen::Vector3d projection = centroid + line_direction * line_direction.dot(curr_point - centroid);
            
            // Calculate point-to-line distance vector
            Eigen::Vector3d distance_vector = curr_point - projection;
            
            // Skip if the distance is too large
            if (distance_vector.norm() > 1.0) continue;
            
            // Calculate Jacobian matrix
            Eigen::Matrix<double, 3, 6> jacobian;
            
            // Translation part of Jacobian
            jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            
            // Rotation part - cross product of point and distance vector
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                Eigen::Vector3d point_transformed = q_w_curr_.toRotationMatrix() * curr_point;
                jacobian.block<3, 3>(0, 3) = -skewSymmetric(point_transformed);
            }
            
            // Weight for robust Gauss-Newton (Huber loss)
            double weight = 1.0;
            double dist = distance_vector.norm();
            if (dist > 0.1) {
                weight = 0.1 / dist;
            }
            
            // Update Hessian and gradient
            Eigen::Vector3d dist_unit = distance_vector.normalized();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    A(i, j) += weight * jacobian.col(i).dot(dist_unit) * jacobian.col(j).dot(dist_unit);
                }
                b(i) += weight * jacobian.col(i).dot(dist_unit) * dist;
            }
            
            num_factors++;
        }
        
        return num_factors;
    }
    
    // A-LOAM style surf factors
    int findSurfFactorsALOAM(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_points,
                           Eigen::Matrix<double, 6, 6>& A, 
                           Eigen::Matrix<double, 6, 1>& b) {
        int num_factors = 0;
        
        for (const auto& point : surf_points->points) {
            std::vector<int> point_search_idx;
            std::vector<float> point_search_sq_dist;
            
            kdtree_surf_map_->nearestKSearch(point, 5, point_search_idx, point_search_sq_dist);
            
            if (point_search_idx.size() < 5) continue;
            
            if (point_search_sq_dist[4] < 0.01) continue; // Too close
            
            // Calculate plane parameters using PCA
            Eigen::Matrix<double, 5, 3> matA0;
            
            // Fill matrix with points
            for (int i = 0; i < 5; i++) {
                matA0(i, 0) = surf_points_map_local_->points[point_search_idx[i]].x;
                matA0(i, 1) = surf_points_map_local_->points[point_search_idx[i]].y;
                matA0(i, 2) = surf_points_map_local_->points[point_search_idx[i]].z;
            }
            
            // Compute centroid
            Eigen::Vector3d centroid(0, 0, 0);
            for (int i = 0; i < 5; i++) {
                centroid += Eigen::Vector3d(matA0(i, 0), matA0(i, 1), matA0(i, 2));
            }
            centroid /= 5.0;
            
            // Compute covariance
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (int i = 0; i < 5; i++) {
                Eigen::Vector3d point_i(matA0(i, 0), matA0(i, 1), matA0(i, 2));
                Eigen::Vector3d zero_mean = point_i - centroid;
                covariance += zero_mean * zero_mean.transpose();
            }
            
            // Compute eigenvectors
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
            
            // Check if the smallest eigenvalue is small enough (indicates a plane)
            if (eigen_solver.eigenvalues()[0] > 0.02 * eigen_solver.eigenvalues()[2]) {
                continue; // Not flat enough
            }
            
            // Get the normal (eigenvector of smallest eigenvalue)
            Eigen::Vector3d normal = eigen_solver.eigenvectors().col(0);
            
            // Ensure normal points outward (toward origin)
            if (normal.dot(centroid) < 0) normal = -normal;
            
            // d component of plane equation ax + by + cz + d = 0
            double d = -normal.dot(centroid);
            
            // Current point
            Eigen::Vector3d curr_point(point.x, point.y, point.z);
            
            // Calculate signed point-to-plane distance
            double dist = normal.dot(curr_point) + d;
            
            // Skip if the distance is too large
            if (std::abs(dist) > 1.0) continue;
            
            // Calculate Jacobian
            Eigen::Matrix<double, 1, 6> jacobian;
            
            // Translation part of Jacobian
            jacobian.block<1, 3>(0, 0) = normal.transpose();
            
            // Rotation part
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                Eigen::Vector3d point_transformed = q_w_curr_.toRotationMatrix() * curr_point;
                jacobian.block<1, 3>(0, 3) = (-skewSymmetric(point_transformed) * normal).transpose();
            }
            
            // Weight for robust Gauss-Newton (Huber loss)
            double weight = 1.0;
            if (std::abs(dist) > 0.1) {
                weight = 0.1 / std::abs(dist);
            }
            
            // Update Hessian and gradient
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    A(i, j) += weight * jacobian(0, i) * jacobian(0, j);
                }
                b(i) += weight * jacobian(0, i) * dist;
            }
            
            num_factors++;
        }
        
        return num_factors;
    }
    
    // Traditional correspondence finders (from original code)
    int findEdgeFeatureCorrespondences(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_points,
                                     Eigen::Matrix<double, 6, 6>& A,
                                     Eigen::Matrix<double, 6, 1>& b) {
        int num_factors = 0;
        
        for (const auto& point : edge_points->points) {
            std::vector<int> point_search_idx;
            std::vector<float> point_search_sq_dist;
            
            // Find closest points in the map
            kdtree_edge_map_->nearestKSearch(point, 5, point_search_idx, point_search_sq_dist);
            
            if (point_search_idx.size() < 2 || point_search_sq_dist[1] > 0.4) {  // Stricter threshold for HDL-32E
                continue; // Need at least 2 close points
            }
            
            // Line parameters
            Eigen::Vector3d p0(edge_points_map_local_->points[point_search_idx[0]].x,
                             edge_points_map_local_->points[point_search_idx[0]].y,
                             edge_points_map_local_->points[point_search_idx[0]].z);
            
            Eigen::Vector3d p1(edge_points_map_local_->points[point_search_idx[1]].x,
                             edge_points_map_local_->points[point_search_idx[1]].y,
                             edge_points_map_local_->points[point_search_idx[1]].z);
            
            Eigen::Vector3d p(point.x, point.y, point.z);
            
            // Line direction
            Eigen::Vector3d line_dir = (p1 - p0).normalized();
            
            // Point-to-line distance vector
            Eigen::Vector3d distance_vector = p - p0 - line_dir * line_dir.dot(p - p0);
            float distance = distance_vector.norm();
            
            // Reject outliers
            if (distance > 0.4) {  // Stricter threshold for HDL-32E
                continue;
            }
            
            // Normalize distance vector
            Eigen::Vector3d norm_distance = distance_vector.normalized();
            
            // Calculate Jacobian
            Eigen::Matrix<double, 3, 6> J;
            J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // Translation part
            
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                J.block<3, 3>(0, 3) = -skewSymmetric(q_w_curr_.toRotationMatrix() * Eigen::Vector3d(point.x, point.y, point.z));  // Rotation part
            }
            
            // Apply robust kernel to reduce influence of outliers - Huber or Cauchy-like
            double weight = 1.0;
            if (distance > 0.1) {
                weight = 0.1 / distance;  // Huber-like weight
            }
            
            // Update Hessian and gradient
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    A(i, j) += weight * J.col(i).dot(norm_distance) * J.col(j).dot(norm_distance);
                }
                b(i) += weight * J.col(i).dot(norm_distance) * distance;
            }
            
            num_factors++;
        }
        
        return num_factors;
    }
    
    int findSurfFeatureCorrespondences(const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_points,
                                     Eigen::Matrix<double, 6, 6>& A,
                                     Eigen::Matrix<double, 6, 1>& b) {
        int num_factors = 0;
        
        for (const auto& point : surf_points->points) {
            std::vector<int> point_search_idx;
            std::vector<float> point_search_sq_dist;
            
            // Find closest points in the map
            kdtree_surf_map_->nearestKSearch(point, 5, point_search_idx, point_search_sq_dist);
            
            if (point_search_idx.size() < 3 || point_search_sq_dist[2] > 0.8) {  // Adjusted for HDL-32E
                continue; // Need at least 3 close points
            }
            
            // Plane parameters
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            
            // Get 3 closest points
            for (int i = 0; i < 3; i++) {
                matA0(i, 0) = surf_points_map_local_->points[point_search_idx[i]].x;
                matA0(i, 1) = surf_points_map_local_->points[point_search_idx[i]].y;
                matA0(i, 2) = surf_points_map_local_->points[point_search_idx[i]].z;
            }
            
            // Calculate plane normal using SVD
            Eigen::Vector3d norm;
            Eigen::Vector3d centroid(0, 0, 0);
            
            for (int i = 0; i < 3; i++) {
                centroid += Eigen::Vector3d(matA0(i, 0), matA0(i, 1), matA0(i, 2));
            }
            centroid /= 3.0;
            
            // Center the points around their centroid
            Eigen::Matrix3d centered_mat;
            for (int i = 0; i < 3; i++) {
                centered_mat.row(i) = Eigen::Vector3d(matA0(i, 0), matA0(i, 1), matA0(i, 2)) - centroid;
            }
            
            // Use SVD to find the normal
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(centered_mat, Eigen::ComputeFullV);
            norm = svd.matrixV().col(2);
            
            // Make sure normal points outward
            if (norm.dot(centroid) < 0) {
                norm = -norm;
            }
            
            // Plane equation: ax + by + cz + d = 0
            double d = -norm.dot(centroid);
            
            // Calculate point-to-plane distance
            double distance = std::abs(norm.dot(Eigen::Vector3d(point.x, point.y, point.z)) + d);
            
            // Reject outliers
            if (distance > 0.2) {
                continue;
            }
            
            // Calculate Jacobian
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = norm.transpose();  // Translation part
            
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                J.block<1, 3>(0, 3) = (-skewSymmetric(q_w_curr_.toRotationMatrix() * Eigen::Vector3d(point.x, point.y, point.z)) * norm).transpose();  // Rotation part
            }
            
            // Apply robust kernel - Huber or Cauchy-like
            double weight = 1.0;
            if (distance > 0.05) {
                weight = 0.05 / distance;  // Huber-like weight
            }
            
            // Update Hessian and gradient
            for (int i = 0; i < 6; i++) {
                for (int j = i; j < 6; j++) {
                    A(i, j) += weight * J(0, i) * J(0, j);
                    if (i != j) {
                        A(j, i) = A(i, j);  // Symmetry
                    }
                }
                b(i) += weight * J(0, i) * distance;
            }
            
            num_factors++;
        }
        
        return num_factors;
    }
    
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }
    
    void transformPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out,
                           const Eigen::Quaterniond& q,
                           const Eigen::Vector3d& t) {
        // Initialize output cloud
        if (cloud_in != cloud_out) {
            cloud_out->header = cloud_in->header;
            cloud_out->points.resize(cloud_in->points.size());
        }
        
        Eigen::Matrix3d R = q.toRotationMatrix();
        
        // Transform each point
        for (size_t i = 0; i < cloud_in->points.size(); i++) {
            const auto& point_in = cloud_in->points[i];
            auto& point_out = cloud_out->points[i];
            
            // Apply rotation and translation
            Eigen::Vector3d point(point_in.x, point_in.y, point_in.z);
            Eigen::Vector3d point_transformed = R * point + t;
            
            // Store result
            point_out.x = point_transformed.x();
            point_out.y = point_transformed.y();
            point_out.z = point_transformed.z();
            point_out.intensity = point_in.intensity;
        }
    }
    
    bool isKeyframe() {
        // Check if current pose is sufficiently different from the last keyframe
        Eigen::Quaterniond q_delta = q_w_curr_ * last_keyframe_q_.inverse();
        Eigen::Vector3d t_delta = t_w_curr_ - last_keyframe_t_;
        
        // Compute angle from quaternion
        double angle = 2.0 * std::acos(std::min(1.0, std::abs(q_delta.w())));
        double dist = t_delta.norm();
        
        // Also make every Nth frame a keyframe to ensure regular updates
        bool time_keyframe = (frame_count_ % keyframe_time_interval_) == 0;
        
        // Return true if either rotation or translation exceeds threshold, or it's a time-based keyframe
        if (angle > keyframe_angle_threshold_ || dist > keyframe_distance_threshold_ || time_keyframe) {
            ROS_INFO("New keyframe: angle=%.2f, dist=%.2f, frame=%d", angle, dist, frame_count_);
            return true;
        }
        return false;
    }
    
    void updateLocalMap() {
        // Create new clouds for map update
        pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud_keyframe(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_cloud_keyframe(new pcl::PointCloud<pcl::PointXYZI>());
        
        // Transform current features to world frame
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            transformPointCloud(edge_points_less_sharp_, edge_cloud_keyframe, q_w_curr_, t_w_curr_);
            transformPointCloud(surf_points_less_flat_, surf_cloud_keyframe, q_w_curr_, t_w_curr_);
        }
        
        // Add to local map
        *edge_points_map_local_ += *edge_cloud_keyframe;
        *surf_points_map_local_ += *surf_cloud_keyframe;
        
        // Downsample local map
        pcl::VoxelGrid<pcl::PointXYZI> edge_downsize_filter;
        edge_downsize_filter.setLeafSize(map_resolution_ * 0.75, map_resolution_ * 0.75, map_resolution_ * 0.75);
        edge_downsize_filter.setInputCloud(edge_points_map_local_);
        pcl::PointCloud<pcl::PointXYZI>::Ptr edge_map_ds(new pcl::PointCloud<pcl::PointXYZI>());
        edge_downsize_filter.filter(*edge_map_ds);
        
        pcl::VoxelGrid<pcl::PointXYZI> surf_downsize_filter;
        surf_downsize_filter.setLeafSize(map_resolution_ * 1.5, map_resolution_ * 1.5, map_resolution_ * 1.5);
        surf_downsize_filter.setInputCloud(surf_points_map_local_);
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_map_ds(new pcl::PointCloud<pcl::PointXYZI>());
        surf_downsize_filter.filter(*surf_map_ds);
        
        // Update local map
        edge_points_map_local_ = edge_map_ds;
        surf_points_map_local_ = surf_map_ds;
        
        // Update KD-trees
        kdtree_edge_map_->setInputCloud(edge_points_map_local_);
        kdtree_surf_map_->setInputCloud(surf_points_map_local_);
        
        // Add to map update queue for global map
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            edge_map_update_queue_.push(edge_cloud_keyframe);
            surf_map_update_queue_.push(surf_cloud_keyframe);
        }
        
        // Publish local map for visualization
        publishLocalMap();
    }
    
    void publishLocalMap() {
        if (pub_local_map_.getNumSubscribers() > 0) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZI>());
            *local_map = *edge_points_map_local_;
            *local_map += *surf_points_map_local_;
            
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*local_map, map_msg);
            map_msg.header.frame_id = map_frame_;
            map_msg.header.stamp = ros::Time::now();
            pub_local_map_.publish(map_msg);
        }
    }
    
    void publishFeatureClouds() {
        if (pub_edge_points_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 edge_msg;
            pcl::toROSMsg(*edge_points_less_sharp_, edge_msg);
            edge_msg.header.frame_id = lidar_frame_;
            edge_msg.header.stamp = ros::Time::now();
            pub_edge_points_.publish(edge_msg);
        }
        
        if (pub_surf_points_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 surf_msg;
            pcl::toROSMsg(*surf_points_less_flat_, surf_msg);
            surf_msg.header.frame_id = lidar_frame_;
            surf_msg.header.stamp = ros::Time::now();
            pub_surf_points_.publish(surf_msg);
        }
    }
    
    void publishResults(const ros::Time& timestamp) {
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            q = q_w_curr_;
            t = t_w_curr_;
        }
        
        // Publish odometry
        nav_msgs::Odometry odom;
        odom.header.frame_id = map_frame_;
        odom.child_frame_id = lidar_frame_;
        odom.header.stamp = timestamp;
        
        // Set pose
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.position.x = t.x();
        odom.pose.pose.position.y = t.y();
        odom.pose.pose.position.z = t.z();
        
        // Set covariance
        for (int i = 0; i < 36; i++) {
            odom.pose.covariance[i] = 0;
            odom.twist.covariance[i] = 0;
        }
        
        // Diagonal elements of covariance
        odom.pose.covariance[0] = 1e-3;
        odom.pose.covariance[7] = 1e-3;
        odom.pose.covariance[14] = 1e-3;
        odom.pose.covariance[21] = 1e-4;
        odom.pose.covariance[28] = 1e-4;
        odom.pose.covariance[35] = 1e-4;
        
        pub_odom_.publish(odom);
        
        // Publish path
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odom.header;
        pose_stamped.pose = odom.pose.pose;
        
        path_.poses.push_back(pose_stamped);
        path_.header = odom.header;
        pub_path_.publish(path_);
        
        // Publish TF
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
        tf::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
        transform.setRotation(tf_q);
        tf_broadcaster_->sendTransform(tf::StampedTransform(transform, timestamp, map_frame_, lidar_frame_));
        
        // Log pose
        ROS_INFO("Current pose: [%.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]",
                t.x(), t.y(), t.z(),
                q.w(), q.x(), q.y(), q.z());
    }
    
    void saveTrajectory(const ros::Time& timestamp) {
        if (!trajectory_file_.is_open()) {
            return;
        }
        
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            q = q_w_curr_;
            t = t_w_curr_;
        }
        
        // Format: timestamp tx ty tz qx qy qz qw
        trajectory_file_ << std::fixed << std::setprecision(6)
                       << timestamp.toSec() << " "
                       << t.x() << " " << t.y() << " " << t.z() << " "
                       << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                       << std::endl;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "loam_mapping_node");
    ros::NodeHandle nh("~");
    
    ROS_INFO("Starting TASLO node optimized for HDL-32E with A-LOAM/FLOAM approaches...");
    
    TASLO taslo(nh);
    
    ros::spin();
    
    return 0;
}