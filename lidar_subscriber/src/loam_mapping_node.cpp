/**
 * ROS Node for FAST-LOAM: Direct Scan-to-Map LiDAR Odometry and Mapping
 * Optimized for 32-channel Velodyne LiDAR (HDL-32E, VLP-32C)
 * With improved accuracy and pose output
 */

 #include <ros/ros.h>
 #include <sensor_msgs/PointCloud2.h>
 #include <nav_msgs/Odometry.h>
 #include <nav_msgs/Path.h>
 #include <tf2_ros/transform_broadcaster.h>
 #include <tf2_ros/transform_listener.h>
 #include <tf2_eigen/tf2_eigen.h>
 #include <geometry_msgs/TransformStamped.h>
 #include <pcl_conversions/pcl_conversions.h>
 #include <pcl/point_cloud.h>
 #include <pcl/point_types.h>
 #include <pcl/filters/voxel_grid.h>
 #include <pcl/kdtree/kdtree_flann.h>
 #include <pcl/common/transforms.h>
 #include <pcl/filters/extract_indices.h>
 #include <pcl/filters/approximate_voxel_grid.h>
 #include <pcl/sample_consensus/method_types.h>
 #include <pcl/sample_consensus/model_types.h>
 #include <pcl/segmentation/sac_segmentation.h>
 #include <pcl/filters/crop_box.h>
 #include <pcl/filters/radius_outlier_removal.h>
 #include <pcl/filters/statistical_outlier_removal.h>
 #include <pcl/io/pcd_io.h>
 #include <Eigen/Dense>
 #include <ceres/ceres.h>
 #include <mutex>
 #include <thread>
 #include <atomic>
 #include <deque>
 #include <vector>
 #include <cmath>
 #include <chrono>
 #include <string>
 #include <iostream>
 #include <fstream>
 #include <ctime>
 #include <unordered_map>
 #include <memory>
 #include <algorithm>
 #include <iomanip>
 
 // Define custom point types for Velodyne 32 LiDAR
 struct PointXYZIR {
     PCL_ADD_POINT4D;
     float intensity;
     uint16_t ring;
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 } EIGEN_ALIGN16;
 
 POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
     (float, x, x) (float, y, y) (float, z, z) 
     (float, intensity, intensity) (uint16_t, ring, ring)
 )
 
 // More comprehensive custom point type for Velodyne 32
 struct VelodynePoint {
     PCL_ADD_POINT4D;
     float intensity;
     uint16_t ring;
     float time;
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 } EIGEN_ALIGN16;
 
 POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePoint,
     (float, x, x) (float, y, y) (float, z, z) 
     (float, intensity, intensity) (uint16_t, ring, ring) (float, time, time)
 )
 
 // Default point type for backward compatibility
 typedef pcl::PointXYZI PointType;
 
 // Improved LidarEdgeFactor with enhanced numerical stability
 class LidarEdgeFactor {
 private:
     Eigen::Vector3d curr_point_;
     Eigen::Vector3d last_point_a_;
     Eigen::Vector3d last_point_b_;
     double s_;
 
 public:
     LidarEdgeFactor(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& last_point_a,
                     const Eigen::Vector3d& last_point_b, double s = 1.0)
         : curr_point_(curr_point), last_point_a_(last_point_a), last_point_b_(last_point_b), s_(s) {}
 
     template <typename T>
     bool operator()(const T* q, const T* t, T* residual) const {
         Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
         Eigen::Matrix<T, 3, 1> lpa{T(last_point_a_.x()), T(last_point_a_.y()), T(last_point_a_.z())};
         Eigen::Matrix<T, 3, 1> lpb{T(last_point_b_.x()), T(last_point_b_.y()), T(last_point_b_.z())};
 
         // Ensure normalized quaternion
         T qnorm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
         Eigen::Quaternion<T> q_eigen(q[3]/qnorm, q[0]/qnorm, q[1]/qnorm, q[2]/qnorm);
         Eigen::Matrix<T, 3, 3> R = q_eigen.toRotationMatrix();
         Eigen::Matrix<T, 3, 1> t_eigen(t[0], t[1], t[2]);
 
         // Transform current point to map frame
         Eigen::Matrix<T, 3, 1> cp_transformed = R * cp + t_eigen;
 
         // Compute line direction and length
         Eigen::Matrix<T, 3, 1> line_direction = lpb - lpa;
         T line_length_sq = line_direction.squaredNorm();
         
         // Check for degenerate line (close to zero length)
         if (line_length_sq < T(1e-8)) {
             // Fall back to point-to-point distance
             residual[0] = T(s_) * (cp_transformed - lpa).norm();
             return true;
         }
         
         // Project point onto line
         Eigen::Matrix<T, 3, 1> ap = cp_transformed - lpa;
         T t_proj = ap.dot(line_direction) / line_length_sq;
         t_proj = t_proj < T(0) ? T(0) : (t_proj > T(1) ? T(1) : t_proj);
         
         // Find closest point on line segment
         Eigen::Matrix<T, 3, 1> closest_point = lpa + t_proj * line_direction;
         
         // Calculate distance (more numerically stable)
         T distance = (cp_transformed - closest_point).norm();
         
         // Apply robust weighting based on distance
         T scaled_distance = distance / T(0.1);  // Scale by 10cm
         if (scaled_distance > T(5.0)) {  // If more than 50cm, cap the influence
             distance = T(0.5);
         }
 
         // Compute residual with weight
         residual[0] = T(s_) * distance;
 
         return true;
     }
 
     static ceres::CostFunction* Create(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& last_point_a,
                                        const Eigen::Vector3d& last_point_b, double s = 1.0) {
         return new ceres::AutoDiffCostFunction<LidarEdgeFactor, 1, 4, 3>(
             new LidarEdgeFactor(curr_point, last_point_a, last_point_b, s));
     }
 };
 
 // Improved LidarPlaneFactor with enhanced numerical stability
 class LidarPlaneFactor {
 private:
     Eigen::Vector3d curr_point_;
     Eigen::Vector3d plane_unit_normal_;
     double negative_plane_d_;
     double s_;
 
 public:
     LidarPlaneFactor(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& plane_unit_normal,
                     double negative_plane_d, double s = 1.0)
         : curr_point_(curr_point), plane_unit_normal_(plane_unit_normal), 
           negative_plane_d_(negative_plane_d), s_(s) {}
 
     template <typename T>
     bool operator()(const T* q, const T* t, T* residual) const {
         Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
         Eigen::Matrix<T, 3, 1> plane_normal{T(plane_unit_normal_.x()), 
                                           T(plane_unit_normal_.y()), 
                                           T(plane_unit_normal_.z())};
         T nd = T(negative_plane_d_);
 
         // Ensure normalized quaternion
         T qnorm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
         Eigen::Quaternion<T> q_eigen(q[3]/qnorm, q[0]/qnorm, q[1]/qnorm, q[2]/qnorm);
         Eigen::Matrix<T, 3, 3> R = q_eigen.toRotationMatrix();
         Eigen::Matrix<T, 3, 1> t_eigen(t[0], t[1], t[2]);
 
         // Transform current point to map frame
         Eigen::Matrix<T, 3, 1> cp_transformed = R * cp + t_eigen;
 
         // Compute point to plane distance (numerically stable)
         T distance = ceres::abs(plane_normal.dot(cp_transformed) + nd);
         
         // Apply robust weighting based on distance
         T scaled_distance = distance / T(0.05);  // Scale by 5cm
         if (scaled_distance > T(5.0)) {  // If more than 25cm, cap the influence
             distance = T(0.25);
         }
 
         // Compute residual with weight
         residual[0] = T(s_) * distance;
 
         return true;
     }
 
     static ceres::CostFunction* Create(const Eigen::Vector3d& curr_point, 
                                      const Eigen::Vector3d& plane_unit_normal,
                                      double negative_plane_d, double s = 1.0) {
         return new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
             new LidarPlaneFactor(curr_point, plane_unit_normal, negative_plane_d, s));
     }
 };
 
 // Efficient ring buffer for point cloud management
 template <typename PointT>
 class CloudRingBuffer {
 private:
     std::vector<typename pcl::PointCloud<PointT>::Ptr> clouds_;
     size_t capacity_;
     size_t current_idx_;
     size_t size_;
     
     pcl::VoxelGrid<PointT> voxel_filter_;
     
     // Combined cloud stored in memory for efficient access
     typename pcl::PointCloud<PointT>::Ptr combined_cloud_;
     bool combined_cloud_updated_;
     
 public:
     // Default constructor
     CloudRingBuffer() : capacity_(0), current_idx_(0), size_(0), combined_cloud_updated_(false) {
         combined_cloud_.reset(new pcl::PointCloud<PointT>());
     }
     
     CloudRingBuffer(size_t capacity, float leaf_size) 
         : capacity_(capacity), current_idx_(0), size_(0), combined_cloud_updated_(false) {
         clouds_.resize(capacity);
         for (size_t i = 0; i < capacity; ++i) {
             clouds_[i].reset(new pcl::PointCloud<PointT>());
         }
         
         combined_cloud_.reset(new pcl::PointCloud<PointT>());
         
         voxel_filter_.setLeafSize(leaf_size, leaf_size, leaf_size);
     }
     
     void push(const typename pcl::PointCloud<PointT>::Ptr& cloud) {
         // Replace the oldest cloud with the new one
         *clouds_[current_idx_] = *cloud;
         
         // Update indices
         current_idx_ = (current_idx_ + 1) % capacity_;
         if (size_ < capacity_) {
             size_++;
         }
         
         combined_cloud_updated_ = false;
     }
     
     typename pcl::PointCloud<PointT>::Ptr getCombinedCloud() {
         if (!combined_cloud_updated_) {
             // Regenerate the combined cloud
             combined_cloud_->clear();
             
             for (size_t i = 0; i < size_; ++i) {
                 *combined_cloud_ += *clouds_[(current_idx_ + capacity_ - i - 1) % capacity_];
             }
             
             // Downsample the combined cloud to maintain reasonable density
             if (combined_cloud_->size() > 20000) {  // Increased threshold for 32-channel
                 typename pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>());
                 voxel_filter_.setInputCloud(combined_cloud_);
                 voxel_filter_.filter(*filtered_cloud);
                 combined_cloud_ = filtered_cloud;
             }
             
             combined_cloud_updated_ = true;
         }
         
         return combined_cloud_;
     }
     
     size_t size() const {
         return size_;
     }
     
     bool empty() const {
         return size_ == 0;
     }
     
     void clear() {
         for (auto& cloud : clouds_) {
             cloud->clear();
         }
         combined_cloud_->clear();
         current_idx_ = 0;
         size_ = 0;
         combined_cloud_updated_ = true;
     }
 };
 
 // Enhanced motion predictor with robust dynamics modeling for 32-channel data
 class MotionPredictor {
 private:
     Eigen::Quaterniond last_q_;
     Eigen::Vector3d last_t_;
     Eigen::Quaterniond last_delta_q_;
     Eigen::Vector3d last_delta_t_;
     double last_timestamp_;
     bool has_previous_estimate_;
     
     // Motion statistics for adaptive filtering
     double avg_linear_velocity_;
     double avg_angular_velocity_;
     int sample_count_;
     
 public:
     MotionPredictor() : 
         last_q_(1, 0, 0, 0), 
         last_t_(0, 0, 0), 
         last_delta_q_(1, 0, 0, 0), 
         last_delta_t_(0, 0, 0), 
         last_timestamp_(0), 
         has_previous_estimate_(false),
         avg_linear_velocity_(0),
         avg_angular_velocity_(0),
         sample_count_(0) {}
     
     void reset() {
         has_previous_estimate_ = false;
         avg_linear_velocity_ = 0;
         avg_angular_velocity_ = 0;
         sample_count_ = 0;
     }
     
     // Update the predictor with a new pose estimate
     void update(const Eigen::Quaterniond& q, const Eigen::Vector3d& t, double timestamp) {
         if (has_previous_estimate_) {
             // Calculate time difference
             double dt = timestamp - last_timestamp_;
             if (dt <= 0) dt = 0.1;  // Safeguard
             
             // Calculate motion between last two poses
             last_delta_q_ = last_q_.inverse() * q;
             last_delta_t_ = t - last_t_;
             
             // Calculate velocities
             double linear_vel = last_delta_t_.norm() / dt;
             double angle = 2.0 * acos(std::min(1.0, std::abs(last_delta_q_.w())));
             double angular_vel = angle / dt;
             
             // Update running averages
             if (sample_count_ < 10) {
                 // During initial phase, simple averaging
                 avg_linear_velocity_ = (avg_linear_velocity_ * sample_count_ + linear_vel) / (sample_count_ + 1);
                 avg_angular_velocity_ = (avg_angular_velocity_ * sample_count_ + angular_vel) / (sample_count_ + 1);
                 sample_count_++;
             } else {
                 // After initialization, use exponential moving average
                 double alpha = 0.3;  // Weighting factor for new samples
                 avg_linear_velocity_ = (1 - alpha) * avg_linear_velocity_ + alpha * linear_vel;
                 avg_angular_velocity_ = (1 - alpha) * avg_angular_velocity_ + alpha * angular_vel;
             }
         }
         
         // Store current pose
         last_q_ = q;
         last_t_ = t;
         last_timestamp_ = timestamp;
         has_previous_estimate_ = true;
     }
     
     // Predict the next pose based on enhanced motion model
     void predict(Eigen::Quaterniond& q_pred, Eigen::Vector3d& t_pred, double timestamp) {
         if (!has_previous_estimate_) {
             // No previous data, return identity
             q_pred = Eigen::Quaterniond(1, 0, 0, 0);
             t_pred = Eigen::Vector3d(0, 0, 0);
             return;
         }
         
         // Calculate time difference
         double dt = timestamp - last_timestamp_;
         if (dt <= 0) dt = 0.1;  // Safeguard
         
         // Detect if motion is too large (potential outlier)
         double expected_translation = avg_linear_velocity_ * dt;
         double max_allowed_translation = expected_translation * 3.0;  // 3x average is max
         
         double translation_scale = 1.0;
         if (last_delta_t_.norm() > max_allowed_translation && max_allowed_translation > 0) {
             // Scale down to reasonable range
             translation_scale = max_allowed_translation / last_delta_t_.norm();
         }
         
         // Predict translation with scaled velocity
         t_pred = last_t_ + translation_scale * last_delta_t_ * (dt / 0.1);  // Normalize by typical frame interval (0.1s)
         
         // Predict rotation using slerp for smoother interpolation
         double angle = 2.0 * acos(std::min(1.0, std::abs(last_delta_q_.w())));
         double expected_angle = avg_angular_velocity_ * dt;
         double max_allowed_angle = expected_angle * 3.0;  // 3x average is max
         
         if (angle > max_allowed_angle && max_allowed_angle > 0) {
             // Limit rotation to reasonable range
             double scale = max_allowed_angle / angle;
             Eigen::Quaterniond limited_q;
             limited_q.w() = 1.0;
             limited_q.x() = last_delta_q_.x() * scale;
             limited_q.y() = last_delta_q_.y() * scale;
             limited_q.z() = last_delta_q_.z() * scale;
             limited_q.normalize();
             
             q_pred = last_q_ * limited_q;
         } else {
             // Normal prediction
             q_pred = last_q_ * last_delta_q_;
         }
         
         // Ensure the quaternion is normalized
         q_pred.normalize();
     }
     
     // Get the quality of the motion model (higher is better)
     double getModelQuality() const {
         if (sample_count_ < 5) {
             return 0.5;  // Medium confidence during initialization
         }
         
         // Calculate confidence based on consistency of motion
         // If standard deviations are low, confidence is high
         return std::min(1.0, 5.0 / sample_count_);
     }
 };
 
 // Helper function to detect and get ring field in Velodyne point cloud
 bool detectVelodyneRingField(const pcl::PointCloud<PointType>& cloud, bool& is_velodyne32, uint8_t& ring_offset) {
     // Check if using custom Velodyne point type (ideal case)
     std::vector<pcl::PCLPointField> fields;
     
     // First try to get fields directly from PCL
     pcl::getFields<PointType>(fields);
     
     // Check for ring field
     bool has_ring = false;
     
     for (const auto& field : fields) {
         if (field.name == "ring") {
             has_ring = true;
             ring_offset = field.offset;
             break;
         }
     }
     
     // Try to determine if it's a Velodyne 32 by checking point distribution
     if (has_ring) {
         // Sample points to check ring distribution
         std::vector<bool> ring_presence(64, false);  // Up to 64 possible rings
         int max_ring = 0;
         
         // Check up to 1000 points
         int check_count = std::min(1000, static_cast<int>(cloud.size()));
         for (int i = 0; i < check_count; i++) {
             const PointType& pt = cloud[i];
             const uint8_t* pt_data = reinterpret_cast<const uint8_t*>(&pt);
             uint16_t ring = *reinterpret_cast<const uint16_t*>(pt_data + ring_offset);
             
             if (ring < 64) {
                 ring_presence[ring] = true;
                 max_ring = std::max(max_ring, static_cast<int>(ring));
             }
         }
         
         // Count how many different rings we saw
         int ring_count = 0;
         for (bool present : ring_presence) {
             if (present) ring_count++;
         }
         
         // If we detected around 32 rings and max ring is approximately 31, it's likely Velodyne 32
         is_velodyne32 = (ring_count >= 30 && ring_count <= 34 && max_ring <= 32);
         return has_ring;
     }
     
     return false;
 }
 
 // Convert quaternion to Euler angles in degrees
 Eigen::Vector3d quaternionToEulerDegrees(const Eigen::Quaterniond& q) {
     Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order (yaw, pitch, roll)
     return Eigen::Vector3d(euler[2] * 180.0 / M_PI,  // Roll
                            euler[1] * 180.0 / M_PI,  // Pitch
                            euler[0] * 180.0 / M_PI); // Yaw
 }
 
 class Fast32Loam {
 public:
     Fast32Loam() : nh_("~"), tf_listener_(tf_buffer_) {
         // Initialize parameters with defaults for 32-channel LiDAR
         initializeParameters();
 
         // Setup subscribers and publishers
         setupROSFramework();
 
         // Initialize variables and data structures
         initialize();
         
         ROS_INFO("FAST-32-LOAM node initialized with %d vertical scans and %d horizontal points", 
                  vertical_scans_, horizontal_scans_);
     }
 
     ~Fast32Loam() {
         // Signal threads to stop
         running_ = false;
         
         if (save_map_) {
             saveGlobalMap();
         }
     }
 
 private:
     // ROS
     ros::NodeHandle nh_;
     ros::Subscriber cloud_sub_;
     ros::Publisher cloud_edge_pub_;
     ros::Publisher cloud_surf_pub_;
     ros::Publisher cloud_full_pub_;
     ros::Publisher odom_pub_;
     ros::Publisher path_pub_;
     ros::Publisher map_pub_;
     tf2_ros::TransformBroadcaster tf_broadcaster_;
     tf2_ros::Buffer tf_buffer_;
     tf2_ros::TransformListener tf_listener_;
     
     std::string fixed_frame_id_;
     std::string lidar_frame_id_;
 
     // Parameters - optimized defaults for Velodyne 32
     double scan_period_;
     int vertical_scans_;
     int horizontal_scans_;
     double vertical_angle_top_;
     double vertical_angle_bottom_;
     double edge_threshold_;
     double surf_threshold_;
     double nearest_feature_dist_sq_;
     int edge_feature_min_valid_num_;
     int surf_feature_min_valid_num_;
     double filter_corner_leaf_size_;
     double filter_surf_leaf_size_;
     double filter_map_leaf_size_;
     bool save_map_;
     std::string map_save_path_;
     double edge_correspondence_threshold_;
     double plane_correspondence_threshold_;
     int max_iterations_;
     bool undistortion_flag_;
     int local_map_size_;
     
     // Multi-resolution parameters
     bool use_multi_resolution_;
     double high_res_leaf_size_;
     double low_res_leaf_size_;
     
     // Hierarchical registration parameters
     bool use_hierarchical_registration_;
     int hierarchical_levels_;
     
     // Motion prediction parameters
     bool use_motion_prediction_;
     double motion_prediction_weight_;
 
     // Data structures
     using CloudPtr = pcl::PointCloud<PointType>::Ptr;
     using CloudConstPtr = pcl::PointCloud<PointType>::ConstPtr;
     
     // Feature map management using ring buffers for efficient updates
     CloudRingBuffer<PointType> corner_map_buffer_;
     CloudRingBuffer<PointType> surf_map_buffer_;
     
     CloudPtr global_map_;
     
     // Voxel grid filters
     pcl::VoxelGrid<PointType> downsize_filter_corner_;
     pcl::VoxelGrid<PointType> downsize_filter_surf_;
     pcl::VoxelGrid<PointType> downsize_filter_map_;
     
     // More efficient approximate voxel grid for high-speed filtering
     pcl::ApproximateVoxelGrid<PointType> approx_filter_corner_;
     pcl::ApproximateVoxelGrid<PointType> approx_filter_surf_;
     
     // KD-trees
     pcl::KdTreeFLANN<PointType>::Ptr kdtree_corner_map_;
     pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf_map_;
     bool kdtree_needs_update_;
 
     // Current scan data
     pcl::PointCloud<PointType> laser_cloud_in_;
     pcl::PointCloud<PointType> corner_points_sharp_;
     pcl::PointCloud<PointType> corner_points_less_sharp_;
     pcl::PointCloud<PointType> surface_points_flat_;
     pcl::PointCloud<PointType> surface_points_less_flat_;
     
     // Multi-resolution versions
     pcl::PointCloud<PointType> corner_points_high_res_;
     pcl::PointCloud<PointType> corner_points_low_res_;
     pcl::PointCloud<PointType> surf_points_high_res_;
     pcl::PointCloud<PointType> surf_points_low_res_;
 
     // Navigation data
     nav_msgs::Path path_;
 
     // Transformation
     Eigen::Quaterniond q_w_curr_;  // World to current orientation
     Eigen::Vector3d t_w_curr_;     // World to current position
     
     // Motion prediction
     MotionPredictor motion_predictor_;
 
     // Thread management
     std::atomic<bool> running_;
     
     // Mutex for thread safety
     std::mutex map_mutex_;
 
     // Timing
     double time_laser_cloud_last_;
     double time_laser_cloud_curr_;
     bool system_initialized_;
     bool first_scan_;
     int frame_count_;
     
     // LiDAR type detection
     bool is_velodyne32_;
     bool has_ring_data_;
     uint8_t ring_field_offset_;
     
     // Pose output
     std::ofstream pose_file_;
     bool write_poses_to_file_;
     std::string pose_file_path_;
     
     // Performance tracking
     struct TimerEntry {
         std::chrono::high_resolution_clock::time_point start;
         std::string name;
         
         TimerEntry(const std::string& n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
         
         double elapsed() const {
             auto end = std::chrono::high_resolution_clock::now();
             return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
         }
     };
     std::vector<TimerEntry> timers_;
     
     // Debug
     bool debug_output_;
     bool timing_output_;
     bool pose_output_;
 
     // Methods
     void initializeParameters() {
         // 32-channel specific defaults
         nh_.param<double>("scan_period", scan_period_, 0.1);
         nh_.param<int>("vertical_scans", vertical_scans_, 32);  // Default for 32-channel LiDAR
         nh_.param<int>("horizontal_scans", horizontal_scans_, 2048);  // Higher horizontal resolution
         nh_.param<double>("vertical_angle_top", vertical_angle_top_, 10.67);  // Velodyne 32E defaults
         nh_.param<double>("vertical_angle_bottom", vertical_angle_bottom_, -30.67);  // Velodyne 32E defaults
         nh_.param<double>("edge_threshold", edge_threshold_, 0.15);  // Slightly higher for 32-channel
         nh_.param<double>("surf_threshold", surf_threshold_, 0.08);  // Adjusted for 32-channel
         nh_.param<double>("nearest_feature_dist_sq", nearest_feature_dist_sq_, 36.0);  // Increased radius
         nh_.param<int>("edge_feature_min_valid_num", edge_feature_min_valid_num_, 10);
         nh_.param<int>("surf_feature_min_valid_num", surf_feature_min_valid_num_, 100);
         nh_.param<double>("filter_corner_leaf_size", filter_corner_leaf_size_, 0.3);  // Larger for 32-channel
         nh_.param<double>("filter_surf_leaf_size", filter_surf_leaf_size_, 0.6);  // Larger for 32-channel
         nh_.param<double>("filter_map_leaf_size", filter_map_leaf_size_, 0.8);  // Larger for 32-channel
         nh_.param<bool>("save_map", save_map_, true);
         nh_.param<std::string>("map_save_path", map_save_path_, "/tmp/loam_map.pcd");
         nh_.param<std::string>("fixed_frame_id", fixed_frame_id_, "map");
         nh_.param<std::string>("lidar_frame_id", lidar_frame_id_, "lidar_link");
         nh_.param<int>("max_iterations", max_iterations_, 6);  // Increased for better accuracy
         nh_.param<bool>("undistortion_flag", undistortion_flag_, true);
         nh_.param<bool>("debug_output", debug_output_, false);
         nh_.param<bool>("timing_output", timing_output_, false);
         nh_.param<bool>("pose_output", pose_output_, true);  // Enable pose output by default
         nh_.param<int>("local_map_size", local_map_size_, 30);  // Larger for 32-channel
         
         // File output
         nh_.param<bool>("write_poses_to_file", write_poses_to_file_, false);
         nh_.param<std::string>("pose_file_path", pose_file_path_, "/tmp/loam_poses.txt");
         
         // Multi-resolution parameters
         nh_.param<bool>("use_multi_resolution", use_multi_resolution_, true);
         nh_.param<double>("high_res_leaf_size", high_res_leaf_size_, 0.3);  // Adjusted for 32-channel
         nh_.param<double>("low_res_leaf_size", low_res_leaf_size_, 1.0);  // Adjusted for 32-channel
         
         // Hierarchical registration parameters
         nh_.param<bool>("use_hierarchical_registration", use_hierarchical_registration_, true);
         nh_.param<int>("hierarchical_levels", hierarchical_levels_, 3);
         
         // Motion prediction parameters
         nh_.param<bool>("use_motion_prediction", use_motion_prediction_, true);
         nh_.param<double>("motion_prediction_weight", motion_prediction_weight_, 0.6);  // Increased weight for smoother motion
 
         system_initialized_ = false;
         first_scan_ = true;
         frame_count_ = 0;
         running_ = true;
         kdtree_needs_update_ = true;
         is_velodyne32_ = false;
         has_ring_data_ = false;
         ring_field_offset_ = 16;  // Default guess for ring field offset
         
         // Open pose file if needed
         if (write_poses_to_file_) {
             pose_file_.open(pose_file_path_);
             if (pose_file_.is_open()) {
                 pose_file_ << "# timestamp tx ty tz qx qy qz qw roll pitch yaw" << std::endl;
                 ROS_INFO("Saving poses to: %s", pose_file_path_.c_str());
             } else {
                 ROS_WARN("Failed to open pose file for writing: %s", pose_file_path_.c_str());
                 write_poses_to_file_ = false;
             }
         }
     }
 
     void setupROSFramework() {
         cloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 5, &Fast32Loam::cloudCallback, this);
         cloud_edge_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/loam/edge_points", 1);
         cloud_surf_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/loam/surf_points", 1);
         cloud_full_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/loam/full_points", 1);
         odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/loam/odometry", 1);
         path_pub_ = nh_.advertise<nav_msgs::Path>("/loam/path", 1);
         map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/loam/map", 1);
     }
     
     void initialize() {
         // Initialize transformation values
         q_w_curr_ = Eigen::Quaterniond(1, 0, 0, 0);  // identity quaternion
         t_w_curr_ = Eigen::Vector3d(0, 0, 0);        // zero translation
         
         // Initialize point cloud containers with ring buffers for efficient management
         corner_map_buffer_ = CloudRingBuffer<PointType>(local_map_size_, filter_corner_leaf_size_);
         surf_map_buffer_ = CloudRingBuffer<PointType>(local_map_size_, filter_surf_leaf_size_);
         
         // Initialize global map
         global_map_.reset(new pcl::PointCloud<PointType>());
         
         // Initialize KD-trees
         kdtree_corner_map_.reset(new pcl::KdTreeFLANN<PointType>());
         kdtree_surf_map_.reset(new pcl::KdTreeFLANN<PointType>());
         
         // Set up voxel grid filters
         downsize_filter_corner_.setLeafSize(filter_corner_leaf_size_, filter_corner_leaf_size_, filter_corner_leaf_size_);
         downsize_filter_surf_.setLeafSize(filter_surf_leaf_size_, filter_surf_leaf_size_, filter_surf_leaf_size_);
         downsize_filter_map_.setLeafSize(filter_map_leaf_size_, filter_map_leaf_size_, filter_map_leaf_size_);
         
         // Set up approximate voxel grid filters for faster processing
         approx_filter_corner_.setLeafSize(filter_corner_leaf_size_, filter_corner_leaf_size_, filter_corner_leaf_size_);
         approx_filter_surf_.setLeafSize(filter_surf_leaf_size_, filter_surf_leaf_size_, filter_surf_leaf_size_);
         
         // Initialize timing variables
         time_laser_cloud_last_ = 0;
         time_laser_cloud_curr_ = 0;
         
         // Initialize threshold values
         edge_correspondence_threshold_ = 0.1;  // 10cm
         plane_correspondence_threshold_ = 0.05;  // 5cm
         
         // Initialize path message
         path_.header.frame_id = fixed_frame_id_;
     }
 
     void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
         startTimer("cloudCallback");
         
         time_laser_cloud_curr_ = cloud_msg->header.stamp.toSec();
 
         // Convert ROS message to PCL point cloud
         startTimer("cloudConversion");
         pcl::PointCloud<PointType> laser_cloud_in;
         pcl::fromROSMsg(*cloud_msg, laser_cloud_in);
         stopTimer();
 
         // Skip if point cloud is empty
         if (laser_cloud_in.empty()) {
             ROS_WARN("Received empty point cloud! Skipping.");
             stopTimer(); // Stop the cloudCallback timer
             return;
         }
         
         // Preprocess point cloud to remove outliers
         startTimer("cloudPreprocessing");
         preprocessPointCloud(laser_cloud_in);
         stopTimer();
         
         // Detect Velodyne LiDAR type and ring field on first scan
         if (first_scan_) {
             startTimer("detectLidarType");
             has_ring_data_ = detectVelodyneRingField(laser_cloud_in, is_velodyne32_, ring_field_offset_);
             
             if (is_velodyne32_) {
                 ROS_INFO("Detected Velodyne 32-channel LiDAR with ring field.");
                 
                 // Adjust parameters if auto-detected
                 if (vertical_scans_ != 32) {
                     vertical_scans_ = 32;
                     ROS_INFO("Automatically adjusted vertical_scans to 32.");
                 }
             } else if (has_ring_data_) {
                 ROS_INFO("Detected LiDAR with ring field, but not identified as Velodyne 32-channel.");
             } else {
                 ROS_WARN("No ring field detected. Using angle-based ring estimation.");
             }
             stopTimer();
         }
 
         // Extract features
         startTimer("featureExtraction");
         extractFeatures(laser_cloud_in);
         stopTimer();
 
         if (first_scan_) {
             // Initialize the system with the first scan
             initializeSystem();
             first_scan_ = false;
             stopTimer(); // Stop the cloudCallback timer
             return;
         }
 
         // FAST-LOAM approach: direct scan-to-map matching
         startTimer("scanToMapMatching");
         scanToMapMatching();
         stopTimer();
         
         // Update maps with current scan
         startTimer("updateMaps");
         updateMaps();
         stopTimer();
 
         // Publish results
         startTimer("publishResults");
         publishResults(cloud_msg->header.stamp);
         stopTimer();
         
         // Output pose information
         if (pose_output_) {
             printCurrentPose();
         }
         
         // Write pose to file if enabled
         if (write_poses_to_file_ && pose_file_.is_open()) {
             writePoseToFile(cloud_msg->header.stamp);
         }
         
         // Update motion predictor with the new pose
         if (use_motion_prediction_) {
             motion_predictor_.update(q_w_curr_, t_w_curr_, time_laser_cloud_curr_);
         }
         
         time_laser_cloud_last_ = time_laser_cloud_curr_;
         frame_count_++;
         
         stopTimer(); // Stop the cloudCallback timer
         
         if (timing_output_) {
             printTimers();
         }
     }
     
     void preprocessPointCloud(pcl::PointCloud<PointType>& cloud) {
         if (cloud.empty()) {
             return;
         }
         
         // Basic distance filtering to remove very close or far points
         CloudPtr filtered_cloud(new pcl::PointCloud<PointType>());
         for (const auto& point : cloud) {
             float range_sq = point.x * point.x + point.y * point.y + point.z * point.z;
             // Keep points between 0.5m and 100m
             if (range_sq > 0.25 && range_sq < 10000.0) {
                 filtered_cloud->push_back(point);
             }
         }
         
         if (filtered_cloud->empty()) {
             ROS_WARN("All points filtered out by distance filter!");
             return;
         }
         
         // Statistical outlier removal for more robust feature extraction
         if (is_velodyne32_ && filtered_cloud->size() > 1000) {
             pcl::StatisticalOutlierRemoval<PointType> sor;
             sor.setInputCloud(filtered_cloud);
             sor.setMeanK(20);           // Consider 20 neighbors
             sor.setStddevMulThresh(1.5); // Threshold: 1.5 standard deviations
             
             pcl::PointCloud<PointType> cloud_filtered;
             sor.filter(cloud_filtered);
             
             // Only use the filter if it doesn't remove too many points
             if (cloud_filtered.size() > filtered_cloud->size() * 0.5) {
                 cloud = cloud_filtered;
                 return;
             }
         }
         
         // If no statistical filtering applied, still use the distance-filtered cloud
         cloud = *filtered_cloud;
     }
     
     void initializeSystem() {
         // Create initial feature maps
         CloudPtr initial_corner_cloud(new pcl::PointCloud<PointType>(corner_points_less_sharp_));
         CloudPtr initial_surf_cloud(new pcl::PointCloud<PointType>(surface_points_less_flat_));
         CloudPtr initial_full_cloud(new pcl::PointCloud<PointType>(laser_cloud_in_));
         
         // Add to the ring buffers
         corner_map_buffer_.push(initial_corner_cloud);
         surf_map_buffer_.push(initial_surf_cloud);
         
         // Add to global map
         *global_map_ += laser_cloud_in_;
         
         // Initialize KD-trees with the first scan
         kdtree_corner_map_->setInputCloud(corner_map_buffer_.getCombinedCloud());
         kdtree_surf_map_->setInputCloud(surf_map_buffer_.getCombinedCloud());
         
         // Set the initial scan timestamp
         time_laser_cloud_last_ = time_laser_cloud_curr_;
         
         // Mark system as initialized
         system_initialized_ = true;
         
         ROS_INFO("FAST-32-LOAM system initialized with %zu corner and %zu surface points.",
                 corner_points_less_sharp_.size(), surface_points_less_flat_.size());
     }
 
     void extractFeatures(const pcl::PointCloud<PointType>& laser_cloud_in) {
         // Clear feature clouds
         corner_points_sharp_.clear();
         corner_points_less_sharp_.clear();
         surface_points_flat_.clear();
         surface_points_less_flat_.clear();
         
         // Store input cloud
         laser_cloud_in_ = laser_cloud_in;
         
         // Organize points by scan ring
         std::vector<pcl::PointCloud<PointType>> laser_cloud_scan_ring(vertical_scans_);
         
         // Distribute points to scan rings
         for (const auto& pt : laser_cloud_in) {
             // Skip invalid points
             if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                 continue;
             }
             
             int scan_id;
             if (has_ring_data_) {
                 // Extract ring ID directly from point data
                 // This accesses the ring field at the appropriate offset
                 const uint8_t* pt_data = reinterpret_cast<const uint8_t*>(&pt);
                 uint16_t ring = *reinterpret_cast<const uint16_t*>(pt_data + ring_field_offset_);
                 
                 // Ensure the ring ID is in valid range for our processing
                 // For Velodyne 32, map 0-31 to 0-31 directly
                 scan_id = ring;
                 
                 if (scan_id < 0 || scan_id >= vertical_scans_) {
                     continue;
                 }
             } else {
                 // Calculate scan ring from point coordinates (fallback method)
                 float vertical_angle = std::atan2(pt.z, std::sqrt(pt.x * pt.x + pt.y * pt.y)) * 180.0f / M_PI;
                 
                 // Map angle to scan ring
                 scan_id = std::round((vertical_angle - vertical_angle_bottom_) / 
                                     (vertical_angle_top_ - vertical_angle_bottom_) * 
                                     (vertical_scans_ - 1));
                 
                 if (scan_id < 0 || scan_id >= vertical_scans_) {
                     continue;
                 }
             }
             
             // Add point to corresponding scan ring
             laser_cloud_scan_ring[scan_id].push_back(pt);
         }
         
         // Temporary storage for feature points
         std::vector<PointType> sharp_points;
         std::vector<PointType> less_sharp_points;
         std::vector<PointType> flat_points;
         std::vector<PointType> less_flat_points;
         
         // Enhanced feature extraction for 32-channel LiDAR
         // Process each scan ring for feature extraction
         for (int i = 0; i < vertical_scans_; i++) {
             pcl::PointCloud<PointType>& cloud_scan = laser_cloud_scan_ring[i];
             
             if (cloud_scan.size() < 30) {  // Require more points for 32-channel
                 continue;  // Skip if scan has too few points
             }
             
             // Calculate curvature for each point with increased neighborhood
             std::vector<float> cloud_curvature(cloud_scan.size(), 0);
             std::vector<int> cloud_sort_idx(cloud_scan.size());
             std::vector<int> cloud_neighbor_picked(cloud_scan.size(), 0);
             
             int window_size = is_velodyne32_ ? 7 : 5;  // Larger window for 32-channel LiDAR
             
             // Calculate curvature with larger window for 32-channel data
             for (size_t j = window_size; j < cloud_scan.size() - window_size; j++) {
                 float diff_x = 0, diff_y = 0, diff_z = 0;
                 
                 // Use a small neighborhood for curvature calculation
                 for (int k = -window_size; k <= window_size; k++) {
                     if (k == 0) continue;
                     diff_x += cloud_scan[j + k].x - cloud_scan[j].x;
                     diff_y += cloud_scan[j + k].y - cloud_scan[j].y;
                     diff_z += cloud_scan[j + k].z - cloud_scan[j].z;
                 }
                 
                 // Normalize curvature by point distance for more consistent feature detection
                 float dist_sq = cloud_scan[j].x * cloud_scan[j].x + 
                                cloud_scan[j].y * cloud_scan[j].y + 
                                cloud_scan[j].z * cloud_scan[j].z;
                 
                 // Avoid division by zero or very small values
                 float norm_factor = std::max(1.0f, dist_sq * 0.01f);  // Adjusted scale factor
                 
                 // Store normalized curvature
                 cloud_curvature[j] = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / norm_factor;
                 cloud_sort_idx[j] = j;
             }
             
             // Divide the scan into segments for better feature distribution
             // More segments for 32-channel LiDAR to capture more details
             int total_points = cloud_scan.size() - window_size * 2;
             const int num_sectors = is_velodyne32_ ? 8 : 6;  // More sectors for 32-channel
             
             for (int j = 0; j < num_sectors; j++) {
                 int start_idx = j * total_points / num_sectors + window_size;
                 int end_idx = (j + 1) * total_points / num_sectors + window_size - 1;
                 
                 // Sort points by curvature within each segment
                 std::sort(cloud_sort_idx.begin() + start_idx, cloud_sort_idx.begin() + end_idx + 1,
                          [&cloud_curvature](int a, int b) { return cloud_curvature[a] < cloud_curvature[b]; });
                 
                 // Select surface (flat) features - low curvature points
                 int flat_count = 0;
                 for (int k = start_idx; k <= end_idx && flat_count < 4; k++) {
                     int idx = cloud_sort_idx[k];
                     
                     if (cloud_curvature[idx] < surf_threshold_ && cloud_neighbor_picked[idx] == 0) {
                         flat_count++;
                         cloud_neighbor_picked[idx] = 1;
                         flat_points.push_back(cloud_scan[idx]);
                         
                         // Mark nearby points as picked to ensure feature distribution
                         for (int l = 1; l <= window_size; l++) {
                             if (idx + l >= cloud_scan.size()) break;
                             float dx = cloud_scan[idx + l].x - cloud_scan[idx].x;
                             float dy = cloud_scan[idx + l].y - cloud_scan[idx].y;
                             float dz = cloud_scan[idx + l].z - cloud_scan[idx].z;
                             if (dx * dx + dy * dy + dz * dz > 0.05) break;
                             cloud_neighbor_picked[idx + l] = 1;
                         }
                         for (int l = -1; l >= -window_size; l--) {
                             if (idx + l < 0) break;
                             float dx = cloud_scan[idx + l].x - cloud_scan[idx].x;
                             float dy = cloud_scan[idx + l].y - cloud_scan[idx].y;
                             float dz = cloud_scan[idx + l].z - cloud_scan[idx].z;
                             if (dx * dx + dy * dy + dz * dz > 0.05) break;
                             cloud_neighbor_picked[idx + l] = 1;
                         }
                     }
                 }
                 
                 // Select more less-flat features
                 for (int k = start_idx; k <= end_idx; k++) {
                     int idx = cloud_sort_idx[k];
                     
                     if (cloud_curvature[idx] < surf_threshold_ * 2 && cloud_neighbor_picked[idx] == 0) {
                         cloud_neighbor_picked[idx] = 1;
                         less_flat_points.push_back(cloud_scan[idx]);
                     }
                 }
                 
                 // Select edge (sharp) features - high curvature points
                 // Resort in descending order for edge features
                 std::sort(cloud_sort_idx.begin() + start_idx, cloud_sort_idx.begin() + end_idx + 1,
                          [&cloud_curvature](int a, int b) { return cloud_curvature[a] > cloud_curvature[b]; });
                 
                 int sharp_count = 0;
                 for (int k = start_idx; k <= end_idx && sharp_count < 2; k++) {
                     int idx = cloud_sort_idx[k];
                     
                     if (cloud_curvature[idx] > edge_threshold_ && cloud_neighbor_picked[idx] == 0) {
                         sharp_count++;
                         cloud_neighbor_picked[idx] = 1;
                         sharp_points.push_back(cloud_scan[idx]);
                         
                         // Mark nearby points as picked to ensure feature distribution
                         for (int l = 1; l <= window_size; l++) {
                             if (idx + l >= cloud_scan.size()) break;
                             float dx = cloud_scan[idx + l].x - cloud_scan[idx].x;
                             float dy = cloud_scan[idx + l].y - cloud_scan[idx].y;
                             float dz = cloud_scan[idx + l].z - cloud_scan[idx].z;
                             if (dx * dx + dy * dy + dz * dz > 0.05) break;
                             cloud_neighbor_picked[idx + l] = 1;
                         }
                         for (int l = -1; l >= -window_size; l--) {
                             if (idx + l < 0) break;
                             float dx = cloud_scan[idx + l].x - cloud_scan[idx].x;
                             float dy = cloud_scan[idx + l].y - cloud_scan[idx].y;
                             float dz = cloud_scan[idx + l].z - cloud_scan[idx].z;
                             if (dx * dx + dy * dy + dz * dz > 0.05) break;
                             cloud_neighbor_picked[idx + l] = 1;
                         }
                     }
                 }
                 
                 // Select more less-sharp edge features
                 for (int k = start_idx; k <= end_idx; k++) {
                     int idx = cloud_sort_idx[k];
                     
                     if (cloud_curvature[idx] > edge_threshold_ * 0.5 && cloud_neighbor_picked[idx] == 0) {
                         cloud_neighbor_picked[idx] = 1;
                         less_sharp_points.push_back(cloud_scan[idx]);
                     }
                 }
             }
         }
         
         // Copy to output clouds
         corner_points_sharp_.insert(corner_points_sharp_.end(), sharp_points.begin(), sharp_points.end());
         corner_points_less_sharp_.insert(corner_points_less_sharp_.end(), sharp_points.begin(), sharp_points.end());
         corner_points_less_sharp_.insert(corner_points_less_sharp_.end(), less_sharp_points.begin(), less_sharp_points.end());
         surface_points_flat_.insert(surface_points_flat_.end(), flat_points.begin(), flat_points.end());
         
         // Downsample less-flat surface points using voxel grid filter
         pcl::PointCloud<PointType> less_flat_cloud;
         less_flat_cloud.insert(less_flat_cloud.end(), less_flat_points.begin(), less_flat_points.end());
         
         pcl::VoxelGrid<PointType> downsize_filter;
         downsize_filter.setLeafSize(filter_surf_leaf_size_, filter_surf_leaf_size_, filter_surf_leaf_size_);
         downsize_filter.setInputCloud(less_flat_cloud.makeShared());
         downsize_filter.filter(surface_points_less_flat_);
         
         // Add flat points to less-flat cloud
         surface_points_less_flat_.insert(surface_points_less_flat_.end(), flat_points.begin(), flat_points.end());
         
         // More aggressive downsampling for 32-channel LiDAR to manage the larger point count
         if (is_velodyne32_ && surface_points_less_flat_.size() > 8000) {
             pcl::PointCloud<PointType> temp_cloud = surface_points_less_flat_;
             pcl::VoxelGrid<PointType> secondary_filter;
             secondary_filter.setLeafSize(filter_surf_leaf_size_ * 1.2, filter_surf_leaf_size_ * 1.2, filter_surf_leaf_size_ * 1.2);
             secondary_filter.setInputCloud(temp_cloud.makeShared());
             secondary_filter.filter(surface_points_less_flat_);
         }
         
         if (debug_output_) {
             ROS_INFO("Feature extraction: %zu sharp edge, %zu less sharp edge, %zu flat surface, %zu less flat surface",
                     corner_points_sharp_.size(), corner_points_less_sharp_.size(), 
                     surface_points_flat_.size(), surface_points_less_flat_.size());
         }
     }
     
     void createMultiResolutionFeatures() {
         // Create high-resolution and low-resolution versions of the feature clouds
         // High resolution for fine alignment
         corner_points_high_res_.clear();
         surf_points_high_res_.clear();
         
         // Low resolution for coarse alignment
         corner_points_low_res_.clear();
         surf_points_low_res_.clear();
         
         // Downsample to high resolution
         pcl::VoxelGrid<PointType> high_res_filter;
         high_res_filter.setLeafSize(high_res_leaf_size_, high_res_leaf_size_, high_res_leaf_size_);
         
         // Downsample to low resolution (use faster approximate voxel grid)
         pcl::ApproximateVoxelGrid<PointType> low_res_filter;
         low_res_filter.setLeafSize(low_res_leaf_size_, low_res_leaf_size_, low_res_leaf_size_);
         
         // Process corner points
         if (!corner_points_less_sharp_.empty()) {
             high_res_filter.setInputCloud(corner_points_less_sharp_.makeShared());
             high_res_filter.filter(corner_points_high_res_);
             
             low_res_filter.setInputCloud(corner_points_less_sharp_.makeShared());
             low_res_filter.filter(corner_points_low_res_);
         }
         
         // Process surface points
         if (!surface_points_less_flat_.empty()) {
             high_res_filter.setInputCloud(surface_points_less_flat_.makeShared());
             high_res_filter.filter(surf_points_high_res_);
             
             low_res_filter.setInputCloud(surface_points_less_flat_.makeShared());
             low_res_filter.filter(surf_points_low_res_);
         }
     }
     
     void scanToMapMatching() {
         std::lock_guard<std::mutex> lock(map_mutex_);
         
         if (corner_map_buffer_.empty() || surf_map_buffer_.empty()) {
             ROS_WARN_THROTTLE(1, "Feature maps are empty, skipping scan-to-map matching");
             return;
         }
         
         // Get combined feature maps
         CloudPtr corner_map = corner_map_buffer_.getCombinedCloud();
         CloudPtr surf_map = surf_map_buffer_.getCombinedCloud();
         
         // Update KD-trees only when needed (not in every frame)
         if (kdtree_needs_update_) {
             if (!corner_map->empty()) {
                 kdtree_corner_map_->setInputCloud(corner_map);
             }
             
             if (!surf_map->empty()) {
                 kdtree_surf_map_->setInputCloud(surf_map);
             }
             
             kdtree_needs_update_ = false;
         }
         
         // Predict initial pose using motion model
         Eigen::Quaterniond q_pred;
         Eigen::Vector3d t_pred;
         
         if (use_motion_prediction_ && system_initialized_) {
             motion_predictor_.predict(q_pred, t_pred, time_laser_cloud_curr_);
             
             // Use a weighted combination of the last pose and the predicted pose
             q_pred = q_w_curr_.slerp(motion_prediction_weight_, q_pred);
             t_pred = (1 - motion_prediction_weight_) * t_w_curr_ + motion_prediction_weight_ * t_pred;
         } else {
             // If no motion prediction, use the last pose
             q_pred = q_w_curr_;
             t_pred = t_w_curr_;
         }
         
         // Create multi-resolution feature point clouds
         if (use_multi_resolution_ || use_hierarchical_registration_) {
             createMultiResolutionFeatures();
         }
         
         // Hierarchical registration (from coarse to fine)
         if (use_hierarchical_registration_) {
             hierarchicalRegistration(q_pred, t_pred, corner_map, surf_map);
         } else {
             // Standard registration
             singleLevelRegistration(q_pred, t_pred, corner_map, surf_map);
         }
         
         // Update the global pose
         q_w_curr_ = q_pred;
         t_w_curr_ = t_pred;
     }
     
     void hierarchicalRegistration(Eigen::Quaterniond& q, Eigen::Vector3d& t, 
                                  CloudPtr& corner_map, CloudPtr& surf_map) {
         // Multi-level registration from coarse to fine
         // Adjusted leaf sizes for 32-channel LiDAR (larger for performance)
         std::vector<double> leaf_sizes = {1.5, 0.8, 0.3};  // Adjusted for 32-channel
         std::vector<int> max_iterations = {5, 10, 20};     // More iterations for 32-channel
         std::vector<double> convergence_thresholds = {0.08, 0.03, 0.01};
         
         // Adjust based on parameter
         int levels = std::min(hierarchical_levels_, static_cast<int>(leaf_sizes.size()));
         
         // Store initial pose for fallback
         Eigen::Quaterniond q_initial = q;
         Eigen::Vector3d t_initial = t;
         bool optimization_success = false;
         
         for (int level = 0; level < levels; level++) {
             // Downsample clouds for this level
             pcl::PointCloud<PointType> corner_cloud_level;
             pcl::PointCloud<PointType> surf_cloud_level;
             
             pcl::VoxelGrid<PointType> downsample_filter;
             downsample_filter.setLeafSize(leaf_sizes[level], leaf_sizes[level], leaf_sizes[level]);
             
             // Downsample feature clouds for this level
             if (!corner_points_less_sharp_.empty()) {
                 downsample_filter.setInputCloud(corner_points_less_sharp_.makeShared());
                 downsample_filter.filter(corner_cloud_level);
             }
             
             if (!surface_points_less_flat_.empty()) {
                 downsample_filter.setInputCloud(surface_points_less_flat_.makeShared());
                 downsample_filter.filter(surf_cloud_level);
             }
             
             // Keep track of the previous pose for convergence check
             Eigen::Quaterniond q_prev = q;
             Eigen::Vector3d t_prev = t;
             
             // Initialize optimization parameters with current pose
             double parameters[7] = {
                 q.x(), q.y(), q.z(), q.w(),
                 t.x(), t.y(), t.z()
             };
             
             // Setup ceres problem
             ceres::Problem problem;
             ceres::LossFunction* loss_function = new ceres::HuberLoss(leaf_sizes[level] * 0.1);  // Scale with level
             ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
             
             // Add parameter blocks with appropriate parameterization
             problem.AddParameterBlock(parameters, 4, quaternion_parameterization);
             problem.AddParameterBlock(parameters + 4, 3);
             
             // Add factors from the downsampled clouds
             int edge_factor_count = addEdgeFactorsToOptimization(problem, loss_function, parameters, 
                                                              corner_cloud_level, corner_map, level > 0);
             
             int plane_factor_count = addPlaneFactorsToOptimization(problem, loss_function, parameters, 
                                                                surf_cloud_level, surf_map, level > 0);
             
             // Solve optimization if we have enough constraints
             if (edge_factor_count + plane_factor_count >= 10) {
                 ceres::Solver::Options options;
                 options.linear_solver_type = ceres::DENSE_QR;
                 options.max_num_iterations = max_iterations[level];
                 options.function_tolerance = 1e-7;  // Tightened tolerance
                 options.gradient_tolerance = 1e-11;  // Tightened tolerance
                 options.parameter_tolerance = 1e-9;  // Tightened tolerance
                 options.minimizer_progress_to_stdout = debug_output_;
                 
                 ceres::Solver::Summary summary;
                 ceres::Solve(options, &problem, &summary);
                 
                 if (debug_output_) {
                     ROS_INFO("Level %d optimization: %d factors, cost %.5f -> %.5f, iterations: %ld", 
                             level, edge_factor_count + plane_factor_count, 
                             summary.initial_cost, summary.final_cost, 
                             static_cast<long>(summary.iterations.size()));
                 }
                 
                 // Update pose for next level
                 q = Eigen::Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).normalized();
                 t = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
                 
                 // Check for convergence
                 double rot_diff = 2.0 * acos(std::min(1.0, std::abs(q.dot(q_prev))));
                 double trans_diff = (t - t_prev).norm();
                 
                 if (rot_diff < convergence_thresholds[level] && trans_diff < convergence_thresholds[level]) {
                     if (debug_output_) {
                         ROS_INFO("Early convergence at level %d: rot_diff=%.6f, trans_diff=%.6f",
                                 level, rot_diff, trans_diff);
                     }
                     optimization_success = true;
                     break;  // Early termination if converged
                 }
                 
                 // Check if the optimization made reasonable progress
                 if (summary.final_cost < summary.initial_cost * 0.8) {
                     optimization_success = true;
                 } else {
                     // If the cost didn't decrease significantly, the optimization might be stuck
                     if (level > 0) {
                         ROS_WARN("Level %d optimization did not make sufficient progress.", level);
                     }
                 }
             } else {
                 ROS_WARN_THROTTLE(1, "Level %d: Not enough constraints for optimization! Edge: %d, Plane: %d", 
                             level, edge_factor_count, plane_factor_count);
                 
                 // Use previous level's result if available
                 if (level > 0) {
                     q = q_prev;
                     t = t_prev;
                 }
             }
         }
         
         // Sanity check on final pose - if too far from initial prediction, fall back
         double translation_diff = (t - t_initial).norm();
         double rotation_diff = 2.0 * acos(std::min(1.0, std::abs(q.dot(q_initial))));
         
         if (!optimization_success || translation_diff > 2.0 || rotation_diff > M_PI / 4) {
             ROS_WARN("Registration result looks suspicious (%.2fm, %.2f°), using motion prediction instead.",
                   translation_diff, rotation_diff * 180 / M_PI);
             
             // Fall back to motion prediction
             q = q_initial;
             t = t_initial;
         }
     }
     
     void singleLevelRegistration(Eigen::Quaterniond& q, Eigen::Vector3d& t, 
                                 CloudPtr& corner_map, CloudPtr& surf_map) {
         // Initialize optimization parameters with current pose
         double parameters[7] = {
             q.x(), q.y(), q.z(), q.w(),
             t.x(), t.y(), t.z()
         };
         
         // Store initial pose for fallback
         Eigen::Quaterniond q_initial = q;
         Eigen::Vector3d t_initial = t;
         bool optimization_success = false;
         
         // Multi-resolution approach
         int edge_factor_count = 0;
         int plane_factor_count = 0;
         
         if (use_multi_resolution_) {
             // First use low-resolution features for coarse alignment
             {
                 // Setup ceres problem in its own scope to ensure cleanup
                 ceres::Problem problem;
                 ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);  // Increased robustness
                 ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
                 
                 // Add parameter blocks with appropriate parameterization
                 problem.AddParameterBlock(parameters, 4, quaternion_parameterization);
                 problem.AddParameterBlock(parameters + 4, 3);
                 
                 edge_factor_count = addEdgeFactorsToOptimization(problem, loss_function, parameters, 
                                                              corner_points_low_res_, corner_map, true);
                 
                 plane_factor_count = addPlaneFactorsToOptimization(problem, loss_function, parameters, 
                                                                surf_points_low_res_, surf_map, true);
                 
                 // Solve optimization if we have enough constraints
                 if (edge_factor_count + plane_factor_count >= 10) {
                     ceres::Solver::Options options;
                     options.linear_solver_type = ceres::DENSE_QR;
                     options.max_num_iterations = max_iterations_ / 2;  // Fewer iterations for coarse alignment
                     options.minimizer_progress_to_stdout = debug_output_;
                     
                     ceres::Solver::Summary summary;
                     ceres::Solve(options, &problem, &summary);
                     
                     if (debug_output_) {
                         ROS_INFO("Coarse optimization: %d factors, cost %.5f -> %.5f", 
                                 edge_factor_count + plane_factor_count, 
                                 summary.initial_cost, summary.final_cost);
                     }
                     
                     // Update parameters for fine alignment
                     q = Eigen::Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).normalized();
                     t = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
                     
                     parameters[0] = q.x();
                     parameters[1] = q.y();
                     parameters[2] = q.z();
                     parameters[3] = q.w();
                     parameters[4] = t.x();
                     parameters[5] = t.y();
                     parameters[6] = t.z();
                     
                     // Check if coarse optimization made reasonable progress
                     if (summary.final_cost < summary.initial_cost * 0.8) {
                         optimization_success = true;
                     }
                 }
             }
             
             // Then use high-resolution features for fine alignment with a new problem instance
             {
                 ceres::Problem problem;
                 ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
                 ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
                 
                 problem.AddParameterBlock(parameters, 4, quaternion_parameterization);
                 problem.AddParameterBlock(parameters + 4, 3);
                 
                 edge_factor_count = addEdgeFactorsToOptimization(problem, loss_function, parameters, 
                                                              corner_points_high_res_, corner_map, false);
                 
                 plane_factor_count = addPlaneFactorsToOptimization(problem, loss_function, parameters, 
                                                                surf_points_high_res_, surf_map, false);
                 
                 // Solve optimization if we have enough constraints
                 if (edge_factor_count + plane_factor_count >= 10) {
                     ceres::Solver::Options options;
                     options.linear_solver_type = ceres::DENSE_QR;
                     options.max_num_iterations = max_iterations_;
                     options.minimizer_progress_to_stdout = debug_output_;
                     
                     ceres::Solver::Summary summary;
                     ceres::Solve(options, &problem, &summary);
                     
                     if (debug_output_) {
                         ROS_INFO("Fine optimization: %d factors, cost %.5f -> %.5f", 
                                 edge_factor_count + plane_factor_count, 
                                 summary.initial_cost, summary.final_cost);
                     }
                     
                     // Update the pose with the optimization result
                     q = Eigen::Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).normalized();
                     t = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
                     
                     // Check if fine optimization made reasonable progress
                     if (summary.final_cost < summary.initial_cost * 0.9) {
                         optimization_success = true;
                     }
                 }
             }
         } else {
             // Standard single-resolution approach
             ceres::Problem problem;
             ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
             ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization();
             
             // Add parameter blocks with appropriate parameterization
             problem.AddParameterBlock(parameters, 4, quaternion_parameterization);
             problem.AddParameterBlock(parameters + 4, 3);
             
             edge_factor_count = addEdgeFactorsToOptimization(problem, loss_function, parameters, 
                                                          corner_points_less_sharp_, corner_map, false);
             
             plane_factor_count = addPlaneFactorsToOptimization(problem, loss_function, parameters, 
                                                            surface_points_less_flat_, surf_map, false);
             
             // Solve optimization if we have enough constraints
             if (edge_factor_count + plane_factor_count >= 10) {
                 ceres::Solver::Options options;
                 options.linear_solver_type = ceres::DENSE_QR;
                 options.max_num_iterations = max_iterations_;
                 options.minimizer_progress_to_stdout = debug_output_;
                 
                 ceres::Solver::Summary summary;
                 ceres::Solve(options, &problem, &summary);
                 
                 if (debug_output_) {
                     ROS_INFO("Scan-to-map optimization: %d factors, cost %.5f -> %.5f", 
                             edge_factor_count + plane_factor_count, 
                             summary.initial_cost, summary.final_cost);
                 }
                 
                 // Update the pose with the optimization result
                 q = Eigen::Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).normalized();
                 t = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
                 
                 // Check if optimization made reasonable progress
                 if (summary.final_cost < summary.initial_cost * 0.8) {
                     optimization_success = true;
                 }
             } else {
                 ROS_WARN_THROTTLE(1, "Not enough constraints for scan-to-map optimization! Edge: %d, Plane: %d", 
                             edge_factor_count, plane_factor_count);
             }
         }
         
         // Sanity check on final pose - if too far from initial prediction, fall back
         double translation_diff = (t - t_initial).norm();
         double rotation_diff = 2.0 * acos(std::min(1.0, std::abs(q.dot(q_initial))));
         
         if (!optimization_success || translation_diff > 1.0 || rotation_diff > M_PI / 6) {
             ROS_WARN("Registration result looks suspicious (%.2fm, %.2f°), using motion prediction instead.",
                   translation_diff, rotation_diff * 180 / M_PI);
             
             // Fall back to motion prediction
             q = q_initial;
             t = t_initial;
         }
     }
     
     // Improved line fitting function
     bool fitLine(const pcl::PointCloud<PointType>& points, const std::vector<int>& indices,
                Eigen::Vector3d& line_point, Eigen::Vector3d& line_direction, double& fit_error) {
         if (indices.size() < 2) {
             return false;
         }
         
         // Compute centroid
         Eigen::Vector3d centroid(0, 0, 0);
         for (const auto& idx : indices) {
             centroid += Eigen::Vector3d(points[idx].x, points[idx].y, points[idx].z);
         }
         centroid /= indices.size();
         
         // Compute covariance matrix
         Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
         for (const auto& idx : indices) {
             Eigen::Vector3d pt(points[idx].x, points[idx].y, points[idx].z);
             Eigen::Vector3d diff = pt - centroid;
             cov += diff * diff.transpose();
         }
         
         // Perform eigen decomposition
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
         if (eigensolver.info() != Eigen::Success) {
             return false;
         }
         
         // The line direction is the eigenvector with the largest eigenvalue
         line_direction = eigensolver.eigenvectors().col(2).normalized();
         line_point = centroid;
         
         // Check if the eigenvalues indicate a line structure
         double e1 = eigensolver.eigenvalues()[0];
         double e2 = eigensolver.eigenvalues()[1];
         double e3 = eigensolver.eigenvalues()[2];
         
         // Check planarity and linearity metrics
         double planarity = (e2 - e1) / e3;
         double linearity = (e3 - e2) / e3;
         
         if (linearity < 0.5 || planarity > 0.5) {
             return false;  // Not a strong line structure
         }
         
         // Compute fitting error
         fit_error = 0;
         for (const auto& idx : indices) {
             Eigen::Vector3d pt(points[idx].x, points[idx].y, points[idx].z);
             Eigen::Vector3d diff = pt - centroid;
             Eigen::Vector3d cross = diff.cross(line_direction);
             double dist_sq = cross.squaredNorm();
             fit_error += dist_sq;
         }
         fit_error = sqrt(fit_error / indices.size());
         
         // Reject if error is too large
         if (fit_error > 0.2) {  // 20cm max error
             return false;
         }
         
         return true;
     }
     
     // Improved plane fitting function
     bool fitPlane(const pcl::PointCloud<PointType>& points, const std::vector<int>& indices,
                  Eigen::Vector3d& plane_normal, double& plane_d, double& fit_error) {
         if (indices.size() < 3) {
             return false;
         }
         
         // Compute centroid
         Eigen::Vector3d centroid(0, 0, 0);
         for (const auto& idx : indices) {
             centroid += Eigen::Vector3d(points[idx].x, points[idx].y, points[idx].z);
         }
         centroid /= indices.size();
         
         // Compute covariance matrix
         Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
         for (const auto& idx : indices) {
             Eigen::Vector3d pt(points[idx].x, points[idx].y, points[idx].z);
             Eigen::Vector3d diff = pt - centroid;
             cov += diff * diff.transpose();
         }
         
         // Perform eigen decomposition
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);
         if (eigensolver.info() != Eigen::Success) {
             return false;
         }
         
         // The normal is the eigenvector corresponding to the smallest eigenvalue
         plane_normal = eigensolver.eigenvectors().col(0).normalized();
         
         // Compute plane equation: ax + by + cz + d = 0
         plane_d = -plane_normal.dot(centroid);
         
         // Check if the eigenvalues indicate a plane structure
         double e1 = eigensolver.eigenvalues()[0];
         double e2 = eigensolver.eigenvalues()[1];
         double e3 = eigensolver.eigenvalues()[2];
         
         // Compute planarity metric (how planar the points are)
         double planarity = (e2 - e1) / e3;
         
         // Require strong planarity
         if (planarity < 0.6) {
             return false;
         }
         
         // Compute fitting error
         fit_error = 0;
         for (const auto& idx : indices) {
             Eigen::Vector3d pt(points[idx].x, points[idx].y, points[idx].z);
             double dist = std::abs(plane_normal.dot(pt) + plane_d);
             fit_error += dist * dist;
         }
         fit_error = sqrt(fit_error / indices.size());
         
         // Reject if error is too large
         if (fit_error > 0.1) {  // 10cm max error
             return false;
         }
         
         return true;
     }
     
     // Improved edge factors addition optimized for 32-channel LiDAR
     int addEdgeFactorsToOptimization(ceres::Problem& problem, ceres::LossFunction* loss_function,
                                     double* parameters, const pcl::PointCloud<PointType>& corner_cloud, 
                                     const CloudPtr& corner_map, bool use_low_res = false) {
         int factor_count = 0;
         
         // Parameters adjusted for 32-channel LiDAR
         int max_points = use_low_res ? 150 : 300;  // Increased for 32-channel
         int step = std::max(1, static_cast<int>(corner_cloud.size() / max_points));
         
         // Track the quality of each factor to select the best ones
         std::vector<std::pair<double, size_t>> factor_errors;
         factor_errors.reserve(corner_cloud.size() / step + 1);
         
         for (size_t i = 0; i < corner_cloud.size(); i += step) {
             const auto& point = corner_cloud[i];
             
             // Skip points that are too close or too far (common with 32-channel LiDARs)
             float range_sq = point.x * point.x + point.y * point.y + point.z * point.z;
             if (range_sq < 1.0 || range_sq > 10000.0) {  // 1m to 100m range check
                 continue;
             }
             
             // Transform point to map frame for search
             PointType point_transformed;
             transformPointToMap(point, point_transformed, parameters);
             
             std::vector<int> point_search_idx;
             std::vector<float> point_search_dist;
             
             // Search for correspondence in the map (use adaptive search radius)
             double search_radius = use_low_res ? 1.2 : 0.6;  // Larger radius for low resolution
             kdtree_corner_map_->radiusSearch(point_transformed, search_radius, point_search_idx, point_search_dist);
             
             if (point_search_idx.size() < 5) {
                 // Not enough points, try nearest neighbor search
                 kdtree_corner_map_->nearestKSearch(point_transformed, 5, point_search_idx, point_search_dist);
                 
                 if (point_search_idx.size() < 5 || point_search_dist[4] > nearest_feature_dist_sq_) {
                     continue;  // Still not enough good points
                 }
             }
             
             // Fit line to the neighborhood points
             Eigen::Vector3d line_point, line_direction;
             double fit_error;
             
             if (!fitLine(*corner_map, point_search_idx, line_point, line_direction, fit_error)) {
                 continue;  // Failed to fit a good line
             }
             
             // Store fit error and point index
             factor_errors.push_back(std::make_pair(fit_error, i));
         }
         
         // Sort factors by increasing error (best first)
         std::sort(factor_errors.begin(), factor_errors.end());
         
         // Select best factors up to a limit - bias toward higher quality matches
         int max_factors = use_low_res ? 150 : 300;  // Increased for 32-channel
         for (size_t i = 0; i < std::min(max_factors, static_cast<int>(factor_errors.size())); i++) {
             const auto& error_pair = factor_errors[i];
             double error = error_pair.first;
             size_t point_idx = error_pair.second;
             
             // Skip lower quality matches at the end of the list
             if (i > max_factors * 0.8 && error > 0.1) {
                 continue;
             }
             
             const auto& point = corner_cloud[point_idx];
             
             // Re-process the point to get correspondence and line
             PointType point_transformed;
             transformPointToMap(point, point_transformed, parameters);
             
             std::vector<int> point_search_idx;
             std::vector<float> point_search_dist;
             
             // We know this will succeed because we already checked during the first pass
             kdtree_corner_map_->nearestKSearch(point_transformed, 5, point_search_idx, point_search_dist);
             
             Eigen::Vector3d line_point, line_direction;
             double fit_error;
             
             if (!fitLine(*corner_map, point_search_idx, line_point, line_direction, fit_error)) {
                 continue;  // Just in case, this shouldn't happen
             }
             
             // Use line endpoints that are a reasonable distance from center
             Eigen::Vector3d point_a = line_point + 0.1 * line_direction;
             Eigen::Vector3d point_b = line_point - 0.1 * line_direction;
             
             // Weight inversely proportional to error and distance from center
             double range = point.x * point.x + point.y * point.y + point.z * point.z;
             double range_weight = 1.0 / (1.0 + range * 0.01);  // Favor closer points slightly
             double weight = (1.0 / (1.0 + error * 10.0)) * range_weight;
             
             // Create current point vector
             Eigen::Vector3d curr_point(point.x, point.y, point.z);
             
             ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, weight);
             problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
             factor_count++;
         }
         
         return factor_count;
     }
     
     // Improved plane factors addition optimized for 32-channel LiDAR
     int addPlaneFactorsToOptimization(ceres::Problem& problem, ceres::LossFunction* loss_function,
                                      double* parameters, const pcl::PointCloud<PointType>& surf_cloud, 
                                      const CloudPtr& surf_map, bool use_low_res = false) {
         int factor_count = 0;
         
         // Parameters adjusted for 32-channel LiDAR
         int max_points = use_low_res ? 300 : 600;  // Increased for 32-channel
         int step = std::max(1, static_cast<int>(surf_cloud.size() / max_points));
         
         // Track the quality of each factor to select the best ones
         std::vector<std::pair<double, size_t>> factor_errors;
         factor_errors.reserve(surf_cloud.size() / step + 1);
         
         for (size_t i = 0; i < surf_cloud.size(); i += step) {
             const auto& point = surf_cloud[i];
             
             // Skip points that are too close or too far (common with 32-channel LiDARs)
             float range_sq = point.x * point.x + point.y * point.y + point.z * point.z;
             if (range_sq < 1.0 || range_sq > 10000.0) {  // 1m to 100m range check
                 continue;
             }
             
             // Transform point to map frame for search
             PointType point_transformed;
             transformPointToMap(point, point_transformed, parameters);
             
             std::vector<int> point_search_idx;
             std::vector<float> point_search_dist;
             
             // Search for correspondences in the map (use adaptive search radius)
             double search_radius = use_low_res ? 1.2 : 0.6;  // Larger radius for low resolution
             kdtree_surf_map_->radiusSearch(point_transformed, search_radius, point_search_idx, point_search_dist);
             
             if (point_search_idx.size() < 5) {
                 // Not enough points, try nearest neighbor search
                 kdtree_surf_map_->nearestKSearch(point_transformed, 5, point_search_idx, point_search_dist);
                 
                 if (point_search_idx.size() < 5 || point_search_dist[4] > nearest_feature_dist_sq_) {
                     continue;  // Still not enough good points
                 }
             }
             
             // Fit plane to the neighborhood points
             Eigen::Vector3d plane_normal;
             double plane_d, fit_error;
             
             if (!fitPlane(*surf_map, point_search_idx, plane_normal, plane_d, fit_error)) {
                 continue;  // Failed to fit a good plane
             }
             
             // Store fit error and point index
             factor_errors.push_back(std::make_pair(fit_error, i));
         }
         
         // Sort factors by increasing error (best first)
         std::sort(factor_errors.begin(), factor_errors.end());
         
         // Select best factors up to a limit - bias toward higher quality matches
         int max_factors = use_low_res ? 300 : 600;  // Increased for 32-channel
         for (size_t i = 0; i < std::min(max_factors, static_cast<int>(factor_errors.size())); i++) {
             const auto& error_pair = factor_errors[i];
             double error = error_pair.first;
             size_t point_idx = error_pair.second;
             
             // Skip lower quality matches at the end of the list
             if (i > max_factors * 0.8 && error > 0.05) {
                 continue;
             }
             
             const auto& point = surf_cloud[point_idx];
             
             // Re-process the point to get correspondence and plane
             PointType point_transformed;
             transformPointToMap(point, point_transformed, parameters);
             
             std::vector<int> point_search_idx;
             std::vector<float> point_search_dist;
             
             // We know this will succeed because we already checked during the first pass
             kdtree_surf_map_->nearestKSearch(point_transformed, 5, point_search_idx, point_search_dist);
             
             Eigen::Vector3d plane_normal;
             double plane_d, fit_error;
             
             if (!fitPlane(*surf_map, point_search_idx, plane_normal, plane_d, fit_error)) {
                 continue;  // Just in case, this shouldn't happen
             }
             
             // Weight inversely proportional to error and distance from center
             double range = point.x * point.x + point.y * point.y + point.z * point.z;
             double range_weight = 1.0 / (1.0 + range * 0.01);  // Favor closer points slightly
             double weight = (1.0 / (1.0 + error * 20.0)) * range_weight;
             
             // Create current point vector
             Eigen::Vector3d curr_point(point.x, point.y, point.z);
             
             ceres::CostFunction* cost_function = 
                 LidarPlaneFactor::Create(curr_point, plane_normal, plane_d, weight);
             problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
             factor_count++;
         }
         
         return factor_count;
     }
     
     void transformPointToMap(const PointType& point_in, PointType& point_out, const double* parameters) {
         // Extract transformation parameters
         Eigen::Quaterniond q(parameters[3], parameters[0], parameters[1], parameters[2]);
         Eigen::Vector3d t(parameters[4], parameters[5], parameters[6]);
         
         // Transform point from LiDAR frame to map frame
         Eigen::Vector3d p_lidar(point_in.x, point_in.y, point_in.z);
         Eigen::Vector3d p_map = q * p_lidar + t;
         
         point_out.x = p_map.x();
         point_out.y = p_map.y();
         point_out.z = p_map.z();
         point_out.intensity = point_in.intensity;
     }
     
     void updateMaps() {
         std::lock_guard<std::mutex> lock(map_mutex_);
         
         // Add the optimized clouds to the feature maps
         CloudPtr corner_cloud_map(new pcl::PointCloud<PointType>());
         CloudPtr surf_cloud_map(new pcl::PointCloud<PointType>());
         CloudPtr full_cloud_map(new pcl::PointCloud<PointType>());
         
         // Transform the clouds to map frame
         pcl::transformPointCloud(corner_points_less_sharp_, *corner_cloud_map, 
                                t_w_curr_.cast<float>(), q_w_curr_.cast<float>());
         
         pcl::transformPointCloud(surface_points_less_flat_, *surf_cloud_map, 
                                t_w_curr_.cast<float>(), q_w_curr_.cast<float>());
         
         pcl::transformPointCloud(laser_cloud_in_, *full_cloud_map, 
                                t_w_curr_.cast<float>(), q_w_curr_.cast<float>());
         
         // Downsample before adding to maps - more aggressive for 32-channel
         CloudPtr corner_cloud_downsampled(new pcl::PointCloud<PointType>());
         CloudPtr surf_cloud_downsampled(new pcl::PointCloud<PointType>());
         
         downsize_filter_corner_.setInputCloud(corner_cloud_map);
         downsize_filter_corner_.filter(*corner_cloud_downsampled);
         
         downsize_filter_surf_.setInputCloud(surf_cloud_map);
         downsize_filter_surf_.filter(*surf_cloud_downsampled);
         
         // Add to ring buffers
         corner_map_buffer_.push(corner_cloud_downsampled);
         surf_map_buffer_.push(surf_cloud_downsampled);
         
         // Handle global map more efficiently for 32-channel LiDAR
         // Randomly sample points to add to global map to prevent excessive growth
         if (is_velodyne32_) {
             // Add only a subset of points to global map
             CloudPtr sampled_cloud(new pcl::PointCloud<PointType>());
             int sample_step = 5;  // Add only every 5th point
             
             for (size_t i = 0; i < full_cloud_map->size(); i += sample_step) {
                 sampled_cloud->push_back((*full_cloud_map)[i]);
             }
             
             *global_map_ += *sampled_cloud;
         } else {
             *global_map_ += *full_cloud_map;
         }
         
         // Control global map size by occasional downsampling - more frequent for 32-channel
         static int map_update_counter = 0;
         int downsample_frequency = is_velodyne32_ ? 5 : 10;  // More frequent for 32-channel
         
         if (map_update_counter++ % downsample_frequency == 0 && global_map_->size() > 100000) {
             CloudPtr global_map_downsampled(new pcl::PointCloud<PointType>());
             downsize_filter_map_.setInputCloud(global_map_);
             downsize_filter_map_.filter(*global_map_downsampled);
             global_map_ = global_map_downsampled;
         }
         
         // Mark KD-trees for update
         kdtree_needs_update_ = true;
     }
     
     void publishResults(const ros::Time& stamp) {
         // Publish odometry
         nav_msgs::Odometry odom;
         odom.header.stamp = stamp;
         odom.header.frame_id = fixed_frame_id_;
         odom.child_frame_id = lidar_frame_id_;
         
         // Set odometry pose
         odom.pose.pose.orientation.x = q_w_curr_.x();
         odom.pose.pose.orientation.y = q_w_curr_.y();
         odom.pose.pose.orientation.z = q_w_curr_.z();
         odom.pose.pose.orientation.w = q_w_curr_.w();
         odom.pose.pose.position.x = t_w_curr_.x();
         odom.pose.pose.position.y = t_w_curr_.y();
         odom.pose.pose.position.z = t_w_curr_.z();
         
         // Velocity is not computed in direct scan-to-map approach
         odom_pub_.publish(odom);
         
         // Publish TF transform using tf2
         geometry_msgs::TransformStamped transform_stamped;
         transform_stamped.header.stamp = stamp;
         transform_stamped.header.frame_id = fixed_frame_id_;
         transform_stamped.child_frame_id = lidar_frame_id_;
         
         transform_stamped.transform.translation.x = t_w_curr_.x();
         transform_stamped.transform.translation.y = t_w_curr_.y();
         transform_stamped.transform.translation.z = t_w_curr_.z();
         
         transform_stamped.transform.rotation.w = q_w_curr_.w();
         transform_stamped.transform.rotation.x = q_w_curr_.x();
         transform_stamped.transform.rotation.y = q_w_curr_.y();
         transform_stamped.transform.rotation.z = q_w_curr_.z();
         
         tf_broadcaster_.sendTransform(transform_stamped);
         
         // Publish path
         geometry_msgs::PoseStamped pose_stamped;
         pose_stamped.header.stamp = stamp;
         pose_stamped.header.frame_id = fixed_frame_id_;
         pose_stamped.pose.orientation.x = q_w_curr_.x();
         pose_stamped.pose.orientation.y = q_w_curr_.y();
         pose_stamped.pose.orientation.z = q_w_curr_.z();
         pose_stamped.pose.orientation.w = q_w_curr_.w();
         pose_stamped.pose.position.x = t_w_curr_.x();
         pose_stamped.pose.position.y = t_w_curr_.y();
         pose_stamped.pose.position.z = t_w_curr_.z();
         
         path_.header.stamp = stamp;
         path_.header.frame_id = fixed_frame_id_;
         path_.poses.push_back(pose_stamped);
         
         // Limit path size for efficiency
         if (path_.poses.size() > 1000) {
             path_.poses.erase(path_.poses.begin());
         }
         
         path_pub_.publish(path_);
         
         // Publish feature clouds
         publishCloud(cloud_edge_pub_, corner_points_sharp_, stamp, lidar_frame_id_);
         publishCloud(cloud_surf_pub_, surface_points_flat_, stamp, lidar_frame_id_);
         publishCloud(cloud_full_pub_, laser_cloud_in_, stamp, lidar_frame_id_);
         
         // Publish map occasionally for visualization - less frequent for 32-channel
         static int map_publish_counter = 0;
         int publish_frequency = is_velodyne32_ ? 20 : 10;  // Less frequent for 32-channel
         
         if (map_publish_counter++ % publish_frequency == 0 && map_pub_.getNumSubscribers() > 0) {
             CloudPtr map_cloud_ds(new pcl::PointCloud<PointType>());
             // More aggressive downsampling for visualization
             pcl::VoxelGrid<PointType> vis_filter;
             vis_filter.setLeafSize(filter_map_leaf_size_ * 1.5, filter_map_leaf_size_ * 1.5, filter_map_leaf_size_ * 1.5);
             vis_filter.setInputCloud(global_map_);
             vis_filter.filter(*map_cloud_ds);
             publishCloud(map_pub_, *map_cloud_ds, stamp, fixed_frame_id_);
         }
     }
 
     void publishCloud(const ros::Publisher& publisher, const pcl::PointCloud<PointType>& cloud, 
                      const ros::Time& stamp, const std::string& frame_id) {
         if (cloud.empty() || publisher.getNumSubscribers() == 0) {
             return;
         }
         
         sensor_msgs::PointCloud2 cloud_msg;
         pcl::toROSMsg(cloud, cloud_msg);
         cloud_msg.header.stamp = stamp;
         cloud_msg.header.frame_id = frame_id;
         publisher.publish(cloud_msg);
     }
 
     void saveGlobalMap() {
         if (global_map_->empty()) {
             ROS_WARN("Global map is empty, not saving.");
             return;
         }
         
         // Downsample the global map for saving
         CloudPtr global_map_filtered(new pcl::PointCloud<PointType>());
         pcl::VoxelGrid<PointType> downsize_filter_save;
         downsize_filter_save.setLeafSize(filter_map_leaf_size_ * 2, filter_map_leaf_size_ * 2, filter_map_leaf_size_ * 2);
         downsize_filter_save.setInputCloud(global_map_);
         downsize_filter_save.filter(*global_map_filtered);
         
         // Save to PCD file
         pcl::io::savePCDFileBinary(map_save_path_, *global_map_filtered);
         ROS_INFO("Global map saved to %s with %zu points.", map_save_path_.c_str(), global_map_filtered->size());
     }
     
     // Print current pose to terminal
     void printCurrentPose() {
         // Get Euler angles in degrees
         Eigen::Vector3d euler_angles = quaternionToEulerDegrees(q_w_curr_);
         
         // Print to terminal with fixed formatting
         std::stringstream ss;
         ss << std::fixed << std::setprecision(3);
         ss << "Position [x,y,z]: [" << t_w_curr_.x() << ", " << t_w_curr_.y() << ", " << t_w_curr_.z() << "] m, ";
         ss << "Orientation [roll,pitch,yaw]: [" << euler_angles.x() << ", " << euler_angles.y() << ", " << euler_angles.z() << "] deg";
         
         std::cout << "\r" << ss.str() << std::flush;
     }
     
     // Write pose to file
     void writePoseToFile(const ros::Time& stamp) {
         if (!pose_file_.is_open()) return;
         
         Eigen::Vector3d euler = quaternionToEulerDegrees(q_w_curr_);
         
         pose_file_ << std::fixed << std::setprecision(9)
                   << stamp.toSec() << " "
                   << std::setprecision(6) 
                   << t_w_curr_.x() << " " << t_w_curr_.y() << " " << t_w_curr_.z() << " "
                   << q_w_curr_.x() << " " << q_w_curr_.y() << " " << q_w_curr_.z() << " " << q_w_curr_.w() << " "
                   << euler.x() << " " << euler.y() << " " << euler.z() << std::endl;
     }
     
     // Timer utilities
     void startTimer(const std::string& name) {
         if (!timing_output_) return;
         timers_.emplace_back(name);
     }
     
     void stopTimer() {
         if (!timing_output_ || timers_.empty()) return;
         timers_.pop_back();
     }
     
     void printTimers() {
         if (!timing_output_ || timers_.empty()) return;
         
         std::stringstream ss;
         ss << "Timing: ";
         for (const auto& timer : timers_) {
             ss << timer.name << "=" << timer.elapsed() << "ms ";
         }
         ROS_INFO_STREAM(ss.str());
     }
 };
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "fast32_loam");
     Fast32Loam fast32_loam;
     ros::spin();
     return 0;
 }