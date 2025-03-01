#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

// ROS headers
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

// NDT cell structure
struct NDTCell {
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix3d inverse_covariance;
    double det_covariance;
    int point_count;

    NDTCell() : 
        mean(Eigen::Vector3d::Zero()), 
        covariance(Eigen::Matrix3d::Identity()),
        inverse_covariance(Eigen::Matrix3d::Identity()),
        det_covariance(1.0), 
        point_count(0) {}
};

class NDTCostFunction : public ceres::SizedCostFunction<1, 6> {
public:
    NDTCostFunction(const Eigen::Vector3d& point, const NDTCell& cell)
        : point_(point), cell_(cell) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        
        // Extract transformation parameters (3 for translation, 3 for rotation in angle-axis form)
        Eigen::Vector3d translation(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d rotation_aa(parameters[0][3], parameters[0][4], parameters[0][5]);
        
        // Convert angle-axis to rotation matrix
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
        Eigen::Vector3d diff = transformed_point - cell_.mean;
        
        // Compute the NDT score: negative log-likelihood based on Normal distribution
        double md2 = diff.transpose() * cell_.inverse_covariance * diff;
        
        // Apply robust loss function
        if (md2 > 1.0) {
            md2 = 2.0 * sqrt(md2) - 1.0;  // Smooth L1 Loss (Huber-like)
        }
        
        // We minimize the negative log-likelihood
        residuals[0] = 0.5 * md2;
        
        // Compute Jacobians if requested
        if (jacobians && jacobians[0]) {
            // The gradient of the score with respect to the transformed point
            Eigen::Vector3d score_gradient;
            
            if (md2 > 1.0) {
                // For robust loss, adjust gradient
                score_gradient = (cell_.inverse_covariance * diff) / sqrt(diff.transpose() * cell_.inverse_covariance * diff);
            } else {
                score_gradient = cell_.inverse_covariance * diff;
            }
            
            // Jacobian with respect to translation (direct mapping)
            jacobians[0][0] = score_gradient.x();
            jacobians[0][1] = score_gradient.y();
            jacobians[0][2] = score_gradient.z();
            
            // Jacobian with respect to rotation parameters
            // Create skew-symmetric matrix for the cross product
            Eigen::Matrix3d skew_p;
            skew_p << 0, -point_.z(), point_.y(),
                      point_.z(), 0, -point_.x(),
                      -point_.y(), point_.x(), 0;
            
            // Compute rotation Jacobian
            Eigen::Vector3d rot_jacobian;
            if (angle < 1e-10) {
                // For small angles use direct cross product
                rot_jacobian = -skew_p.transpose() * score_gradient;
            } else {
                // For larger angles, the Jacobian is more complex
                // We use rotated point in the cross product
                Eigen::Vector3d p_rot = rotation_matrix * point_;
                Eigen::Matrix3d skew_rot_p;
                skew_rot_p << 0, -p_rot.z(), p_rot.y(),
                               p_rot.z(), 0, -p_rot.x(),
                               -p_rot.y(), p_rot.x(), 0;
                
                rot_jacobian = -skew_rot_p.transpose() * score_gradient;
            }
            
            jacobians[0][3] = rot_jacobian.x();
            jacobians[0][4] = rot_jacobian.y();
            jacobians[0][5] = rot_jacobian.z();
        }
        
        return true;
    }

private:
    Eigen::Vector3d point_;
    NDTCell cell_;
};

class HighAccuracyNDT {
public:
    HighAccuracyNDT(ros::NodeHandle& nh) : 
        nh_(nh), 
        has_target_cloud_(false), 
        processing_thread_active_(true),
        frames_processed_(0),
        rng_(std::random_device()()) {
            
        // Parameters for high-accuracy NDT
        downsample_resolution_ = 0.1;    // 10cm for downsampling
        ndt_resolution_coarse_ = 2.0;    // 2m for coarse grid
        ndt_resolution_fine_ = 0.5;      // 0.5m for fine grid
        max_iterations_ = 35;            // Optimization iterations
        min_range_ = 0.5;                // Minimum range filter
        max_range_ = 100.0;              // Maximum range filter
        min_points_per_voxel_ = 6;       // Minimum points per voxel
        outlier_ratio_ = 0.55;           // Robust loss parameter
        transformation_epsilon_ = 0.01;  // Convergence threshold
        use_initial_guess_ = true;       // Use previous transform as initial
        fixed_frame_ = "map";            // Fixed frame name
        use_direct_search_ = false;      // Direct search optimization
        
        // Initialize ROS interfaces
        cloud_sub_ = nh_.subscribe("/velodyne_points", 100, &HighAccuracyNDT::cloudCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("ndt_pose", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("ndt_odom", 10);
        queue_size_pub_ = nh_.advertise<std_msgs::Int32>("queue_size", 10);
        frames_processed_pub_ = nh_.advertise<std_msgs::Int32>("frames_processed", 10);
        registered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("registered_cloud", 10);
        
        // Initialize pose
        current_pose_ = Eigen::Matrix4d::Identity();
        previous_transform_ = Eigen::Matrix4d::Identity();
        
        // Initialize cloud pointers
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        keyframe_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        // Start processing thread
        processing_thread_ = std::thread(&HighAccuracyNDT::processQueue, this);
        
        // Start status publisher timer
        status_timer_ = nh_.createTimer(ros::Duration(1.0), &HighAccuracyNDT::publishStatus, this);
        
        ROS_INFO("High-Accuracy NDT Registration initialized");
        ROS_INFO("Downsample: %.2fm, NDT resolutions: [%.1f, %.1f]", 
                downsample_resolution_, ndt_resolution_coarse_, ndt_resolution_fine_);
    }
    
    ~HighAccuracyNDT() {
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
    ros::Publisher queue_size_pub_;
    ros::Publisher frames_processed_pub_;
    ros::Publisher registered_cloud_pub_;
    ros::Timer status_timer_;
    tf::TransformBroadcaster tf_broadcaster_;
    
    // Parameters
    double downsample_resolution_;
    double ndt_resolution_coarse_;
    double ndt_resolution_fine_;
    int max_iterations_;
    double min_range_;
    double max_range_;
    int min_points_per_voxel_;
    double outlier_ratio_;
    double transformation_epsilon_;
    bool use_initial_guess_;
    std::string fixed_frame_;
    bool use_direct_search_;
    
    // State variables
    int frames_processed_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe_cloud_;
    bool has_target_cloud_;
    Eigen::Matrix4d current_pose_;
    Eigen::Matrix4d previous_transform_;
    ros::Time last_cloud_time_;
    
    // Random number generator for sampling
    std::mt19937 rng_;
    
    // Thread-related
    std::queue<sensor_msgs::PointCloud2::ConstPtr> cloud_queue_;
    std::mutex queue_mutex_;
    std::thread processing_thread_;
    bool processing_thread_active_;
    
    // NDT Grid structure
    struct VoxelGrid {
        double resolution;
        std::unordered_map<uint64_t, NDTCell> cells;
        
        VoxelGrid(double res) : resolution(res) {}
        
        // Get voxel index from point coordinates
        uint64_t getVoxelIndex(double x, double y, double z) const {
            int i = static_cast<int>(std::floor(x / resolution));
            int j = static_cast<int>(std::floor(y / resolution));
            int k = static_cast<int>(std::floor(z / resolution));
            
            // Use spatial hashing for unique 64-bit index
            uint64_t hash = ((static_cast<uint64_t>(i) << 20) ^ 
                            (static_cast<uint64_t>(j) << 10) ^ 
                            static_cast<uint64_t>(k));
            return hash;
        }
        
        // Get voxel containing a point
        const NDTCell* getVoxel(double x, double y, double z) const {
            uint64_t idx = getVoxelIndex(x, y, z);
            auto it = cells.find(idx);
            if (it != cells.end()) {
                return &(it->second);
            }
            return nullptr;
        }
    };
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Limit queue size
        if (cloud_queue_.size() > 100) {
            cloud_queue_.pop();
        }
        
        cloud_queue_.push(cloud_msg);
    }
    
    void publishStatus(const ros::TimerEvent&) {
        std_msgs::Int32 queue_size_msg;
        std_msgs::Int32 frames_processed_msg;
        
        // Get current queue size
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_size_msg.data = cloud_queue_.size();
        }
        
        frames_processed_msg.data = frames_processed_;
        
        // Publish statistics
        queue_size_pub_.publish(queue_size_msg);
        frames_processed_pub_.publish(frames_processed_msg);
        
        // Log status information
        ROS_INFO("Status: %d frames processed, %d frames in queue", 
                frames_processed_, queue_size_msg.data);
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
                auto start_time = std::chrono::steady_clock::now();
                
                // Convert ROS message to PCL point cloud
                pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::fromROSMsg(*cloud_msg, *raw_cloud);
                
                ROS_INFO("Processing frame %d with %ld points", frames_processed_ + 1, raw_cloud->size());
                
                // Preprocess cloud: filter and downsample
                pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                preProcessPointCloud(raw_cloud, filtered_cloud);
                
                if (filtered_cloud->size() < 100) {
                    ROS_WARN("Too few points after filtering (%ld), skipping frame", filtered_cloud->size());
                    continue;
                }
                
                // Perform registration
                if (!has_target_cloud_) {
                    // First frame, just store it
                    target_cloud_ = filtered_cloud;
                    keyframe_cloud_ = filtered_cloud;
                    has_target_cloud_ = true;
                    frames_processed_++;
                    ROS_INFO("First frame stored as reference");
                    
                    // Publish first frame
                    publishRegisteredCloud(filtered_cloud, cloud_msg->header);
                    publishPose(cloud_msg->header);
                } else {
                    // Perform scan-to-scan registration
                    
                    // Initial guess
                    Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity();
                    if (use_initial_guess_ && frames_processed_ > 0) {
                        initial_guess = previous_transform_;
                    }
                    
                    // Multi-resolution NDT
                    // 1. Create coarse NDT grid
                    VoxelGrid coarse_grid = createNDTGrid(target_cloud_, ndt_resolution_coarse_);
                    
                    // 2. Perform alignment at coarse level
                    Eigen::Matrix4d coarse_transform = alignPointClouds(filtered_cloud, coarse_grid, initial_guess);
                    
                    // 3. Create fine NDT grid
                    VoxelGrid fine_grid = createNDTGrid(target_cloud_, ndt_resolution_fine_);
                    
                    // 4. Perform alignment at fine level
                    Eigen::Matrix4d transform = alignPointClouds(filtered_cloud, fine_grid, coarse_transform);
                    
                    // Store transform for next frame
                    previous_transform_ = transform;
                    
                    // Update global pose
                    current_pose_ = transform * current_pose_;
                    
                    // Transform cloud to global frame
                    pcl::PointCloud<pcl::PointXYZ>::Ptr registered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::transformPointCloud(*filtered_cloud, *registered_cloud, current_pose_);
                    
                    // Update target for next iteration
                    target_cloud_ = filtered_cloud;
                    
                    // Publish results
                    publishRegisteredCloud(registered_cloud, cloud_msg->header);
                    publishPose(cloud_msg->header);
                    
                    frames_processed_++;
                    
                    // Display transform for debugging
                    Eigen::Vector3d translation = transform.block<3,1>(0,3);
                    Eigen::Matrix3d rotation = transform.block<3,3>(0,0);
                    Eigen::AngleAxisd aaxis(rotation);
                    
                    ROS_INFO("Transform: [%.3f, %.3f, %.3f] rot: %.2f degrees", 
                            translation.x(), translation.y(), translation.z(),
                            aaxis.angle() * 180.0 / M_PI);
                }
                
                auto end_time = std::chrono::steady_clock::now();
                double process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count() / 1000.0;
                
                ROS_INFO("Frame processed in %.3f seconds", process_time);
                
                // Update time
                last_cloud_time_ = cloud_msg->header.stamp;
            }
            
            // Short sleep to avoid CPU spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void preProcessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
        
        // Remove NaN points
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_nan_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*input, *no_nan_cloud, indices);
        
        // Filter by range
        pcl::PointCloud<pcl::PointXYZ>::Ptr range_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        range_filtered->reserve(no_nan_cloud->size());
        
        for (const auto& point : no_nan_cloud->points) {
            double range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range >= min_range_ && range <= max_range_) {
                range_filtered->push_back(point);
            }
        }
        
        // Remove statistical outliers
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_outliers(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(range_filtered);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_outliers);
        
        // Downsample with voxel grid
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(filtered_outliers);
        voxel_grid.setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
        voxel_grid.filter(*output);
        
        ROS_INFO("Pre-processing: %ld -> %ld points", input->size(), output->size());
    }
    
    // Create NDT grid from point cloud
    VoxelGrid createNDTGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution) {
        VoxelGrid grid(resolution);
        
        // First pass: collect points for each voxel
        std::unordered_map<uint64_t, std::vector<Eigen::Vector3d>> voxel_points;
        
        for (const auto& point : cloud->points) {
            uint64_t idx = grid.getVoxelIndex(point.x, point.y, point.z);
            voxel_points[idx].push_back(Eigen::Vector3d(point.x, point.y, point.z));
        }
        
        // Second pass: compute NDT cells
        for (const auto& [idx, points] : voxel_points) {
            if (points.size() < min_points_per_voxel_) {
                continue;  // Skip cells with too few points
            }
            
            NDTCell cell;
            cell.point_count = points.size();
            
            // Compute mean
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (const auto& p : points) {
                mean += p;
            }
            mean /= points.size();
            cell.mean = mean;
            
            // Compute covariance
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (const auto& p : points) {
                Eigen::Vector3d diff = p - mean;
                covariance += diff * diff.transpose();
            }
            
            // Unbiased estimate
            if (points.size() > 1) {
                covariance /= (points.size() - 1);
            }
            
            // Regularize covariance matrix
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            Eigen::Vector3d eigenvalues = solver.eigenvalues();
            Eigen::Matrix3d eigenvectors = solver.eigenvectors();
            
            // Apply minimum eigenvalue constraint
            double max_eigenvalue = eigenvalues.maxCoeff();
            double min_allowed = std::max(0.01 * max_eigenvalue, 0.001);
            
            for (int i = 0; i < 3; i++) {
                if (eigenvalues(i) < min_allowed) {
                    eigenvalues(i) = min_allowed;
                }
            }
            
            // Reconstruct covariance
            cell.covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            
            // Compute inverse and determinant
            cell.inverse_covariance = cell.covariance.inverse();
            cell.det_covariance = cell.covariance.determinant();
            
            // Only add if valid
            if (cell.det_covariance > 0 && 
                std::isfinite(cell.inverse_covariance.sum()) &&
                std::isfinite(cell.det_covariance)) {
                
                grid.cells[idx] = cell;
            }
        }
        
        ROS_INFO("Created NDT grid with %zu cells (res=%.2f)", grid.cells.size(), resolution);
        return grid;
    }
    
    // Align a point cloud using Ceres
    Eigen::Matrix4d alignPointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
                                    const VoxelGrid& target_grid,
                                    const Eigen::Matrix4d& initial_guess) {
        
        // Set up Ceres problem
        ceres::Problem problem;
        
        // Extract transformation parameters from initial guess
        Eigen::Matrix3d init_rotation = initial_guess.block<3, 3>(0, 0);
        Eigen::Vector3d init_translation = initial_guess.block<3, 1>(0, 3);
        
        // Convert rotation matrix to angle-axis
        Eigen::AngleAxisd angle_axis(init_rotation);
        double angle = angle_axis.angle();
        Eigen::Vector3d axis = angle_axis.axis();
        Eigen::Vector3d rotation_aa = angle * axis;
        
        // Parameters: tx, ty, tz, rx, ry, rz (angle-axis)
        double transform_params[6] = {
            init_translation.x(), init_translation.y(), init_translation.z(),
            rotation_aa.x(), rotation_aa.y(), rotation_aa.z()
        };
        
        // Add cost functions for point-to-distribution matches
        int num_residuals = 0;
        
        // Select random subset of points for efficiency
        std::vector<int> indices(source->size());
        for (size_t i = 0; i < source->size(); i++) {
            indices[i] = i;
        }
        
        // Shuffle indices for uniform sampling
        std::shuffle(indices.begin(), indices.end(), rng_);
        
        // Use a limited number of points for efficiency
        int max_points = std::min(static_cast<int>(source->size()), 2000);
        
        for (int i = 0; i < max_points; i++) {
            const auto& point = source->points[indices[i]];
            
            // Create point vector
            Eigen::Vector3d p_src(point.x, point.y, point.z);
            
            // Find corresponding voxel 
            // First transform using initial guess for better matches
            Eigen::Vector3d p_init = init_rotation * p_src + init_translation;
            
            // Find corresponding cell in target grid
            const NDTCell* cell = target_grid.getVoxel(p_init.x(), p_init.y(), p_init.z());
            
            if (cell) {
                // Add cost function for this match
                ceres::CostFunction* cost_fn = new NDTCostFunction(p_src, *cell);
                problem.AddResidualBlock(cost_fn, new ceres::HuberLoss(0.1), transform_params);
                num_residuals++;
            }
        }
        
        if (num_residuals < 10) {
            ROS_WARN("Not enough matches (%d), returning initial guess", num_residuals);
            return initial_guess;
        }
        
        // Configure the solver
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = max_iterations_;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // Extract the optimized transform
        Eigen::Vector3d translation(transform_params[0], transform_params[1], transform_params[2]);
        Eigen::Vector3d rot_aa(transform_params[3], transform_params[4], transform_params[5]);
        
        Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
        
        // Set translation
        result.block<3, 1>(0, 3) = translation;
        
        // Set rotation
        double rot_angle = rot_aa.norm();
        if (rot_angle > 1e-10) {
            Eigen::AngleAxisd rotation(rot_angle, rot_aa / rot_angle);
            result.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        }
        
        ROS_INFO("NDT alignment with %d matches converged after %d iterations", 
                 num_residuals, summary.iterations.size());
        
        return result;
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
    
    void publishPose(const std_msgs::Header& header) {
        // Extract translation and rotation
        Eigen::Vector3d translation = current_pose_.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation_matrix = current_pose_.block<3, 3>(0, 0);
        
        // Convert to quaternion
        Eigen::Quaterniond quaternion(rotation_matrix);
        quaternion.normalize();
        
        // Create pose message
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
        
        // Create odometry message
        nav_msgs::Odometry odom_msg;
        odom_msg.header = header;
        odom_msg.header.frame_id = fixed_frame_;
        odom_msg.child_frame_id = "velodyne";
        
        odom_msg.pose.pose = pose_msg.pose;
        
        odom_pub_.publish(odom_msg);
        
        // Publish TF transform
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
        tf::Quaternion tf_quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
        transform.setRotation(tf_quaternion);
        
        tf_broadcaster_.sendTransform(tf::StampedTransform(
            transform, header.stamp, fixed_frame_, "velodyne"));
        
        // Print current pose
        ROS_INFO("Current pose: [%.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]",
                translation.x(), translation.y(), translation.z(),
                quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "high_accuracy_ndt");
    ros::NodeHandle nh("~");
    
    HighAccuracyNDT ndt_registration(nh);
    
    ros::spin();
    
    return 0;
}