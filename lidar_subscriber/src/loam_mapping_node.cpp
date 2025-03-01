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
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
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

// NDT cell structure with proper statistics
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

// Manually optimized NDT implementation for maximum accuracy
class HighAccuracyNDT {
public:
    HighAccuracyNDT(ros::NodeHandle& nh) : 
        nh_(nh), 
        has_target_cloud_(false), 
        processing_thread_active_(true),
        frames_processed_(0),
        rng_(std::random_device()()) {
            
        // ====== Parameters for high-accuracy NDT ======
        
        // Load parameters from ROS parameter server with defaults
        nh_.param<double>("downsample_resolution", downsample_resolution_, 0.1);  // 10cm voxel grid
        nh_.param<double>("ndt_resolution_coarse", ndt_resolution_coarse_, 2.0);  // Coarse resolution
        nh_.param<double>("ndt_resolution_fine", ndt_resolution_fine_, 1.0);      // Fine resolution
        nh_.param<int>("max_iterations", max_iterations_, 50);                    // Optimization iterations
        nh_.param<double>("min_range", min_range_, 0.5);                          // Min range filter
        nh_.param<double>("max_range", max_range_, 100.0);                        // Max range filter
        nh_.param<int>("min_points_per_voxel", min_points_per_voxel_, 6);         // Min points for covariance
        nh_.param<bool>("use_initial_guess", use_initial_guess_, true);           // Use previous transform
        nh_.param<double>("epsilon", epsilon_, 0.01);                             // Convergence threshold
        nh_.param<double>("voxel_importance_coeff", voxel_importance_, 0.1);      // Importance coefficient
        nh_.param<std::string>("fixed_frame", fixed_frame_, "map");               // Fixed frame name
        nh_.param<double>("outlier_ratio", outlier_ratio_, 0.55);                 // Robust loss parameter
        nh_.param<double>("trans_epsilon", transformation_epsilon_, 0.01);        // Transformation threshold
        nh_.param<double>("step_size", step_size_, 0.1);                          // Optimization step size
        nh_.param<bool>("use_direct_search", use_direct_search_, true);           // Use direct search optimization
        
        // Initialize publishers and subscribers
        cloud_sub_ = nh_.subscribe("/velodyne_points", 100, &HighAccuracyNDT::cloudCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("ndt_pose", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("ndt_odom", 10);
        queue_size_pub_ = nh_.advertise<std_msgs::Int32>("queue_size", 10);
        frames_processed_pub_ = nh_.advertise<std_msgs::Int32>("frames_processed", 10);
        registered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("registered_cloud", 10);
        
        // Initialize pose
        current_pose_ = Eigen::Matrix4f::Identity();
        prev_transform_ = Eigen::Matrix4f::Identity();
        
        // Target cloud pointer initialization
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        // Start processing thread
        processing_thread_ = std::thread(&HighAccuracyNDT::processQueue, this);
        
        // Start status publisher timer
        status_timer_ = nh_.createTimer(ros::Duration(1.0), &HighAccuracyNDT::publishStatus, this);
        
        ROS_INFO("High-Accuracy NDT Registration initialized");
        ROS_INFO("Downsample resolution: %.3f, NDT resolutions: [%.2f, %.2f]", 
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
    bool use_initial_guess_;
    double epsilon_;
    double voxel_importance_;
    std::string fixed_frame_;
    double outlier_ratio_;
    double transformation_epsilon_;
    double step_size_;
    bool use_direct_search_;
    
    // State variables
    int frames_processed_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    bool has_target_cloud_;
    Eigen::Matrix4f current_pose_;
    Eigen::Matrix4f prev_transform_;
    ros::Time last_cloud_time_;
    
    // Random number generator for sampling
    std::mt19937 rng_;
    
    // Thread-related
    std::queue<sensor_msgs::PointCloud2::ConstPtr> cloud_queue_;
    std::mutex queue_mutex_;
    std::thread processing_thread_;
    bool processing_thread_active_;
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Check queue size
        if (cloud_queue_.size() > 100) {
            cloud_queue_.pop();  // Drop oldest message if queue too large
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
                    has_target_cloud_ = true;
                    frames_processed_++;
                    ROS_INFO("First frame stored as reference");
                    
                    // Publish first frame
                    publishRegisteredCloud(filtered_cloud, cloud_msg->header);
                    publishPose(cloud_msg->header);
                } else {
                    // Align current frame to target frame
                    
                    // Initial guess
                    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
                    if (use_initial_guess_ && frames_processed_ > 0) {
                        initial_guess = prev_transform_;
                    }
                    
                    // Create voxel grid for target cloud
                    VoxelGrid target_grid = createVoxelGrid(target_cloud_, ndt_resolution_coarse_);
                    
                    // Perform coarse-to-fine alignment
                    Eigen::Matrix4f coarse_transform = alignPointCloud(filtered_cloud, target_grid, initial_guess);
                    
                    // Create fine voxel grid for target
                    VoxelGrid fine_target_grid = createVoxelGrid(target_cloud_, ndt_resolution_fine_);
                    
                    // Perform fine alignment
                    Eigen::Matrix4f transform = alignPointCloud(filtered_cloud, fine_target_grid, coarse_transform);
                    
                    // Store transform for next frame
                    prev_transform_ = transform;
                    
                    // Update global pose
                    current_pose_ = transform * current_pose_;
                    
                    // Transform current frame to global frame
                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::transformPointCloud(*filtered_cloud, *transformed_cloud, current_pose_);
                    
                    // Update target for next iteration
                    target_cloud_ = filtered_cloud;
                    
                    // Publish results
                    publishRegisteredCloud(transformed_cloud, cloud_msg->header);
                    publishPose(cloud_msg->header);
                    
                    frames_processed_++;
                    
                    // Display transform for debugging
                    Eigen::Vector3f translation = transform.block<3,1>(0,3);
                    Eigen::Matrix3f rotation = transform.block<3,3>(0,0);
                    Eigen::AngleAxisf aaxis(rotation);
                    
                    ROS_INFO("Transform: [%.3f, %.3f, %.3f] rot: %.2f degrees", 
                            translation.x(), translation.y(), translation.z(),
                            aaxis.angle() * 180.0f / M_PI);
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
    
    struct VoxelGrid {
        std::unordered_map<size_t, NDTCell> voxels;
        double resolution;
        
        VoxelGrid(double res) : resolution(res) {}
        
        // Hash function for getting voxel index
        size_t getVoxelIndex(double x, double y, double z) const {
            int i = static_cast<int>(std::floor(x / resolution));
            int j = static_cast<int>(std::floor(y / resolution));
            int k = static_cast<int>(std::floor(z / resolution));
            
            // Szudzik's function for spatial hashing
            size_t a = i >= 0 ? 2 * i : -2 * i - 1;
            size_t b = j >= 0 ? 2 * j : -2 * j - 1;
            size_t c = k >= 0 ? 2 * k : -2 * k - 1;
            
            size_t hash = ((a >= b ? a * a + a + b : a + b * b) * 3937) ^ 
                         (c * 257);
                
            return hash;
        }
        
        // Find the cell containing point (x,y,z)
        NDTCell* getCell(double x, double y, double z) {
            size_t idx = getVoxelIndex(x, y, z);
            auto it = voxels.find(idx);
            if (it != voxels.end()) {
                return &(it->second);
            }
            return nullptr;
        }
    };
    
    VoxelGrid createVoxelGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution) {
        VoxelGrid grid(resolution);
        
        // First pass: collect points per voxel
        std::unordered_map<size_t, std::vector<Eigen::Vector3f>> voxel_points;
        
        for (const auto& point : cloud->points) {
            size_t idx = grid.getVoxelIndex(point.x, point.y, point.z);
            voxel_points[idx].push_back(Eigen::Vector3f(point.x, point.y, point.z));
        }
        
        // Second pass: compute statistics for each voxel
        for (const auto& [idx, points] : voxel_points) {
            // Need enough points for good statistics
            if (points.size() < min_points_per_voxel_) {
                continue;
            }
            
            NDTCell cell;
            cell.point_count = points.size();
            
            // Compute mean
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (const auto& p : points) {
                mean += p.cast<double>();
            }
            mean /= static_cast<double>(points.size());
            cell.mean = mean;
            
            // Compute covariance
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (const auto& p : points) {
                Eigen::Vector3d diff = p.cast<double>() - mean;
                covariance += diff * diff.transpose();
            }
            
            // Unbiased estimate
            if (points.size() > 1) {
                covariance /= (points.size() - 1);
            }
            
            // Condition the covariance matrix - regularization
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            Eigen::Vector3d eigenvalues = solver.eigenvalues();
            Eigen::Matrix3d eigenvectors = solver.eigenvectors();
            
            // Apply minimum eigenvalue constraint - addresses degenerate cases
            double max_eigenvalue = eigenvalues.maxCoeff();
            double min_allowed = std::max(0.01 * max_eigenvalue, 0.001);
            
            for (int i = 0; i < 3; i++) {
                if (eigenvalues(i) < min_allowed) {
                    eigenvalues(i) = min_allowed;
                }
            }
            
            // Reconstruct with conditioned eigenvalues
            cell.covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            
            // Compute inverse and determinant
            cell.inverse_covariance = cell.covariance.inverse();
            cell.det_covariance = cell.covariance.determinant();
            
            // Make sure matrix is valid
            if (cell.det_covariance > 0 && 
                std::isfinite(cell.inverse_covariance.sum()) &&
                std::isfinite(cell.det_covariance)) {
                
                grid.voxels[idx] = cell;
            }
        }
        
        ROS_INFO("Created voxel grid with %zu cells (res=%.2f)", grid.voxels.size(), resolution);
        return grid;
    }
    
    void preProcessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
        
        // 1. Remove NaN points
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_nan_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*input, *no_nan_cloud, indices);
        
        // 2. Filter points by range
        pcl::PointCloud<pcl::PointXYZ>::Ptr range_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        range_filtered->reserve(no_nan_cloud->size());
        
        for (const auto& point : no_nan_cloud->points) {
            float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (range >= min_range_ && range <= max_range_) {
                range_filtered->push_back(point);
            }
        }
        
        // 3. Remove statistical outliers
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_outliers(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(range_filtered);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_outliers);
        
        // 4. Downsample with voxel grid
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(filtered_outliers);
        voxel_grid.setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
        voxel_grid.filter(*output);
        
        ROS_INFO("Pre-processing: %ld -> %ld points", input->size(), output->size());
    }
    
    // Core NDT function implementation - manual optimization for highest accuracy
    double computeNDTScore(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                           const VoxelGrid& target_grid,
                           const Eigen::Matrix4f& transform,
                           Eigen::Vector<float, 6>* score_gradient = nullptr,
                           Eigen::Matrix<float, 6, 6>* hessian = nullptr) {
        
        // Initialize score, gradient, and hessian if needed
        double score = 0.0;
        if (score_gradient) {
            score_gradient->setZero();
        }
        if (hessian) {
            hessian->setZero();
        }
        
        // Extract rotation and translation
        Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = transform.block<3, 1>(0, 3);
        
        // Number of points with valid voxel associations
        int valid_points = 0;
        
        // Loop through source cloud points
        for (const auto& point : source_cloud->points) {
            // Transform the point
            Eigen::Vector3f p_src(point.x, point.y, point.z);
            Eigen::Vector3f p_trans = R * p_src + t;
            
            // Find target voxel for this point
            NDTCell* target_cell = target_grid.getCell(p_trans.x(), p_trans.y(), p_trans.z());
            
            if (!target_cell) {
                continue;  // No valid target voxel
            }
            
            // Calculate the difference between the transformed point and voxel mean
            Eigen::Vector3d p_trans_d = p_trans.cast<double>();
            Eigen::Vector3d diff = p_trans_d - target_cell->mean;
            
            // Calculate Mahalanobis distance and score contribution
            double md2 = diff.transpose() * target_cell->inverse_covariance * diff;
            
            // Apply robust loss function to handle outliers
            if (md2 > outlier_ratio_) {
                // If distance is too large, apply constant penalty
                md2 = outlier_ratio_;
            }
            
            // Accumulate score - negative because we want to maximize likelihood
            double cell_score = -0.5 * md2;
            score += cell_score;
            
            // If gradient or hessian are requested, compute derivatives
            if (score_gradient || hessian) {
                // Convert to float for gradient/hessian computation
                Eigen::Vector3f diff_f = diff.cast<float>();
                Eigen::Matrix3f inv_cov_f = target_cell->inverse_covariance.cast<float>();
                
                // Avoid computing for outliers
                if (md2 <= outlier_ratio_) {
                    // Score gradient w.r.t. transformation parameters
                    Eigen::Vector3f grad_trans = -inv_cov_f * diff_f;
                    
                    // Compute partial derivatives
                    if (score_gradient) {
                        // Translation part - directly use the gradient
                        score_gradient->head<3>() += grad_trans;
                        
                        // Rotation part - use cross product
                        Eigen::Vector3f p_cross = p_src.cross(R.transpose() * grad_trans);
                        score_gradient->tail<3>() += p_cross;
                    }
                    
                    // Compute Hessian if needed
                    if (hessian) {
                        // Translation-translation block
                        hessian->block<3, 3>(0, 0) += inv_cov_f;
                        
                        // Rotation-translation and translation-rotation blocks
                        Eigen::Matrix3f skew_p;
                        skew_p << 0, -p_src.z(), p_src.y(),
                                p_src.z(), 0, -p_src.x(),
                                -p_src.y(), p_src.x(), 0;
                        
                        Eigen::Matrix3f rot_trans_hessian = -skew_p * inv_cov_f;
                        hessian->block<3, 3>(3, 0) += rot_trans_hessian;
                        hessian->block<3, 3>(0, 3) += rot_trans_hessian.transpose();
                        
                        // Rotation-rotation block
                        Eigen::Matrix3f p_hat_squared = skew_p * skew_p;
                        hessian->block<3, 3>(3, 3) += p_hat_squared * inv_cov_f.trace();
                    }
                }
            }
            
            valid_points++;
        }
        
        // Return average score
        if (valid_points > 0) {
            double avg_score = score / valid_points;
            
            // Scale gradient and Hessian if computed
            if (score_gradient) {
                *score_gradient /= valid_points;
            }
            if (hessian) {
                *hessian /= valid_points;
                
                // Ensure Hessian is positive definite
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(*hessian);
                Eigen::Matrix<float, 6, 1> eigenvalues = eig.eigenvalues();
                Eigen::Matrix<float, 6, 6> eigenvectors = eig.eigenvectors();
                
                for (int i = 0; i < 6; i++) {
                    if (eigenvalues(i) < 1e-5) {
                        eigenvalues(i) = 1e-5;
                    }
                }
                
                *hessian = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
            }
            
            return avg_score;
        }
        
        return -std::numeric_limits<double>::max(); // No valid points
    }
    
    // Newton's method for optimization
    Eigen::Matrix4f optimizeTransformNewton(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                                          const VoxelGrid& target_grid,
                                          const Eigen::Matrix4f& initial_guess) {
        
        Eigen::Matrix4f current_transform = initial_guess;
        
        // Optimization variables
        Eigen::Vector<float, 6> step;
        Eigen::Vector<float, 6> gradient;
        Eigen::Matrix<float, 6, 6> hessian;
        
        double score = computeNDTScore(source_cloud, target_grid, current_transform, &gradient, &hessian);
        double previous_score = -std::numeric_limits<double>::max();
        
        // Newton optimization
        for (int iter = 0; iter < max_iterations_; iter++) {
            // Break if score isn't improving significantly
            if (iter > 0 && std::abs(score - previous_score) < epsilon_) {
                ROS_INFO("Converged at iteration %d: score diff = %.10f < %.10f", 
                        iter, std::abs(score - previous_score), epsilon_);
                break;
            }
            
            // Compute update step: solve H * step = -g
            Eigen::JacobiSVD<Eigen::Matrix<float, 6, 6>> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
            step = -svd.solve(gradient);
            
            // Apply step size constraint
            float step_norm = step.norm();
            if (step_norm > step_size_) {
                step *= step_size_ / step_norm;
            }
            
            // Convert to transformation matrix
            Eigen::Matrix4f step_transform = Eigen::Matrix4f::Identity();
            
            // Apply translation part
            step_transform.block<3, 1>(0, 3) = step.head<3>();
            
            // Apply rotation part
            Eigen::Vector3f rot_vec = step.tail<3>();
            float rot_angle = rot_vec.norm();
            
            if (rot_angle > 1e-10) {
                Eigen::Vector3f rot_axis = rot_vec / rot_angle;
                Eigen::AngleAxisf rot(rot_angle, rot_axis);
                step_transform.block<3, 3>(0, 0) = rot.toRotationMatrix();
            }
            
            // Apply step: T_new = T_step * T_old
            Eigen::Matrix4f new_transform = step_transform * current_transform;
            
            // Evaluate new score
            previous_score = score;
            score = computeNDTScore(source_cloud, target_grid, new_transform, &gradient, &hessian);
            
            // Update if better
            if (score > previous_score) {
                current_transform = new_transform;
            } else {
                // If not better, try with half the step size
                step *= 0.5;
                
                // Apply smaller step
                step_transform = Eigen::Matrix4f::Identity();
                step_transform.block<3, 1>(0, 3) = step.head<3>();
                
                rot_vec = step.tail<3>();
                rot_angle = rot_vec.norm();
                
                if (rot_angle > 1e-10) {
                    Eigen::Vector3f rot_axis = rot_vec / rot_angle;
                    Eigen::AngleAxisf rot(rot_angle, rot_axis);
                    step_transform.block<3, 3>(0, 0) = rot.toRotationMatrix();
                }
                
                new_transform = step_transform * current_transform;
                score = computeNDTScore(source_cloud, target_grid, new_transform, &gradient, &hessian);
                
                if (score > previous_score) {
                    current_transform = new_transform;
                } else {
                    // Stuck in local minimum or numerical issues
                    break;
                }
            }
            
            // Check if transformation is small enough to indicate convergence
            if (step_norm < transformation_epsilon_) {
                ROS_INFO("Converged at iteration %d: step norm = %.10f < %.10f", 
                        iter, step_norm, transformation_epsilon_);
                break;
            }
        }
        
        return current_transform;
    }
    
    // Direct search algorithm for robustness against local minima
    Eigen::Matrix4f directSearchOptimization(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                                           const VoxelGrid& target_grid,
                                           const Eigen::Matrix4f& initial_guess) {
        
        Eigen::Matrix4f best_transform = initial_guess;
        double best_score = computeNDTScore(source_cloud, target_grid, best_transform);
        
        // Parameters for search
        const int num_directions = 26;  // 3D neighborhood
        
        // Define search directions
        std::vector<Eigen::Vector<float, 6>> directions(num_directions);
        int dir_idx = 0;
        
        // Create search pattern for translations and rotations
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    
                    // Create a direction vector
                    Eigen::Vector<float, 6> dir;
                    dir << 0.05f * dx, 0.05f * dy, 0.05f * dz, 0.02f * dx, 0.02f * dy, 0.02f * dz;
                    directions[dir_idx++] = dir;
                }
            }
        }
        
        // Direct search iterations
        for (int iter = 0; iter < max_iterations_; iter++) {
            bool improved = false;
            
            // Try each direction
            for (const auto& dir : directions) {
                // Convert direction to transformation
                Eigen::Matrix4f dir_transform = Eigen::Matrix4f::Identity();
                
                // Apply translation
                dir_transform.block<3, 1>(0, 3) = dir.head<3>();
                
                // Apply rotation
                Eigen::Vector3f rot_vec = dir.tail<3>();
                float rot_angle = rot_vec.norm();
                
                if (rot_angle > 1e-10) {
                    Eigen::Vector3f rot_axis = rot_vec / rot_angle;
                    Eigen::AngleAxisf rot(rot_angle, rot_axis);
                    dir_transform.block<3, 3>(0, 0) = rot.toRotationMatrix();
                }
                
                // Apply to current best
                Eigen::Matrix4f candidate = dir_transform * best_transform;
                
                // Evaluate score
                double score = computeNDTScore(source_cloud, target_grid, candidate);
                
                // Update if better
                if (score > best_score) {
                    best_score = score;
                    best_transform = candidate;
                    improved = true;
                }
            }
            
            // Stop if no improvement
            if (!improved) {
                break;
            }
        }
        
        return best_transform;
    }
    
    // Main alignment function that calls appropriate optimization
    Eigen::Matrix4f alignPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                                  const VoxelGrid& target_grid,
                                  const Eigen::Matrix4f& initial_guess) {
        
        Eigen::Matrix4f final_transform;
        
        // If direct search is enabled, use it for robustness first
        if (use_direct_search_) {
            ROS_INFO("Using direct search optimization...");
            final_transform = directSearchOptimization(source_cloud, target_grid, initial_guess);
        } else {
            final_transform = initial_guess;
        }
        
        // Then refine with Newton's method
        ROS_INFO("Refining with Newton optimization...");
        final_transform = optimizeTransformNewton(source_cloud, target_grid, final_transform);
        
        return final_transform;
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
        Eigen::Vector3f translation = current_pose_.block<3, 1>(0, 3);
        Eigen::Matrix3f rotation_matrix = current_pose_.block<3, 3>(0, 0);
        
        // Convert to quaternion
        Eigen::Quaternionf quaternion(rotation_matrix);
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