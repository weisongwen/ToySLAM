#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <boost/filesystem.hpp>
#include <algorithm>

class NDT_OMP_Processor {
public:
    NDT_OMP_Processor() : nh_("~"), current_index_(1) {
        initialize_parameters();
        load_pointclouds();
        initialize_ndt();
        initialize_publishers();
    }

    void process_clouds() {
        if (clouds_.size() < 2) {
            ROS_WARN("Need at least 2 clouds for registration");
            return;
        }

        ros::Rate rate(1); // Processing rate (1Hz)
        while (ros::ok() && current_index_ < clouds_.size()) {
            auto& target_cloud = clouds_[current_index_ - 1];
            auto& source_cloud = clouds_[current_index_];

            Eigen::Matrix4f transformation = perform_registration(target_cloud, source_cloud);
            publish_transform(transformation);
            
            ROS_INFO_STREAM("Transform " << current_index_-1 << " to " << current_index_ << ":\n" 
                           << transformation);
            
            current_index_++;
            rate.sleep();
        }
    }

private:
    void initialize_parameters() {
        nh_.param<std::string>("pcd_directory", pcd_directory_, "/home/wws/Download/pcdFile");
        nh_.param<double>("resolution", ndt_resolution_, 1.0);
        nh_.param<double>("step_size", step_size_, 0.1);
        nh_.param<double>("epsilon", epsilon_, 0.01);
        nh_.param<int>("max_iterations", max_iterations_, 64);
        nh_.param<int>("num_threads", num_threads_, 4);
        nh_.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.5);

        transformationSum = Eigen::Matrix4f::Identity();
    }

    void initialize_publishers() {
        path_pub_ = nh_.advertise<nav_msgs::Path>("/ndt_trajectory", 10);
        // map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
        // aligned_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 1);
    }

    void load_pointclouds() {
        namespace fs = boost::filesystem;
        std::vector<fs::path> files;
        
        // Get and sort PCD files
        for (const auto& entry : fs::directory_iterator(pcd_directory_)) {
            if (entry.path().extension() == ".pcd") {
                files.push_back(entry.path());
            }
        }

        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return std::stoi(a.stem().string().substr(6)) < 
                   std::stoi(b.stem().string().substr(6));
        });

        // Load clouds
        for (const auto& file : files) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.string(), *cloud) == -1) {
                ROS_ERROR("Failed to load %s", file.string().c_str());
                continue;
            }
            
            // Downsample cloud
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            voxel_filter.setInputCloud(cloud);
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            voxel_filter.filter(*filtered_cloud);
            
            clouds_.push_back(filtered_cloud);
            ROS_INFO("Loaded and filtered %s (%ld points)", 
                    file.filename().c_str(), filtered_cloud->size());
        }
    }

    void initialize_ndt() {
        ndt_omp_.setResolution(ndt_resolution_);
        ndt_omp_.setStepSize(step_size_);
        ndt_omp_.setTransformationEpsilon(epsilon_);
        ndt_omp_.setMaximumIterations(max_iterations_);
        ndt_omp_.setNumThreads(num_threads_);
        ndt_omp_.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    }

    Eigen::Matrix4f perform_registration(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source) 
    {
        ndt_omp_.setInputTarget(target);
        ndt_omp_.setInputSource(source);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        ndt_omp_.align(*aligned);

        if (!ndt_omp_.hasConverged()) {
            ROS_WARN("NDT failed to converge");
            return Eigen::Matrix4f::Identity();
        }

        return ndt_omp_.getFinalTransformation();
    }

    void publish_transform(const Eigen::Matrix4f& transformation) {
        static tf2_ros::TransformBroadcaster br;
        
        geometry_msgs::TransformStamped transform_msg;
        transform_msg.header.stamp = ros::Time::now();
        transform_msg.header.frame_id = "map";
        transform_msg.child_frame_id = "robot";

        Eigen::Affine3f affine(transformation);
        Eigen::Vector3f translation = affine.translation();
        Eigen::Quaternionf rotation(affine.linear());

        transform_msg.transform.translation.x = translation.x();
        transform_msg.transform.translation.y = translation.y();
        transform_msg.transform.translation.z = translation.z();
        
        transform_msg.transform.rotation.x = rotation.x();
        transform_msg.transform.rotation.y = rotation.y();
        transform_msg.transform.rotation.z = rotation.z();
        transform_msg.transform.rotation.w = rotation.w();

        br.sendTransform(transform_msg);

        transformationSum *= transformation;
        Eigen::Affine3f affineSum(transformationSum);
        Eigen::Vector3f translationSum = affineSum.translation();
        Eigen::Quaternionf rotationSum(affineSum.linear());

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
        pose.pose.position.x = translationSum.x();
        pose.pose.position.y = translationSum.y();
        pose.pose.position.z = translationSum.z();
        pose.pose.orientation.x = rotationSum.x();
        pose.pose.orientation.y = rotationSum.y();
        pose.pose.orientation.z = rotationSum.z();
        pose.pose.orientation.w = rotationSum.w();
        trajectory_poses_.poses.push_back(pose);

        if (trajectory_poses_.poses.empty()) return;

        trajectory_poses_.header.stamp = ros::Time::now();
        trajectory_poses_.header.frame_id = "map";
        path_pub_.publish(trajectory_poses_);

        ROS_INFO_STREAM("TransformSum " << current_index_-1 << " to " << current_index_ << ":\n" 
                           << transformationSum);

    }

    ros::NodeHandle nh_;
    ros::Publisher path_pub_;
    ros::Publisher map_pub_;
    ros::Publisher aligned_cloud_pub_;
    nav_msgs::Path trajectory_poses_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_;
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp_;
    size_t current_index_;
    
    // Parameters
    std::string pcd_directory_;
    double ndt_resolution_, step_size_, epsilon_, voxel_leaf_size_;
    int max_iterations_, num_threads_;

    Eigen::Matrix4f transformationSum;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_omp_node");
    NDT_OMP_Processor processor;
    processor.process_clouds();
    return 0;
}
