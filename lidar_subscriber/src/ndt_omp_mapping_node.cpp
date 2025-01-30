#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>
#include <tf2_ros/transform_broadcaster.h>
#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class NDTMapper {
public:
    NDTMapper() : nh_("~"), current_index_(1), global_map_(new pcl::PointCloud<pcl::PointXYZ>) {
        initialize_parameters();
        initialize_publishers();
        initialize_ndt();
        load_initial_clouds();
    }

    void run() {
        ros::Rate rate(1); // Processing rate (1Hz)
        while (ros::ok()) {
            process_available_clouds();
            publish_trajectory();
            publish_global_map();
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
        nh_.param<int>("num_threads", num_threads_, 40);
        nh_.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.5);
        nh_.param<double>("map_voxel_size", map_voxel_size_, 0.2);
    }

    void initialize_publishers() {
        path_pub_ = nh_.advertise<nav_msgs::Path>("/ndt_trajectory", 10);
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
        aligned_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 1);
    }

    void initialize_ndt() {
        ndt_omp_.setResolution(ndt_resolution_);
        ndt_omp_.setStepSize(step_size_);
        ndt_omp_.setTransformationEpsilon(epsilon_);
        ndt_omp_.setMaximumIterations(max_iterations_);
        ndt_omp_.setNumThreads(num_threads_);
        ndt_omp_.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    }

    void load_initial_clouds() {
        process_new_clouds(true);
        if (!clouds_.empty()) {
            add_to_trajectory(Eigen::Matrix4f::Identity());
            update_global_map(clouds_[0], Eigen::Matrix4f::Identity());
        }
    }

    void process_available_clouds() {
        size_t previous_count = clouds_.size();
        process_new_clouds();
        
        while (current_index_ < clouds_.size()) {
            auto result = align_consecutive_clouds(
                clouds_[current_index_ - 1],
                clouds_[current_index_]
            );

            if (result.hasConverged()) {
                Eigen::Matrix4f transform = result.getFinalTransformation();
                
                ROS_INFO_STREAM("Transform " << current_index_-1 << " to " << current_index_ << ":\n" 
                           << transform);
                
                Eigen::Matrix4f global_transform;
                if(trajectory_.size())
                {
                    global_transform = trajectory_.back() * transform;
                }
                else
                {
                    global_transform = transform;
                }
                
                update_trajectory(global_transform);
                update_global_map(clouds_[current_index_], global_transform);
            }

            current_index_++;

            // repeat the publishment
            publish_trajectory();
            publish_global_map();
        }
    }

    void process_new_clouds(bool initial_load = false) {
        namespace fs = boost::filesystem;
        std::vector<fs::path> new_files;

        for (const auto& entry : fs::directory_iterator(pcd_directory_)) {
            if (entry.path().extension() == ".pcd") {
                int file_number = extract_file_number(entry.path().stem().string());
                if (file_number >= static_cast<int>(clouds_.size()) + 1) {
                    new_files.push_back(entry.path());
                }
            }
        }

        std::sort(new_files.begin(), new_files.end(), [](const fs::path& a, const fs::path& b) {
            return extract_file_number(a.stem().string()) < 
                   extract_file_number(b.stem().string());
        });

        for (const auto& file : new_files) {
            auto cloud = load_and_filter_cloud(file.string());
            if (cloud && !cloud->empty()) {
                clouds_.push_back(cloud);
                ROS_INFO("Loaded %s (%zu points)", 
                        file.filename().c_str(), cloud->size());
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr load_and_filter_cloud(const std::string& path) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1) return nullptr;

        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel_filter.setInputCloud(cloud);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*filtered);
        return filtered;
    }

    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>
    align_consecutive_clouds(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source) 
    {
        ndt_omp_.setInputTarget(target);
        ndt_omp_.setInputSource(source);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        ndt_omp_.align(*aligned);

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*aligned, msg);
        msg.header.frame_id = "map";
        aligned_cloud_pub_.publish(msg);

        // return ndt_omp_.getResult();
        return ndt_omp_;
    }

    void update_trajectory(const Eigen::Matrix4f& global_transform) {
        trajectory_.push_back(global_transform);
        add_to_trajectory(global_transform);
    }

    void add_to_trajectory(const Eigen::Matrix4f& transform) {
        Eigen::Affine3f affine(transform);
        Eigen::Vector3f position = affine.translation();
        Eigen::Quaternionf rotation(affine.linear());

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
        pose.pose.position.x = position.x();
        pose.pose.position.y = position.y();
        pose.pose.position.z = position.z();
        pose.pose.orientation.x = rotation.x();
        pose.pose.orientation.y = rotation.y();
        pose.pose.orientation.z = rotation.z();
        pose.pose.orientation.w = rotation.w();

        trajectory_poses_.poses.push_back(pose);
    }

    void update_global_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                          const Eigen::Matrix4f& transform) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
        
        *global_map_ += *transformed_cloud;
        
        // Downsample global map
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        // voxel_filter.setLeafSize(map_voxel_size_, map_voxel_size_, map_voxel_size_);
        voxel_filter.setLeafSize(0.5, 0.5, 0.5);
        voxel_filter.setInputCloud(global_map_);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*filtered);
        global_map_.swap(filtered);
    }

    void publish_trajectory() {
        if (trajectory_poses_.poses.empty()) return;

        trajectory_poses_.header.stamp = ros::Time::now();
        trajectory_poses_.header.frame_id = "map";
        path_pub_.publish(trajectory_poses_);
    }

    void publish_global_map() {
        if (global_map_->empty()) return;

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global_map_, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "map";
        map_pub_.publish(msg);
    }

    static int extract_file_number(const std::string& filename) {
        size_t underscore = filename.find_last_of('_');
        if (underscore != std::string::npos) {
            try {
                return std::stoi(filename.substr(underscore + 1));
            } catch (...) {}
        }
        return -1;
    }

    ros::NodeHandle nh_;
    ros::Publisher path_pub_;
    ros::Publisher map_pub_;
    ros::Publisher aligned_cloud_pub_;
    nav_msgs::Path trajectory_poses_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp_;
    std::vector<Eigen::Matrix4f> trajectory_;
    size_t current_index_;

    // Parameters
    std::string pcd_directory_;
    double ndt_resolution_, step_size_, epsilon_, voxel_leaf_size_, map_voxel_size_;
    int max_iterations_, num_threads_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_mapping_node");
    NDTMapper mapper;
    mapper.run();
    return 0;
}