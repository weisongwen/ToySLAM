#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Geometry>

class NDTMappingNode {
public:
    NDTMappingNode() : global_map_(new pcl::PointCloud<pcl::PointXYZ>) {
        initialize_parameters();
        initialize_publishers();
        initialize_ndt();
    }

    void process_bag(const std::string& bag_path) {
        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException& e) {
            ROS_ERROR("Failed to open bag file: %s", e.what());
            return;
        }

        std::vector<std::string> topics{input_topic_};
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud;

        for (const rosbag::MessageInstance& msg : view) {
            if (!ros::ok()) break;

            sensor_msgs::PointCloud2::ConstPtr cloud_msg = 
                msg.instantiate<sensor_msgs::PointCloud2>();
            if (!cloud_msg) continue;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*cloud_msg, *cloud);

            // Preprocess cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = downsample_cloud(cloud);
            
            if (!prev_cloud) {
                prev_cloud = filtered;
                update_global_map(filtered, pose);
                continue;
            }

            // Perform registration
            Eigen::Matrix4f transform = perform_registration(prev_cloud, filtered);
            pres_transform = transform;
            pose = pose * transform;

            // Update tracking
            update_trajectory(pose);
            update_global_map(filtered, pose);
            
            prev_cloud = filtered;

            // Publish intermediate results
            publish_global_map();
            publish_trajectory();
        }

        bag.close();
    }

private:
    void initialize_parameters() {
        nh_.param<std::string>("input_topic", input_topic_, "/velodyne_points");
        nh_.param<double>("ndt_resolution", ndt_resolution_, 1.0);
        nh_.param<double>("ndt_step_size", ndt_step_size_, 0.1);
        nh_.param<double>("ndt_epsilon", ndt_epsilon_, 0.01);
        nh_.param<int>("ndt_max_iterations", ndt_max_iterations_, 64);
        nh_.param<int>("ndt_threads", ndt_threads_, 40);
        nh_.param<double>("voxel_leaf", voxel_leaf_, 0.3); // 0.5 is unstable, 0.1 is too slow, 0.3 seems good
        nh_.param<double>("map_voxel", map_voxel_, 0.5); 

        pres_transform = Eigen::Matrix4f::Identity();
    }

    void initialize_publishers() {
        path_pub_ = nh_.advertise<nav_msgs::Path>("/ndt_trajectory", 10);
        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map", 1);
    }

    void initialize_ndt() {
        ndt_omp_.setResolution(ndt_resolution_);
        ndt_omp_.setStepSize(ndt_step_size_);
        ndt_omp_.setTransformationEpsilon(ndt_epsilon_);
        ndt_omp_.setMaximumIterations(ndt_max_iterations_);
        ndt_omp_.setNumThreads(ndt_threads_);
        ndt_omp_.setNeighborhoodSearchMethod(pclomp::DIRECT7);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) 
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        voxel_filter.setInputCloud(cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*filtered);
        return filtered;
    }

    Eigen::Matrix4f perform_registration(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source) 
    {
        ndt_omp_.setInputTarget(target);
        ndt_omp_.setInputSource(source);

        auto t1 = ros::WallTime::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        // ndt_omp_.align(*aligned);
        ndt_omp_.align(*aligned, pres_transform);
        auto t2 = ros::WallTime::now();
        std::cout << "single (t2-t1) : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;
        std::cout << "fitness (t2-t1): " << ndt_omp_.getFitnessScore() << std::endl << std::endl;
        // ndt_omp_.align(*aligned);
        // auto t3 = ros::WallTime::now();
        // std::cout << "single (t3-t2) : " << (t3 - t2).toSec() * 1000 << "[msec]" << std::endl;
        // std::cout << "fitness (t3-t2): " << ndt_omp_.getFitnessScore() << std::endl << std::endl;


        if (ndt_omp_.hasConverged()) {
            return ndt_omp_.getFinalTransformation();
        }
        return Eigen::Matrix4f::Identity();
    }

    void update_global_map(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          const Eigen::Matrix4f& transform) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *transformed, transform);
        
        *global_map_ += *transformed;
        
        // Downsample global map
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(map_voxel_, map_voxel_, map_voxel_);
        voxel_filter.setInputCloud(global_map_);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*filtered);
        global_map_.swap(filtered);
    }

    void update_trajectory(const Eigen::Matrix4f& pose) {
        Eigen::Affine3f affine(pose);
        Eigen::Vector3f position = affine.translation();
        Eigen::Quaternionf rotation(affine.linear());

        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = position.x();
        pose_msg.pose.position.y = position.y();
        pose_msg.pose.position.z = position.z();
        pose_msg.pose.orientation.x = rotation.x();
        pose_msg.pose.orientation.y = rotation.y();
        pose_msg.pose.orientation.z = rotation.z();
        pose_msg.pose.orientation.w = rotation.w();

        trajectory_.poses.push_back(pose_msg);
    }

    void publish_trajectory() {
        trajectory_.header.stamp = ros::Time::now();
        trajectory_.header.frame_id = "map";
        path_pub_.publish(trajectory_);
    }

    void publish_global_map() {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*global_map_, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "map";
        map_pub_.publish(msg);
    }

    ros::NodeHandle nh_;
    ros::Publisher path_pub_;
    ros::Publisher map_pub_;
    nav_msgs::Path trajectory_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp_;

    // Parameters
    std::string input_topic_;
    double ndt_resolution_, ndt_step_size_, ndt_epsilon_;
    double voxel_leaf_, map_voxel_;
    int ndt_max_iterations_, ndt_threads_;

    Eigen::Matrix4f pres_transform;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_mapping_node");
    
    if (argc < 2) {
        ROS_ERROR("Usage: rosrun your_package ndt_mapping_node path_to_bagfile.bag");
        return 1;
    }

    NDTMappingNode mapper;
    mapper.process_bag(argv[1]);
    
    return 0;
}