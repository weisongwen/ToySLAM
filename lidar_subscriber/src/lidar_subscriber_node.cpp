#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <boost/filesystem.hpp>

class PointCloudSaver 
{
public:
    PointCloudSaver() : nh_("~"), save_count_(0)
    {
        // Initialize parameters
        nh_.param<std::string>("save_directory", save_directory_, "/home/wws/Download/pcdFile");
        nh_.param<std::string>("file_prefix", file_prefix_, "cloud_");
        nh_.param<std::string>("input_topic", input_topic_, "/velodyne_points");
        
        // Create directory if it doesn't exist
        boost::filesystem::path dir(save_directory_);
        if(!boost::filesystem::exists(dir)) {
            boost::filesystem::create_directories(dir);
            ROS_INFO("Created directory: %s", save_directory_.c_str());
        }

        // Initialize subscriber
        sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
            input_topic_, 1, &PointCloudSaver::cloudCallback, this);

        ROS_INFO("Point Cloud Saver initialized");
        ROS_INFO("Saving to: %s", save_directory_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        // Convert ROS message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        // Generate filename
        std::string filename = save_directory_ + "/" + file_prefix_ + 
                              std::to_string(++save_count_) + ".pcd";

        // Save to PCD file
        if(pcl::io::savePCDFileBinary(filename, *cloud) == 0) {
            ROS_INFO_ONCE("Saving point clouds...");
            ROS_DEBUG("Saved %s", filename.c_str());
            ROS_INFO("Saved %s", filename.c_str());
        }
        else {
            ROS_ERROR("Failed to save %s", filename.c_str());
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    std::string save_directory_;
    std::string file_prefix_;
    std::string input_topic_;
    unsigned int save_count_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointcloud_saver_node");
    PointCloudSaver saver;
    ros::spin();
    return 0;
}

// rosrun lidar_subscriber lidar_subscriber_node \ _save_directory:=/home/wws/Download/pcdFile \ _file_prefix:=lidar_scan_ \ _input_topic:=/velodyne_points