#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <cmath>

/**
 * Simplified User Position Node
 * 
 * This node continuously publishes a moving user position for testing the UWB ray tracer.
 * The user follows a figure-8 path along the road.
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "user_position_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // Publisher for user position
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("user_pose", 10);
    
    // Parameters
    double path_length = private_nh.param<double>("path_length", 60.0);
    double path_width = private_nh.param<double>("path_width", 4.0);
    double z_height = private_nh.param<double>("z_height", 1.7);
    double speed = private_nh.param<double>("speed", 0.5); // units per second
    double update_rate = private_nh.param<double>("update_rate", 10.0);
    
    // Message for publishing
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.frame_id = "map";
    tf::TransformBroadcaster tf_broadcaster;
    
    // Time parameter for figure-8 path
    double t = 0.0;
    
    ros::Rate rate(update_rate);
    
    ROS_INFO("User position node started. Moving in figure-8 pattern.");
    
    while (ros::ok()) {
        // Calculate position on figure-8 path
        double x = (path_length / 2.0) * sin(t);
        double y = (path_width / 2.0) * sin(2 * t);
        double yaw = std::atan2(2 * (path_width / 2.0) * cos(2 * t), (path_length / 2.0) * cos(t));
        
        // Update message
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.x = x;
        pose_msg.pose.position.y = y;
        pose_msg.pose.position.z = z_height;
        
        // Convert yaw to quaternion
        tf::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        
        // Publish
        pose_pub.publish(pose_msg);
        
        // Broadcast TF
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(x, y, z_height));
        transform.setRotation(q);
        tf_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "user"));
        
        // Increment time parameter
        t += speed / update_rate;
        if (t > 2 * M_PI) t -= 2 * M_PI;
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}