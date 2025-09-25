#include "../include/visualizer.h"
#include <visualization_msgs/Marker.h>
#include <iomanip> // For std::setprecision
#include <sstream> // For std::stringstream
#include <limits>  // For std::numeric_limits

Visualizer::Visualizer(ros::NodeHandle& nh, const ParameterServer& params)
    : params_(params) {

    optimized_pose_pub_ = nh.advertise<nav_msgs::Odometry>(params_.optimized_pose_topic, params_.optimized_pose_queue_size);
    imu_pose_pub_ = nh.advertise<nav_msgs::Odometry>(params_.imu_pose_topic, params_.imu_pose_queue_size);
    imu_path_pub_ = nh.advertise<nav_msgs::Path>("imu_path", 10000);
    
    // Initialize visualization publishers
    gps_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/gps_path", 1, true);
    optimized_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/optimized_path", 1, true);
    final_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/final_path", 1, true);
    gt_path_pub_ = nh.advertise<nav_msgs::Path>("/trajectory/ground_truth_path", 1, true);
    //gt_odom_pub_ = nh.advertise<nav_msgs::Odometry>("/odometry/ground_truth_odom", 1, true);

    position_error_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/errors/position", 1);
    velocity_error_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/errors/velocity", 1);

    // Initialize path messages
    gt_path_msg_.header.frame_id = params_.world_frame_id;
    gps_path_msg_.header.frame_id = params_.world_frame_id;
    optimized_path_msg_.header.frame_id = params_.world_frame_id;
    imu_path_msg_.header.frame_id = params_.world_frame_id;
    final_path_msg_.header.frame_id = params_.world_frame_id;

}

void Visualizer::reset() {
    gps_path_msg_.poses.clear();
    gt_path_msg_.poses.clear();
    optimized_path_msg_.poses.clear();
    imu_path_msg_.poses.clear();
    final_path_msg_.poses.clear();


    visualization_msgs::MarkerArray clear_markers;
    visualization_msgs::Marker marker;
    marker.header.frame_id = params_.world_frame_id;
    marker.action = visualization_msgs::Marker::DELETEALL;
    clear_markers.markers.push_back(marker);
    position_error_pub_.publish(clear_markers);
    velocity_error_pub_.publish(clear_markers);
}

void Visualizer::publishOptimizedPath(const State& latest_state) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(latest_state.timestamp);
    pose_stamped.header.frame_id = params_.world_frame_id;
    
    pose_stamped.pose.position.x = latest_state.position.x();
    pose_stamped.pose.position.y = latest_state.position.y();
    pose_stamped.pose.position.z = latest_state.position.z();
    
    pose_stamped.pose.orientation.w = latest_state.orientation.w();
    pose_stamped.pose.orientation.x = latest_state.orientation.x();
    pose_stamped.pose.orientation.y = latest_state.orientation.y();
    pose_stamped.pose.orientation.z = latest_state.orientation.z();
    
    optimized_path_msg_.header.stamp = ros::Time(latest_state.timestamp);
    optimized_path_msg_.poses.push_back(pose_stamped);
    
    optimized_path_pub_.publish(optimized_path_msg_);
}

void Visualizer::publishFinalPath(const State& old_state) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(old_state.timestamp);
    pose_stamped.header.frame_id = params_.world_frame_id;
    pose_stamped.pose.position.x = old_state.position.x();
    pose_stamped.pose.position.y = old_state.position.y();
    pose_stamped.pose.position.z = old_state.position.z();
    pose_stamped.pose.orientation.w = old_state.orientation.w();
    pose_stamped.pose.orientation.x = old_state.orientation.x();
    pose_stamped.pose.orientation.y = old_state.orientation.y();
    pose_stamped.pose.orientation.z = old_state.orientation.z();
    
    final_path_msg_.header.stamp = ros::Time(old_state.timestamp);
    final_path_msg_.poses.push_back(pose_stamped);
    
    final_path_pub_.publish(final_path_msg_);
}

void Visualizer::publishGpsPath(const GnssMeasurement& measurement) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(measurement.timestamp);
    pose_stamped.header.frame_id = params_.world_frame_id;
    pose_stamped.pose.position.x = measurement.position.x();
    pose_stamped.pose.position.y = measurement.position.y();
    pose_stamped.pose.position.z = measurement.position.z();
    pose_stamped.pose.orientation.w = 1.0;
    
    gps_path_msg_.poses.push_back(pose_stamped);
    gps_path_pub_.publish(gps_path_msg_);
}

void Visualizer::publishGroundTruthPath(const GnssMeasurement& measurement) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(measurement.timestamp);
    pose_stamped.header.frame_id = params_.world_frame_id;
    pose_stamped.pose.position.x = measurement.position.x();
    pose_stamped.pose.position.y = measurement.position.y();
    pose_stamped.pose.position.z = measurement.position.z();
    pose_stamped.pose.orientation.w = measurement.orientation.w();
    pose_stamped.pose.orientation.x = measurement.orientation.x();
    pose_stamped.pose.orientation.y = measurement.orientation.y();
    pose_stamped.pose.orientation.z = measurement.orientation.z();

    gt_path_msg_.header.stamp = pose_stamped.header.stamp;
    gt_path_msg_.poses.push_back(pose_stamped);
    gt_path_pub_.publish(gt_path_msg_);
}

void Visualizer::publishImuPath(const nav_msgs::Odometry& odom_msg) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odom_msg.header;
    pose_stamped.pose = odom_msg.pose.pose;
    imu_path_msg_.header.stamp = pose_stamped.header.stamp;
    imu_path_msg_.poses.push_back(pose_stamped);
    imu_path_pub_.publish(imu_path_msg_);
}

// Publish IMU-predicted pose
void Visualizer::publishImuPose(const State& current_state) {
    try {
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time(current_state.timestamp);
        odom_msg.header.frame_id = params_.world_frame_id; // "map"
        // odom_msg.child_frame_id = body_frame_id_;   // "base_link"
        odom_msg.child_frame_id = params_.world_frame_id;   // "base_link"
        
        // Position
        odom_msg.pose.pose.position.x = current_state.position.x();
        odom_msg.pose.pose.position.y = current_state.position.y();
        odom_msg.pose.pose.position.z = current_state.position.z();
        
        // Orientation
        odom_msg.pose.pose.orientation.w = current_state.orientation.w();
        odom_msg.pose.pose.orientation.x = current_state.orientation.x();
        odom_msg.pose.pose.orientation.y = current_state.orientation.y();
        odom_msg.pose.pose.orientation.z = current_state.orientation.z();
        
        // Velocity
        odom_msg.twist.twist.linear.x = current_state.velocity.x();
        odom_msg.twist.twist.linear.y = current_state.velocity.y();
        odom_msg.twist.twist.linear.z = current_state.velocity.z();
        
        // Publish the message
        imu_pose_pub_.publish(odom_msg);

        // add imu path for debug
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odom_msg.header;
        pose_stamped.pose = odom_msg.pose.pose;

        imu_path_msg_.header.stamp = pose_stamped.header.stamp;
        imu_path_msg_.poses.push_back(pose_stamped);

        imu_path_pub_.publish(imu_path_msg_);
        
        // Publish the TF transform
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = odom_msg.header.stamp;
        transform_stamped.header.frame_id = params_.world_frame_id; // "map"
        transform_stamped.child_frame_id = params_.body_frame_id;   // "base_link"
        
        // Set translation
        transform_stamped.transform.translation.x = current_state.position.x();
        transform_stamped.transform.translation.y = current_state.position.y();
        transform_stamped.transform.translation.z = current_state.position.z();
        
        // Set rotation
        transform_stamped.transform.rotation.w = current_state.orientation.w();
        transform_stamped.transform.rotation.x = current_state.orientation.x();
        transform_stamped.transform.rotation.y = current_state.orientation.y();
        transform_stamped.transform.rotation.z = current_state.orientation.z();
        
        // Publish the transform
        tf_broadcaster_.sendTransform(transform_stamped);
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in publishImuPose: %s", e.what());
    }
}

// Publish optimized pose
void Visualizer::publishOptimizedPose(const State& latest_state) {
    try {
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time(latest_state.timestamp); // 使用最新状态的时间戳
        odom_msg.header.frame_id = params_.world_frame_id; // "map"
        odom_msg.child_frame_id = params_.body_frame_id;   // "base_link"
        
        // Position
        odom_msg.pose.pose.position.x = latest_state.position.x();
        odom_msg.pose.pose.position.y = latest_state.position.y();
        odom_msg.pose.pose.position.z = latest_state.position.z();
        
        // Orientation
        odom_msg.pose.pose.orientation.w = latest_state.orientation.w();
        odom_msg.pose.pose.orientation.x = latest_state.orientation.x();
        odom_msg.pose.pose.orientation.y = latest_state.orientation.y();
        odom_msg.pose.pose.orientation.z = latest_state.orientation.z();

        // Velocity
        odom_msg.twist.twist.linear.x = latest_state.velocity.x();
        odom_msg.twist.twist.linear.y = latest_state.velocity.y();
        odom_msg.twist.twist.linear.z = latest_state.velocity.z();
        
        // Publish the message
        optimized_pose_pub_.publish(odom_msg);
        
        // Publish the TF transform
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = odom_msg.header.stamp; // 与 odom_msg 时间戳一致
        transform_stamped.header.frame_id = params_.world_frame_id; // "map"
        transform_stamped.child_frame_id = params_.body_frame_id;   // "base_link"
        
        // Set translation
        transform_stamped.transform.translation.x = latest_state.position.x();
        transform_stamped.transform.translation.y = latest_state.position.y();
        transform_stamped.transform.translation.z = latest_state.position.z();
        
        // Set rotation
        transform_stamped.transform.rotation.w = latest_state.orientation.w();
        transform_stamped.transform.rotation.x = latest_state.orientation.x();
        transform_stamped.transform.rotation.y = latest_state.orientation.y();
        transform_stamped.transform.rotation.z = latest_state.orientation.z();
        
        // Publish the transform
        tf_broadcaster_.sendTransform(transform_stamped);
        //updateOptimizedPath();
        publishOptimizedPath(latest_state);
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in publishOptimizedPose: %s", e.what());
    }
}


void Visualizer::publishErrorVisualizations(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements) {
    calculateAndVisualizePositionError(current_state, gps_measurements);
    calculateAndVisualizeVelocityError(current_state, gps_measurements);
}

void Visualizer::calculateAndVisualizePositionError(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements) {
    visualization_msgs::MarkerArray position_error_markers;
    
    if (gps_measurements.empty() || current_state.position.norm() < 0.001) {
                return;
            }
            
    // Find closest GPS measurement to the current state
    double min_time_diff = std::numeric_limits<double>::max();
    GnssMeasurement closest_gps;
    bool found_gps = false;
    
    double current_time = current_state.timestamp;
    
    for (const auto& gps : gps_measurements) {
        double time_diff = std::abs(gps.timestamp - current_time);
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            closest_gps = gps;
            found_gps = true;
        }
    }
    
    // If we found a close GPS measurement (within 0.1s)
    if (found_gps && min_time_diff < 0.1) {
        // Calculate position error vector (optimized - GPS)
        Eigen::Vector3d position_error = current_state.position - closest_gps.position;
               
        // Calculate error magnitude
        double error_norm = position_error.norm();
        
        // Create marker for the total error vector (from GPS to optimized)
        visualization_msgs::Marker error_marker;
        error_marker.header.frame_id = params_.world_frame_id;
        error_marker.header.stamp = ros::Time(current_time);
        error_marker.ns = "position_error";
        error_marker.id = 0;
        error_marker.type = visualization_msgs::Marker::ARROW;
        error_marker.action = visualization_msgs::Marker::ADD;
        
        // Start of the arrow is at the GPS position
        error_marker.points.resize(2);
        error_marker.points[0].x = closest_gps.position.x();
        error_marker.points[0].y = closest_gps.position.y();
        error_marker.points[0].z = closest_gps.position.z();
        
        // End of the arrow is at the estimated position
        error_marker.points[1].x = current_state.position.x();
        error_marker.points[1].y = current_state.position.y();
        error_marker.points[1].z = current_state.position.z();
        
        // Set the arrow properties
        error_marker.scale.x = 0.05; // shaft diameter
        error_marker.scale.y = 0.1;  // head diameter
        error_marker.scale.z = 0.1;  // head length
        
        // Color the arrow based on error magnitude (green to red)
        error_marker.color.a = 1.0;
        
        // Scale from green (small error) to red (large error)
        double max_expected_error = 5.0; // meters
        double error_ratio = std::min(1.0, error_norm / max_expected_error);
        error_marker.color.r = error_ratio;
        error_marker.color.g = 1.0 - error_ratio;
        error_marker.color.b = 0.0;
        
        // Add to marker array
        position_error_markers.markers.push_back(error_marker);
        
        // Create text marker to display error value
        visualization_msgs::Marker text_marker;
        text_marker.header = error_marker.header;
        text_marker.ns = "position_error_text";
        text_marker.id = 0;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // Position the text above the error arrow
        text_marker.pose.position.x = (closest_gps.position.x() + current_state.position.x()) / 2.0;
        text_marker.pose.position.y = (closest_gps.position.y() + current_state.position.y()) / 2.0;
        text_marker.pose.position.z = (closest_gps.position.z() + current_state.position.z()) / 2.0 + 0.5;
        text_marker.pose.orientation.w = 1.0;
        
        // Set the text content to show error components
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) 
        << "Error: " << error_norm << "m "
        << "E:" << position_error.x() << " "
        << "N:" << position_error.y() << " "
        << "U:" << position_error.z();
        text_marker.text = ss.str();
        
        // Set text properties
        text_marker.scale.z = 0.3; // text height
        text_marker.color.r = error_ratio;
        text_marker.color.g = 1.0 - error_ratio;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0;
        
        // Add to marker array
        position_error_markers.markers.push_back(text_marker);
        
        // Create component arrows for ENU directions
        std::string components[3] = {"east", "north", "up"};
        
        
        ComponentInfo enu_components[3] = {
            {Eigen::Vector3d(1, 0, 0), {1.0f, 0.0f, 0.0f}, 0}, // East (Red)
            {Eigen::Vector3d(0, 1, 0), {0.0f, 1.0f, 0.0f}, 1}, // North (Green)
            {Eigen::Vector3d(0, 0, 1), {0.0f, 0.0f, 1.0f}, 2}  // Up (Blue)
        };
        
        // Create markers for each component
        for (int i = 0; i < 3; i++) {
            // Create a new marker for this component
            visualization_msgs::Marker component_marker;
            component_marker.header = error_marker.header;
            component_marker.ns = "position_error_" + components[i];
            component_marker.id = i;
            component_marker.type = visualization_msgs::Marker::ARROW;
            component_marker.action = visualization_msgs::Marker::ADD;
            
            // Start at GPS position
            component_marker.points.resize(2);
            component_marker.points[0].x = closest_gps.position.x();
            component_marker.points[0].y = closest_gps.position.y();
            component_marker.points[0].z = closest_gps.position.z();
            
            // Calculate end point: project the error along this component's direction
            double component_error = position_error(enu_components[i].index);
            
            // End at GPS position + error component in specific direction
            component_marker.points[1] = component_marker.points[0];
            component_marker.points[1].x += component_error * enu_components[i].direction.x();
            component_marker.points[1].y += component_error * enu_components[i].direction.y();
            component_marker.points[1].z += component_error * enu_components[i].direction.z();
            
            // Ensure minimum arrow size for visibility (if there is some error)
            const double min_visible_length = 0.1; // meters
            double arrow_length = std::abs(component_error);
            
            if (arrow_length > 0.001 && arrow_length < min_visible_length) {
                // Scale up small errors to be visible
                double scale_factor = min_visible_length / arrow_length;
                
                // Apply scaling to make arrow longer
                component_marker.points[1].x = component_marker.points[0].x + 
                    (component_marker.points[1].x - component_marker.points[0].x) * scale_factor;
                component_marker.points[1].y = component_marker.points[0].y + 
                    (component_marker.points[1].y - component_marker.points[0].y) * scale_factor;
                component_marker.points[1].z = component_marker.points[0].z + 
                    (component_marker.points[1].z - component_marker.points[0].z) * scale_factor;
            }
            
            // Set the arrow properties
            component_marker.scale.x = 0.04; // shaft diameter
            component_marker.scale.y = 0.08; // head diameter
            component_marker.scale.z = 0.08; // head length
            
            // Set color based on component (RGB = ENU)
            component_marker.color.r = enu_components[i].color[0];
            component_marker.color.g = enu_components[i].color[1];
            component_marker.color.b = enu_components[i].color[2];
            component_marker.color.a = 0.8;
            
            // Add label with component error value
            visualization_msgs::Marker text_marker;
            text_marker.header = component_marker.header;
            text_marker.ns = "position_error_text_" + components[i];
            text_marker.id = i;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            // Position the text at the end of the component arrow
            text_marker.pose.position = component_marker.points[1];
            text_marker.pose.position.z += 0.15; // offset slightly above the arrow
            text_marker.pose.orientation.w = 1.0;
            
            // Set the text content
            std::stringstream ss;
            ss << components[i] << ": " << std::fixed << std::setprecision(2) << component_error << "m";
            text_marker.text = ss.str();
            
            // Set text properties
            text_marker.scale.z = 0.2; // text height
            text_marker.color.r = enu_components[i].color[0];
            text_marker.color.g = enu_components[i].color[1];
            text_marker.color.b = enu_components[i].color[2];
            text_marker.color.a = 1.0;
            
            // Only add the marker if there is some error in this component
            if (std::abs(component_error) > 0.001 || arrow_length >= min_visible_length) {
                position_error_markers.markers.push_back(component_marker);
                position_error_markers.markers.push_back(text_marker);
            }
        }
        
        // Publish the position error visualization
        position_error_pub_.publish(position_error_markers);
        
        // Log error statistics periodically
        static double last_log_time = 0;
        if (current_time - last_log_time > 5.0) {
            // ROS_INFO("Position Error (m): %.2f (E:%.2f, N:%.2f, U:%.2f), publishing %zu markers", 
            //          error_norm, position_error.x(), position_error.y(), position_error.z(),
            //          position_error_markers.markers.size());
            last_log_time = current_time;
        }
    } else {
        // No matching GPS data found
        if (found_gps) {
            ROS_WARN_THROTTLE(5.0, "Found closest GPS but time difference too large: %.3f seconds", min_time_diff);
        } else {
            ROS_WARN_THROTTLE(5.0, "No GPS measurements available for error calculation");
        }
    }
}

void Visualizer::calculateAndVisualizeVelocityError(const State& current_state, const std::vector<GnssMeasurement>& gps_measurements) {
    visualization_msgs::MarkerArray velocity_error_markers;
            
    if (gps_measurements.empty() || current_state.velocity.norm() < 0.001) {
        return;
    }
    
    // Find closest GPS measurement to the current state
    double min_time_diff = std::numeric_limits<double>::max();
    GnssMeasurement closest_gps;
    bool found_gps = false;
    
    double current_time = current_state.timestamp;
    
    for (const auto& gps : gps_measurements) {
        double time_diff = std::abs(gps.timestamp - current_time);
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            closest_gps = gps;
            found_gps = true;
        }
    }
    
    // If we found a close GPS measurement (within 0.1s)
    if (found_gps && min_time_diff < 0.1) {
        // Calculate velocity error vector
        Eigen::Vector3d velocity_error = current_state.velocity - closest_gps.velocity;
               
        // Create markers for the velocity vectors
        // 1. Current estimated velocity
        visualization_msgs::Marker est_vel_marker;
        est_vel_marker.header.frame_id = params_.world_frame_id;
        est_vel_marker.header.stamp = ros::Time(current_time);
        est_vel_marker.ns = "velocity";
        est_vel_marker.id = 0;
        est_vel_marker.type = visualization_msgs::Marker::ARROW;
        est_vel_marker.action = visualization_msgs::Marker::ADD;
        
        // Start at current position
        est_vel_marker.points.resize(2);
        est_vel_marker.points[0].x = current_state.position.x();
        est_vel_marker.points[0].y = current_state.position.y();
        est_vel_marker.points[0].z = current_state.position.z();
        
        // Scale velocity for visualization (2x scale)
        double vel_scale = 2.0;
        est_vel_marker.points[1].x = current_state.position.x() + vel_scale * current_state.velocity.x();
        est_vel_marker.points[1].y = current_state.position.y() + vel_scale * current_state.velocity.y();
        est_vel_marker.points[1].z = current_state.position.z() + vel_scale * current_state.velocity.z();
        
        // Set marker properties
        est_vel_marker.scale.x = 0.05; // shaft diameter
        est_vel_marker.scale.y = 0.1;  // head diameter
        est_vel_marker.scale.z = 0.1;  // head length
        est_vel_marker.color.r = 0.0;
        est_vel_marker.color.g = 0.8;
        est_vel_marker.color.b = 0.0;
        est_vel_marker.color.a = 1.0;
        
        // 2. GPS velocity
        visualization_msgs::Marker gps_vel_marker = est_vel_marker;
        gps_vel_marker.id = 1;
        
        // Start at GPS position
        gps_vel_marker.points[0].x = closest_gps.position.x();
        gps_vel_marker.points[0].y = closest_gps.position.y();
        gps_vel_marker.points[0].z = closest_gps.position.z();
        
        // End at GPS position + GPS velocity (scaled)
        gps_vel_marker.points[1].x = closest_gps.position.x() + vel_scale * closest_gps.velocity.x();
        gps_vel_marker.points[1].y = closest_gps.position.y() + vel_scale * closest_gps.velocity.y();
        gps_vel_marker.points[1].z = closest_gps.position.z() + vel_scale * closest_gps.velocity.z();
        
        // Set GPS velocity marker color
        gps_vel_marker.color.r = 0.8;
        gps_vel_marker.color.g = 0.0;
        gps_vel_marker.color.b = 0.0;
        
        // Add to marker array
        velocity_error_markers.markers.push_back(est_vel_marker);
        velocity_error_markers.markers.push_back(gps_vel_marker);
        
        // Create text marker to display velocity error value
        visualization_msgs::Marker text_marker;
        text_marker.header = est_vel_marker.header;
        text_marker.ns = "velocity_error_text";
        text_marker.id = 0;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // Position the text above the current position
        text_marker.pose.position.x = current_state.position.x();
        text_marker.pose.position.y = current_state.position.y();
        text_marker.pose.position.z = current_state.position.z() + 1.0;
        text_marker.pose.orientation.w = 1.0;
        
        // Set the text content to show velocity error components
        double velocity_error_norm = velocity_error.norm();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) 
            << "Vel Error: " << velocity_error_norm << "m/s "
            << "E:" << velocity_error.x() << " "
            << "N:" << velocity_error.y() << " "
            << "U:" << velocity_error.z();
        text_marker.text = ss.str();
        
        // Set text properties
        text_marker.scale.z = 0.3; // text height
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        
        // Add to marker array
        velocity_error_markers.markers.push_back(text_marker);
        
        // Add velocity component visualizations for ENU directions
        std::string components[3] = {"east", "north", "up"};
        Eigen::Vector3d unit_vectors[3] = {
            Eigen::Vector3d(1, 0, 0),
            Eigen::Vector3d(0, 1, 0),
            Eigen::Vector3d(0, 0, 1)
        };
        
        for (int i = 0; i < 3; i++) {
            visualization_msgs::Marker vel_est_component = est_vel_marker;
            vel_est_component.ns = "velocity_est_" + components[i];
            vel_est_component.id = i;
            
            // Start at current position
            vel_est_component.points[0] = est_vel_marker.points[0];
            
            // End at current position + velocity component
            vel_est_component.points[1] = est_vel_marker.points[0];
            vel_est_component.points[1].x += vel_scale * current_state.velocity(i) * unit_vectors[i](0);
            vel_est_component.points[1].y += vel_scale * current_state.velocity(i) * unit_vectors[i](1);
            vel_est_component.points[1].z += vel_scale * current_state.velocity(i) * unit_vectors[i](2);
            
            // Set color based on component (RGB = ENU)
            vel_est_component.color.r = (i == 0) ? 0.8 : 0.0;
            vel_est_component.color.g = (i == 1) ? 0.8 : 0.0;
            vel_est_component.color.b = (i == 2) ? 0.8 : 0.0;
            vel_est_component.color.a = 0.5;
            
            velocity_error_markers.markers.push_back(vel_est_component);
            
            // Do the same for GPS velocity
            visualization_msgs::Marker vel_gps_component = gps_vel_marker;
            vel_gps_component.ns = "velocity_gps_" + components[i];
            vel_gps_component.id = i;
            
            vel_gps_component.points[0] = gps_vel_marker.points[0];
            vel_gps_component.points[1] = gps_vel_marker.points[0];
            vel_gps_component.points[1].x += vel_scale * closest_gps.velocity(i) * unit_vectors[i](0);
            vel_gps_component.points[1].y += vel_scale * closest_gps.velocity(i) * unit_vectors[i](1);
            vel_gps_component.points[1].z += vel_scale * closest_gps.velocity(i) * unit_vectors[i](2);
            
            vel_gps_component.color.r = (i == 0) ? 0.5 : 0.0;
            vel_gps_component.color.g = (i == 1) ? 0.5 : 0.0;
            vel_gps_component.color.b = (i == 2) ? 0.5 : 0.0;
            vel_gps_component.color.a = 0.5;
            
            velocity_error_markers.markers.push_back(vel_gps_component);
        }
        
        // Publish the velocity error visualization
        velocity_error_pub_.publish(velocity_error_markers);
        
        // Log velocity error statistics periodically
        static double last_log_time = 0;
        if (current_time - last_log_time > 5.0) {
            // ROS_INFO("Velocity Error (m/s): %.2f (E:%.2f, N:%.2f, U:%.2f)", 
            //         velocity_error_norm, velocity_error.x(), velocity_error.y(), velocity_error.z());
            last_log_time = current_time;
        }
    }
}