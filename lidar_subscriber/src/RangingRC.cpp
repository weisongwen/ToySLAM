#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Range.h>
#include <visualization_msgs/MarkerArray.h>
#include <signal_reflection/SignalPath.h>
#include <random>

class SignalReflectionNode {
public:
    SignalReflectionNode() {
        // Initialize parameters
        nh_.param("/simulation/frequency", sim_freq_, 10.0);
        nh_.param("/simulation/noise_stddev", noise_stddev_, 0.05);
        nh_.param("/simulation/epsilon", epsilon_, 0.1);

        // Initialize beacon positions (world coordinates)
        beacons_ = {
            {1, Eigen::Vector3d(2.0, 1.0, 0.5)},
            {2, Eigen::Vector3d(-1.0, 2.0, 0.5)},
            {3, Eigen::Vector3d(-2.0, -1.0, 0.5)},
            {4, Eigen::Vector3d(1.0, -2.0, 0.5)},
            {5, Eigen::Vector3d(0.0, 0.0, 0.5)}
        };

        // Setup publishers
        beacon_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/beacons", 1);
        range_pub_ = nh_.advertise<sensor_msgs::Range>("/range_measurements", 10);
        path_pub_ = nh_.advertise<signal_reflection::SignalPath>("/signal_paths", 10);

        // Setup simulation timer
        sim_timer_ = nh_.createTimer(ros::Duration(1.0/sim_freq_), 
                                   &SignalReflectionNode::simulationLoop, this);

        // Publish static beacons once
        publishBeacons();
    }

private:
    ros::NodeHandle nh_;
    ros::Timer sim_timer_;
    ros::Publisher beacon_pub_, range_pub_, path_pub_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    
    std::map<int, Eigen::Vector3d> beacons_;
    double sim_time_ = 0.0;
    double sim_freq_;
    double noise_stddev_;
    double epsilon_;

    void publishBeacons() {
        visualization_msgs::MarkerArray markers;
        int id = 0;
        for (const auto& [bid, pos] : beacons_) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.ns = "beacons";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = pos.x();
            marker.pose.position.y = pos.y();
            marker.pose.position.z = pos.z();
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3;
            marker.color.r = 1.0;
            marker.color.a = 1.0;
            markers.markers.push_back(marker);
        }
        beacon_pub_.publish(markers);
    }

    Eigen::Vector3d generateTrajectory(double t) {
        // Circular trajectory with 2m radius, constant altitude
        return Eigen::Vector3d(2.0 * cos(t), 2.0 * sin(t), 0.5);
    }

    void publishUserTransform(const Eigen::Vector3d& position) {
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "world";
        transform.child_frame_id = "user_base";
        transform.transform.translation.x = position.x();
        transform.transform.translation.y = position.y();
        transform.transform.translation.z = position.z();
        transform.transform.rotation.w = 1.0;
        tf_broadcaster_.sendTransform(transform);
    }

    void processRangeMeasurement(int beacon_id, 
                               const Eigen::Vector3d& user_pos,
                               const Eigen::Vector3d& beacon_pos,
                               double measured_range) {
        signal_reflection::SignalPath path_msg;
        path_msg.header.stamp = ros::Time::now();
        path_msg.beacon_id = beacon_id;
        path_msg.measured_range = measured_range;

        // Calculate theoretical distances
        const double direct_dist = (beacon_pos - user_pos).norm();
        const Eigen::Vector3d virtual_beacon(beacon_pos.x(), beacon_pos.y(), -beacon_pos.z());
        const double reflected_dist = (virtual_beacon - user_pos).norm();

        // Determine path type
        if (std::abs(measured_range - direct_dist) < epsilon_) {
            path_msg.path_type = signal_reflection::SignalPath::DIRECT;
        }
        else if (std::abs(measured_range - reflected_dist) < epsilon_) {
            path_msg.path_type = signal_reflection::SignalPath::REFLECTED;
            
            // Calculate reflection point (intersection with ground plane z=0)
            const double t = user_pos.z() / (user_pos.z() + beacon_pos.z());
            const Eigen::Vector3d refl_point = user_pos + t * (virtual_beacon - user_pos);
            path_msg.reflection_point.x = refl_point.x();
            path_msg.reflection_point.y = refl_point.y();
            path_msg.reflection_point.z = 0.0;
        }
        else {
            ROS_WARN("Measurement from beacon %d doesn't match any known path (Measured: %.2f, Direct: %.2f, Reflected: %.2f)", 
                    beacon_id, measured_range, direct_dist, reflected_dist);
            return;
        }

        path_pub_.publish(path_msg);
    }

    void simulationLoop(const ros::TimerEvent& event) {
        // Update simulation time
        sim_time_ += 1.0/sim_freq_;

        // Generate and publish user trajectory
        const Eigen::Vector3d user_pos = generateTrajectory(sim_time_);
        publishUserTransform(user_pos);

        // Generate simulated measurements for each beacon
        std::default_random_engine generator(ros::Time::now().nsec);
        std::normal_distribution<double> noise_dist(0.0, noise_stddev_);

        for (const auto& [beacon_id, beacon_pos] : beacons_) {
            // Randomly select path type for simulation
            const bool use_reflected = (rand() % 2 == 0);
            
            // Calculate true distance
            const double true_dist = use_reflected ? 
                (Eigen::Vector3d(beacon_pos.x(), beacon_pos.y(), -beacon_pos.z()) - user_pos).norm() :
                (beacon_pos - user_pos).norm();

            // Create noisy measurement
            sensor_msgs::Range range_msg;
            range_msg.header.stamp = ros::Time::now();
            range_msg.header.frame_id = "beacon" + std::to_string(beacon_id);
            range_msg.radiation_type = sensor_msgs::Range::ULTRASOUND;
            range_msg.field_of_view = 0.1;
            range_msg.min_range = 0.1;
            range_msg.max_range = 10.0;
            range_msg.range = std::max(0.1, true_dist + noise_dist(generator));

            // Publish and process the measurement
            range_pub_.publish(range_msg);
            processRangeMeasurement(beacon_id, user_pos, beacon_pos, range_msg.range);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "signal_reflection_node");
    SignalReflectionNode node;
    ros::spin();
    return 0;
}