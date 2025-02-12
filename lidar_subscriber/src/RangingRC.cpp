#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/NavSatFix.h>
#include <Eigen/Dense>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <vector>
#include <random>
#include <cmath>

class GPSSimulator {
    private:
        ros::NodeHandle nh_;
        ros::Publisher path_pub_;
        ros::Publisher building_pub_;
        ros::Publisher satellite_pub_;
        ros::Publisher receiver_pub_;
        ros::Publisher pseudorange_pub_;
        
        // Random number generators
        std::random_device rd_;
        std::default_random_engine generator_;
        std::normal_distribution<double> noise_dist_;
        
        // Constants
        const double SPEED_OF_LIGHT = 299792458.0; // m/s
        const double GPS_FREQ = 1575.42e6; // L1 frequency in Hz
        const double WAVELENGTH = SPEED_OF_LIGHT / GPS_FREQ;
        
        // Simulation parameters
        struct Building {
            Eigen::Vector3d min_point;
            Eigen::Vector3d max_point;
        };
        
        struct Satellite {
            Eigen::Vector3d position;
            int prn;
        };
        
        struct Ray {
            Eigen::Vector3d start;
            Eigen::Vector3d direction;
            double strength;
            std::vector<Eigen::Vector3d> reflection_points;
        };
        
        Eigen::Vector3d receiver_position_;
        std::vector<Building> buildings_;
        std::vector<Satellite> satellites_;
        std::vector<Ray> valid_paths_;
        
        // Noise parameters
        double pseudorange_noise_std_;
        ros::Timer timer ;
    
    public:
        GPSSimulator() : 
            generator_(rd_()),
            noise_dist_(0.0, 1.0)
        {
            // Initialize publishers with correct message type syntax
            path_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/gps/signal_paths", 1);
            building_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/gps/buildings", 1);
            satellite_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/gps/satellites", 1);
            receiver_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gps/receiver", 1);
            pseudorange_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/gps/pseudorange", 1);
            
            // Initialize simulation parameters
            initializeSimulation();
            
            // Start periodic updates
            timer = nh_.createTimer(ros::Duration(1.0), 
                                             &GPSSimulator::updateCallback, this);
        }
    
    
    void initializeSimulation() {
        // Set receiver position
        receiver_position_ = Eigen::Vector3d(2.0, 50.0, 1.7); // 1.7m height
        
        // Initialize buildings (two rows of buildings)
        double street_width = 20.0;
        double building_height = 30.0;
        double building_width = 20.0;
        double building_length = 50.0;
        
        // Left side buildings
        for(int i = 0; i < 3; i++) {
            Building b;
            b.min_point = Eigen::Vector3d(-street_width/2 - building_width, 
                                        i * building_length, 0);
            b.max_point = Eigen::Vector3d(-street_width/2, 
                                        (i+1) * building_length, 
                                        building_height);
            buildings_.push_back(b);
        }
        
        // Right side buildings
        for(int i = 0; i < 3; i++) {
            Building b;
            b.min_point = Eigen::Vector3d(street_width/2, 
                                        i * building_length, 0);
            b.max_point = Eigen::Vector3d(street_width/2 + building_width, 
                                        (i+1) * building_length, 
                                        building_height);
            buildings_.push_back(b);
        }
        
        // Initialize satellites (simulated GPS constellation)
        initializeSatellites();
        
        // Set noise parameter
        pseudorange_noise_std_ = 3.0; // 3 meters standard deviation
    }
    
    void initializeSatellites() {
        // Simulate a basic GPS constellation (simplified)
        double radius = 26600000.0; // Approximate GPS orbit radius
        int num_satellites = 8;
        
        for(int i = 0; i < num_satellites; i++) {
            double angle = 2.0 * M_PI * i / num_satellites;
            double elevation = M_PI/4; // 45 degrees elevation
            
            Satellite sat;
            sat.position = Eigen::Vector3d(
                radius * cos(angle) * cos(elevation),
                radius * sin(angle) * cos(elevation),
                radius * sin(elevation)
            );
            sat.prn = i + 1;
            satellites_.push_back(sat);
        }
    }
    
    bool rayIntersectsBuilding(const Ray& ray, const Building& building, 
                              Eigen::Vector3d& intersection_point) {
        // Simplified ray-box intersection test
        double t_min = -INFINITY;
        double t_max = INFINITY;
        
        for(int i = 0; i < 3; i++) {
            if(abs(ray.direction[i]) < 1e-8) {
                if(ray.start[i] < building.min_point[i] || 
                   ray.start[i] > building.max_point[i])
                    return false;
            } else {
                double t1 = (building.min_point[i] - ray.start[i]) / ray.direction[i];
                double t2 = (building.max_point[i] - ray.start[i]) / ray.direction[i];
                
                if(t1 > t2) std::swap(t1, t2);
                
                t_min = std::max(t_min, t1);
                t_max = std::min(t_max, t2);
                
                if(t_min > t_max) return false;
            }
        }
        
        if(t_min > 0) { // Intersection in positive direction
            intersection_point = ray.start + t_min * ray.direction;
            return true;
        }
        
        return false;
    }
    
    void findValidPaths() {
        valid_paths_.clear();
        
        for(const auto& sat : satellites_) {
            // Direct path
            Ray direct_ray;
            direct_ray.start = sat.position;
            direct_ray.direction = (receiver_position_ - sat.position).normalized();
            direct_ray.strength = 1.0;
            
            bool is_blocked = false;
            for(const auto& building : buildings_) {
                Eigen::Vector3d intersection;
                if(rayIntersectsBuilding(direct_ray, building, intersection)) {
                    is_blocked = true;
                    break;
                }
            }
            
            if(!is_blocked) {
                valid_paths_.push_back(direct_ray);
            }
            
            // Single reflection paths
            for(const auto& building : buildings_) {
                // Check reflection points on building surfaces
                std::vector<Eigen::Vector3d> surfaces = {
                    Eigen::Vector3d(building.min_point.x(), 0, building.max_point.z()),
                    Eigen::Vector3d(building.max_point.x(), 0, building.max_point.z())
                };
                
                for(const auto& reflection_point : surfaces) {
                    Ray reflected_ray;
                    reflected_ray.start = sat.position;
                    reflected_ray.direction = (reflection_point - sat.position).normalized();
                    reflected_ray.strength = 0.5; // Reduced strength for reflected signals
                    reflected_ray.reflection_points.push_back(reflection_point);
                    
                    // Check if reflection path is valid
                    Eigen::Vector3d to_receiver = receiver_position_ - reflection_point;
                    if(to_receiver.normalized().dot(reflected_ray.direction) < 0) {
                        continue; // Invalid reflection angle
                    }
                    
                    bool path_blocked = false;
                    for(const auto& other_building : buildings_) {
                        Eigen::Vector3d intersection;
                        if(rayIntersectsBuilding(reflected_ray, other_building, intersection)) {
                            path_blocked = true;
                            break;
                        }
                    }
                    
                    if(!path_blocked) {
                        valid_paths_.push_back(reflected_ray);
                    }
                }
            }
        }
    }
    
    double calculatePseudorange(const Ray& ray) {
        double geometric_range = 0.0;
        
        if(ray.reflection_points.empty()) {
            // Direct path
            geometric_range = (receiver_position_ - ray.start).norm();
        } else {
            // Reflected path
            geometric_range = (ray.reflection_points[0] - ray.start).norm() +
                            (receiver_position_ - ray.reflection_points[0]).norm();
        }
        
        // Add noise
        double noise = noise_dist_(generator_) * pseudorange_noise_std_;
        
        // Add atmospheric delays (simplified)
        double ionospheric_delay = 5.0; // meters
        double tropospheric_delay = 2.5; // meters
        
        return geometric_range + noise + ionospheric_delay + tropospheric_delay;
    }
    
    void publishVisualization() {
        visualization_msgs::MarkerArray building_markers;
        visualization_msgs::MarkerArray path_markers;
        visualization_msgs::MarkerArray satellite_markers;
        
        // Publish buildings
        for(size_t i = 0; i < buildings_.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.id = i;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            
            marker.pose.position.x = (buildings_[i].min_point.x() + buildings_[i].max_point.x()) / 2;
            marker.pose.position.y = (buildings_[i].min_point.y() + buildings_[i].max_point.y()) / 2;
            marker.pose.position.z = (buildings_[i].min_point.z() + buildings_[i].max_point.z()) / 2;
            
            marker.scale.x = buildings_[i].max_point.x() - buildings_[i].min_point.x();
            marker.scale.y = buildings_[i].max_point.y() - buildings_[i].min_point.y();
            marker.scale.z = buildings_[i].max_point.z() - buildings_[i].min_point.z();
            
            marker.color.r = 0.8;
            marker.color.g = 0.8;
            marker.color.b = 0.8;
            marker.color.a = 0.8;
            
            building_markers.markers.push_back(marker);
        }
        
        // Publish signal paths
        for(size_t i = 0; i < valid_paths_.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.id = i;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            
            geometry_msgs::Point p;
            p.x = valid_paths_[i].start.x();
            p.y = valid_paths_[i].start.y();
            p.z = valid_paths_[i].start.z();
            marker.points.push_back(p);
            
            for(const auto& reflection : valid_paths_[i].reflection_points) {
                p.x = reflection.x();
                p.y = reflection.y();
                p.z = reflection.z();
                marker.points.push_back(p);
            }
            
            p.x = receiver_position_.x();
            p.y = receiver_position_.y();
            p.z = receiver_position_.z();
            marker.points.push_back(p);
            
            marker.scale.x = 0.1;
            marker.color.r = valid_paths_[i].strength;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = valid_paths_[i].strength;
            
            path_markers.markers.push_back(marker);
        }
        
        // Publish satellites
        for(size_t i = 0; i < satellites_.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            
            marker.pose.position.x = satellites_[i].position.x();
            marker.pose.position.y = satellites_[i].position.y();
            marker.pose.position.z = satellites_[i].position.z();
            
            marker.scale.x = marker.scale.y = marker.scale.z = 1000000.0;
            
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.8;
            
            satellite_markers.markers.push_back(marker);
        }
        
        building_pub_.publish(building_markers);
        path_pub_.publish(path_markers);
        satellite_pub_.publish(satellite_markers);
    }
    
    void updateCallback(const ros::TimerEvent& event) {
        // Update satellite positions (simple circular motion)
        std::cout<<"updateCallback-> \n";
        double dt = 1.0;
        double angular_velocity = 2.0 * M_PI / (12 * 3600); // 12-hour orbit
        
        for(auto& sat : satellites_) {
            Eigen::Vector3d pos = sat.position;
            double r = pos.norm();
            double theta = atan2(pos.y(), pos.x()) + angular_velocity * dt;
            
            sat.position.x() = r * cos(theta);
            sat.position.y() = r * sin(theta);
        }
        
        // Find valid signal paths
        findValidPaths();
        
        // Calculate and publish pseudoranges
        sensor_msgs::NavSatFix pseudorange_msg;
        pseudorange_msg.header.stamp = ros::Time::now();
        pseudorange_msg.header.frame_id = "world";
        
        for(const auto& path : valid_paths_) {
            double pseudorange = calculatePseudorange(path);
            // In a real implementation, you would add the pseudorange
            // measurements to the message here
            std::cout<<"pseudorange-> " << pseudorange<<"\n";
        }
        
        pseudorange_pub_.publish(pseudorange_msg);
        
        // Publish visualization
        publishVisualization();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_simulator");
    GPSSimulator simulator;
    ros::spin();
    return 0;
}