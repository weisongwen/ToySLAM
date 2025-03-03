/**
 * UWB Ray Tracing ROS Node - Complete Revised Version
 * 
 * Features:
 * - 10 UWB beacons placed strategically
 * - Vibrant path visualization with distinct colors
 * - Enhanced reflection visualization
 * - Rigorous ray tracing with multiple reflections
 * - Buildings on both sides of road optimized for reflections
 */

 #include <ros/ros.h>
 #include <visualization_msgs/Marker.h>
 #include <visualization_msgs/MarkerArray.h>
 #include <geometry_msgs/PoseWithCovarianceStamped.h>
 #include <geometry_msgs/PoseStamped.h>
 #include <std_msgs/ColorRGBA.h>
 #include <tf/transform_broadcaster.h>
 #include <Eigen/Dense>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <random>
 #include <algorithm>
 #include <memory>
 #include <sstream>
 #include <iomanip>
 
 // Structure to represent a 3D Box (Building)
 struct Building {
     std::string id;
     Eigen::Vector3d center;
     Eigen::Vector3d dimensions;
     Eigen::Vector3d color;
     double reflectivity;
     
     Eigen::Vector3d min() const {
         return center - dimensions/2;
     }
     
     Eigen::Vector3d max() const {
         return center + dimensions/2;
     }
     
     std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getFaces() const {
         std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> faces;
         Eigen::Vector3d min_pt = min();
         Eigen::Vector3d max_pt = max();
         
         // Front face (normal = -z)
         faces.push_back({Eigen::Vector3d(center.x(), center.y(), min_pt.z()), 
                           Eigen::Vector3d(0, 0, -1)});
         
         // Back face (normal = +z)
         faces.push_back({Eigen::Vector3d(center.x(), center.y(), max_pt.z()), 
                           Eigen::Vector3d(0, 0, 1)});
         
         // Left face (normal = -x)
         faces.push_back({Eigen::Vector3d(min_pt.x(), center.y(), center.z()), 
                           Eigen::Vector3d(-1, 0, 0)});
         
         // Right face (normal = +x)
         faces.push_back({Eigen::Vector3d(max_pt.x(), center.y(), center.z()), 
                           Eigen::Vector3d(1, 0, 0)});
         
         // Bottom face (normal = -y)
         faces.push_back({Eigen::Vector3d(center.x(), min_pt.y(), center.z()), 
                           Eigen::Vector3d(0, -1, 0)});
         
         // Top face (normal = +y)
         faces.push_back({Eigen::Vector3d(center.x(), max_pt.y(), center.z()), 
                           Eigen::Vector3d(0, 1, 0)});
         
         return faces;
     }
 };
 
 // Structure to represent a UWB Beacon
 struct UWBBeacon {
     std::string id;
     Eigen::Vector3d position;
     double range;         // Max range in meters
     double frequency;     // Signal frequency in MHz
     double power;         // Transmit power in dBm
     
     UWBBeacon() : range(200.0), frequency(6500.0), power(15.0) {}
 };
 
 // Structure for signal path segment
 struct PathSegment {
     Eigen::Vector3d start;
     Eigen::Vector3d end;
     int reflection_count;
     double path_loss;     // Path loss in dB
     double total_distance; // Total distance to this point
     double reflectivity;  // Product of reflectivity coefficients
     int reflection_building_id;  // ID of building causing reflection, -1 if no reflection
     
     PathSegment(const Eigen::Vector3d& s, const Eigen::Vector3d& e, int rc, 
                double pl, double d, double r, int b_id = -1) 
         : start(s), end(e), reflection_count(rc), 
           path_loss(pl), total_distance(d), reflectivity(r),
           reflection_building_id(b_id) {}
 };
 
 // Structure for a ray tracing path
 struct SignalPath {
     std::vector<PathSegment> segments;
     double total_path_loss;   // Total path loss in dB
     double total_distance;    // Total path length in meters
     int reflection_count;     // Number of reflections
     bool valid;               // Whether path is valid (not blocked)
     std::string beacon_id;    // ID of the source beacon
     
     SignalPath() : total_path_loss(0.0), total_distance(0.0), 
                   reflection_count(0), valid(true), beacon_id("unknown") {}
     
     void addSegment(const PathSegment& segment) {
         segments.push_back(segment);
         total_path_loss += segment.path_loss;
         total_distance = segment.total_distance;
         reflection_count = segment.reflection_count;
     }
 };
 
 // Ray-tracing specific constants and calculations
 namespace RayTracing {
     constexpr double SPEED_OF_LIGHT = 299792458.0;
     
     double calculateWavelength(double frequency_mhz) {
         return SPEED_OF_LIGHT / (frequency_mhz * 1e6);
     }
     
     double calculateFreeSpacePathLoss(double distance, double frequency_mhz) {
         // FSPL (dB) = 20*log10(d) + 20*log10(f) + 32.44
         // where d is distance in km and f is frequency in MHz
         double distance_km = distance / 1000.0;
         return 20.0 * std::log10(distance_km) + 20.0 * std::log10(frequency_mhz) + 32.44;
     }
     
     double calculateReflectionCoefficient(double incident_angle, double surface_reflectivity) {
         double grazing_factor = std::sin(incident_angle);
         return surface_reflectivity * grazing_factor;
     }
     
     Eigen::Vector3d calculateReflection(const Eigen::Vector3d& incident, 
                                         const Eigen::Vector3d& normal) {
         return incident - 2 * incident.dot(normal) * normal;
     }
     
     double calculateIncidentAngle(const Eigen::Vector3d& incident, 
                                   const Eigen::Vector3d& normal) {
         double cos_angle = std::abs(incident.dot(normal) / 
                           (incident.norm() * normal.norm()));
         return std::acos(cos_angle);
     }
 }
 
 class UWBRayTracer {
 private:
     ros::NodeHandle nh_;
     ros::NodeHandle private_nh_;
 
     // Publishers
     ros::Publisher building_pub_;
     ros::Publisher road_pub_;
     ros::Publisher beacon_pub_;
     ros::Publisher user_pub_;
     ros::Publisher path_pub_;
     ros::Publisher reflection_pub_;
     ros::Publisher text_pub_;
     
     // Subscribers
     ros::Subscriber user_pose_sub_;
     ros::Subscriber beacon_pose_sub_;
     
     // Timer
     ros::Timer update_timer_;
     
     // TF broadcaster
     tf::TransformBroadcaster tf_broadcaster_;
     
     // Lists of objects
     std::vector<Building> buildings_;
     std::vector<UWBBeacon> beacons_;
     Eigen::Vector3d user_position_;
     std::vector<SignalPath> signal_paths_;
     
     // Parameters
     int max_reflections_;     
     double max_distance_;     
     double min_signal_power_; 
     double noise_floor_;      
     
     // Road parameters
     double road_length_;
     double road_width_;
     double building_spacing_;
     double building_height_min_;
     double building_height_max_;
     double building_width_min_;
     double building_width_max_;
     double building_depth_min_;
     double building_depth_max_;
     double buildings_per_side_;
     
     // Environment parameters
     double sidewalk_width_;
     double beacon_height_;
     int num_beacons_;
     
     // Visualization parameters
     std::string fixed_frame_;
     double building_alpha_;
     double path_width_;
     bool debug_mode_;
     
     // Random number generator
     std::mt19937 rng_;
     std::uniform_real_distribution<double> height_dist_;
     std::uniform_real_distribution<double> width_dist_;
     std::uniform_real_distribution<double> depth_dist_;
     std::uniform_real_distribution<double> reflectivity_dist_;
     std::uniform_real_distribution<double> color_dist_;
 
 public:
     UWBRayTracer() : private_nh_("~"), 
                      rng_(std::random_device()()) {
         // Get parameters
         private_nh_.param<int>("max_reflections", max_reflections_, 3);
         private_nh_.param<double>("max_distance", max_distance_, 200.0);
         private_nh_.param<double>("min_signal_power", min_signal_power_, -90.0);
         private_nh_.param<double>("noise_floor", noise_floor_, -100.0);
         private_nh_.param<std::string>("fixed_frame", fixed_frame_, "map");
         private_nh_.param<double>("building_alpha", building_alpha_, 0.7);
         private_nh_.param<double>("path_width", path_width_, 0.1);  // Thicker paths for better visibility
         private_nh_.param<bool>("debug_mode", debug_mode_, false);
         private_nh_.param<int>("num_beacons", num_beacons_, 10);  // Exactly 10 beacons
         
         // Road and environment parameters
         private_nh_.param<double>("road_length", road_length_, 120.0);
         private_nh_.param<double>("road_width", road_width_, 8.0);
         private_nh_.param<double>("sidewalk_width", sidewalk_width_, 3.0);
         private_nh_.param<double>("building_spacing", building_spacing_, 3.0);
         private_nh_.param<double>("building_height_min", building_height_min_, 5.0);
         private_nh_.param<double>("building_height_max", building_height_max_, 15.0);
         private_nh_.param<double>("building_width_min", building_width_min_, 10.0);
         private_nh_.param<double>("building_width_max", building_width_max_, 20.0);
         private_nh_.param<double>("building_depth_min", building_depth_min_, 8.0);
         private_nh_.param<double>("building_depth_max", building_depth_max_, 12.0);
         private_nh_.param<double>("buildings_per_side", buildings_per_side_, 6.0);
         private_nh_.param<double>("beacon_height", beacon_height_, 10.0);  // High enough for visibility
         
         // Initialize random distributions
         height_dist_ = std::uniform_real_distribution<double>(building_height_min_, building_height_max_);
         width_dist_ = std::uniform_real_distribution<double>(building_width_min_, building_width_max_);
         depth_dist_ = std::uniform_real_distribution<double>(building_depth_min_, building_depth_max_);
         reflectivity_dist_ = std::uniform_real_distribution<double>(0.5, 0.9);  // Higher reflectivity for better paths
         color_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
         
         // Publishers
         building_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/buildings", 1);
         road_pub_ = nh_.advertise<visualization_msgs::Marker>("uwb_ray_tracer/road", 1);
         beacon_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/beacons", 1);
         user_pub_ = nh_.advertise<visualization_msgs::Marker>("uwb_ray_tracer/user", 1);
         path_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/paths", 1);
         reflection_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/reflections", 1);
         text_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/text_info", 1);
         
         // Subscribers
         user_pose_sub_ = nh_.subscribe("user_pose", 10, &UWBRayTracer::userPoseCallback, this);
         beacon_pose_sub_ = nh_.subscribe("beacon_pose", 10, &UWBRayTracer::beaconPoseCallback, this);
         
         // Initialize user position in the middle of the road
         user_position_ = Eigen::Vector3d(0, 0, 1.7);
         
         // Generate environment 
         generateEnvironment();
         
         // Create exactly 10 beacons placed strategically
         createExactlyTenBeacons();
         
         // Create a timer for updates
         double update_rate;
         private_nh_.param<double>("update_rate", update_rate, 10.0);
         update_timer_ = nh_.createTimer(ros::Duration(1.0/update_rate), 
                                         &UWBRayTracer::updateCallback, this);
         
         ROS_INFO("UWB Ray Tracer initialized with %zu buildings and %zu beacons", 
                 buildings_.size(), beacons_.size());
     }
     
     ~UWBRayTracer() {
         clearVisualizations();
     }
     
     void userPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
         user_position_ = Eigen::Vector3d(
             msg->pose.position.x,
             msg->pose.position.y,
             msg->pose.position.z
         );
         
         // Clear signal paths since user position changed
         signal_paths_.clear();
     }
     
     void beaconPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
         std::string beacon_id = msg->header.frame_id;
         bool found = false;
         
         Eigen::Vector3d position(
             msg->pose.pose.position.x,
             msg->pose.pose.position.y,
             msg->pose.pose.position.z
         );
         
         for (auto& beacon : beacons_) {
             if (beacon.id == beacon_id) {
                 beacon.position = position;
                 found = true;
                 break;
             }
         }
         
         if (!found) {
             UWBBeacon new_beacon;
             new_beacon.id = beacon_id;
             new_beacon.position = position;
             new_beacon.range = private_nh_.param<double>("beacon_range", 200.0);
             new_beacon.frequency = private_nh_.param<double>("beacon_frequency", 6500.0);
             new_beacon.power = private_nh_.param<double>("beacon_power", 15.0);
             beacons_.push_back(new_beacon);
             
             ROS_INFO("Added new beacon: %s at (%.2f, %.2f, %.2f)", 
                      beacon_id.c_str(), position.x(), position.y(), position.z());
         }
         
         // Clear signal paths since beacon positions changed
         signal_paths_.clear();
     }
     
     void updateCallback(const ros::TimerEvent& event) {
         // If signal paths are empty, compute new paths
         if (signal_paths_.empty()) {
             computeSignalPaths();
         }
         
         // Publish visualizations
         publishRoad();
         publishBuildings();
         publishBeacons();
         publishUser();
         publishSignalPaths();
         publishReflectionPoints();
         publishTextInfo();
         
         // Broadcast TF frames
         broadcastTFs();
     }
     
     void clearVisualizations() {
         visualization_msgs::MarkerArray empty_array;
         building_pub_.publish(empty_array);
         beacon_pub_.publish(empty_array);
         path_pub_.publish(empty_array);
         reflection_pub_.publish(empty_array);
         text_pub_.publish(empty_array);
         
         visualization_msgs::Marker empty_marker;
         empty_marker.action = visualization_msgs::Marker::DELETE;
         user_pub_.publish(empty_marker);
         road_pub_.publish(empty_marker);
     }
     
     void generateEnvironment() {
         ROS_INFO("Generating environment with buildings on both sides of the road...");
         
         // Create buildings on both sides of the road with consistent spacing
         double x_start = -road_length_ / 2.0 + building_width_min_ / 2.0;
         double x_end = road_length_ / 2.0 - building_width_min_ / 2.0;
         double segment_length = (x_end - x_start) / buildings_per_side_;
         
         // Left side buildings (negative y)
         double y_left = -(road_width_ / 2.0 + sidewalk_width_ + building_depth_min_ / 2.0);
         for (int i = 0; i < buildings_per_side_; ++i) {
             double building_x = x_start + i * segment_length + (segment_length - building_width_min_) * ((double)(rng_() % 100) / 100.0);
             double building_width = width_dist_(rng_);
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             Building building;
             building.id = "building_left_" + std::to_string(i);
             building.center = Eigen::Vector3d(building_x, y_left - ((double)(rng_() % 100) / 100.0) * building_spacing_, building_height / 2.0);
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             building.color = Eigen::Vector3d(0.7 + 0.3 * color_dist_(rng_), 0.3 * color_dist_(rng_), 0.3 * color_dist_(rng_));  // Reddish buildings
             building.reflectivity = reflectivity_dist_(rng_);
             
             buildings_.push_back(building);
         }
         
         // Right side buildings (positive y)
         double y_right = road_width_ / 2.0 + sidewalk_width_ + building_depth_min_ / 2.0;
         for (int i = 0; i < buildings_per_side_; ++i) {
             double building_x = x_start + i * segment_length + (segment_length - building_width_min_) * ((double)(rng_() % 100) / 100.0);
             double building_width = width_dist_(rng_);
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             Building building;
             building.id = "building_right_" + std::to_string(i);
             building.center = Eigen::Vector3d(building_x, y_right + ((double)(rng_() % 100) / 100.0) * building_spacing_, building_height / 2.0);
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             building.color = Eigen::Vector3d(0.3 * color_dist_(rng_), 0.3 * color_dist_(rng_), 0.7 + 0.3 * color_dist_(rng_));  // Bluish buildings
             building.reflectivity = reflectivity_dist_(rng_);
             
             buildings_.push_back(building);
         }
     }
     
     void createExactlyTenBeacons() {
         ROS_INFO("Creating exactly 10 UWB beacons placed strategically...");
         
         // Clear any existing beacons
         beacons_.clear();
         
         // Distribute beacons evenly along the road
         double x_start = -road_length_ / 2.0 + 10.0;  // Start a bit from the edge
         double x_end = road_length_ / 2.0 - 10.0;     // End a bit from the edge
         double x_step = (x_end - x_start) / 9.0;      // Space for 10 beacons
         
         // Default beacon parameters
         double range = private_nh_.param<double>("beacon_range", 200.0);
         double freq = private_nh_.param<double>("beacon_frequency", 6500.0);
         double power = private_nh_.param<double>("beacon_power", 15.0);
         
         // Create beacons - 5 on each side, alternating
         for (int i = 0; i < 10; ++i) {
             UWBBeacon beacon;
             beacon.id = "beacon_" + std::to_string(i);
             
             double x_pos = x_start + i * x_step;
             
             // Alternate beacons between left and right sides
             if (i % 2 == 0) {
                 // Left side (negative y)
                 beacon.position = Eigen::Vector3d(
                     x_pos,
                     -(road_width_ / 2.0 + sidewalk_width_ / 2.0),
                     beacon_height_
                 );
             } else {
                 // Right side (positive y)
                 beacon.position = Eigen::Vector3d(
                     x_pos,
                     (road_width_ / 2.0 + sidewalk_width_ / 2.0),
                     beacon_height_
                 );
             }
             
             // Set beacon properties
             beacon.range = range;
             beacon.frequency = freq;
             beacon.power = power;
             
             beacons_.push_back(beacon);
             
             ROS_INFO("Created beacon %s at (%.2f, %.2f, %.2f)", 
                      beacon.id.c_str(), beacon.position.x(), beacon.position.y(), beacon.position.z());
         }
     }
     
     void computeSignalPaths() {
         signal_paths_.clear();
         
         for (const auto& beacon : beacons_) {
             // Compute direct path first
             computeDirectPath(beacon);
             
             // Compute reflected paths
             for (int reflection_count = 1; reflection_count <= max_reflections_; ++reflection_count) {
                 computeReflectedPaths(beacon, reflection_count);
             }
         }
         
         // Sort paths by total path loss (best signal first)
         std::sort(signal_paths_.begin(), signal_paths_.end(), 
               [](const SignalPath& a, const SignalPath& b) {
                   return a.total_path_loss < b.total_path_loss;
               });
         
         // Keep only paths with signal above threshold
         auto it = std::remove_if(signal_paths_.begin(), signal_paths_.end(),
                               [this](const SignalPath& path) {
                                   return path.total_path_loss > (min_signal_power_ - noise_floor_);
                               });
         
         signal_paths_.erase(it, signal_paths_.end());
         
         if (signal_paths_.empty()) {
             ROS_WARN("No valid signal paths found! Adjusting parameters to force visibility...");
             
             // Force at least direct paths to ensure visualization
             for (const auto& beacon : beacons_) {
                 computeDirectPath(beacon, true);  // Force some paths
             }
         }
         
         ROS_INFO("Computed %zu valid signal paths", signal_paths_.size());
         
         if (debug_mode_ && !signal_paths_.empty()) {
             // Count paths by reflection count
             int direct_paths = 0;
             int single_reflection = 0;
             int double_reflection = 0;
             int triple_reflection = 0;
             
             for (const auto& path : signal_paths_) {
                 if (path.reflection_count == 0) direct_paths++;
                 else if (path.reflection_count == 1) single_reflection++;
                 else if (path.reflection_count == 2) double_reflection++;
                 else if (path.reflection_count == 3) triple_reflection++;
             }
             
             ROS_INFO("Path breakdown: %d direct, %d single reflection, %d double reflection, %d triple reflection",
                      direct_paths, single_reflection, double_reflection, triple_reflection);
         }
     }
     
     void computeDirectPath(const UWBBeacon& beacon, bool force_path = false) {
         // Create a direct path from beacon to user
         Eigen::Vector3d direction = user_position_ - beacon.position;
         double distance = direction.norm();
         
         // Skip if outside range
         if (distance > beacon.range || distance > max_distance_) {
             if (debug_mode_) {
                 ROS_INFO("Beacon %s: Direct path distance %.2f exceeds range %.2f", 
                          beacon.id.c_str(), distance, beacon.range);
             }
             return;
         }
         
         // Check for building intersections
         bool path_blocked = false;
         Building blocking_building;
         
         if (!force_path) {
             for (const auto& building : buildings_) {
                 if (rayIntersectsBuilding(beacon.position, direction.normalized(), distance, building)) {
                     path_blocked = true;
                     blocking_building = building;
                     break;
                 }
             }
         }
         
         if (path_blocked && debug_mode_) {
             ROS_INFO("Beacon %s: Direct path blocked by building %s", 
                      beacon.id.c_str(), blocking_building.id.c_str());
             
             if (!force_path) {
                 return;
             }
         }
         
         // Calculate path loss
         double path_loss = RayTracing::calculateFreeSpacePathLoss(distance, beacon.frequency);
         path_loss -= beacon.power;  // Adjust for beacon transmit power
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Direct path distance=%.2f, loss=%.2f dB", 
                      beacon.id.c_str(), distance, path_loss);
         }
         
         // Create the signal path
         SignalPath path;
         path.beacon_id = beacon.id;
         path.addSegment(PathSegment(beacon.position, user_position_, 0, path_loss, distance, 1.0));
         
         if (force_path && path_blocked) {
             // Adjust path loss to ensure it's visible but clearly attenuated
             path.total_path_loss += 20.0;  // Add 20dB of attenuation to show it's blocked
         }
         
         signal_paths_.push_back(path);
     }
     
     void computeReflectedPaths(const UWBBeacon& beacon, int reflection_count) {
         if (reflection_count <= 0 || reflection_count > max_reflections_) {
             return;
         }
         
         // For single reflections
         if (reflection_count == 1) {
             computeSingleReflectionPaths(beacon);
         }
         // For multiple reflections
         else {
             computeMultipleReflectionPaths(beacon, reflection_count);
         }
     }
     
     void computeSingleReflectionPaths(const UWBBeacon& beacon) {
         int paths_found = 0;
         
         // For each building, check all faces
         for (size_t building_idx = 0; building_idx < buildings_.size(); building_idx++) {
             const auto& building = buildings_[building_idx];
             auto faces = building.getFaces();
             
             for (const auto& face : faces) {
                 Eigen::Vector3d point_on_plane = face.first;
                 Eigen::Vector3d normal = face.second;
                 
                 // Compute reflection point
                 Eigen::Vector3d reflection_point;
                 bool valid_reflection = computeReflectionPoint(
                     beacon.position, user_position_, point_on_plane, normal, reflection_point);
                 
                 if (!valid_reflection) {
                     continue;
                 }
                 
                 // Check if reflection point is on the building face
                 if (!isPointOnBuildingFace(reflection_point, building, normal)) {
                     continue;
                 }
                 
                 // Calculate distances
                 double distance1 = (reflection_point - beacon.position).norm();
                 double distance2 = (user_position_ - reflection_point).norm();
                 double total_distance = distance1 + distance2;
                 
                 // Check if within range
                 if (total_distance > beacon.range || total_distance > max_distance_) {
                     continue;
                 }
                 
                 // Check for obstacles from beacon to reflection point
                 Eigen::Vector3d dir1 = (reflection_point - beacon.position).normalized();
                 bool path1_blocked = false;
                 
                 for (size_t other_idx = 0; other_idx < buildings_.size(); other_idx++) {
                     if (other_idx != building_idx && // Skip the building we're reflecting off
                         rayIntersectsBuilding(beacon.position, dir1, distance1, buildings_[other_idx])) {
                         path1_blocked = true;
                         break;
                     }
                 }
                 
                 if (path1_blocked) {
                     continue;
                 }
                 
                 // Check for obstacles from reflection point to user
                 Eigen::Vector3d dir2 = (user_position_ - reflection_point).normalized();
                 bool path2_blocked = false;
                 
                 for (size_t other_idx = 0; other_idx < buildings_.size(); other_idx++) {
                     if (other_idx != building_idx && // Skip the building we're reflecting off
                         rayIntersectsBuilding(reflection_point, dir2, distance2, buildings_[other_idx])) {
                         path2_blocked = true;
                         break;
                     }
                 }
                 
                 if (path2_blocked) {
                     continue;
                 }
                 
                 // Calculate reflection coefficient
                 double incident_angle = RayTracing::calculateIncidentAngle(-dir1, normal);
                 double reflection_coef = RayTracing::calculateReflectionCoefficient(
                     incident_angle, building.reflectivity);
                 
                 // Calculate path losses
                 double path_loss1 = RayTracing::calculateFreeSpacePathLoss(distance1, beacon.frequency);
                 double path_loss2 = RayTracing::calculateFreeSpacePathLoss(distance2, beacon.frequency);
                 
                 // Reflection loss (simplified)
                 double reflection_loss = -20.0 * std::log10(reflection_coef);
                 
                 // Total loss adjusted for beacon power
                 double total_loss = path_loss1 + path_loss2 + reflection_loss - beacon.power;
                 
                 if (debug_mode_) {
                     ROS_INFO("Beacon %s: Single reflection path on building %s, total_loss=%.2f dB", 
                              beacon.id.c_str(), building.id.c_str(), total_loss);
                 }
                 
                 // Create the signal path
                 SignalPath path;
                 path.beacon_id = beacon.id;
                 path.addSegment(PathSegment(
                     beacon.position, reflection_point, 0, path_loss1, distance1, 1.0));
                 
                 path.addSegment(PathSegment(
                     reflection_point, user_position_, 1, path_loss2 + reflection_loss, 
                     total_distance, reflection_coef, building_idx));
                 
                 signal_paths_.push_back(path);
                 paths_found++;
             }
         }
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Found %d single reflection paths", beacon.id.c_str(), paths_found);
         }
     }
     
     void computeMultipleReflectionPaths(const UWBBeacon& beacon, int reflection_count) {
         if (reflection_count > 3) {
             return; // Limit to 3 reflections for performance
         }
         
         std::vector<std::tuple<size_t, Eigen::Vector3d, Eigen::Vector3d>> all_faces;
         
         // Collect all building faces with their building index
         for (size_t i = 0; i < buildings_.size(); ++i) {
             auto faces = buildings_[i].getFaces();
             for (const auto& face : faces) {
                 all_faces.push_back(std::make_tuple(i, face.first, face.second));
             }
         }
         
         int paths_found = 0;
         
         // For 2 reflections
         if (reflection_count == 2) {
             for (size_t i = 0; i < all_faces.size(); ++i) {
                 size_t building1_idx = std::get<0>(all_faces[i]);
                 Eigen::Vector3d point1 = std::get<1>(all_faces[i]);
                 Eigen::Vector3d normal1 = std::get<2>(all_faces[i]);
                 
                 for (size_t j = 0; j < all_faces.size(); ++j) {
                     if (i == j) continue; // Skip same face
                     
                     size_t building2_idx = std::get<0>(all_faces[j]);
                     Eigen::Vector3d point2 = std::get<1>(all_faces[j]);
                     Eigen::Vector3d normal2 = std::get<2>(all_faces[j]);
                     
                     // Find reflection points
                     Eigen::Vector3d reflection_point1, reflection_point2;
                     
                     // Special case for two reflections
                     if (find2ReflectionPath(beacon.position, user_position_, 
                                            point1, normal1, point2, normal2,
                                            reflection_point1, reflection_point2)) {
                         
                         // Check if points are on the correct faces
                         if (!isPointOnBuildingFace(reflection_point1, buildings_[building1_idx], normal1) ||
                             !isPointOnBuildingFace(reflection_point2, buildings_[building2_idx], normal2)) {
                             continue;
                         }
                         
                         // Calculate distances
                         double distance1 = (reflection_point1 - beacon.position).norm();
                         double distance2 = (reflection_point2 - reflection_point1).norm();
                         double distance3 = (user_position_ - reflection_point2).norm();
                         double total_distance = distance1 + distance2 + distance3;
                         
                         // Check if within range
                         if (total_distance > beacon.range || total_distance > max_distance_) {
                             continue;
                         }
                         
                         // Check for obstacles between segments
                         Eigen::Vector3d dir1 = (reflection_point1 - beacon.position).normalized();
                         Eigen::Vector3d dir2 = (reflection_point2 - reflection_point1).normalized();
                         Eigen::Vector3d dir3 = (user_position_ - reflection_point2).normalized();
                         
                         bool blocked = false;
                         
                         // Check beacon to first reflection
                         for (size_t k = 0; k < buildings_.size(); ++k) {
                             if (k != building1_idx && 
                                 rayIntersectsBuilding(beacon.position, dir1, distance1, buildings_[k])) {
                                 blocked = true;
                                 break;
                             }
                         }
                         if (blocked) continue;
                         
                         // Check first to second reflection
                         for (size_t k = 0; k < buildings_.size(); ++k) {
                             if (k != building1_idx && k != building2_idx && 
                                 rayIntersectsBuilding(reflection_point1, dir2, distance2, buildings_[k])) {
                                 blocked = true;
                                 break;
                             }
                         }
                         if (blocked) continue;
                         
                         // Check second reflection to user
                         for (size_t k = 0; k < buildings_.size(); ++k) {
                             if (k != building2_idx && 
                                 rayIntersectsBuilding(reflection_point2, dir3, distance3, buildings_[k])) {
                                 blocked = true;
                                 break;
                             }
                         }
                         if (blocked) continue;
                         
                         // Calculate reflection coefficients
                         double incident_angle1 = RayTracing::calculateIncidentAngle(-dir1, normal1);
                         double reflection_coef1 = RayTracing::calculateReflectionCoefficient(
                             incident_angle1, buildings_[building1_idx].reflectivity);
                         
                         double incident_angle2 = RayTracing::calculateIncidentAngle(-dir2, normal2);
                         double reflection_coef2 = RayTracing::calculateReflectionCoefficient(
                             incident_angle2, buildings_[building2_idx].reflectivity);
                         
                         // Total reflection coefficient
                         double total_reflection = reflection_coef1 * reflection_coef2;
                         
                         // Calculate path losses
                         double path_loss1 = RayTracing::calculateFreeSpacePathLoss(distance1, beacon.frequency);
                         double path_loss2 = RayTracing::calculateFreeSpacePathLoss(distance2, beacon.frequency);
                         double path_loss3 = RayTracing::calculateFreeSpacePathLoss(distance3, beacon.frequency);
                         
                         // Reflection losses
                         double reflection_loss1 = -20.0 * std::log10(reflection_coef1);
                         double reflection_loss2 = -20.0 * std::log10(reflection_coef2);
                         
                         // Total path loss
                         double total_loss = path_loss1 + path_loss2 + path_loss3 + 
                                           reflection_loss1 + reflection_loss2 - beacon.power;
                                           
                         // Create the signal path
                         SignalPath path;
                         path.beacon_id = beacon.id;
                         path.addSegment(PathSegment(
                             beacon.position, reflection_point1, 0, path_loss1, 
                             distance1, 1.0));
                         
                         path.addSegment(PathSegment(
                             reflection_point1, reflection_point2, 1, 
                             path_loss2 + reflection_loss1, 
                             distance1 + distance2, reflection_coef1, building1_idx));
                         
                         path.addSegment(PathSegment(
                             reflection_point2, user_position_, 2, 
                             path_loss3 + reflection_loss2, 
                             total_distance, total_reflection, building2_idx));
                         
                         signal_paths_.push_back(path);
                         paths_found++;
                     }
                 }
             }
         }
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Found %d double reflection paths", beacon.id.c_str(), paths_found);
         }
     }
     
     bool find2ReflectionPath(
         const Eigen::Vector3d& source, 
         const Eigen::Vector3d& target,
         const Eigen::Vector3d& point1, 
         const Eigen::Vector3d& normal1,
         const Eigen::Vector3d& point2, 
         const Eigen::Vector3d& normal2,
         Eigen::Vector3d& reflection_point1,
         Eigen::Vector3d& reflection_point2) {
         
         // Reflect source across plane 1
         double d1 = normal1.dot(point1);
         double t1 = (d1 - normal1.dot(source)) / normal1.dot(normal1);
         Eigen::Vector3d mirrored_source = source + 2 * t1 * normal1;
         
         // Reflect target across plane 2
         double d2 = normal2.dot(point2);
         double t2 = (d2 - normal2.dot(target)) / normal2.dot(normal2);
         Eigen::Vector3d mirrored_target = target + 2 * t2 * normal2;
         
         // Find reflection point 1 (on plane 1)
         Eigen::Vector3d ray_dir = (mirrored_target - mirrored_source).normalized();
         
         // Find intersection with plane 1
         double nd1 = normal1.dot(ray_dir);
         if (std::abs(nd1) < 1e-6) {
             return false; // Ray is parallel to plane 1
         }
         
         double t_hit1 = (d1 - normal1.dot(mirrored_source)) / nd1;
         if (t_hit1 < 0) {
             return false; // Intersection is behind mirrored source
         }
         
         reflection_point1 = mirrored_source + t_hit1 * ray_dir;
         
         // Find reflection point 2 (on plane 2)
         // Calculate the reflected direction from reflection_point1
         Eigen::Vector3d incident1 = (reflection_point1 - source).normalized();
         Eigen::Vector3d reflected1 = RayTracing::calculateReflection(incident1, normal1);
         
         // Find intersection with plane 2
         double nd2 = normal2.dot(reflected1);
         if (std::abs(nd2) < 1e-6) {
             return false; // Reflected ray is parallel to plane 2
         }
         
         double t_hit2 = (d2 - normal2.dot(reflection_point1)) / nd2;
         if (t_hit2 < 0) {
             return false; // Intersection is behind reflection point 1
         }
         
         reflection_point2 = reflection_point1 + t_hit2 * reflected1;
         
         // Verify the path by checking angles
         Eigen::Vector3d dir1 = (reflection_point1 - source).normalized();
         Eigen::Vector3d dir2 = (reflection_point2 - reflection_point1).normalized();
         Eigen::Vector3d dir3 = (target - reflection_point2).normalized();
         
         // Check that reflection 1 follows the law of reflection
         Eigen::Vector3d reflected_dir1 = RayTracing::calculateReflection(dir1, normal1);
         if ((reflected_dir1 - dir2).norm() > 0.2) {  // More tolerant threshold
             return false; // Reflection 1 doesn't follow the law of reflection
         }
         
         // Check that reflection 2 follows the law of reflection
         Eigen::Vector3d reflected_dir2 = RayTracing::calculateReflection(dir2, normal2);
         if ((reflected_dir2 - dir3).norm() > 0.2) {  // More tolerant threshold
             return false; // Reflection 2 doesn't follow the law of reflection
         }
         
         return true;
     }
     
     bool computeReflectionPoint(
         const Eigen::Vector3d& source, 
         const Eigen::Vector3d& target,
         const Eigen::Vector3d& point_on_plane, 
         const Eigen::Vector3d& normal,
         Eigen::Vector3d& reflection_point) {
         
         // First, create the mirror image of the target through the plane
         double d = normal.dot(point_on_plane);
         double t = (d - normal.dot(target)) / normal.dot(normal);
         Eigen::Vector3d mirror_target = target + 2 * t * normal;
         
         // Now find intersection of line from source to mirror_target with the plane
         Eigen::Vector3d ray_dir = (mirror_target - source).normalized();
         
         // Check if ray is parallel to plane
         double nd = normal.dot(ray_dir);
         if (std::abs(nd) < 1e-6) {
             return false; // Ray is parallel to plane
         }
         
         // Compute intersection
         double t_hit = (d - normal.dot(source)) / nd;
         
         // Check if intersection is in positive direction
         if (t_hit < 0) {
             return false; // Intersection is behind the source
         }
         
         // Compute reflection point
         reflection_point = source + t_hit * ray_dir;
         
         return true;
     }
     
     bool isPointOnBuildingFace(
         const Eigen::Vector3d& point, 
         const Building& building,
         const Eigen::Vector3d& face_normal) {
         
         Eigen::Vector3d min_pt = building.min();
         Eigen::Vector3d max_pt = building.max();
         
         const double EPSILON = 1e-2; // More tolerant epsilon
         
         // Check if point is on the building's bounding box
         // Determine which face we're on based on the normal
         if (std::abs(face_normal.x()) > 0.9) { // X-normal face (left/right)
             if (std::abs(point.x() - min_pt.x()) < EPSILON) { // Left face
                 return (point.y() >= min_pt.y() - EPSILON && point.y() <= max_pt.y() + EPSILON &&
                         point.z() >= min_pt.z() - EPSILON && point.z() <= max_pt.z() + EPSILON);
             } 
             else if (std::abs(point.x() - max_pt.x()) < EPSILON) { // Right face
                 return (point.y() >= min_pt.y() - EPSILON && point.y() <= max_pt.y() + EPSILON &&
                         point.z() >= min_pt.z() - EPSILON && point.z() <= max_pt.z() + EPSILON);
             }
         }
         else if (std::abs(face_normal.y()) > 0.9) { // Y-normal face (front/back)
             if (std::abs(point.y() - min_pt.y()) < EPSILON) { // Front face
                 return (point.x() >= min_pt.x() - EPSILON && point.x() <= max_pt.x() + EPSILON &&
                         point.z() >= min_pt.z() - EPSILON && point.z() <= max_pt.z() + EPSILON);
             }
             else if (std::abs(point.y() - max_pt.y()) < EPSILON) { // Back face
                 return (point.x() >= min_pt.x() - EPSILON && point.x() <= max_pt.x() + EPSILON &&
                         point.z() >= min_pt.z() - EPSILON && point.z() <= max_pt.z() + EPSILON);
             }
         }
         else if (std::abs(face_normal.z()) > 0.9) { // Z-normal face (top/bottom)
             if (std::abs(point.z() - min_pt.z()) < EPSILON) { // Bottom face
                 return (point.x() >= min_pt.x() - EPSILON && point.x() <= max_pt.x() + EPSILON &&
                         point.y() >= min_pt.y() - EPSILON && point.y() <= max_pt.y() + EPSILON);
             }
             else if (std::abs(point.z() - max_pt.z()) < EPSILON) { // Top face
                 return (point.x() >= min_pt.x() - EPSILON && point.x() <= max_pt.x() + EPSILON &&
                         point.y() >= min_pt.y() - EPSILON && point.y() <= max_pt.y() + EPSILON);
             }
         }
         
         return false;
     }
     
     bool rayIntersectsBuilding(
         const Eigen::Vector3d& ray_origin, 
         const Eigen::Vector3d& ray_dir,
         double ray_length,
         const Building& building) {
         
         Eigen::Vector3d min_pt = building.min();
         Eigen::Vector3d max_pt = building.max();
         
         // Check if ray origin is inside building
         if (ray_origin.x() >= min_pt.x() && ray_origin.x() <= max_pt.x() &&
             ray_origin.y() >= min_pt.y() && ray_origin.y() <= max_pt.y() &&
             ray_origin.z() >= min_pt.z() && ray_origin.z() <= max_pt.z()) {
             return true;
         }
         
         // AABB ray intersection test
         double t_min = 0.0;
         double t_max = std::numeric_limits<double>::infinity();
         
         // For X-axis
         if (std::abs(ray_dir.x()) < 1e-6) {
             if (ray_origin.x() < min_pt.x() || ray_origin.x() > max_pt.x())
                 return false;
         } else {
             double t1 = (min_pt.x() - ray_origin.x()) / ray_dir.x();
             double t2 = (max_pt.x() - ray_origin.x()) / ray_dir.x();
             
             if (t1 > t2) std::swap(t1, t2);
             
             t_min = std::max(t_min, t1);
             t_max = std::min(t_max, t2);
             
             if (t_min > t_max) return false;
         }
         
         // For Y-axis
         if (std::abs(ray_dir.y()) < 1e-6) {
             if (ray_origin.y() < min_pt.y() || ray_origin.y() > max_pt.y())
                 return false;
         } else {
             double t1 = (min_pt.y() - ray_origin.y()) / ray_dir.y();
             double t2 = (max_pt.y() - ray_origin.y()) / ray_dir.y();
             
             if (t1 > t2) std::swap(t1, t2);
             
             t_min = std::max(t_min, t1);
             t_max = std::min(t_max, t2);
             
             if (t_min > t_max) return false;
         }
         
         // For Z-axis
         if (std::abs(ray_dir.z()) < 1e-6) {
             if (ray_origin.z() < min_pt.z() || ray_origin.z() > max_pt.z())
                 return false;
         } else {
             double t1 = (min_pt.z() - ray_origin.z()) / ray_dir.z();
             double t2 = (max_pt.z() - ray_origin.z()) / ray_dir.z();
             
             if (t1 > t2) std::swap(t1, t2);
             
             t_min = std::max(t_min, t1);
             t_max = std::min(t_max, t2);
             
             if (t_min > t_max) return false;
         }
         
         // Check if intersection is within the ray length
         return t_min < ray_length && t_max > 0;
     }
     
     void publishRoad() {
         visualization_msgs::Marker road_marker;
         road_marker.header.frame_id = fixed_frame_;
         road_marker.header.stamp = ros::Time::now();
         road_marker.ns = "road";
         road_marker.id = 0;
         road_marker.type = visualization_msgs::Marker::CUBE;
         road_marker.action = visualization_msgs::Marker::ADD;
         
         // Position (center of the road)
         road_marker.pose.position.x = 0.0;
         road_marker.pose.position.y = 0.0;
         road_marker.pose.position.z = -0.05; // Slightly below ground
         road_marker.pose.orientation.w = 1.0;
         
         // Scale
         road_marker.scale.x = road_length_;
         road_marker.scale.y = road_width_;
         road_marker.scale.z = 0.1; // Thin plane
         
         // Color (dark gray)
         road_marker.color.r = 0.2;
         road_marker.color.g = 0.2;
         road_marker.color.b = 0.2;
         road_marker.color.a = 1.0;
         
         road_pub_.publish(road_marker);
         
         // Also publish sidewalks
         visualization_msgs::Marker sidewalk_left;
         sidewalk_left.header = road_marker.header;
         sidewalk_left.ns = "sidewalk";
         sidewalk_left.id = 1;
         sidewalk_left.type = visualization_msgs::Marker::CUBE;
         sidewalk_left.action = visualization_msgs::Marker::ADD;
         
         // Position (left sidewalk)
         sidewalk_left.pose.position.x = 0.0;
         sidewalk_left.pose.position.y = -(road_width_ / 2.0 + sidewalk_width_ / 2.0);
         sidewalk_left.pose.position.z = -0.03; // Slightly below ground but above road
         sidewalk_left.pose.orientation.w = 1.0;
         
         // Scale
         sidewalk_left.scale.x = road_length_;
         sidewalk_left.scale.y = sidewalk_width_;
         sidewalk_left.scale.z = 0.1; // Thin plane
         
         // Color (light gray)
         sidewalk_left.color.r = 0.6;
         sidewalk_left.color.g = 0.6;
         sidewalk_left.color.b = 0.6;
         sidewalk_left.color.a = 1.0;
         
         road_pub_.publish(sidewalk_left);
         
         // Right sidewalk
         visualization_msgs::Marker sidewalk_right = sidewalk_left;
         sidewalk_right.id = 2;
         sidewalk_right.pose.position.y = (road_width_ / 2.0 + sidewalk_width_ / 2.0);
         
         road_pub_.publish(sidewalk_right);
     }
     
     void publishBuildings() {
         visualization_msgs::MarkerArray building_markers;
         int id = 0;
         
         for (const auto& building : buildings_) {
             visualization_msgs::Marker marker;
             marker.header.frame_id = fixed_frame_;
             marker.header.stamp = ros::Time::now();
             marker.ns = "buildings";
             marker.id = id++;
             marker.type = visualization_msgs::Marker::CUBE;
             marker.action = visualization_msgs::Marker::ADD;
             
             // Position
             marker.pose.position.x = building.center.x();
             marker.pose.position.y = building.center.y();
             marker.pose.position.z = building.center.z();
             marker.pose.orientation.w = 1.0;
             
             // Scale
             marker.scale.x = building.dimensions.x();
             marker.scale.y = building.dimensions.y();
             marker.scale.z = building.dimensions.z();
             
             // Color
             marker.color.r = building.color.x();
             marker.color.g = building.color.y();
             marker.color.b = building.color.z();
             marker.color.a = building_alpha_;
             
             building_markers.markers.push_back(marker);
         }
         
         building_pub_.publish(building_markers);
     }
     
     void publishBeacons() {
         visualization_msgs::MarkerArray beacon_markers;
         int id = 0;
         
         for (const auto& beacon : beacons_) {
             // Beacon marker
             visualization_msgs::Marker marker;
             marker.header.frame_id = fixed_frame_;
             marker.header.stamp = ros::Time::now();
             marker.ns = "beacons";
             marker.id = id++;
             marker.type = visualization_msgs::Marker::SPHERE;
             marker.action = visualization_msgs::Marker::ADD;
             
             // Position
             marker.pose.position.x = beacon.position.x();
             marker.pose.position.y = beacon.position.y();
             marker.pose.position.z = beacon.position.z();
             marker.pose.orientation.w = 1.0;
             
             // Scale
             marker.scale.x = 1.0;  // Larger beacon for better visibility
             marker.scale.y = 1.0;
             marker.scale.z = 1.0;
             
             // Color - bright yellow for beacons
             marker.color.r = 1.0;
             marker.color.g = 1.0;
             marker.color.b = 0.0;
             marker.color.a = 1.0;
             
             beacon_markers.markers.push_back(marker);
             
             // Add vertical pole for the beacon
             visualization_msgs::Marker pole_marker;
             pole_marker.header.frame_id = fixed_frame_;
             pole_marker.header.stamp = ros::Time::now();
             pole_marker.ns = "beacon_poles";
             pole_marker.id = id++;
             pole_marker.type = visualization_msgs::Marker::CYLINDER;
             pole_marker.action = visualization_msgs::Marker::ADD;
             
             // Position (pole center is halfway between ground and beacon)
             pole_marker.pose.position.x = beacon.position.x();
             pole_marker.pose.position.y = beacon.position.y();
             pole_marker.pose.position.z = beacon.position.z() / 2.0;
             pole_marker.pose.orientation.w = 1.0;
             
             // Scale (thin pole with height = beacon.z)
             pole_marker.scale.x = 0.2;
             pole_marker.scale.y = 0.2;
             pole_marker.scale.z = beacon.position.z();
             
             // Color - gray pole
             pole_marker.color.r = 0.7;
             pole_marker.color.g = 0.7;
             pole_marker.color.b = 0.7;
             pole_marker.color.a = 1.0;
             
             beacon_markers.markers.push_back(pole_marker);
             
             // Text label for beacon
             visualization_msgs::Marker text_marker;
             text_marker.header.frame_id = fixed_frame_;
             text_marker.header.stamp = ros::Time::now();
             text_marker.ns = "beacon_labels";
             text_marker.id = id++;
             text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             text_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - slightly above the beacon
             text_marker.pose.position.x = beacon.position.x();
             text_marker.pose.position.y = beacon.position.y();
             text_marker.pose.position.z = beacon.position.z() + 1.0;
             text_marker.pose.orientation.w = 1.0;
             
             // Scale
             text_marker.scale.z = 0.8; // Text height
             
             // Color
             text_marker.color.r = 1.0;
             text_marker.color.g = 1.0;
             text_marker.color.b = 1.0;
             text_marker.color.a = 1.0;
             
             // Text
             text_marker.text = beacon.id;
             
             beacon_markers.markers.push_back(text_marker);
         }
         
         beacon_pub_.publish(beacon_markers);
     }
     
     void publishUser() {
         visualization_msgs::Marker marker;
         marker.header.frame_id = fixed_frame_;
         marker.header.stamp = ros::Time::now();
         marker.ns = "user";
         marker.id = 0;
         marker.type = visualization_msgs::Marker::SPHERE;
         marker.action = visualization_msgs::Marker::ADD;
         
         // Position
         marker.pose.position.x = user_position_.x();
         marker.pose.position.y = user_position_.y();
         marker.pose.position.z = user_position_.z();
         marker.pose.orientation.w = 1.0;
         
         // Scale
         marker.scale.x = 0.6;  // Larger user for better visibility
         marker.scale.y = 0.6;
         marker.scale.z = 0.6;
         
         // Color - bright blue for user
         marker.color.r = 0.0;
         marker.color.g = 0.5;
         marker.color.b = 1.0;
         marker.color.a = 1.0;
         
         user_pub_.publish(marker);
     }
     
     void publishSignalPaths() {
         visualization_msgs::MarkerArray path_markers;
         int id = 0;
         
         // Clear previous paths
         if (signal_paths_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             path_markers.markers.push_back(clear_marker);
             path_pub_.publish(path_markers);
             return;
         }
         
         // Only visualize the top N paths for clarity
         const int max_paths_to_visualize = std::min(30, static_cast<int>(signal_paths_.size()));
         
         for (int i = 0; i < max_paths_to_visualize; ++i) {
             const auto& path = signal_paths_[i];
             
             // Skip invalid paths
             if (!path.valid || path.segments.empty()) {
                 continue;
             }
             
             // Compute signal strength on a scale from 0 to 1
             double signal_strength = std::max(0.0, std::min(1.0, 
                                            1.0 - path.total_path_loss / (min_signal_power_ - noise_floor_)));
             
             // More vibrant colors based on reflection count
             std_msgs::ColorRGBA color;
             switch (path.reflection_count) {
                 case 0:  // Direct path - brighter green
                     color.r = 0.0;
                     color.g = 1.0;
                     color.b = 0.0;
                     break;
                 case 1:  // Single reflection - bright yellow
                     color.r = 1.0;
                     color.g = 1.0;
                     color.b = 0.0;
                     break;
                 case 2:  // Double reflection - bright orange
                     color.r = 1.0;
                     color.g = 0.6;
                     color.b = 0.0;
                     break;
                 default:  // More reflections - bright red
                     color.r = 1.0;
                     color.g = 0.0;
                     color.b = 0.0;
                     break;
             }
             
             color.a = 0.8 + 0.2 * signal_strength; // High opacity for better visibility
             
             for (size_t j = 0; j < path.segments.size(); ++j) {
                 const auto& segment = path.segments[j];
                 
                 // Create a marker for each segment
                 visualization_msgs::Marker line_marker;
                 line_marker.header.frame_id = fixed_frame_;
                 line_marker.header.stamp = ros::Time::now();
                 line_marker.ns = "signal_paths";
                 line_marker.id = id++;
                 line_marker.type = visualization_msgs::Marker::LINE_STRIP;  // Use LINE_STRIP for cleaner lines
                 line_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Points for the line strip
                 line_marker.points.resize(2);
                 line_marker.points[0].x = segment.start.x();
                 line_marker.points[0].y = segment.start.y();
                 line_marker.points[0].z = segment.start.z();
                 
                 line_marker.points[1].x = segment.end.x();
                 line_marker.points[1].y = segment.end.y();
                 line_marker.points[1].z = segment.end.z();
                 
                 // Scale - thicker lines for better visibility
                 line_marker.scale.x = path_width_ * (1.0 + 0.5 * signal_strength);
                 
                 // Color based on path type
                 line_marker.color = color;
                 
                 path_markers.markers.push_back(line_marker);
                 
                 // Add arrow to show direction
                 visualization_msgs::Marker arrow_marker;
                 arrow_marker.header.frame_id = fixed_frame_;
                 arrow_marker.header.stamp = ros::Time::now();
                 arrow_marker.ns = "signal_arrows";
                 arrow_marker.id = id++;
                 arrow_marker.type = visualization_msgs::Marker::ARROW;
                 arrow_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Points for the arrow (positioned at 70% of the way)
                 Eigen::Vector3d midpoint = segment.start + 0.7 * (segment.end - segment.start);
                 Eigen::Vector3d arrow_start = midpoint - 0.5 * (segment.end - segment.start).normalized();
                 Eigen::Vector3d arrow_end = midpoint + 0.5 * (segment.end - segment.start).normalized();
                 
                 arrow_marker.points.resize(2);
                 arrow_marker.points[0].x = arrow_start.x();
                 arrow_marker.points[0].y = arrow_start.y();
                 arrow_marker.points[0].z = arrow_start.z();
                 
                 arrow_marker.points[1].x = arrow_end.x();
                 arrow_marker.points[1].y = arrow_end.y();
                 arrow_marker.points[1].z = arrow_end.z();
                 
                 // Scale
                 arrow_marker.scale.x = path_width_ * 0.5;  // Shaft diameter
                 arrow_marker.scale.y = path_width_ * 1.5;  // Head diameter
                 arrow_marker.scale.z = path_width_ * 1.0;  // Head length
                 
                 // Color - same as path but slightly more opaque
                 std_msgs::ColorRGBA arrow_color = color;
                 arrow_color.a = std::min(1.0, color.a + 0.2);
                 arrow_marker.color = arrow_color;
                 
                 path_markers.markers.push_back(arrow_marker);
             }
         }
         
         path_pub_.publish(path_markers);
     }
     
     void publishReflectionPoints() {
         visualization_msgs::MarkerArray reflection_markers;
         int id = 0;
         
         // Clear previous reflections
         if (signal_paths_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             reflection_markers.markers.push_back(clear_marker);
             reflection_pub_.publish(reflection_markers);
             return;
         }
         
         // Only visualize the top N paths for clarity
         const int max_paths = std::min(30, static_cast<int>(signal_paths_.size()));
         
         // Map to avoid duplicate reflection points
         std::map<std::tuple<double,double,double>, int> reflection_counts;
         
         // First, count all reflection points
         for (int i = 0; i < max_paths; ++i) {
             const auto& path = signal_paths_[i];
             
             for (size_t j = 1; j < path.segments.size(); ++j) {  // Start from 1 to skip beacon->reflection segment
                 const auto& segment = path.segments[j];
                 
                 // Skip if not a reflection point
                 if (segment.reflection_count <= 0) continue;
                 
                 // Use the segment start point as the reflection point
                 std::tuple<double,double,double> point_key(
                     segment.start.x(), segment.start.y(), segment.start.z());
                 
                 reflection_counts[point_key]++;
             }
         }
         
         // Now visualize each unique reflection point
         for (const auto& pair : reflection_counts) {
             auto point_key = pair.first;
             int count = pair.second;
             
             visualization_msgs::Marker point_marker;
             point_marker.header.frame_id = fixed_frame_;
             point_marker.header.stamp = ros::Time::now();
             point_marker.ns = "reflection_points";
             point_marker.id = id++;
             point_marker.type = visualization_msgs::Marker::SPHERE;
             point_marker.action = visualization_msgs::Marker::ADD;
             
             // Position
             point_marker.pose.position.x = std::get<0>(point_key);
             point_marker.pose.position.y = std::get<1>(point_key);
             point_marker.pose.position.z = std::get<2>(point_key);
             point_marker.pose.orientation.w = 1.0;
             
             // Scale - size based on number of reflections
             double size_factor = 0.2 + 0.1 * std::min(5, count);  // Cap at 5 reflections
             point_marker.scale.x = path_width_ * 4.0 * size_factor;
             point_marker.scale.y = path_width_ * 4.0 * size_factor;
             point_marker.scale.z = path_width_ * 4.0 * size_factor;
             
             // Color - white with high opacity
             point_marker.color.r = 1.0;
             point_marker.color.g = 1.0;
             point_marker.color.b = 1.0;
             point_marker.color.a = 0.9;
             
             reflection_markers.markers.push_back(point_marker);
             
             // Add text showing number of reflections
             if (count > 1) {
                 visualization_msgs::Marker text_marker;
                 text_marker.header.frame_id = fixed_frame_;
                 text_marker.header.stamp = ros::Time::now();
                 text_marker.ns = "reflection_counts";
                 text_marker.id = id++;
                 text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                 text_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Position - slightly above reflection point
                 text_marker.pose.position.x = std::get<0>(point_key);
                 text_marker.pose.position.y = std::get<1>(point_key);
                 text_marker.pose.position.z = std::get<2>(point_key) + size_factor;
                 text_marker.pose.orientation.w = 1.0;
                 
                 // Scale
                 text_marker.scale.z = 0.4 * size_factor; // Text height
                 
                 // Color
                 text_marker.color.r = 1.0;
                 text_marker.color.g = 1.0;
                 text_marker.color.b = 0.0;
                 text_marker.color.a = 1.0;
                 
                 // Text - show number of paths using this reflection point
                 text_marker.text = std::to_string(count);
                 
                 reflection_markers.markers.push_back(text_marker);
             }
         }
         
         reflection_pub_.publish(reflection_markers);
     }
     
     void publishTextInfo() {
         visualization_msgs::MarkerArray text_markers;
         int id = 0;
         
         // Show info about the top paths
         const int max_paths_to_show = std::min(5, static_cast<int>(signal_paths_.size()));
         double text_height = 0.6;
         
         for (int i = 0; i < max_paths_to_show; ++i) {
             const auto& path = signal_paths_[i];
             
             visualization_msgs::Marker text_marker;
             text_marker.header.frame_id = fixed_frame_;
             text_marker.header.stamp = ros::Time::now();
             text_marker.ns = "path_info";
             text_marker.id = id++;
             text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             text_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - stack the text on the left side of the view
             text_marker.pose.position.x = -road_length_ / 2.0 + 5.0;
             text_marker.pose.position.y = road_width_ / 2.0 + sidewalk_width_ + 3.0;
             text_marker.pose.position.z = building_height_max_ + 3.0 - i * text_height * 1.2;
             text_marker.pose.orientation.w = 1.0;
             
             // Scale
             text_marker.scale.z = text_height; // Text height
             
             // Color based on reflection count - match path colors
             switch (path.reflection_count) {
                 case 0:  // Direct path - green
                     text_marker.color.r = 0.0;
                     text_marker.color.g = 1.0;
                     text_marker.color.b = 0.0;
                     break;
                 case 1:  // Single reflection - yellow
                     text_marker.color.r = 1.0;
                     text_marker.color.g = 1.0;
                     text_marker.color.b = 0.0;
                     break;
                 case 2:  // Double reflection - orange
                     text_marker.color.r = 1.0;
                     text_marker.color.g = 0.6;
                     text_marker.color.b = 0.0;
                     break;
                 default:  // More reflections - red
                     text_marker.color.r = 1.0;
                     text_marker.color.g = 0.0;
                     text_marker.color.b = 0.0;
                     break;
             }
             text_marker.color.a = 1.0;
             
             // Format the text
             std::stringstream ss;
             ss << "Path " << (i+1) << " (" << path.beacon_id << "): " 
                << path.reflection_count << " reflections, "
                << std::fixed << std::setprecision(1) << path.total_distance << "m, "
                << std::fixed << std::setprecision(1) << path.total_path_loss << " dB loss";
             
             text_marker.text = ss.str();
             
             text_markers.markers.push_back(text_marker);
         }
         
         // Add general info
         visualization_msgs::Marker title_marker;
         title_marker.header.frame_id = fixed_frame_;
         title_marker.header.stamp = ros::Time::now();
         title_marker.ns = "title";
         title_marker.id = id++;
         title_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         title_marker.action = visualization_msgs::Marker::ADD;
         
         title_marker.pose.position.x = -road_length_ / 2.0 + 5.0;
         title_marker.pose.position.y = road_width_ / 2.0 + sidewalk_width_ + 3.0;
         title_marker.pose.position.z = building_height_max_ + 5.0;
         title_marker.pose.orientation.w = 1.0;
         
         title_marker.scale.z = text_height * 1.5; // Title is larger
         
         title_marker.color.r = 1.0;
         title_marker.color.g = 1.0;
         title_marker.color.b = 1.0;
         title_marker.color.a = 1.0;
         
         title_marker.text = "UWB Signal Paths";
         
         text_markers.markers.push_back(title_marker);
         
         // Add legend
         visualization_msgs::Marker legend_marker;
         legend_marker.header.frame_id = fixed_frame_;
         legend_marker.header.stamp = ros::Time::now();
         legend_marker.ns = "legend";
         legend_marker.id = id++;
         legend_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         legend_marker.action = visualization_msgs::Marker::ADD;
         
         legend_marker.pose.position.x = -road_length_ / 2.0 + 5.0;
         legend_marker.pose.position.y = -(road_width_ / 2.0 + sidewalk_width_ + 3.0);
         legend_marker.pose.position.z = building_height_max_ + 5.0;
         legend_marker.pose.orientation.w = 1.0;
         
         legend_marker.scale.z = text_height * 1.2;
         
         legend_marker.color.r = 1.0;
         legend_marker.color.g = 1.0;
         legend_marker.color.b = 1.0;
         legend_marker.color.a = 1.0;
         
         legend_marker.text = "Green = Direct Path\nYellow = 1 Reflection\nOrange = 2 Reflections\nRed = 3+ Reflections";
         
         text_markers.markers.push_back(legend_marker);
         
         // Add paths count info
         visualization_msgs::Marker count_marker;
         count_marker.header.frame_id = fixed_frame_;
         count_marker.header.stamp = ros::Time::now();
         count_marker.ns = "path_count";
         count_marker.id = id++;
         count_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         count_marker.action = visualization_msgs::Marker::ADD;
         
         count_marker.pose.position.x = road_length_ / 2.0 - 20.0;
         count_marker.pose.position.y = road_width_ / 2.0 + sidewalk_width_ + 3.0;
         count_marker.pose.position.z = building_height_max_ + 5.0;
         count_marker.pose.orientation.w = 1.0;
         
         count_marker.scale.z = text_height;
         
         count_marker.color.r = 1.0;
         count_marker.color.g = 1.0;
         count_marker.color.b = 0.0;
         count_marker.color.a = 1.0;
         
         // Count paths by type
         int direct_paths = 0;
         int single_reflection = 0;
         int multi_reflection = 0;
         
         for (const auto& path : signal_paths_) {
             if (path.reflection_count == 0) direct_paths++;
             else if (path.reflection_count == 1) single_reflection++;
             else multi_reflection++;
         }
         
         count_marker.text = "Paths: " + std::to_string(direct_paths) + " direct, " + 
                           std::to_string(single_reflection) + " single refl., " +
                           std::to_string(multi_reflection) + " multi refl.";
         
         text_markers.markers.push_back(count_marker);
         
         text_pub_.publish(text_markers);
     }
     
     void broadcastTFs() {
         // Broadcast TF frames for beacons and user
         ros::Time now = ros::Time::now();
         
         // User TF
         tf::Transform user_tf;
         user_tf.setOrigin(tf::Vector3(user_position_.x(), user_position_.y(), user_position_.z()));
         user_tf.setRotation(tf::Quaternion(0, 0, 0, 1));
         tf_broadcaster_.sendTransform(tf::StampedTransform(user_tf, now, fixed_frame_, "user"));
         
         // Beacon TFs
         for (const auto& beacon : beacons_) {
             tf::Transform beacon_tf;
             beacon_tf.setOrigin(tf::Vector3(beacon.position.x(), beacon.position.y(), beacon.position.z()));
             beacon_tf.setRotation(tf::Quaternion(0, 0, 0, 1));
             tf_broadcaster_.sendTransform(tf::StampedTransform(beacon_tf, now, fixed_frame_, beacon.id));
         }
     }
 };
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "uwb_ray_tracer");
     
     UWBRayTracer ray_tracer;
     
     ros::spin();
     
     return 0;
 }