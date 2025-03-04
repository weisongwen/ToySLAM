/**
 * UWB Ray Tracing ROS Node - With Moving User and Building Penetration Prevention
 * 
 * Features:
 * - User moves along a configurable slow trajectory
 * - Fixed building penetration detection for all paths
 * - Signal paths update as user moves
 * - Reflections follow the laws of physics
 * - 10 UWB beacons with direct and reflection paths
 * - Visualization of all signal paths and reflection points
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
 #include <limits>
 
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
 
     // Get the faces of the building
     struct BuildingFace {
         Eigen::Vector3d center;     // Center of the face
         Eigen::Vector3d normal;     // Normal vector of the face
         Eigen::Vector3d dimensions; // Dimensions of the face (width, height)
         Eigen::Vector3d tangent;    // Tangent vector (horizontal)
         Eigen::Vector3d bitangent;  // Bitangent vector (vertical)
         bool is_active;             // Whether this face is currently reflecting
     };
     
     std::vector<BuildingFace> getFaces() const {
         std::vector<BuildingFace> faces;
         Eigen::Vector3d min_pt = min();
         Eigen::Vector3d max_pt = max();
         
         // Front face (normal = -y)
         faces.push_back({
             Eigen::Vector3d(center.x(), min_pt.y(), center.z()),  // Center
             Eigen::Vector3d(0, -1, 0),                           // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 0, 1),                            // Bitangent (z-direction)
             false                                                // Not active initially
         });
         
         // Back face (normal = +y)
         faces.push_back({
             Eigen::Vector3d(center.x(), max_pt.y(), center.z()),  // Center
             Eigen::Vector3d(0, 1, 0),                            // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 0, 1),                            // Bitangent (z-direction)
             false                                                // Not active initially
         });
         
         // Left face (normal = -x)
         faces.push_back({
             Eigen::Vector3d(min_pt.x(), center.y(), center.z()),  // Center
             Eigen::Vector3d(-1, 0, 0),                           // Normal
             Eigen::Vector3d(dimensions.y(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(0, 1, 0),                            // Tangent (y-direction)
             Eigen::Vector3d(0, 0, 1),                            // Bitangent (z-direction)
             false                                                // Not active initially
         });
         
         // Right face (normal = +x)
         faces.push_back({
             Eigen::Vector3d(max_pt.x(), center.y(), center.z()),  // Center
             Eigen::Vector3d(1, 0, 0),                            // Normal
             Eigen::Vector3d(dimensions.y(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(0, 1, 0),                            // Tangent (y-direction)
             Eigen::Vector3d(0, 0, 1),                            // Bitangent (z-direction)
             false                                                // Not active initially
         });
         
         // Bottom face (normal = -z)
         faces.push_back({
             Eigen::Vector3d(center.x(), center.y(), min_pt.z()),  // Center
             Eigen::Vector3d(0, 0, -1),                           // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.y(), 0),  // Dimensions (width, depth)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 1, 0),                            // Bitangent (y-direction)
             false                                                // Not active initially
         });
         
         // Top face (normal = +z)
         faces.push_back({
             Eigen::Vector3d(center.x(), center.y(), max_pt.z()),  // Center
             Eigen::Vector3d(0, 0, 1),                            // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.y(), 0),  // Dimensions (width, depth)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 1, 0),                            // Bitangent (y-direction)
             false                                                // Not active initially
         });
         
         return faces;
     }
     
     // Check if a point is inside or very close to the building
     bool containsPoint(const Eigen::Vector3d& point, double epsilon = 1e-3) const {
         Eigen::Vector3d min_p = min();
         Eigen::Vector3d max_p = max();
         
         return (point.x() >= min_p.x() - epsilon) && (point.x() <= max_p.x() + epsilon) &&
                (point.y() >= min_p.y() - epsilon) && (point.y() <= max_p.y() + epsilon) && 
                (point.z() >= min_p.z() - epsilon) && (point.z() <= max_p.z() + epsilon);
     }
 };
 
 // Structure to represent a UWB Beacon
 struct UWBBeacon {
     std::string id;
     Eigen::Vector3d position;
     double range;         // Max range in meters
     double frequency;     // Signal frequency in MHz
     double power;         // Transmit power in dBm
     
     UWBBeacon() : range(150.0), frequency(6500.0), power(20.0) {}
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
     int reflection_face_id;     // ID of the face causing reflection, -1 if no reflection
     double incident_angle;      // Angle of incidence (degrees)
     bool penetrates_building;   // Flag to indicate if this segment penetrates any building
     
     PathSegment(const Eigen::Vector3d& s, const Eigen::Vector3d& e, int rc, 
                 double pl, double d, double r, int b_id = -1, int f_id = -1, double angle = 0.0) 
         : start(s), end(e), reflection_count(rc), 
           path_loss(pl), total_distance(d), reflectivity(r),
           reflection_building_id(b_id), reflection_face_id(f_id),
           incident_angle(angle), penetrates_building(false) {}
 };
 
 // Structure for a ray tracing path
 struct SignalPath {
     std::vector<PathSegment> segments;
     double total_path_loss;   // Total path loss in dB
     double total_distance;    // Total path length in meters
     int reflection_count;     // Number of reflections
     bool valid;               // Whether path is valid (not blocked)
     std::string beacon_id;    // ID of the source beacon
     bool forced;              // Whether this is a forced path
     bool penetrates_building; // Whether any segment penetrates a building
     
     SignalPath() : total_path_loss(0.0), total_distance(0.0), 
                   reflection_count(0), valid(true), beacon_id("unknown"),
                   forced(false), penetrates_building(false) {}
     
     void addSegment(const PathSegment& segment) {
         segments.push_back(segment);
         total_path_loss += segment.path_loss;
         total_distance = segment.total_distance;
         reflection_count = segment.reflection_count;
         if (segment.penetrates_building) {
             penetrates_building = true;
         }
     }
 };
 
 // Ray-tracing specific constants and calculations
 namespace RayTracing {
     constexpr double SPEED_OF_LIGHT = 299792458.0;
     constexpr double EPSILON = 1e-4;         // Epsilon for numerical stability
     constexpr double ANGLE_EPSILON = 0.15;   // Angle verification tolerance
     constexpr double FACE_EPSILON = 0.25;    // Face boundary tolerance
     
     double calculateWavelength(double frequency_mhz) {
         return SPEED_OF_LIGHT / (frequency_mhz * 1e6);
     }
     
     double calculateFreeSpacePathLoss(double distance, double frequency_mhz) {
         // FSPL (dB) = 20*log10(d) + 20*log10(f) + 32.44
         // where d is distance in km and f is frequency in MHz
         double distance_km = distance / 1000.0;
         return 20.0 * std::log10(distance_km) + 20.0 * std::log10(frequency_mhz) + 32.44;
     }
     
     double calculateReflectionCoefficient(double incident_angle_rad, double surface_reflectivity) {
         double grazing_factor = std::sin(incident_angle_rad);
         return surface_reflectivity * grazing_factor;
     }
     
     Eigen::Vector3d calculateReflection(const Eigen::Vector3d& incident, 
                                         const Eigen::Vector3d& normal) {
         // Ensure normal is normalized
         Eigen::Vector3d unit_normal = normal.normalized();
         
         // Calculate reflection vector: r = i - 2(i·n)n
         return incident - 2 * incident.dot(unit_normal) * unit_normal;
     }
     
     double calculateIncidentAngle(const Eigen::Vector3d& incident, 
                                    const Eigen::Vector3d& normal) {
         // Calculate angle between incident ray and normal
         // Make sure vectors are normalized
         Eigen::Vector3d unit_incident = incident.normalized();
         Eigen::Vector3d unit_normal = normal.normalized();
         
         // Get cosine of angle between vectors
         double cos_angle = std::abs(unit_incident.dot(unit_normal));
         
         // Return angle in radians
         return std::acos(cos_angle);
     }
     
     // Verify reflection satisfies the law of reflection
     bool verifyReflectionLaw(
         const Eigen::Vector3d& incident,
         const Eigen::Vector3d& reflected,
         const Eigen::Vector3d& normal,
         double tolerance = ANGLE_EPSILON) {
         
         // Normalize all vectors
         Eigen::Vector3d unit_incident = incident.normalized();
         Eigen::Vector3d unit_reflected = reflected.normalized();
         Eigen::Vector3d unit_normal = normal.normalized();
         
         // Compute theoretical reflection
         Eigen::Vector3d theoretical_reflection = calculateReflection(unit_incident, unit_normal);
         
         // Check if actual reflection matches theoretical
         double deviation = (theoretical_reflection - unit_reflected).norm();
         
         return deviation < tolerance;
     }
     
     // Check if a point is within a rectangular face
     bool isPointOnFace(
         const Eigen::Vector3d& point,
         const Eigen::Vector3d& face_center,
         const Eigen::Vector3d& face_normal,
         const Eigen::Vector3d& face_tangent,
         const Eigen::Vector3d& face_bitangent,
         const Eigen::Vector3d& face_dimensions,
         double epsilon = FACE_EPSILON) {
         
         // Check if point is on the plane
         double distance_from_plane = std::abs((point - face_center).dot(face_normal));
         if (distance_from_plane > epsilon) {
             return false;
         }
         
         // Project point onto face plane
         Eigen::Vector3d point_on_plane = point - face_normal * 
                                         ((point - face_center).dot(face_normal));
         
         // Calculate vector from face center to projected point
         Eigen::Vector3d to_point = point_on_plane - face_center;
         
         // Project this vector onto tangent and bitangent
         double tangent_proj = to_point.dot(face_tangent);
         double bitangent_proj = to_point.dot(face_bitangent);
         
         // Check if projected point is within the face bounds
         double half_width = face_dimensions.x() / 2.0 + epsilon;
         double half_height = face_dimensions.y() / 2.0 + epsilon;
         
         return (std::abs(tangent_proj) <= half_width) && 
                (std::abs(bitangent_proj) <= half_height);
     }
 
     // Safe calculation of inverse with checks for division by zero
     double safeInverse(double value) {
         const double MIN_VALUE = 1e-10;
         if (std::abs(value) < MIN_VALUE) {
             return std::copysign(1.0/MIN_VALUE, value);
         }
         return 1.0 / value;
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
     ros::Publisher reflection_surface_pub_;
     ros::Publisher text_pub_;
     ros::Publisher debug_pub_;
     ros::Publisher trajectory_pub_;
     
     // Timers
     ros::Timer update_timer_;
     ros::Timer movement_timer_;
     
     // TF broadcaster
     tf::TransformBroadcaster tf_broadcaster_;
     
     // Lists of objects
     std::vector<Building> buildings_;
     std::vector<UWBBeacon> beacons_;
     Eigen::Vector3d user_position_;
     std::vector<SignalPath> signal_paths_;
     std::vector<Eigen::Vector3d> user_trajectory_;
     
     // Track actively reflecting faces for visualization
     std::vector<std::pair<int, int>> active_reflection_faces_; // (building_idx, face_idx)
     
     // Debug visualization
     std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> debug_penetration_points_;
     
     // Parameters
     int max_reflections_;     
     double max_distance_;     
     double min_signal_power_; 
     double noise_floor_;      
     
     // Road parameters
     double road_length_;
     double road_width_;
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
     
     // User movement parameters
     bool enable_user_movement_;
     std::string movement_type_;
     double movement_speed_;
     double movement_radius_;
     double movement_height_;
     double movement_period_;
     double movement_phase_;
     double current_time_;
     int current_trajectory_point_;
     
     // Visualization parameters
     std::string fixed_frame_;
     double building_alpha_;
     double path_width_;
     bool debug_mode_;
     bool slow_mode_;
     bool force_reflections_;
     bool special_check_enabled_;
     int penetration_samples_;
     
     // Random number generator
     std::mt19937 rng_;
     std::uniform_real_distribution<double> height_dist_;
     std::uniform_real_distribution<double> width_dist_;
     std::uniform_real_distribution<double> depth_dist_;
     std::uniform_real_distribution<double> reflectivity_dist_;
     std::uniform_real_distribution<double> color_dist_;
 
 public:
     UWBRayTracer() : private_nh_("~"), 
                      rng_(std::random_device()()),
                      current_time_(0.0),
                      current_trajectory_point_(0) {
         // Get parameters
         private_nh_.param<int>("max_reflections", max_reflections_, 2);
         private_nh_.param<double>("max_distance", max_distance_, 150.0);
         private_nh_.param<double>("min_signal_power", min_signal_power_, -90.0);
         private_nh_.param<double>("noise_floor", noise_floor_, -100.0);
         private_nh_.param<std::string>("fixed_frame", fixed_frame_, "map");
         private_nh_.param<double>("building_alpha", building_alpha_, 0.5);
         private_nh_.param<double>("path_width", path_width_, 0.2);
         private_nh_.param<bool>("debug_mode", debug_mode_, true);
         private_nh_.param<bool>("slow_mode", slow_mode_, true);
         private_nh_.param<bool>("force_reflections", force_reflections_, true);
         private_nh_.param<bool>("special_check_enabled", special_check_enabled_, true);
         private_nh_.param<int>("penetration_samples", penetration_samples_, 20);
         
         // Road and environment parameters
         private_nh_.param<double>("road_length", road_length_, 100.0);
         private_nh_.param<double>("road_width", road_width_, 10.0);
         private_nh_.param<double>("sidewalk_width", sidewalk_width_, 3.0);
         private_nh_.param<double>("building_height_min", building_height_min_, 10.0);
         private_nh_.param<double>("building_height_max", building_height_max_, 20.0);
         private_nh_.param<double>("building_width_min", building_width_min_, 10.0);
         private_nh_.param<double>("building_width_max", building_width_max_, 15.0);
         private_nh_.param<double>("building_depth_min", building_depth_min_, 8.0);
         private_nh_.param<double>("building_depth_max", building_depth_max_, 12.0);
         private_nh_.param<double>("buildings_per_side", buildings_per_side_, 12.0);
         private_nh_.param<double>("beacon_height", beacon_height_, 15.0);
         
         // User movement parameters
         private_nh_.param<bool>("enable_user_movement", enable_user_movement_, true);
         private_nh_.param<std::string>("movement_type", movement_type_, "circuit");
         private_nh_.param<double>("movement_speed", movement_speed_, 0.5);  // meters per second
         private_nh_.param<double>("movement_radius", movement_radius_, road_width_ * 0.4);
         private_nh_.param<double>("movement_height", movement_height_, 1.7);
         private_nh_.param<double>("movement_period", movement_period_, 60.0);  // seconds for one cycle
         private_nh_.param<double>("movement_phase", movement_phase_, 0.0);
         
         // Initialize user position to starting position based on movement type
         if (movement_type_ == "figure8") {
             user_position_ = Eigen::Vector3d(0.0, 0.0, movement_height_);
         } else if (movement_type_ == "circle") {
             user_position_ = Eigen::Vector3d(movement_radius_, 0.0, movement_height_);
         } else if (movement_type_ == "circuit") {
             // Start at beginning of circular track
             user_position_ = Eigen::Vector3d(-road_length_/4.0, -road_width_/3.0, movement_height_);
         } else {
             // Default position at center
             user_position_ = Eigen::Vector3d(0.0, 0.0, movement_height_);
         }
         
         // Initialize random distributions
         height_dist_ = std::uniform_real_distribution<double>(building_height_min_, building_height_max_);
         width_dist_ = std::uniform_real_distribution<double>(building_width_min_, building_width_max_);
         depth_dist_ = std::uniform_real_distribution<double>(building_depth_min_, building_depth_max_);
         reflectivity_dist_ = std::uniform_real_distribution<double>(0.6, 0.9);
         color_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
         
         // Publishers
         building_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/buildings", 1);
         road_pub_ = nh_.advertise<visualization_msgs::Marker>("uwb_ray_tracer/road", 1);
         beacon_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/beacons", 1);
         user_pub_ = nh_.advertise<visualization_msgs::Marker>("uwb_ray_tracer/user", 1);
         path_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/paths", 1);
         reflection_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/reflections", 1);
         reflection_surface_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/reflection_surfaces", 1);
         text_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/text_info", 1);
         debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("uwb_ray_tracer/debug", 1);
         trajectory_pub_ = nh_.advertise<visualization_msgs::Marker>("uwb_ray_tracer/trajectory", 1);
         
         // Generate environment
         generateEnvironment();
         
         // Create 10 UWB beacons
         createTenBeacons();
         
         // Generate trajectory if using circuit mode
         if (movement_type_ == "circuit") {
             generateCircuitTrajectory();
         }
         
         // Create a timer for updates - slower in slow mode
         double update_rate = slow_mode_ ? 2.0 : 10.0;
         private_nh_.param<double>("update_rate", update_rate, update_rate);
         update_timer_ = nh_.createTimer(ros::Duration(1.0/update_rate), 
                                         &UWBRayTracer::updateCallback, this);
         
         // Create a timer for movement - faster updates for smoother motion
         double movement_update_rate = 10.0;
         if (enable_user_movement_) {
             movement_timer_ = nh_.createTimer(ros::Duration(1.0/movement_update_rate), 
                                           &UWBRayTracer::movementCallback, this);
         }
         
         ROS_INFO("UWB Ray Tracer initialized with %zu buildings and %zu beacons", 
                 buildings_.size(), beacons_.size());
         ROS_INFO("User will move with %s trajectory at %.2f m/s", 
                 movement_type_.c_str(), movement_speed_);
     }
     
     ~UWBRayTracer() {
         clearVisualizations();
     }
     
     void updateCallback(const ros::TimerEvent& event) {
         // Clear debug markers
         debug_penetration_points_.clear();
         
         // Clear paths when user moves significantly
         static Eigen::Vector3d last_compute_position = user_position_;
         double distance_moved = (user_position_ - last_compute_position).norm();
         
         if (distance_moved > 1.0 || signal_paths_.empty()) {
             // User has moved enough to require recomputing paths
             signal_paths_.clear();
             last_compute_position = user_position_;
             
             // In slow mode, add artificial delay for easier visualization
             if (slow_mode_) {
                 ROS_INFO("Computing signal paths for user at (%.2f, %.2f, %.2f)...", 
                          user_position_.x(), user_position_.y(), user_position_.z());
                 ros::Duration(0.1).sleep();  // Add delay to simulate computation time
             }
             
             // Compute paths
             computeSignalPaths();
             
             // Verify all paths don't pass through buildings
             validateAllPaths();
             
             // Ensure direct paths exist for all beacons
             ensureDirectPaths();
             
             // If not enough reflection paths found, add more
             if (force_reflections_ && !hasEnoughReflectionPaths()) {
                 ROS_INFO("Adding more reflection paths for visualization...");
                 addForcedReflections();
                 
                 // Re-validate all paths
                 validateAllPaths();
             }
         }
         
         // Publish visualizations
         publishRoad();
         publishBuildings();
         publishBeacons();
         publishUser();
         publishReflectionSurfaces();
         publishSignalPaths();
         publishReflectionPoints();
         publishTextInfo();
         
         // Publish trajectory visualization
         if (movement_type_ == "circuit" && !user_trajectory_.empty()) {
             publishTrajectory();
         }
         
         // Publish debug visualizations
         if (debug_mode_) {
             publishDebugMarkers();
         }
         
         // Broadcast TF frames
         broadcastTFs();
     }
     
     void movementCallback(const ros::TimerEvent& event) {
         if (!enable_user_movement_) {
             return;
         }
         
         // Update time
         double dt = event.current_real.toSec() - event.last_real.toSec();
         current_time_ += dt;
         
         if (movement_type_ == "figure8") {
             updateFigure8Motion(dt);
         } else if (movement_type_ == "circle") {
             updateCircularMotion(dt);
         } else if (movement_type_ == "circuit") {
             updateCircuitMotion(dt);
         }
     }
     
     void updateFigure8Motion(double dt) {
         // Figure-8 motion in XY plane
         double phase = 2 * M_PI * current_time_ / movement_period_ + movement_phase_;
         
         double x = movement_radius_ * std::sin(phase);
         double y = movement_radius_ * std::sin(phase) * std::cos(phase);
         
         user_position_ = Eigen::Vector3d(x, y, movement_height_);
     }
     
     void updateCircularMotion(double dt) {
         // Circular motion in XY plane
         double phase = 2 * M_PI * current_time_ / movement_period_ + movement_phase_;
         
         double x = movement_radius_ * std::cos(phase);
         double y = movement_radius_ * std::sin(phase);
         
         user_position_ = Eigen::Vector3d(x, y, movement_height_);
     }
     
     void generateCircuitTrajectory() {
         // Generate a circuit around the whole road area
         user_trajectory_.clear();
         
         // Calculate the number of points based on the circuit perimeter and desired speed
         double circuit_width = road_width_ * 0.7;  // Stay within road bounds
         double circuit_length = road_length_ * 0.5; // Use half the road length
         
         // Calculate perimeter (approximately)
         double perimeter = 2.0 * (circuit_width + circuit_length);
         
         // Time to complete the circuit at the specified speed
         double circuit_time = perimeter / movement_speed_;
         
         // Number of points to generate (at least 100, but more for longer circuits)
         int num_points = std::max(100, static_cast<int>(perimeter / 0.5));
         
         // Set movement period based on the circuit time
         movement_period_ = circuit_time;
         
         // Offset for the circuit center
         double x_offset = -road_length_/4.0;
         double y_offset = 0.0;
         
         // Generate points for the circuit
         for (int i = 0; i < num_points; i++) {
             double t = static_cast<double>(i) / num_points;
             double angle = 2.0 * M_PI * t;
             
             // Parametric equation for a rounded rectangle/oval
             double x, y;
             
             if (angle < M_PI / 2.0) {
                 // Top-right quadrant
                 double a = angle / (M_PI / 2.0);
                 x = circuit_length/2.0 * (1.0 - a) + circuit_length/2.0 * std::cos(M_PI/2.0 * a);
                 y = circuit_width/2.0 * std::sin(M_PI/2.0 * a);
             } else if (angle < M_PI) {
                 // Top-left quadrant
                 double a = (angle - M_PI/2.0) / (M_PI/2.0);
                 x = -circuit_length/2.0 * a - circuit_length/2.0 * std::sin(M_PI/2.0 * a);
                 y = circuit_width/2.0 * std::cos(M_PI/2.0 * a);
             } else if (angle < 3.0 * M_PI / 2.0) {
                 // Bottom-left quadrant
                 double a = (angle - M_PI) / (M_PI/2.0);
                 x = -circuit_length/2.0 * (1.0 - a) - circuit_length/2.0 * std::cos(M_PI/2.0 * a);
                 y = -circuit_width/2.0 * std::sin(M_PI/2.0 * a);
             } else {
                 // Bottom-right quadrant
                 double a = (angle - 3.0*M_PI/2.0) / (M_PI/2.0);
                 x = circuit_length/2.0 * a + circuit_length/2.0 * std::sin(M_PI/2.0 * a);
                 y = -circuit_width/2.0 * std::cos(M_PI/2.0 * a);
             }
             
             // Add offset
             x += x_offset;
             y += y_offset;
             
             // Add point to trajectory
             user_trajectory_.push_back(Eigen::Vector3d(x, y, movement_height_));
         }
         
         // Close the loop
         if (!user_trajectory_.empty()) {
             user_trajectory_.push_back(user_trajectory_[0]);
         }
         
         ROS_INFO("Generated circuit trajectory with %zu points, period: %.1f seconds",
                  user_trajectory_.size(), movement_period_);
     }
     
     void updateCircuitMotion(double dt) {
         if (user_trajectory_.empty()) {
             return;
         }
         
         // Move along predefined trajectory
         double distance_increment = movement_speed_ * dt;
         double total_distance = 0.0;
         
         // Find next trajectory point
         while (total_distance < distance_increment) {
             // Calculate distance to next point
             int next_point = (current_trajectory_point_ + 1) % user_trajectory_.size();
             
             Eigen::Vector3d current = user_trajectory_[current_trajectory_point_];
             Eigen::Vector3d next = user_trajectory_[next_point];
             
             double segment_length = (next - current).norm();
             
             if (total_distance + segment_length <= distance_increment) {
                 // Move to next point completely
                 total_distance += segment_length;
                 current_trajectory_point_ = next_point;
                 
                 // If we've completed a circuit, reset timer
                 if (current_trajectory_point_ == 0) {
                     current_time_ = 0.0;
                 }
             } else {
                 // Partially move along current segment
                 double remaining = distance_increment - total_distance;
                 double fraction = remaining / segment_length;
                 
                 // Interpolate position
                 user_position_ = current + fraction * (next - current);
                 break;
             }
             
             // Safety check to prevent infinite loops
             if (current_trajectory_point_ == 0) {
                 break;
             }
         }
     }
     
     void validateAllPaths() {
         int invalid_paths = 0;
         
         for (auto& path : signal_paths_) {
             bool valid_path = true;
             
             for (auto& segment : path.segments) {
                 // Check segment for building penetration using our improved method
                 if (checkSegmentPenetration(
                     segment.start, segment.end, 
                     segment.reflection_building_id, -1)) {
                     
                     segment.penetrates_building = true;
                     valid_path = false;
                     
                     if (debug_mode_) {
                         ROS_WARN("Path from beacon %s has a segment that penetrates a building",
                                  path.beacon_id.c_str());
                     }
                 }
             }
             
             path.penetrates_building = !valid_path;
             if (path.penetrates_building) {
                 invalid_paths++;
             }
         }
         
         // Remove penetrating paths
         auto it = std::remove_if(signal_paths_.begin(), signal_paths_.end(),
                                  [](const SignalPath& path) {
                                      return path.penetrates_building;
                                  });
         
         if (it != signal_paths_.end()) {
             signal_paths_.erase(it, signal_paths_.end());
             
             if (invalid_paths > 0) {
                 ROS_INFO("Removed %d paths that penetrate buildings", invalid_paths);
             }
         }
     }
     
     bool hasEnoughReflectionPaths() {
         int reflection_paths = 0;
         
         for (const auto& path : signal_paths_) {
             if (path.reflection_count > 0) {
                 reflection_paths++;
                 if (reflection_paths >= 5) { // At least 5 reflection paths
                     return true;
                 }
             }
         }
         return false;
     }
     
     void clearVisualizations() {
         // Create empty marker arrays to clear visualizations
         visualization_msgs::MarkerArray empty_array;
         building_pub_.publish(empty_array);
         beacon_pub_.publish(empty_array);
         path_pub_.publish(empty_array);
         reflection_pub_.publish(empty_array);
         reflection_surface_pub_.publish(empty_array);
         text_pub_.publish(empty_array);
         debug_pub_.publish(empty_array);
         
         // Delete user and road markers
         visualization_msgs::Marker empty_marker;
         empty_marker.action = visualization_msgs::Marker::DELETE;
         user_pub_.publish(empty_marker);
         road_pub_.publish(empty_marker);
         trajectory_pub_.publish(empty_marker);
     }
     
     void generateEnvironment() {
         ROS_INFO("Generating environment with buildings on both sides...");
         
         // Clear any existing buildings
         buildings_.clear();
         
         // Distance from road centerline to building face
         double road_side = road_width_ / 2.0 + sidewalk_width_;
         
         // Number of buildings per side
         int num_buildings = static_cast<int>(buildings_per_side_);
         double segment_length = road_length_ / num_buildings;
         
         // 1. LEFT SIDE (negative Y)
         for (int i = 0; i < num_buildings; ++i) {
             // Create a building with specific spacing
             double building_width = width_dist_(rng_) * 0.8;  // Slightly narrower
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             // Position building with clear gap from road
             double building_x = -road_length_/2.0 + (i + 0.5) * segment_length;
             
             Building building;
             building.id = "left_" + std::to_string(i);
             building.center = Eigen::Vector3d(
                 building_x, 
                 -(road_side + building_depth/2.0), 
                 building_height/2.0
             );
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             
             // Red-ish color for left side buildings
             building.color = Eigen::Vector3d(0.8, 0.2, 0.2);
             building.reflectivity = 0.8;
             buildings_.push_back(building);
         }
         
         // 2. RIGHT SIDE (positive Y)
         for (int i = 0; i < num_buildings; ++i) {
             // Create a building with specific spacing
             double building_width = width_dist_(rng_) * 0.8;
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             // Position building with clear gap from road and offset from left side
             double building_x = -road_length_/2.0 + (i + 0.3) * segment_length;
             
             Building building;
             building.id = "right_" + std::to_string(i);
             building.center = Eigen::Vector3d(
                 building_x, 
                 (road_side + building_depth/2.0), 
                 building_height/2.0
             );
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             
             // Blue-ish color for right side buildings
             building.color = Eigen::Vector3d(0.2, 0.2, 0.8);
             building.reflectivity = 0.8;
             buildings_.push_back(building);
         }
         
         ROS_INFO("Created %zu buildings with clear spacing between them", buildings_.size());
     }
     
     void createTenBeacons() {
         ROS_INFO("Creating exactly 10 UWB beacons...");
         
         // Clear any existing beacons
         beacons_.clear();
         
         // Distance from centerline to beacon position
         double road_side = road_width_ / 2.0 + 1.0;  // 1m into sidewalk
         
         // Create 5 beacons on each side of the road
         // LEFT SIDE beacons (negative Y)
         for (int i = 0; i < 5; ++i) {
             double beacon_x = -road_length_/2.0 + (i + 0.5) * road_length_/5.0;
             
             UWBBeacon beacon;
             beacon.id = "beacon_left_" + std::to_string(i);
             beacon.position = Eigen::Vector3d(beacon_x, -road_side, beacon_height_);
             beacon.range = max_distance_;
             beacon.frequency = 6500.0;
             beacon.power = 20.0;
             
             beacons_.push_back(beacon);
             
             ROS_INFO("Created beacon %s at (%.2f, %.2f, %.2f)", 
                      beacon.id.c_str(), beacon.position.x(), beacon.position.y(), beacon.position.z());
         }
         
         // RIGHT SIDE beacons (positive Y)
         for (int i = 0; i < 5; ++i) {
             double beacon_x = -road_length_/2.0 + (i + 0.5) * road_length_/5.0 - 4.0;  // Offset by 4m
             
             UWBBeacon beacon;
             beacon.id = "beacon_right_" + std::to_string(i);
             beacon.position = Eigen::Vector3d(beacon_x, road_side, beacon_height_);
             beacon.range = max_distance_;
             beacon.frequency = 6500.0;
             beacon.power = 20.0;
             
             beacons_.push_back(beacon);
             
             ROS_INFO("Created beacon %s at (%.2f, %.2f, %.2f)", 
                      beacon.id.c_str(), beacon.position.x(), beacon.position.y(), beacon.position.z());
         }
         
         ROS_INFO("Created exactly %zu UWB beacons", beacons_.size());
     }
     
     void computeSignalPaths() {
         ROS_INFO("Computing signal paths with robust building penetration prevention...");
         
         signal_paths_.clear();
         active_reflection_faces_.clear();
         
         for (const auto& beacon : beacons_) {
             // Compute direct path first
             computeDirectPath(beacon);
             
             // Compute reflected paths
             for (int reflection_count = 1; reflection_count <= max_reflections_; ++reflection_count) {
                 if (slow_mode_) {
                     ros::Duration(0.05).sleep();
                 }
                 computeReflectedPaths(beacon, reflection_count);
             }
         }
         
         // Sort paths by total path loss (best signal first)
         std::sort(signal_paths_.begin(), signal_paths_.end(), 
               [](const SignalPath& a, const SignalPath& b) {
                   return a.total_path_loss < b.total_path_loss;
               });
         
         // Keep only paths with signal above threshold and no building penetration
         auto it = std::remove_if(signal_paths_.begin(), signal_paths_.end(),
                               [this](const SignalPath& path) {
                                   if (path.penetrates_building) {
                                       return true;  // Remove paths that penetrate buildings
                                   }
                                   
                                   if (path.forced) {
                                       return false;  // Keep forced paths regardless of signal power
                                   }
                                   
                                   return path.total_path_loss > (min_signal_power_ - noise_floor_);
                               });
         
         signal_paths_.erase(it, signal_paths_.end());
         
         // Limit number of paths to visualize for clarity
         const int max_paths_to_keep = 50;
         
         if (signal_paths_.size() > max_paths_to_keep) {
             signal_paths_.resize(max_paths_to_keep);
         }
         
         // Count paths by type
         int direct_paths = 0;
         int single_reflection = 0;
         int double_reflection = 0;
         int forced_paths = 0;
         
         for (const auto& path : signal_paths_) {
             if (path.forced) {
                 forced_paths++;
             } else if (path.reflection_count == 0) {
                 direct_paths++;
             } else if (path.reflection_count == 1) {
                 single_reflection++;
             } else if (path.reflection_count == 2) {
                 double_reflection++;
             }
         }
         
         ROS_INFO("Computed %zu valid signal paths (%d direct, %d single, %d double, %d forced)",
                 signal_paths_.size(), direct_paths, single_reflection, double_reflection, forced_paths);
     }
     
     void ensureDirectPaths() {
         // Make sure every beacon has a direct path to the user if not blocked by buildings
         std::set<std::string> beacons_with_direct_paths;
         
         // Find beacons that already have direct paths
         for (const auto& path : signal_paths_) {
             if (path.reflection_count == 0) {
                 beacons_with_direct_paths.insert(path.beacon_id);
             }
         }
         
         // For every beacon without a direct path, check if one is possible
         for (const auto& beacon : beacons_) {
             if (beacons_with_direct_paths.find(beacon.id) != beacons_with_direct_paths.end()) {
                 continue;  // Already has a direct path
             }
             
             // Check if there's a clear direct path
             double distance = (user_position_ - beacon.position).norm();
             
             // Skip if outside range
             if (distance > beacon.range || distance > max_distance_) {
                 continue;
             }
             
             // Check for building penetration
             bool penetrates = checkSegmentPenetration(beacon.position, user_position_);
             
             if (!penetrates) {
                 // Clear path exists, add it
                 double path_loss = RayTracing::calculateFreeSpacePathLoss(distance, beacon.frequency);
                 path_loss -= beacon.power;  // Adjust for beacon power
                 
                 SignalPath path;
                 path.beacon_id = beacon.id;
                 
                 PathSegment segment(beacon.position, user_position_, 0, path_loss, distance, 1.0);
                 segment.penetrates_building = false;  // We verified it doesn't penetrate
                 
                 path.addSegment(segment);
                 signal_paths_.push_back(path);
                 
                 if (debug_mode_) {
                     ROS_INFO("Added direct path from beacon %s, distance=%.2f m", 
                              beacon.id.c_str(), distance);
                 }
             }
         }
     }
     
     void addForcedReflections() {
         // For each beacon with no reflections, try to find a valid reflection path
         std::map<std::string, int> beacon_reflection_counts;
         
         for (const auto& path : signal_paths_) {
             if (path.reflection_count > 0) {
                 beacon_reflection_counts[path.beacon_id]++;
             }
         }
         
         for (const auto& beacon : beacons_) {
             if (beacon_reflection_counts[beacon.id] > 0) {
                 continue;  // Already has reflection paths
             }
             
             // Find a valid reflection path
             addForcedReflectionForBeacon(beacon);
         }
     }
     
     void addForcedReflectionForBeacon(const UWBBeacon& beacon) {
         // Try to find the best building and face for a reflection
         size_t best_building_idx = SIZE_MAX;
         size_t best_face_idx = SIZE_MAX;
         Eigen::Vector3d best_reflection_point;
         double best_score = -std::numeric_limits<double>::max();
         
         // Try each building
         for (size_t building_idx = 0; building_idx < buildings_.size(); building_idx++) {
             const auto& building = buildings_[building_idx];
             auto faces = building.getFaces();
             
             // Try each face
             for (size_t face_idx = 0; face_idx < faces.size(); face_idx++) {
                 const auto& face = faces[face_idx];
                 
                 // Skip bottom face
                 if (face.normal.z() < -0.9) {
                     continue;
                 }
                 
                 // Try several points on the face to find a good reflection
                 const int grid_size = 3;  // 3x3 grid
                 double width = face.dimensions.x();
                 double height = face.dimensions.y();
                 
                 for (int i = 0; i < grid_size; i++) {
                     for (int j = 0; j < grid_size; j++) {
                         // Calculate grid point
                         double u = (2.0 * i / (grid_size - 1.0) - 1.0) * (width / 2.0 * 0.8);
                         double v = (2.0 * j / (grid_size - 1.0) - 1.0) * (height / 2.0 * 0.8);
                         
                         Eigen::Vector3d reflection_point = face.center + u * face.tangent + v * face.bitangent;
                         
                         // Calculate segments
                         double dist1 = (reflection_point - beacon.position).norm();
                         double dist2 = (user_position_ - reflection_point).norm();
                         double total_dist = dist1 + dist2;
                         
                         // Skip if too far
                         if (total_dist > max_distance_ * 1.2) {
                             continue;
                         }
                         
                         // Check if both segments are clear of buildings
                         bool segment1_penetrates = checkSegmentPenetration(
                             beacon.position, reflection_point, building_idx);
                         
                         if (segment1_penetrates) {
                             continue;  // First segment penetrates a building
                         }
                         
                         bool segment2_penetrates = checkSegmentPenetration(
                             reflection_point, user_position_, building_idx);
                         
                         if (segment2_penetrates) {
                             continue;  // Second segment penetrates a building
                         }
                         
                         // Calculate score (lower distance = better, less building penetration risk)
                         double score = 1000.0 / total_dist;
                         
                         if (score > best_score) {
                             best_score = score;
                             best_building_idx = building_idx;
                             best_face_idx = face_idx;
                             best_reflection_point = reflection_point;
                         }
                     }
                 }
             }
         }
         
         // If found a good reflection point, create a forced path
         if (best_building_idx != SIZE_MAX) {
             const auto& building = buildings_[best_building_idx];
             auto faces = building.getFaces();
             const auto& face = faces[best_face_idx];
             
             double dist1 = (best_reflection_point - beacon.position).norm();
             double dist2 = (user_position_ - best_reflection_point).norm();
             double total_dist = dist1 + dist2;
             
             // Calculate incident angle
             Eigen::Vector3d dir1 = (best_reflection_point - beacon.position).normalized();
             double incident_angle_rad = RayTracing::calculateIncidentAngle(dir1, face.normal);
             double incident_angle_deg = incident_angle_rad * 180.0 / M_PI;
             
             // Create the path
             SignalPath path;
             path.beacon_id = beacon.id;
             path.forced = true;
             
             // First segment: beacon to reflection point
             PathSegment segment1(
                 beacon.position, best_reflection_point, 0, 30.0,  // Fixed loss for visualization
                 dist1, 1.0, -1, -1, 0.0);
             
             // Set penetration flag
             segment1.penetrates_building = checkSegmentPenetration(
                 beacon.position, best_reflection_point, best_building_idx);
             
             path.addSegment(segment1);
             
             // Second segment: reflection point to user
             PathSegment segment2(
                 best_reflection_point, user_position_, 1, 20.0,  // Fixed loss for visualization
                 total_dist, 0.8, best_building_idx, best_face_idx, incident_angle_deg);
             
             // Set penetration flag
             segment2.penetrates_building = checkSegmentPenetration(
                 best_reflection_point, user_position_, best_building_idx);
             
             path.addSegment(segment2);
             
             // Double-check paths with our advanced penetration algorithm
             bool path_ok = true;
             for (const auto& segment : path.segments) {
                 // Final validation with most rigorous test
                 if (checkSegmentPenetration(segment.start, segment.end, 
                                          segment.reflection_building_id)) {
                     path_ok = false;
                     break;
                 }
             }
             
             // Only add the path if it's valid
             if (path_ok) {
                 signal_paths_.push_back(path);
                 
                 // Mark face as active
                 active_reflection_faces_.push_back(std::make_pair(best_building_idx, best_face_idx));
                 
                 if (debug_mode_) {
                     ROS_INFO("Added forced reflection path for beacon %s via building %s face %zu",
                              beacon.id.c_str(), building.id.c_str(), best_face_idx);
                 }
             } else if (debug_mode_) {
                 ROS_WARN("Forced reflection path for beacon %s would penetrate a building, not adding",
                          beacon.id.c_str());
             }
         }
     }
     
     void computeDirectPath(const UWBBeacon& beacon) {
         // Direct line-of-sight path from beacon to user
         Eigen::Vector3d direction = (user_position_ - beacon.position).normalized();
         double distance = (user_position_ - beacon.position).norm();
         
         // Skip if outside beacon range
         if (distance > beacon.range || distance > max_distance_) {
             if (debug_mode_) {
                 ROS_INFO("Beacon %s: Direct path distance %.2f exceeds range %.2f", 
                          beacon.id.c_str(), distance, beacon.range);
             }
             return;
         }
         
         // Check if the direct path penetrates any building
         bool penetrates = checkSegmentPenetration(beacon.position, user_position_);
         
         if (penetrates) {
             if (debug_mode_) {
                 ROS_INFO("Beacon %s: Direct path blocked by building", beacon.id.c_str());
             }
             return;
         }
         
         // Calculate path loss
         double path_loss = RayTracing::calculateFreeSpacePathLoss(distance, beacon.frequency);
         path_loss -= beacon.power;  // Adjust for beacon transmit power
         
         // Create the signal path
         SignalPath path;
         path.beacon_id = beacon.id;
         
         PathSegment segment(beacon.position, user_position_, 0, path_loss, distance, 1.0);
         segment.penetrates_building = false;  // We verified it doesn't penetrate
         
         path.addSegment(segment);
         signal_paths_.push_back(path);
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Added direct path, distance=%.2f m, loss=%.2f dB", 
                      beacon.id.c_str(), distance, path_loss);
         }
     }
     
     void computeReflectedPaths(const UWBBeacon& beacon, int reflection_count) {
         if (reflection_count <= 0 || reflection_count > max_reflections_) {
             return;
         }
         
         if (debug_mode_) {
             ROS_INFO("Searching for %s reflection paths from beacon %s...", 
                      (reflection_count == 1) ? "single" : "double",
                      beacon.id.c_str());
         }
         
         if (reflection_count == 1) {
             computeSingleReflectionPaths(beacon);
         } else if (reflection_count == 2) {
             computeDoubleReflectionPaths(beacon);
         }
     }
     
     void computeSingleReflectionPaths(const UWBBeacon& beacon) {
         int paths_found = 0;
         
         // For each building
         for (size_t building_idx = 0; building_idx < buildings_.size(); building_idx++) {
             const auto& building = buildings_[building_idx];
             auto faces = building.getFaces();
             
             // For each face
             for (size_t face_idx = 0; face_idx < faces.size(); face_idx++) {
                 const auto& face = faces[face_idx];
                 
                 // Skip bottom face
                 if (face.normal.z() < -0.9) {
                     continue;
                 }
                 
                 // Try a grid of points on the face for better reflection finding
                 const int grid_size = 3;
                 double width = face.dimensions.x();
                 double height = face.dimensions.y();
                 
                 for (int i = 0; i < grid_size; i++) {
                     for (int j = 0; j < grid_size; j++) {
                         // Calculate grid point
                         double u = (2.0 * i / (grid_size - 1.0) - 1.0) * (width / 2.0 * 0.8);
                         double v = (2.0 * j / (grid_size - 1.0) - 1.0) * (height / 2.0 * 0.8);
                         
                         Eigen::Vector3d reflection_point = face.center + u * face.tangent + v * face.bitangent;
                         
                         // Calculate directions
                         Eigen::Vector3d dir1 = (reflection_point - beacon.position).normalized();
                         Eigen::Vector3d dir2 = (user_position_ - reflection_point).normalized();
                         
                         // Calculate incident angle
                         double incident_angle_rad = RayTracing::calculateIncidentAngle(dir1, face.normal);
                         
                         // Verify reflection law
                         if (!RayTracing::verifyReflectionLaw(-dir1, dir2, face.normal)) {
                             continue;  // Law of reflection not satisfied
                         }
                         
                         // Calculate distances
                         double distance1 = (reflection_point - beacon.position).norm();
                         double distance2 = (user_position_ - reflection_point).norm();
                         double total_distance = distance1 + distance2;
                         
                         // Skip if too far
                         if (total_distance > beacon.range || total_distance > max_distance_) {
                             continue;
                         }
                         
                         // Critical: check for building penetration
                         bool segment1_penetrates = checkSegmentPenetration(
                             beacon.position, reflection_point, building_idx);
                         
                         if (segment1_penetrates) {
                             continue;  // First segment penetrates a building
                         }
                         
                         bool segment2_penetrates = checkSegmentPenetration(
                             reflection_point, user_position_, building_idx);
                         
                         if (segment2_penetrates) {
                             continue;  // Second segment penetrates a building
                         }
                         
                         // Convert incident angle to degrees for display
                         double incident_angle_deg = incident_angle_rad * 180.0 / M_PI;
                         
                         // Calculate reflection coefficient
                         double reflection_coef = RayTracing::calculateReflectionCoefficient(
                             incident_angle_rad, building.reflectivity);
                         
                         // Calculate path losses
                         double path_loss1 = RayTracing::calculateFreeSpacePathLoss(distance1, beacon.frequency);
                         double path_loss2 = RayTracing::calculateFreeSpacePathLoss(distance2, beacon.frequency);
                         
                         // Reflection loss (based on reflection coefficient)
                         double reflection_loss = -20.0 * std::log10(reflection_coef);
                         
                         // Total loss adjusted for beacon power
                         double total_loss = path_loss1 + path_loss2 + reflection_loss - beacon.power;
                         
                         // Create the signal path
                         SignalPath path;
                         path.beacon_id = beacon.id;
                         
                         // First segment: beacon to reflection point
                         PathSegment segment1(
                             beacon.position, reflection_point, 0, path_loss1, 
                             distance1, 1.0, -1, -1, 0.0);
                         
                         segment1.penetrates_building = segment1_penetrates;
                         path.addSegment(segment1);
                         
                         // Second segment: reflection point to user
                         PathSegment segment2(
                             reflection_point, user_position_, 1, path_loss2 + reflection_loss, 
                             total_distance, reflection_coef, building_idx, face_idx, incident_angle_deg);
                         
                         segment2.penetrates_building = segment2_penetrates;
                         path.addSegment(segment2);
                         
                         signal_paths_.push_back(path);
                         
                         // Mark face as active
                         active_reflection_faces_.push_back(std::make_pair(building_idx, face_idx));
                         
                         paths_found++;
                         
                         if (debug_mode_) {
                             ROS_INFO("Beacon %s: Found reflection on building %s face %zu, angle=%.1f°, loss=%.1f dB",
                                      beacon.id.c_str(), building.id.c_str(), face_idx, 
                                      incident_angle_deg, total_loss);
                         }
                     }
                 }
             }
         }
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Found %d single reflection paths", beacon.id.c_str(), paths_found);
         }
     }
     
     void computeDoubleReflectionPaths(const UWBBeacon& beacon) {
         int paths_found = 0;
         
         // For first reflection building
         for (size_t building1_idx = 0; building1_idx < buildings_.size() && paths_found < 2; building1_idx++) {
             const auto& building1 = buildings_[building1_idx];
             auto faces1 = building1.getFaces();
             
             // For first reflection face
             for (size_t face1_idx = 0; face1_idx < faces1.size() && paths_found < 2; face1_idx++) {
                 const auto& face1 = faces1[face1_idx];
                 
                 // Skip bottom face
                 if (face1.normal.z() < -0.9) {
                     continue;
                 }
                 
                 // Try a point in the center of the face for simplicity
                 Eigen::Vector3d reflection_point1 = face1.center;
                 
                 // Check if beacon can see this point
                 Eigen::Vector3d dir1 = (reflection_point1 - beacon.position).normalized();
                 double dist1 = (reflection_point1 - beacon.position).norm();
                 
                 // Check if segment from beacon to first reflection penetrates any building
                 bool segment1_penetrates = checkSegmentPenetration(
                     beacon.position, reflection_point1, building1_idx);
                 
                 if (segment1_penetrates) {
                     continue;  // First segment penetrates a building
                 }
                 
                 // Calculate incident angle and reflection direction
                 double incident_angle1 = RayTracing::calculateIncidentAngle(dir1, face1.normal);
                 Eigen::Vector3d reflection_dir1 = RayTracing::calculateReflection(dir1, face1.normal);
                 
                 // For second reflection building
                 for (size_t building2_idx = 0; building2_idx < buildings_.size() && paths_found < 2; building2_idx++) {
                     // Skip same building unless it's in a special geometry
                     if (building1_idx == building2_idx) {
                         continue;
                     }
                     
                     const auto& building2 = buildings_[building2_idx];
                     auto faces2 = building2.getFaces();
                     
                     // For second reflection face
                     for (size_t face2_idx = 0; face2_idx < faces2.size() && paths_found < 2; face2_idx++) {
                         const auto& face2 = faces2[face2_idx];
                         
                         // Skip bottom face
                         if (face2.normal.z() < -0.9) {
                             continue;
                         }
                         
                         // Find intersection with second face plane
                         double d2 = face2.normal.dot(face2.center);
                         double nd2 = face2.normal.dot(reflection_dir1);
                         
                         // Skip if parallel
                         if (std::abs(nd2) < RayTracing::EPSILON) {
                             continue;
                         }
                         
                         // Calculate intersection parameter
                         double t2 = (d2 - face2.normal.dot(reflection_point1)) / nd2;
                         
                         // Skip if intersection is behind first reflection
                         if (t2 < 0) {
                             continue;
                         }
                         
                         // Calculate second reflection point
                         Eigen::Vector3d reflection_point2 = reflection_point1 + t2 * reflection_dir1;
                         
                         // Check if point is on the face
                         if (!RayTracing::isPointOnFace(
                             reflection_point2, face2.center, face2.normal,
                             face2.tangent, face2.bitangent, face2.dimensions)) {
                             continue;
                         }
                         
                         // Calculate directions and distances
                         Eigen::Vector3d dir2 = (reflection_point2 - reflection_point1).normalized();
                         double dist2 = (reflection_point2 - reflection_point1).norm();
                         
                         // Check if segment from first to second reflection penetrates any building
                         bool segment2_penetrates = checkSegmentPenetration(
                             reflection_point1, reflection_point2, building1_idx, building2_idx);
                         
                         if (segment2_penetrates) {
                             continue;  // Second segment penetrates a building
                         }
                         
                         // Calculate second incident angle and reflection
                         double incident_angle2 = RayTracing::calculateIncidentAngle(dir2, face2.normal);
                         Eigen::Vector3d reflection_dir2 = RayTracing::calculateReflection(dir2, face2.normal);
                         
                         // Calculate direction to user
                         Eigen::Vector3d dir3 = (user_position_ - reflection_point2).normalized();
                         double dist3 = (user_position_ - reflection_point2).norm();
                         
                         // Verify reflection law
                         if (!RayTracing::verifyReflectionLaw(-dir2, dir3, face2.normal)) {
                             continue;  // Law of reflection not satisfied
                         }
                         
                         // Check if segment from second reflection to user penetrates any building
                         bool segment3_penetrates = checkSegmentPenetration(
                             reflection_point2, user_position_, building2_idx);
                         
                         if (segment3_penetrates) {
                             continue;  // Third segment penetrates a building
                         }
                         
                         // Calculate total distance
                         double total_dist = dist1 + dist2 + dist3;
                         
                         // Skip if too far
                         if (total_dist > beacon.range || total_dist > max_distance_) {
                             continue;
                         }
                         
                         // Calculate angles in degrees for display
                         double incident_angle1_deg = incident_angle1 * 180.0 / M_PI;
                         double incident_angle2_deg = incident_angle2 * 180.0 / M_PI;
                         
                         // Calculate reflection coefficients
                         double reflection_coef1 = RayTracing::calculateReflectionCoefficient(
                             incident_angle1, building1.reflectivity);
                         double reflection_coef2 = RayTracing::calculateReflectionCoefficient(
                             incident_angle2, building2.reflectivity);
                         
                         // Path losses
                         double path_loss1 = RayTracing::calculateFreeSpacePathLoss(dist1, beacon.frequency);
                         double path_loss2 = RayTracing::calculateFreeSpacePathLoss(dist2, beacon.frequency);
                         double path_loss3 = RayTracing::calculateFreeSpacePathLoss(dist3, beacon.frequency);
                         
                         // Reflection losses
                         double reflection_loss1 = -20.0 * std::log10(reflection_coef1);
                         double reflection_loss2 = -20.0 * std::log10(reflection_coef2);
                         
                         // Total loss
                         double total_loss = path_loss1 + path_loss2 + path_loss3 + 
                                           reflection_loss1 + reflection_loss2 - beacon.power;
                         
                         // Create the path
                         SignalPath path;
                         path.beacon_id = beacon.id;
                         
                         // First segment: beacon to first reflection
                         PathSegment segment1(
                             beacon.position, reflection_point1, 0, path_loss1, 
                             dist1, 1.0, -1, -1, 0.0);
                         
                         segment1.penetrates_building = segment1_penetrates;
                         path.addSegment(segment1);
                         
                         // Second segment: first to second reflection
                         PathSegment segment2(
                             reflection_point1, reflection_point2, 1, 
                             path_loss2 + reflection_loss1, 
                             dist1 + dist2, reflection_coef1, 
                             building1_idx, face1_idx, incident_angle1_deg);
                         
                         segment2.penetrates_building = segment2_penetrates;
                         path.addSegment(segment2);
                         
                         // Third segment: second reflection to user
                         PathSegment segment3(
                             reflection_point2, user_position_, 2, 
                             path_loss3 + reflection_loss2, 
                             total_dist, reflection_coef1 * reflection_coef2, 
                             building2_idx, face2_idx, incident_angle2_deg);
                         
                         segment3.penetrates_building = segment3_penetrates;
                         path.addSegment(segment3);
                         
                         signal_paths_.push_back(path);
                         
                         // Mark both faces as active
                         active_reflection_faces_.push_back(std::make_pair(building1_idx, face1_idx));
                         active_reflection_faces_.push_back(std::make_pair(building2_idx, face2_idx));
                         
                         paths_found++;
                         
                         if (debug_mode_) {
                             ROS_INFO("Beacon %s: Found double reflection path via buildings %s→%s, loss=%.1f dB",
                                      beacon.id.c_str(), 
                                      building1.id.c_str(), building2.id.c_str(),
                                      total_loss);
                         }
                     }
                 }
             }
         }
         
         if (debug_mode_) {
             ROS_INFO("Beacon %s: Found %d double reflection paths", beacon.id.c_str(), paths_found);
         }
     }
     
     // Completely revised building penetration check
     bool checkSegmentPenetration(
         const Eigen::Vector3d& start, 
         const Eigen::Vector3d& end,
         int exclude_building_idx = -1,
         int exclude_building_idx2 = -1) {
         
         // Calculate ray direction and length
         Eigen::Vector3d direction = end - start;
         double segment_length = direction.norm();
         
         // Normalize direction (avoid division by zero)
         if (segment_length < 1e-6) {
             return false;  // Extremely short segment, not penetrating
         }
         direction.normalize();
         
         // Special case for problematic geometries
         if (special_check_enabled_) {
             for (size_t i = 0; i < buildings_.size(); i++) {
                 if (static_cast<int>(i) == exclude_building_idx || 
                     static_cast<int>(i) == exclude_building_idx2) {
                     continue;  // Skip excluded buildings
                 }
                 
                 const auto& building = buildings_[i];
                 
                 // Special intense check for known problem cases
                 if (building.id == "left_5" || building.id == "left_6") {
                     // Do a very thorough multi-point check along this segment
                     const int extra_samples = penetration_samples_;  // High sample count
                     for (int j = 1; j < extra_samples; j++) {
                         double t = static_cast<double>(j) / extra_samples;  // Avoid endpoints
                         Eigen::Vector3d sample_point = start + t * (end - start);
                         
                         if (building.containsPoint(sample_point, 0.05)) {
                             if (debug_mode_&&false) {
                                 debug_penetration_points_.push_back(
                                     std::make_pair(sample_point, building.color));
                                 
                                 ROS_WARN("Special case: Segment penetrates building %s at sample point %d/%d",
                                          building.id.c_str(), j, extra_samples);
                             }
                             return true;
                         }
                     }
                 }
             }
         }
         
         // Method 1: Check if any building contains points along the segment
         // Use more sample points for better accuracy
         int sample_count = std::max(5, std::min(10, static_cast<int>(segment_length / 2.0)));
         
         for (int i = 1; i < sample_count; i++) {  // Skip endpoints to avoid false positives
             double t = static_cast<double>(i) / sample_count;
             Eigen::Vector3d sample_point = start + t * (end - start);
             
             for (size_t j = 0; j < buildings_.size(); j++) {
                 if (static_cast<int>(j) == exclude_building_idx || 
                     static_cast<int>(j) == exclude_building_idx2) {
                     continue;  // Skip excluded buildings
                 }
                 
                 const auto& building = buildings_[j];
                 
                 if (building.containsPoint(sample_point)) {
                     if (debug_mode_ &&false) {
                         debug_penetration_points_.push_back(std::make_pair(sample_point, building.color));
                         
                         ROS_WARN("Segment penetrates building %s (point containment)",
                                  building.id.c_str());
                     }
                     return true;
                 }
             }
         }
         
         // Method 2: Use ray-AABB intersection for more precision
         for (size_t i = 0; i < buildings_.size(); i++) {
             if (static_cast<int>(i) == exclude_building_idx || 
                 static_cast<int>(i) == exclude_building_idx2) {
                 continue;  // Skip excluded buildings
             }
             
             const auto& building = buildings_[i];
             
             // Skip if either endpoint is inside or very close to this building
             if (building.containsPoint(start, 0.1) || building.containsPoint(end, 0.1)) {
                 continue;
             }
             
             // Check for ray-AABB intersection
             if (rayIntersectsAABB(start, direction, segment_length, building)) {
                 if (debug_mode_&&false) {
                     // Calculate an approximate penetration point for visualization
                     Eigen::Vector3d mid_point = (start + end) * 0.5;
                     debug_penetration_points_.push_back(std::make_pair(mid_point, building.color));
                     
                     ROS_WARN("Segment penetrates building %s (ray-AABB intersection)",
                              building.id.c_str());
                 }
                 return true;
             }
         }
         
         return false;  // No penetration detected
     }
     
     bool rayIntersectsAABB(
         const Eigen::Vector3d& ray_origin, 
         const Eigen::Vector3d& ray_dir,
         double ray_length,
         const Building& building) {
         
         const Eigen::Vector3d min_pt = building.min();
         const Eigen::Vector3d max_pt = building.max();
         
         // Calculate inverses safely
         const double invDirX = RayTracing::safeInverse(ray_dir.x());
         const double invDirY = RayTracing::safeInverse(ray_dir.y());
         const double invDirZ = RayTracing::safeInverse(ray_dir.z());
         
         // Calculate slab intersections
         double tx1 = (min_pt.x() - ray_origin.x()) * invDirX;
         double tx2 = (max_pt.x() - ray_origin.x()) * invDirX;
         
         double ty1 = (min_pt.y() - ray_origin.y()) * invDirY;
         double ty2 = (max_pt.y() - ray_origin.y()) * invDirY;
         
         double tz1 = (min_pt.z() - ray_origin.z()) * invDirZ;
         double tz2 = (max_pt.z() - ray_origin.z()) * invDirZ;
         
         // Swap to ensure t1 <= t2
         if (std::isnan(tx1) || std::isnan(tx2)) { tx1 = -std::numeric_limits<double>::infinity(); tx2 = std::numeric_limits<double>::infinity(); }
         if (std::isnan(ty1) || std::isnan(ty2)) { ty1 = -std::numeric_limits<double>::infinity(); ty2 = std::numeric_limits<double>::infinity(); }
         if (std::isnan(tz1) || std::isnan(tz2)) { tz1 = -std::numeric_limits<double>::infinity(); tz2 = std::numeric_limits<double>::infinity(); }
         
         if (tx1 > tx2) std::swap(tx1, tx2);
         if (ty1 > ty2) std::swap(ty1, ty2);
         if (tz1 > tz2) std::swap(tz1, tz2);
         
         // Find entrance and exit points
         double tmin = std::max(std::max(tx1, ty1), tz1);
         double tmax = std::min(std::min(tx2, ty2), tz2);
         
         // Special case for segments close to building edges
         const double EDGE_EPSILON = 1e-3;
         if (std::abs(tmin - tmax) < EDGE_EPSILON) {
             return false;  // Grazing the surface, not penetrating
         }
         
         // Check intersection validity
         if (tmax < 0 || tmin > tmax || tmin > ray_length) {
             return false;  // No valid intersection
         }
         
         // If we got here, the ray intersects the AABB within its length
         return true;
     }
     
     // Visualization methods remain the same as before, providing clean visualization
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
             
             // Add wireframe outline for better visibility
             visualization_msgs::Marker outline;
             outline.header = marker.header;
             outline.ns = "building_outlines";
             outline.id = id++;
             outline.type = visualization_msgs::Marker::LINE_LIST;
             outline.action = visualization_msgs::Marker::ADD;
             outline.pose = marker.pose;
             
             // Scale - line width
             outline.scale.x = 0.05;
             
             // Color - black
             outline.color.r = 0.0;
             outline.color.g = 0.0;
             outline.color.b = 0.0;
             outline.color.a = 0.7;
             
             // Calculate vertices
             Eigen::Vector3d min_pt = building.min();
             Eigen::Vector3d max_pt = building.max();
             
             // Bottom face
             addBoxLine(outline, min_pt.x(), min_pt.y(), min_pt.z(), max_pt.x(), min_pt.y(), min_pt.z());
             addBoxLine(outline, max_pt.x(), min_pt.y(), min_pt.z(), max_pt.x(), max_pt.y(), min_pt.z());
             addBoxLine(outline, max_pt.x(), max_pt.y(), min_pt.z(), min_pt.x(), max_pt.y(), min_pt.z());
             addBoxLine(outline, min_pt.x(), max_pt.y(), min_pt.z(), min_pt.x(), min_pt.y(), min_pt.z());
             
             // Top face
             addBoxLine(outline, min_pt.x(), min_pt.y(), max_pt.z(), max_pt.x(), min_pt.y(), max_pt.z());
             addBoxLine(outline, max_pt.x(), min_pt.y(), max_pt.z(), max_pt.x(), max_pt.y(), max_pt.z());
             addBoxLine(outline, max_pt.x(), max_pt.y(), max_pt.z(), min_pt.x(), max_pt.y(), max_pt.z());
             addBoxLine(outline, min_pt.x(), max_pt.y(), max_pt.z(), min_pt.x(), min_pt.y(), max_pt.z());
             
             // Vertical edges
             addBoxLine(outline, min_pt.x(), min_pt.y(), min_pt.z(), min_pt.x(), min_pt.y(), max_pt.z());
             addBoxLine(outline, max_pt.x(), min_pt.y(), min_pt.z(), max_pt.x(), min_pt.y(), max_pt.z());
             addBoxLine(outline, max_pt.x(), max_pt.y(), min_pt.z(), max_pt.x(), max_pt.y(), max_pt.z());
             addBoxLine(outline, min_pt.x(), max_pt.y(), min_pt.z(), min_pt.x(), max_pt.y(), max_pt.z());
             
             building_markers.markers.push_back(outline);
             
             // Add building ID text
             visualization_msgs::Marker text_marker;
             text_marker.header = marker.header;
             text_marker.ns = "building_ids";
             text_marker.id = id++;
             text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             text_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - above building
             text_marker.pose.position.x = building.center.x();
             text_marker.pose.position.y = building.center.y();
             text_marker.pose.position.z = building.center.z() + building.dimensions.z() / 2.0 + 1.0;
             text_marker.pose.orientation.w = 1.0;
             
             // Scale
             text_marker.scale.z = 0.8; // Text height
             
             // Color
             text_marker.color.r = building.color.x();
             text_marker.color.g = building.color.y();
             text_marker.color.b = building.color.z();
             text_marker.color.a = 1.0;
             
             // Text
             text_marker.text = building.id;
             
             building_markers.markers.push_back(text_marker);
         }
         
         building_pub_.publish(building_markers);
     }
     
     void addBoxLine(visualization_msgs::Marker& marker, 
                    double x1, double y1, double z1, 
                    double x2, double y2, double z2) {
         geometry_msgs::Point p1, p2;
         p1.x = x1; p1.y = y1; p1.z = z1;
         p2.x = x2; p2.y = y2; p2.z = z2;
         marker.points.push_back(p1);
         marker.points.push_back(p2);
     }
     
     void publishBeacons() {
         visualization_msgs::MarkerArray beacon_markers;
         int id = 0;
         
         for (const auto& beacon : beacons_) {
             // Beacon sphere
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
             marker.scale.x = 0.8;
             marker.scale.y = 0.8;
             marker.scale.z = 0.8;
             
             // Color - gold for beacons
             marker.color.r = 1.0;
             marker.color.g = 0.84;
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
             pole_marker.scale.x = 0.1;
             pole_marker.scale.y = 0.1;
             pole_marker.scale.z = beacon.position.z();
             
             // Color - gray pole
             pole_marker.color.r = 0.7;
             pole_marker.color.g = 0.7;
             pole_marker.color.b = 0.7;
             pole_marker.color.a = 1.0;
             
             beacon_markers.markers.push_back(pole_marker);
             
             // Beacon label
             visualization_msgs::Marker text_marker;
             text_marker.header.frame_id = fixed_frame_;
             text_marker.header.stamp = ros::Time::now();
             text_marker.ns = "beacon_labels";
             text_marker.id = id++;
             text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             text_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - above beacon
             text_marker.pose.position.x = beacon.position.x();
             text_marker.pose.position.y = beacon.position.y();
             text_marker.pose.position.z = beacon.position.z() + 0.8;
             text_marker.pose.orientation.w = 1.0;
             
             // Scale
             text_marker.scale.z = 0.5; // Text height
             
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
         marker.scale.x = 0.7;
         marker.scale.y = 0.7;
         marker.scale.z = 0.7;
         
         // Color - bright cyan for user
         marker.color.r = 0.0;
         marker.color.g = 0.8;
         marker.color.b = 1.0;
         marker.color.a = 1.0;
         
         user_pub_.publish(marker);
         
         // Add user label
         visualization_msgs::Marker text_marker;
         text_marker.header.frame_id = fixed_frame_;
         text_marker.header.stamp = ros::Time::now();
         text_marker.ns = "user_label";
         text_marker.id = 1;
         text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         text_marker.action = visualization_msgs::Marker::ADD;
         
         // Position - above user
         text_marker.pose.position.x = user_position_.x();
         text_marker.pose.position.y = user_position_.y();
         text_marker.pose.position.z = user_position_.z() + 1.0;
         text_marker.pose.orientation.w = 1.0;
         
         // Scale
         text_marker.scale.z = 0.6; // Text height
         
         // Color
         text_marker.color.r = 0.0;
         text_marker.color.g = 0.8;
         text_marker.color.b = 1.0;
         text_marker.color.a = 1.0;
         
         // Text
         text_marker.text = "USER";
         
         user_pub_.publish(text_marker);
     }
     
     void publishTrajectory() {
         if (user_trajectory_.empty()) {
             return;
         }
         
         visualization_msgs::Marker traj_marker;
         traj_marker.header.frame_id = fixed_frame_;
         traj_marker.header.stamp = ros::Time::now();
         traj_marker.ns = "user_trajectory";
         traj_marker.id = 0;
         traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
         traj_marker.action = visualization_msgs::Marker::ADD;
         
         // Add trajectory points
         for (const auto& point : user_trajectory_) {
             geometry_msgs::Point p;
             p.x = point.x();
             p.y = point.y();
             p.z = point.z();
             traj_marker.points.push_back(p);
         }
         
         // Line width
         traj_marker.scale.x = 0.1;
         
         // Color - light blue
         traj_marker.color.r = 0.3;
         traj_marker.color.g = 0.8;
         traj_marker.color.b = 1.0;
         traj_marker.color.a = 0.5;
         
         trajectory_pub_.publish(traj_marker);
         
         // Add current position marker
         visualization_msgs::Marker current_point;
         current_point.header.frame_id = fixed_frame_;
         current_point.header.stamp = ros::Time::now();
         current_point.ns = "trajectory_current";
         current_point.id = 1;
         current_point.type = visualization_msgs::Marker::SPHERE;
         current_point.action = visualization_msgs::Marker::ADD;
         
         // Current position on trajectory
         current_point.pose.position.x = user_trajectory_[current_trajectory_point_].x();
         current_point.pose.position.y = user_trajectory_[current_trajectory_point_].y();
         current_point.pose.position.z = user_trajectory_[current_trajectory_point_].z();
         current_point.pose.orientation.w = 1.0;
         
         // Scale
         current_point.scale.x = 0.2;
         current_point.scale.y = 0.2;
         current_point.scale.z = 0.2;
         
         // Color - bright green
         current_point.color.r = 0.0;
         current_point.color.g = 1.0;
         current_point.color.b = 0.0;
         current_point.color.a = 1.0;
         
         trajectory_pub_.publish(current_point);
     }
     
     void publishReflectionSurfaces() {
         visualization_msgs::MarkerArray surface_markers;
         int id = 0;
         
         // Clear previous active surfaces if no paths
         if (active_reflection_faces_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             surface_markers.markers.push_back(clear_marker);
             reflection_surface_pub_.publish(surface_markers);
             return;
         }
         
         // Remove duplicates from the active reflection faces list
         std::set<std::pair<int, int>> unique_faces(
             active_reflection_faces_.begin(), active_reflection_faces_.end());
         
         // Visualize each active reflection face
         for (const auto& face_pair : unique_faces) {
             int building_idx = face_pair.first;
             int face_idx = face_pair.second;
             
             // Skip invalid indices
             if (building_idx < 0 || building_idx >= buildings_.size()) {
                 continue;
             }
             
             const auto& building = buildings_[building_idx];
             auto faces = building.getFaces();
             
             if (face_idx < 0 || face_idx >= faces.size()) {
                 continue;
             }
             
             const auto& face = faces[face_idx];
             
             // Create a thin box to represent the reflecting face
             visualization_msgs::Marker surface_marker;
             surface_marker.header.frame_id = fixed_frame_;
             surface_marker.header.stamp = ros::Time::now();
             surface_marker.ns = "reflection_surfaces";
             surface_marker.id = id++;
             surface_marker.type = visualization_msgs::Marker::CUBE;
             surface_marker.action = visualization_msgs::Marker::ADD;
             
             // Position at the face center
             surface_marker.pose.position.x = face.center.x();
             surface_marker.pose.position.y = face.center.y();
             surface_marker.pose.position.z = face.center.z();
             
             // Orientation to align with the face
             Eigen::Vector3d z_axis(0, 0, 1);
             Eigen::Vector3d rotation_axis = z_axis.cross(face.normal);
             double rotation_angle = std::acos(z_axis.dot(face.normal));
             
             if (rotation_axis.norm() > RayTracing::EPSILON) {
                 Eigen::Vector3d unit_axis = rotation_axis.normalized();
                 tf::Quaternion q(unit_axis.x(), unit_axis.y(), unit_axis.z(), rotation_angle);
                 surface_marker.pose.orientation.x = q.x();
                 surface_marker.pose.orientation.y = q.y();
                 surface_marker.pose.orientation.z = q.z();
                 surface_marker.pose.orientation.w = q.w();
             } else {
                 // Handle case where normal is parallel to z-axis
                 if (face.normal.z() < 0) {
                     tf::Quaternion q;
                     q.setRPY(M_PI, 0, 0);
                     surface_marker.pose.orientation.x = q.x();
                     surface_marker.pose.orientation.y = q.y();
                     surface_marker.pose.orientation.z = q.z();
                     surface_marker.pose.orientation.w = q.w();
                 } else {
                     surface_marker.pose.orientation.w = 1.0;
                 }
             }
             
             // Scale - slightly smaller than the actual face to be visible
             surface_marker.scale.x = face.dimensions.x() * 0.95;
             surface_marker.scale.y = face.dimensions.y() * 0.95;
             surface_marker.scale.z = 0.05; // Very thin to appear as a surface
             
             // Color - bright red to highlight the reflection surface
             surface_marker.color.r = 1.0;
             surface_marker.color.g = 0.0;
             surface_marker.color.b = 0.0;
             surface_marker.color.a = 0.6;
             
             surface_markers.markers.push_back(surface_marker);
             
             // Add normal vector indicator
             visualization_msgs::Marker normal_marker;
             normal_marker.header.frame_id = fixed_frame_;
             normal_marker.header.stamp = ros::Time::now();
             normal_marker.ns = "reflection_normals";
             normal_marker.id = id++;
             normal_marker.type = visualization_msgs::Marker::ARROW;
             normal_marker.action = visualization_msgs::Marker::ADD;
             
             // Use the face center as start
             normal_marker.points.resize(2);
             normal_marker.points[0].x = face.center.x();
             normal_marker.points[0].y = face.center.y();
             normal_marker.points[0].z = face.center.z();
             
             // Normal direction (1m length)
             Eigen::Vector3d normal_end = face.center + face.normal;
             normal_marker.points[1].x = normal_end.x();
             normal_marker.points[1].y = normal_end.y();
             normal_marker.points[1].z = normal_end.z();
             
             // Scale - arrow size
             normal_marker.scale.x = 0.1;  // Shaft diameter
             normal_marker.scale.y = 0.2;  // Head diameter
             normal_marker.scale.z = 0.2;  // Head length
             
             // Color - yellow for normal vector
             normal_marker.color.r = 1.0;
             normal_marker.color.g = 1.0;
             normal_marker.color.b = 0.0;
             normal_marker.color.a = 1.0;
             
             surface_markers.markers.push_back(normal_marker);
         }
         
         reflection_surface_pub_.publish(surface_markers);
     }
     
     void publishSignalPaths() {
         visualization_msgs::MarkerArray path_markers;
         int id = 0;
         
         // Clear previous paths if there are none
         if (signal_paths_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             path_markers.markers.push_back(clear_marker);
             path_pub_.publish(path_markers);
             return;
         }
         
         // Color scheme for paths
         std_msgs::ColorRGBA direct_color; // Green
         direct_color.r = 0.0;
         direct_color.g = 1.0;
         direct_color.b = 0.0;
         direct_color.a = 1.0;
         
         std_msgs::ColorRGBA single_refl_color; // Yellow
         single_refl_color.r = 1.0;
         single_refl_color.g = 1.0;
         single_refl_color.b = 0.0;
         single_refl_color.a = 1.0;
         
         std_msgs::ColorRGBA multi_refl_color; // Red
         multi_refl_color.r = 1.0;
         multi_refl_color.g = 0.0;
         multi_refl_color.b = 0.0;
         multi_refl_color.a = 1.0;
         
         // Visualize each signal path
         for (const auto& path : signal_paths_) {
             // Skip invalid paths
             if (!path.valid || path.segments.empty() || path.penetrates_building) {
                 continue;
             }
             
             // Set color based on reflection count
             std_msgs::ColorRGBA path_color;
             switch (path.reflection_count) {
                 case 0:
                     path_color = direct_color;
                     break;
                 case 1:
                     path_color = single_refl_color;
                     break;
                 default:
                     path_color = multi_refl_color;
                     break;
             }
             
             // Visualize each segment of the path
             for (size_t j = 0; j < path.segments.size(); ++j) {
                 const auto& segment = path.segments[j];
                 
                 // Skip segments that penetrate buildings
                 if (segment.penetrates_building) {
                     continue;
                 }
                 
                 // Create a line segment
                 visualization_msgs::Marker segment_marker;
                 segment_marker.header.frame_id = fixed_frame_;
                 segment_marker.header.stamp = ros::Time::now();
                 segment_marker.ns = "path_segments";
                 segment_marker.id = id++;
                 segment_marker.type = visualization_msgs::Marker::LINE_STRIP;
                 segment_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Add both endpoints
                 geometry_msgs::Point start_point, end_point;
                 start_point.x = segment.start.x();
                 start_point.y = segment.start.y();
                 start_point.z = segment.start.z();
                 
                 end_point.x = segment.end.x();
                 end_point.y = segment.end.y();
                 end_point.z = segment.end.z();
                 
                 segment_marker.points.push_back(start_point);
                 segment_marker.points.push_back(end_point);
                 
                 // Line width
                 segment_marker.scale.x = path_width_;
                 
                 // Color - use path color
                 segment_marker.color = path_color;
                 
                 // If this is a forced path, use a dashed line effect
                 if (path.forced) {
                     // Use a dashed style by making the line semi-transparent
                     segment_marker.color.a = 0.8;
                 }
                 
                 path_markers.markers.push_back(segment_marker);
                 
                 // Add small arrow in the middle to indicate direction
                 visualization_msgs::Marker arrow_marker;
                 arrow_marker.header.frame_id = fixed_frame_;
                 arrow_marker.header.stamp = ros::Time::now();
                 arrow_marker.ns = "path_arrows";
                 arrow_marker.id = id++;
                 arrow_marker.type = visualization_msgs::Marker::ARROW;
                 arrow_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Position the arrow at the middle of the segment
                 Eigen::Vector3d midpoint = (segment.start + segment.end) / 2.0;
                 Eigen::Vector3d direction = (segment.end - segment.start).normalized();
                 
                 // Scale arrow size based on segment length
                 double segment_length = (segment.end - segment.start).norm();
                 double arrow_length = std::min(segment_length * 0.3, 1.0);
                 
                 Eigen::Vector3d arrow_start = midpoint - direction * arrow_length * 0.5;
                 Eigen::Vector3d arrow_end = midpoint + direction * arrow_length * 0.5;
                 
                 geometry_msgs::Point arrow_start_pt, arrow_end_pt;
                 arrow_start_pt.x = arrow_start.x();
                 arrow_start_pt.y = arrow_start.y();
                 arrow_start_pt.z = arrow_start.z();
                 
                 arrow_end_pt.x = arrow_end.x();
                 arrow_end_pt.y = arrow_end.y();
                 arrow_end_pt.z = arrow_end.z();
                 
                 arrow_marker.points.push_back(arrow_start_pt);
                 arrow_marker.points.push_back(arrow_end_pt);
                 
                 // Arrow proportions
                 arrow_marker.scale.x = path_width_ * 0.5;  // Shaft width
                 arrow_marker.scale.y = path_width_ * 1.5;  // Head width
                 arrow_marker.scale.z = path_width_ * 1.0;  // Head length
                 
                 // Color - same as segment
                 arrow_marker.color = path_color;
                 
                 path_markers.markers.push_back(arrow_marker);
                 
                 // For reflection segments, add angle information
                 if (segment.reflection_count > 0) {
                     // Add text with reflection angle info
                     visualization_msgs::Marker angle_marker;
                     angle_marker.header.frame_id = fixed_frame_;
                     angle_marker.header.stamp = ros::Time::now();
                     angle_marker.ns = "reflection_angles";
                     angle_marker.id = id++;
                     angle_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                     angle_marker.action = visualization_msgs::Marker::ADD;
                     
                     // Position - near reflection point
                     angle_marker.pose.position.x = segment.start.x();
                     angle_marker.pose.position.y = segment.start.y();
                     angle_marker.pose.position.z = segment.start.z() + 0.5;
                     angle_marker.pose.orientation.w = 1.0;
                     
                     // Scale
                     angle_marker.scale.z = 0.4; // Text height
                     
                     // Color - white
                     angle_marker.color.r = 1.0;
                     angle_marker.color.g = 1.0;
                     angle_marker.color.b = 1.0;
                     angle_marker.color.a = 1.0;
                     
                     // Text - show incident angle
                     std::stringstream ss;
                     ss << std::fixed << std::setprecision(1) << segment.incident_angle << "°";
                     angle_marker.text = ss.str();
                     
                     path_markers.markers.push_back(angle_marker);
                 }
             }
             
             // Add text showing path details
             visualization_msgs::Marker path_info_marker;
             path_info_marker.header.frame_id = fixed_frame_;
             path_info_marker.header.stamp = ros::Time::now();
             path_info_marker.ns = "path_info";
             path_info_marker.id = id++;
             path_info_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             path_info_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - near the beacon
             const auto& first_segment = path.segments.front();
             path_info_marker.pose.position.x = first_segment.start.x() + 0.5;
             path_info_marker.pose.position.y = first_segment.start.y() + 0.5;
             path_info_marker.pose.position.z = first_segment.start.z() + 0.5;
             path_info_marker.pose.orientation.w = 1.0;
             
             // Scale
             path_info_marker.scale.z = 0.4; // Text height
             
             // Color - same as path
             path_info_marker.color = path_color;
             
             // Text - brief path info
             std::stringstream ss;
             ss << std::fixed << std::setprecision(1);
             if (path.reflection_count == 0) {
                 ss << "Direct: " << path.total_distance << "m";
             } else {
                 ss << path.reflection_count << " refl: " << path.total_distance << "m";
             }
             if (path.forced) {
                 ss << " (forced)";
             }
             path_info_marker.text = ss.str();
             
             path_markers.markers.push_back(path_info_marker);
         }
         
         path_pub_.publish(path_markers);
     }
     
     void publishReflectionPoints() {
         visualization_msgs::MarkerArray reflection_markers;
         int id = 0;
         
         // Clear if no signal paths
         if (signal_paths_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             reflection_markers.markers.push_back(clear_marker);
             reflection_pub_.publish(reflection_markers);
             return;
         }
         
         // Track unique reflection points to avoid duplicates
         std::map<std::tuple<double, double, double>, int> unique_points;
         
         // For each path with reflections
         for (const auto& path : signal_paths_) {
             if (path.reflection_count == 0 || path.penetrates_building) {
                 continue; // Skip direct paths and invalid paths
             }
             
             // Start from segment 1 (the first reflection point is at the end of segment 0)
             for (size_t j = 1; j < path.segments.size(); ++j) {
                 const auto& prev_segment = path.segments[j-1];
                 
                 // Skip if this segment penetrates a building
                 if (prev_segment.penetrates_building) {
                     continue;
                 }
                 
                 // The reflection point is the end of the previous segment
                 Eigen::Vector3d refl_point = prev_segment.end;
                 
                 // Create a tuple key from the point coordinates
                 auto point_key = std::make_tuple(refl_point.x(), refl_point.y(), refl_point.z());
                 
                 // Count points at this location
                 unique_points[point_key]++;
             }
         }
         
         // Visualize unique reflection points
         for (const auto& pair : unique_points) {
             auto point_key = pair.first;
             int count = pair.second;
             
             // Get the coordinates
             double x = std::get<0>(point_key);
             double y = std::get<1>(point_key);
             double z = std::get<2>(point_key);
             
             // 1. Sphere marker for the reflection point
             visualization_msgs::Marker point_marker;
             point_marker.header.frame_id = fixed_frame_;
             point_marker.header.stamp = ros::Time::now();
             point_marker.ns = "reflection_points";
             point_marker.id = id++;
             point_marker.type = visualization_msgs::Marker::SPHERE;
             point_marker.action = visualization_msgs::Marker::ADD;
             
             point_marker.pose.position.x = x;
             point_marker.pose.position.y = y;
             point_marker.pose.position.z = z;
             point_marker.pose.orientation.w = 1.0;
             
             // Size based on number of reflections
             double size = 0.2 + 0.05 * std::min(5, count);
             point_marker.scale.x = size;
             point_marker.scale.y = size;
             point_marker.scale.z = size;
             
             // Color - bright white
             point_marker.color.r = 1.0;
             point_marker.color.g = 1.0;
             point_marker.color.b = 1.0;
             point_marker.color.a = 1.0;
             
             reflection_markers.markers.push_back(point_marker);
             
             // 2. Number of reflections label if more than 1
             if (count > 1) {
                 visualization_msgs::Marker count_marker;
                 count_marker.header.frame_id = fixed_frame_;
                 count_marker.header.stamp = ros::Time::now();
                 count_marker.ns = "reflection_counts";
                 count_marker.id = id++;
                 count_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                 count_marker.action = visualization_msgs::Marker::ADD;
                 
                 count_marker.pose.position.x = x;
                 count_marker.pose.position.y = y;
                 count_marker.pose.position.z = z + size * 1.5;
                 count_marker.pose.orientation.w = 1.0;
                 
                 count_marker.scale.z = 0.3; // Text height
                 
                 // Color - same as reflection point
                 count_marker.color.r = 1.0;
                 count_marker.color.g = 1.0;
                 count_marker.color.b = 1.0;
                 count_marker.color.a = 1.0;
                 
                 count_marker.text = std::to_string(count);
                 
                 reflection_markers.markers.push_back(count_marker);
             }
             
             // 3. Add "REFL" label
             visualization_msgs::Marker label_marker;
             label_marker.header.frame_id = fixed_frame_;
             label_marker.header.stamp = ros::Time::now();
             label_marker.ns = "reflection_labels";
             label_marker.id = id++;
             label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             label_marker.action = visualization_msgs::Marker::ADD;
             
             label_marker.pose.position.x = x;
             label_marker.pose.position.y = y;
             label_marker.pose.position.z = z + size * 0.7;
             label_marker.pose.orientation.w = 1.0;
             
             label_marker.scale.z = 0.25; // Text height
             
             // Color - yellow for high visibility
             label_marker.color.r = 1.0;
             label_marker.color.g = 1.0;
             label_marker.color.b = 0.0;
             label_marker.color.a = 1.0;
             
             label_marker.text = "REFL";
             
             reflection_markers.markers.push_back(label_marker);
         }
         
         reflection_pub_.publish(reflection_markers);
     }
     
     void publishDebugMarkers() {
         // Publish markers for debug visualization
         visualization_msgs::MarkerArray debug_markers;
         int id = 0;
         
         // Clear previous markers if no debug info
         if (debug_penetration_points_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             debug_markers.markers.push_back(clear_marker);
             debug_pub_.publish(debug_markers);
             return;
         }
         
         // Visualize penetration points for debugging
         for (const auto& point_pair : debug_penetration_points_) {
             const auto& point = point_pair.first;
             const auto& color = point_pair.second;
             
             // Sphere marker for penetration point
             visualization_msgs::Marker point_marker;
             point_marker.header.frame_id = fixed_frame_;
             point_marker.header.stamp = ros::Time::now();
             point_marker.ns = "penetration_points";
             point_marker.id = id++;
             point_marker.type = visualization_msgs::Marker::SPHERE;
             point_marker.action = visualization_msgs::Marker::ADD;
             
             point_marker.pose.position.x = point.x();
             point_marker.pose.position.y = point.y();
             point_marker.pose.position.z = point.z();
             point_marker.pose.orientation.w = 1.0;
             
             // Size for visibility
             point_marker.scale.x = 0.3;
             point_marker.scale.y = 0.3;
             point_marker.scale.z = 0.3;
             
             // Color - based on building color but with transparency
             point_marker.color.r = color.x();
             point_marker.color.g = color.y();
             point_marker.color.b = color.z();
             point_marker.color.a = 0.7;
             
             debug_markers.markers.push_back(point_marker);
             
             // Add a "PENETRATION" label
             visualization_msgs::Marker label_marker;
             label_marker.header.frame_id = fixed_frame_;
             label_marker.header.stamp = ros::Time::now();
             label_marker.ns = "penetration_labels";
             label_marker.id = id++;
             label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             label_marker.action = visualization_msgs::Marker::ADD;
             
             label_marker.pose.position.x = point.x();
             label_marker.pose.position.y = point.y();
             label_marker.pose.position.z = point.z() + 0.3;
             label_marker.pose.orientation.w = 1.0;
             
             label_marker.scale.z = 0.2; // Text height
             
             // Color - red for warning
             label_marker.color.r = 1.0;
             label_marker.color.g = 0.0;
             label_marker.color.b = 0.0;
             label_marker.color.a = 1.0;
             
             label_marker.text = "PENETRATION";
             
             debug_markers.markers.push_back(label_marker);
         }
         
         debug_pub_.publish(debug_markers);
     }
     
     void publishTextInfo() {
         visualization_msgs::MarkerArray text_markers;
         int id = 0;
         
         // Title text
         visualization_msgs::Marker title_marker;
         title_marker.header.frame_id = fixed_frame_;
         title_marker.header.stamp = ros::Time::now();
         title_marker.ns = "text_info";
         title_marker.id = id++;
         title_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         title_marker.action = visualization_msgs::Marker::ADD;
         
         title_marker.pose.position.x = 0.0;
         title_marker.pose.position.y = 0.0;
         title_marker.pose.position.z = 30.0;
         title_marker.pose.orientation.w = 1.0;
         
         title_marker.scale.z = 1.0; // Text height
         
         title_marker.color.r = 1.0;
         title_marker.color.g = 1.0;
         title_marker.color.b = 1.0;
         title_marker.color.a = 1.0;
         
         title_marker.text = "UWB Signal Reflection Physics - Moving User";
         
         text_markers.markers.push_back(title_marker);
         
         // Legend text
         visualization_msgs::Marker legend_marker;
         legend_marker.header.frame_id = fixed_frame_;
         legend_marker.header.stamp = ros::Time::now();
         legend_marker.ns = "legend";
         legend_marker.id = id++;
         legend_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         legend_marker.action = visualization_msgs::Marker::ADD;
         
         legend_marker.pose.position.x = -road_length_ / 2.0 + 5.0;
         legend_marker.pose.position.y = -road_width_ / 2.0 - 5.0;
         legend_marker.pose.position.z = 5.0;
         legend_marker.pose.orientation.w = 1.0;
         
         legend_marker.scale.z = 0.5; // Text height
         
         legend_marker.color.r = 1.0;
         legend_marker.color.g = 1.0;
         legend_marker.color.b = 1.0;
         legend_marker.color.a = 1.0;
         
         legend_marker.text = 
             "GREEN: Direct Path\n"
             "YELLOW: Single Reflection\n"
             "RED: Multi-Reflection\n\n"
             "RED SURFACES: Reflecting Faces\n"
             "WHITE SPHERES: Reflection Points\n"
             "BLUE LINE: User Trajectory";
         
         text_markers.markers.push_back(legend_marker);
         
         // Count summary
         visualization_msgs::Marker count_marker;
         count_marker.header.frame_id = fixed_frame_;
         count_marker.header.stamp = ros::Time::now();
         count_marker.ns = "counts";
         count_marker.id = id++;
         count_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
         count_marker.action = visualization_msgs::Marker::ADD;
         
         count_marker.pose.position.x = road_length_ / 2.0 - 15.0;
        count_marker.pose.position.y = -road_width_ / 2.0 - 5.0;
        count_marker.pose.position.z = 5.0;
        count_marker.pose.orientation.w = 1.0;
        
        count_marker.scale.z = 0.5; // Text height
        
        count_marker.color.r = 1.0;
        count_marker.color.g = 1.0;
        count_marker.color.b = 0.0;
        count_marker.color.a = 1.0;
        
        // Count paths by type
        int direct_paths = 0;
        int single_reflection = 0;
        int double_reflection = 0;
        int forced_paths = 0;
        
        for (const auto& path : signal_paths_) {
            if (path.penetrates_building) {
                continue;  // Skip invalid paths
            }
            
            if (path.forced) {
                forced_paths++;
            } else if (path.reflection_count == 0) {
                direct_paths++;
            } else if (path.reflection_count == 1) {
                single_reflection++;
            } else if (path.reflection_count == 2) {
                double_reflection++;
            }
        }
        
        std::stringstream ss;
        ss << "Paths: " << direct_paths << " direct\n"
           << single_reflection << " single reflection\n"
           << double_reflection << " double reflection\n";
        
        if (forced_paths > 0) {
            ss << forced_paths << " forced paths\n";
        }
        
        ss << "\nMoving with " << movement_type_ << " trajectory";
        
        count_marker.text = ss.str();
        
        text_markers.markers.push_back(count_marker);
        
        // User position info
        visualization_msgs::Marker user_info;
        user_info.header.frame_id = fixed_frame_;
        user_info.header.stamp = ros::Time::now();
        user_info.ns = "user_info";
        user_info.id = id++;
        user_info.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        user_info.action = visualization_msgs::Marker::ADD;
        
        user_info.pose.position.x = road_length_ / 2.0 - 15.0;
        user_info.pose.position.y = road_width_ / 2.0 + 5.0;
        user_info.pose.position.z = 5.0;
        user_info.pose.orientation.w = 1.0;
        
        user_info.scale.z = 0.5;
        
        user_info.color.r = 0.0;
        user_info.color.g = 0.8;
        user_info.color.b = 1.0;
        user_info.color.a = 1.0;
        
        std::stringstream user_ss;
        user_ss << "Current User Position:\n"
                << "X: " << std::fixed << std::setprecision(2) << user_position_.x() << "\n"
                << "Y: " << std::fixed << std::setprecision(2) << user_position_.y() << "\n"
                << "Z: " << std::fixed << std::setprecision(2) << user_position_.z() << "\n"
                << "Speed: " << std::fixed << std::setprecision(2) << movement_speed_ << " m/s";
        
        user_info.text = user_ss.str();
        
        text_markers.markers.push_back(user_info);
        
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
    ros::init(argc, argv, "RangingRC");
    
    ROS_INFO("Starting UWB Ray Tracer with moving user");
    UWBRayTracer ray_tracer;
    
    ros::spin();
    
    return 0;
}