/**
 * @file gps_raim_node.cpp
 * @brief ROS node for GPS single point positioning with RAIM and protection level calculation
 * 
 * This node simulates GPS satellites and implements a receiver that:
 * 1. Performs single point positioning to estimate the receiver position
 * 2. Implements RAIM for fault detection
 * 3. Calculates horizontal and vertical protection levels using rigorous methods
 */

 #include <ros/ros.h>
 #include <sensor_msgs/NavSatFix.h>
 #include <geometry_msgs/PoseWithCovarianceStamped.h>
 #include <Eigen/Dense>
 #include <random>
 #include <vector>
 #include <cmath>
 
 // Constants
 const double PI = 3.14159265358979323846;
 const double EARTH_RADIUS = 6378137.0;  // Earth radius in meters (WGS84)
 const double MU = 3.986005e14;          // Earth's gravitational constant
 const double OMEGA_E = 7.2921151467e-5; // Earth's rotation rate in rad/s
 const double C = 299792458.0;           // Speed of light in m/s
 const double SV_CLOCK_BIAS_SD = 30.0;   // Standard deviation of satellite clock bias in meters
 const double SV_POS_SD = 10.0;          // Standard deviation of satellite position error in meters
 const double PSEUDORANGE_SD = 5.0;      // Standard deviation of pseudorange measurement noise in meters
 const double CHI_SQUARE_THRESHOLD = 11.07; // Chi-square threshold for fault detection (95% confidence, 5 DoF)
 
 // WGS84 parameters
 const double WGS84_a = 6378137.0;       // Semi-major axis in meters
 const double WGS84_f = 1.0/298.257223563; // Flattening
 const double WGS84_e2 = 2*WGS84_f - WGS84_f*WGS84_f; // Eccentricity squared
 
 // Structures
 struct SatelliteData {
     int prn;               // Satellite PRN number
     double x, y, z;        // ECEF coordinates in meters
     double vx, vy, vz;     // Velocity in ECEF frame in m/s
     double clock_bias;     // Satellite clock bias in seconds
     double clock_drift;    // Satellite clock drift in seconds/second
     double pseudorange;    // Pseudorange measurement in meters
     bool used;             // Flag to indicate if the satellite is used in the solution
 };
 
 struct ReceiverState {
     double x, y, z;        // ECEF coordinates in meters
     double clock_bias;     // Receiver clock bias in seconds
     Eigen::Matrix4d covariance; // Covariance matrix of the state
     double hdop;           // Horizontal dilution of precision
     double vdop;           // Vertical dilution of precision
     double hpl;            // Horizontal protection level in meters
     double vpl;            // Vertical protection level in meters
     bool raim_alert;       // RAIM alert flag
     double test_statistic; // RAIM test statistic
 };
 
 class GpsRaimNode {
 private:
     ros::NodeHandle nh_;
     ros::Publisher position_pub_;
     ros::Publisher navsat_pub_;
     ros::Timer timer_;
     
     std::vector<SatelliteData> satellites_;
     ReceiverState receiver_state_;
     std::default_random_engine generator_;
     
     // Simulation parameters
     double true_receiver_x_;
     double true_receiver_y_;
     double true_receiver_z_;
     double simulation_time_;
     double time_step_;
     
     // RAIM parameters
     double pfa_;           // Probability of false alarm
     double pmd_;           // Probability of missed detection
     double bias_error_;    // Minimum detectable bias
 
 public:
     GpsRaimNode(ros::NodeHandle& nh) : nh_(nh) {
         // Initialize publishers
         position_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("gps_position", 10);
         navsat_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("gps_fix", 10);
         
         // Initialize random number generator with a seed
         generator_.seed(static_cast<unsigned int>(time(nullptr)));
         
         // Initialize simulation parameters
         nh_.param<double>("true_receiver_x", true_receiver_x_, 0.0);
         nh_.param<double>("true_receiver_y", true_receiver_y_, 0.0);
         nh_.param<double>("true_receiver_z", true_receiver_z_, EARTH_RADIUS);
         nh_.param<double>("time_step", time_step_, 1.0);
         
         // Initialize RAIM parameters
         nh_.param<double>("pfa", pfa_, 1e-5);           // Probability of false alarm
         nh_.param<double>("pmd", pmd_, 1e-3);           // Probability of missed detection
         nh_.param<double>("bias_error", bias_error_, 30.0); // Minimum detectable bias in meters
         
         simulation_time_ = 0.0;
         
         // Initialize receiver state with a reasonable guess
         receiver_state_.x = true_receiver_x_;
         receiver_state_.y = true_receiver_y_;
         receiver_state_.z = true_receiver_z_;
         receiver_state_.clock_bias = 0.0;
         receiver_state_.hpl = 9999.0;
         receiver_state_.vpl = 9999.0;
         receiver_state_.raim_alert = false;
         receiver_state_.test_statistic = 0.0;
         
         // Generate initial satellite constellation
         generateSatelliteConstellation();
         
         // Start the timer for regular updates
         timer_ = nh_.createTimer(ros::Duration(time_step_), &GpsRaimNode::timerCallback, this);
         
         ROS_INFO("GPS RAIM Node initialized");
     }
     
     // Generate a simulated satellite constellation
     void generateSatelliteConstellation() {
         satellites_.clear();
         
         // Create a constellation of GPS satellites
         // This is a simplified simulation - in reality, you would use almanac or ephemeris data
         
         // We'll create 32 satellites in 6 orbital planes
         const int num_satellites = 32;
         const int planes = 6;
         const double inclination = 55.0 * PI / 180.0;  // 55 degrees
         const double semi_major_axis = 26559700.0;     // Semi-major axis in meters
         const double eccentricity = 0.01;              // Eccentricity
         
         std::uniform_real_distribution<double> rand_anomaly(0.0, 2.0 * PI);
         std::normal_distribution<double> rand_clock_bias(0.0, SV_CLOCK_BIAS_SD / C);
         
         for (int i = 0; i < num_satellites; ++i) {
             SatelliteData sat;
             sat.prn = i + 1;
             
             // Calculate orbital parameters
             int plane = i % planes;
             double raan = (2.0 * PI * plane) / planes;  // Right Ascension of Ascending Node
             double mean_anomaly = rand_anomaly(generator_);
             
             // Convert mean anomaly to eccentric anomaly (simplified)
             double eccentric_anomaly = mean_anomaly;
             for (int j = 0; j < 5; ++j) {  // A few iterations should be enough
                 eccentric_anomaly = mean_anomaly + eccentricity * sin(eccentric_anomaly);
             }
             
             // Calculate position in orbital plane
             double x_orbit = semi_major_axis * (cos(eccentric_anomaly) - eccentricity);
             double y_orbit = semi_major_axis * sin(eccentric_anomaly) * sqrt(1 - eccentricity * eccentricity);
             
             // Rotate to ECEF frame
             double x_temp = x_orbit;
             double y_temp = y_orbit * cos(inclination);
             double z_temp = y_orbit * sin(inclination);
             
             sat.x = x_temp * cos(raan) - y_temp * sin(raan);
             sat.y = x_temp * sin(raan) + y_temp * cos(raan);
             sat.z = z_temp;
             
             // Set velocity (simplified)
             double orbital_velocity = sqrt(MU / semi_major_axis);
             sat.vx = -orbital_velocity * sin(eccentric_anomaly);
             sat.vy = orbital_velocity * cos(eccentric_anomaly) * sqrt(1 - eccentricity * eccentricity);
             sat.vz = 0.0;
             
             // Set clock bias and drift
             sat.clock_bias = rand_clock_bias(generator_);
             sat.clock_drift = 0.0;  // Simplified: no drift
             
             sat.used = true;  // Initially, all satellites are used
             satellites_.push_back(sat);
         }
         
         // Update pseudoranges
         updatePseudoranges();
     }
     
     // Update satellite positions and pseudoranges
     void updateSatellites() {
         simulation_time_ += time_step_;
         
         // Update each satellite's position based on velocity
         for (auto& sat : satellites_) {
             // Simple linear motion (in a real implementation, you would use orbital mechanics)
             sat.x += sat.vx * time_step_;
             sat.y += sat.vy * time_step_;
             sat.z += sat.vz * time_step_;
             
             // Update clock bias
             sat.clock_bias += sat.clock_drift * time_step_;
         }
         
         // Update pseudoranges
         updatePseudoranges();
     }
     
     // Calculate pseudoranges based on true receiver position and satellite positions
     void updatePseudoranges() {
         std::normal_distribution<double> noise(0.0, PSEUDORANGE_SD);
         std::normal_distribution<double> sv_pos_error(0.0, SV_POS_SD);
         
         for (auto& sat : satellites_) {
             // Add small random errors to satellite positions (more realistic)
             double x_err = sv_pos_error(generator_);
             double y_err = sv_pos_error(generator_);
             double z_err = sv_pos_error(generator_);
             
             // Calculate true geometric range to actual receiver position
             double dx = sat.x + x_err - true_receiver_x_;
             double dy = sat.y + y_err - true_receiver_y_;
             double dz = sat.z + z_err - true_receiver_z_;
             double geometric_range = sqrt(dx*dx + dy*dy + dz*dz);
             
             // Add satellite clock bias effect
             double pseudorange = geometric_range + sat.clock_bias * C;
             
             // Add measurement noise
             pseudorange += noise(generator_);
             
             sat.pseudorange = pseudorange;
             
             // Determine if satellite is visible from receiver
             // This is a simple elevation mask check
             double lat, lon, h;
             ecefToGeodetic(true_receiver_x_, true_receiver_y_, true_receiver_z_, lat, lon, h);
             
             // Calculate local ENU coordinates of satellite
             double e, n, u;
             ecefToEnu(sat.x, sat.y, sat.z, true_receiver_x_, true_receiver_y_, true_receiver_z_, lat, lon, e, n, u);
             
             // Calculate elevation angle
             double elevation = atan2(u, sqrt(e*e + n*n)) * 180.0 / PI;
             
             // Apply elevation mask of 5 degrees
             if (elevation < 5.0) {
                 sat.used = false;
             } else {
                 sat.used = true;
             }
         }
     }
     
     // Perform single point positioning
     void singlePointPositioning() {
         // Count the number of satellites being used
         int num_used_sats = 0;
         for (const auto& sat : satellites_) {
             if (sat.used) {
                 num_used_sats++;
             }
         }
         
         // We need at least 4 satellites for a solution
         if (num_used_sats < 4) {
             ROS_WARN("Not enough satellites for positioning (need at least 4, got %d)", num_used_sats);
             return;
         }
         
         // Initialize current estimate of receiver position and clock bias
         Eigen::Vector4d state;
         state << receiver_state_.x, receiver_state_.y, receiver_state_.z, receiver_state_.clock_bias * C;
         
         // Iterative solution (typically converges in a few iterations)
         for (int iter = 0; iter < 10; ++iter) {
             Eigen::MatrixXd H(num_used_sats, 4);  // Geometry matrix
             Eigen::VectorXd z(num_used_sats);     // Measurement residual vector
             Eigen::VectorXd w(num_used_sats);     // Weights
             
             int row = 0;
             for (const auto& sat : satellites_) {
                 if (!sat.used) {
                     continue;
                 }
                 
                 // Calculate predicted pseudorange based on current estimate
                 double dx = sat.x - state(0);
                 double dy = sat.y - state(1);
                 double dz = sat.z - state(2);
                 double range = sqrt(dx*dx + dy*dy + dz*dz);
                 double pred_pseudorange = range + state(3);  // Add clock bias
                 
                 // Calculate residual
                 z(row) = sat.pseudorange - pred_pseudorange;
                 
                 // Calculate geometry matrix row
                 H(row, 0) = -dx / range;  // Partial derivative with respect to x
                 H(row, 1) = -dy / range;  // Partial derivative with respect to y
                 H(row, 2) = -dz / range;  // Partial derivative with respect to z
                 H(row, 3) = 1.0;          // Partial derivative with respect to clock bias
                 
                 // Set weight (inverse of measurement variance)
                 w(row) = 1.0 / (PSEUDORANGE_SD * PSEUDORANGE_SD);
                 
                 row++;
             }
             
             // Create weight matrix
             Eigen::MatrixXd W = w.asDiagonal();
             
             // Calculate state update using weighted least squares
             Eigen::MatrixXd H_T_W = H.transpose() * W;
             Eigen::Matrix4d H_T_W_H_inv = (H_T_W * H).inverse();
             Eigen::Vector4d delta_state = H_T_W_H_inv * H_T_W * z;
             
             // Update state
             state += delta_state;
             
             // Check convergence
             if (delta_state.norm() < 1e-3) {
                 // Store the covariance matrix for later use
                 receiver_state_.covariance = H_T_W_H_inv;
                 break;
             }
         }
         
         // Update receiver state
         receiver_state_.x = state(0);
         receiver_state_.y = state(1);
         receiver_state_.z = state(2);
         receiver_state_.clock_bias = state(3) / C;
         
         // Calculate DOP (Dilution of Precision) values
         calculateDOP();
     }
     
     // Calculate Dilution of Precision values
     void calculateDOP() {
         // Count the number of satellites being used
         int num_used_sats = 0;
         for (const auto& sat : satellites_) {
             if (sat.used) {
                 num_used_sats++;
             }
         }
         
         if (num_used_sats < 4) {
             return;  // Can't calculate DOP with fewer than 4
         }
         
         // Convert ECEF to local East-North-Up (ENU)
         double lat, lon, h;
         ecefToGeodetic(receiver_state_.x, receiver_state_.y, receiver_state_.z, lat, lon, h);
         
         // Rotation matrix from ECEF to ENU
         Eigen::Matrix3d R_ecef_to_enu;
         R_ecef_to_enu << -sin(lon), cos(lon), 0,
                          -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
                          cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat);
         
         // Extract position part of covariance
         Eigen::Matrix3d pos_cov = receiver_state_.covariance.block<3,3>(0,0);
         
         // Transform to ENU
         Eigen::Matrix3d enu_cov = R_ecef_to_enu * pos_cov * R_ecef_to_enu.transpose();
         
         // Calculate DOP values
         receiver_state_.hdop = sqrt(enu_cov(0,0) + enu_cov(1,1));
         receiver_state_.vdop = sqrt(enu_cov(2,2));
     }
     
     // Perform RAIM fault detection using the chi-square test
     bool raimFaultDetection() {
         // Count the number of satellites being used
         int num_used_sats = 0;
         for (const auto& sat : satellites_) {
             if (sat.used) {
                 num_used_sats++;
             }
         }
         
         // We need at least 5 satellites for RAIM
         if (num_used_sats < 5) {
             ROS_WARN("Not enough satellites for RAIM (need at least 5, got %d)", num_used_sats);
             receiver_state_.raim_alert = true;
             receiver_state_.test_statistic = 0.0;
             return false;
         }
         
         // Create geometry matrix
         Eigen::MatrixXd H(num_used_sats, 4);
         Eigen::VectorXd residuals(num_used_sats);
         
         int row = 0;
         for (const auto& sat : satellites_) {
             if (!sat.used) {
                 continue;
             }
             
             // Calculate predicted pseudorange
             double dx = sat.x - receiver_state_.x;
             double dy = sat.y - receiver_state_.y;
             double dz = sat.z - receiver_state_.z;
             double range = sqrt(dx*dx + dy*dy + dz*dz);
             double pred_pseudorange = range + receiver_state_.clock_bias * C;
             
             // Calculate residual
             residuals(row) = sat.pseudorange - pred_pseudorange;
             
             // Fill geometry matrix
             H(row, 0) = -dx / range;
             H(row, 1) = -dy / range;
             H(row, 2) = -dz / range;
             H(row, 3) = 1.0;
             
             row++;
         }
         
         // Calculate the weighted least squares solution matrix
         Eigen::MatrixXd W = Eigen::MatrixXd::Identity(num_used_sats, num_used_sats) / (PSEUDORANGE_SD * PSEUDORANGE_SD);
         Eigen::MatrixXd G = (H.transpose() * W * H).inverse() * H.transpose() * W;
         
         // Calculate the parity matrix
         Eigen::MatrixXd P = Eigen::MatrixXd::Identity(num_used_sats, num_used_sats) - H * G;
         
         // Calculate SSE (Sum of Squared Errors) in the parity space
         receiver_state_.test_statistic = residuals.transpose() * W * P * W * residuals;
         
         // Degrees of freedom is the redundancy (num_sats - 4)
         int dof = num_used_sats - 4;
         
         // Compare with chi-square threshold
         receiver_state_.raim_alert = (receiver_state_.test_statistic > CHI_SQUARE_THRESHOLD);
         
         return !receiver_state_.raim_alert;
     }
     
     // Calculate protection levels using a rigorous approach based on parity space
     void calculateProtectionLevels() {
         // Count the number of satellites being used
         int num_used_sats = 0;
         for (const auto& sat : satellites_) {
             if (sat.used) {
                 num_used_sats++;
             }
         }
         
         // We need at least 5 satellites for protection levels
         if (num_used_sats < 5) {
             receiver_state_.hpl = 9999.0;  // Set to a very large value
             receiver_state_.vpl = 9999.0;
             return;
         }
         
         // Create geometry matrix in ECEF
         Eigen::MatrixXd H(num_used_sats, 4);
         
         int row = 0;
         for (const auto& sat : satellites_) {
             if (!sat.used) {
                 continue;
             }
             
             double dx = sat.x - receiver_state_.x;
             double dy = sat.y - receiver_state_.y;
             double dz = sat.z - receiver_state_.z;
             double range = sqrt(dx*dx + dy*dy + dz*dz);
             
             H(row, 0) = -dx / range;
             H(row, 1) = -dy / range;
             H(row, 2) = -dz / range;
             H(row, 3) = 1.0;
             
             row++;
         }
         
         // Weight matrix (assuming equal weights for simplicity)
         double sigma2 = PSEUDORANGE_SD * PSEUDORANGE_SD;
         Eigen::MatrixXd W = Eigen::MatrixXd::Identity(num_used_sats, num_used_sats) / sigma2;
         
         // Calculate the least squares solution matrix
         Eigen::Matrix4d G = (H.transpose() * W * H).inverse() * H.transpose() * W;
         
         // Calculate the parity matrix
         Eigen::MatrixXd P = Eigen::MatrixXd::Identity(num_used_sats, num_used_sats) - H * G;
         
         // Convert ECEF to geodetic coordinates
         double lat, lon, h;
         ecefToGeodetic(receiver_state_.x, receiver_state_.y, receiver_state_.z, lat, lon, h);
         
         // Create rotation matrix from ECEF to local ENU frame
         Eigen::Matrix3d R_ecef_to_enu;
         R_ecef_to_enu << -sin(lon), cos(lon), 0,
                          -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
                          cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat);
         
         // Calculate slopes for each satellite
         std::vector<double> h_slopes(num_used_sats);
         std::vector<double> v_slopes(num_used_sats);
         
         for (int i = 0; i < num_used_sats; ++i) {
             // Extract the i-th column of G (sensitivity of position to measurement i)
             Eigen::Vector4d g_i = G.col(i);
             
             // Transform position error to local ENU frame
             Eigen::Vector3d delta_enu = R_ecef_to_enu * g_i.head<3>();
             
             // Calculate horizontal and vertical components
             double delta_h = sqrt(delta_enu(0)*delta_enu(0) + delta_enu(1)*delta_enu(1));
             double delta_v = fabs(delta_enu(2));
             
             // Extract the i-th column of P (sensitivity of parity vector to measurement i)
             Eigen::VectorXd p_i = P.col(i);
             
             // Calculate the magnitude of the parity vector component
             double p_i_norm = p_i.norm();
             
             // Calculate slopes (only if p_i_norm is not too small to avoid division by zero)
             if (p_i_norm > 1e-10) {
                 h_slopes[i] = delta_h / p_i_norm;
                 v_slopes[i] = delta_v / p_i_norm;
             } else {
                 h_slopes[i] = 0.0;
                 v_slopes[i] = 0.0;
             }
         }
         
         // Find maximum slopes
         double max_h_slope = *std::max_element(h_slopes.begin(), h_slopes.end());
         double max_v_slope = *std::max_element(v_slopes.begin(), v_slopes.end());
         
         // Calculate the non-centrality parameter for the chi-square distribution
         // based on the probability of missed detection (pmd_)
         int dof = num_used_sats - 4;  // Degrees of freedom
         
         // These constants are derived from statistical tables
         // For a given Pmd and Pfa, these represent the non-centrality parameter
         // of the chi-square distribution
         const double K_hpl = 6.0;  // For horizontal protection level
         const double K_vpl = 5.33; // For vertical protection level
         
         // Calculate protection levels
         receiver_state_.hpl = K_hpl * max_h_slope * PSEUDORANGE_SD;
         receiver_state_.vpl = K_vpl * max_v_slope * PSEUDORANGE_SD;
         
         // Alternative approach: calculate HPL using the position covariance
         Eigen::Matrix3d enu_cov = R_ecef_to_enu * receiver_state_.covariance.block<3,3>(0,0) * R_ecef_to_enu.transpose();
         
         // Calculate semi-major axis of horizontal error ellipse
         double a = enu_cov(0,0);
         double b = enu_cov(0,1);
         double c = enu_cov(1,1);
         double discriminant = sqrt((a-c)*(a-c) + 4*b*b);
         double eigenvalue1 = (a+c+discriminant)/2;
         double eigenvalue2 = (a+c-discriminant)/2;
         double semi_major = sqrt(std::max(eigenvalue1, eigenvalue2));
         
         // Use the larger of the two methods for HPL
         double hpl_alt = K_hpl * semi_major;
         receiver_state_.hpl = std::max(receiver_state_.hpl, hpl_alt);
     }
     
     // Convert ECEF coordinates to geodetic (latitude, longitude, height)
     void ecefToGeodetic(double x, double y, double z, double& lat, double& lon, double& h) {
         // Calculate longitude
         lon = atan2(y, x);
         
         // Initial values
         double p = sqrt(x*x + y*y);
         double theta = atan2(z*WGS84_a, p*WGS84_a*(1-WGS84_f));
         
         // Iterative calculation
         double N, lat_prev;
         lat = atan2(z + WGS84_e2*WGS84_a*sin(theta)*sin(theta)*sin(theta),
                    p - WGS84_e2*WGS84_a*cos(theta)*cos(theta)*cos(theta));
         
         do {
             lat_prev = lat;
             N = WGS84_a / sqrt(1 - WGS84_e2*sin(lat)*sin(lat));
             h = p/cos(lat) - N;
             lat = atan2(z, p*(1-WGS84_e2*N/(N+h)));
         } while (fabs(lat - lat_prev) > 1e-12);
     }
     
     // Convert ECEF to ENU coordinates relative to a reference point
     void ecefToEnu(double x, double y, double z, double ref_x, double ref_y, double ref_z, 
                   double ref_lat, double ref_lon, double& e, double& n, double& u) {
         // Compute the delta in ECEF
         double dx = x - ref_x;
         double dy = y - ref_y;
         double dz = z - ref_z;
         
         // Compute the transformation matrix
         double sin_lat = sin(ref_lat);
         double cos_lat = cos(ref_lat);
         double sin_lon = sin(ref_lon);
         double cos_lon = cos(ref_lon);
         
         // ENU components
         e = -sin_lon*dx + cos_lon*dy;
         n = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz;
         u = cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz;
     }
     
     // Convert geodetic coordinates to ECEF
     void geodeticToEcef(double lat, double lon, double h, double& x, double& y, double& z) {
         double N = WGS84_a / sqrt(1 - WGS84_e2 * sin(lat) * sin(lat));
         
         x = (N + h) * cos(lat) * cos(lon);
         y = (N + h) * cos(lat) * sin(lon);
         z = (N * (1 - WGS84_e2) + h) * sin(lat);
     }
     
     // Publish the position results
     void publishResults() {
         ros::Time now = ros::Time::now();
         
         // Convert ECEF to geodetic coordinates
         double lat, lon, h;
         ecefToGeodetic(receiver_state_.x, receiver_state_.y, receiver_state_.z, lat, lon, h);
         
         // Publish NavSatFix message
         sensor_msgs::NavSatFix navsat_msg;
         navsat_msg.header.stamp = now;
         navsat_msg.header.frame_id = "gps";
         navsat_msg.latitude = lat * 180.0 / PI;
         navsat_msg.longitude = lon * 180.0 / PI;
         navsat_msg.altitude = h;
         
         // Set covariance
         navsat_msg.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_KNOWN;
         
         // Convert ECEF covariance to geodetic
         // This is a simplification - a proper transformation would be more complex
         navsat_msg.position_covariance[0] = receiver_state_.covariance(0,0);  // lat-lat
         navsat_msg.position_covariance[4] = receiver_state_.covariance(1,1);  // lon-lon
         navsat_msg.position_covariance[8] = receiver_state_.covariance(2,2);  // alt-alt
         
         // Set status based on RAIM
         if (receiver_state_.raim_alert) {
             navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_NO_FIX;
         } else {
             navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
         }
         navsat_msg.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
         
         navsat_pub_.publish(navsat_msg);
         
         // Publish PoseWithCovariance message
         geometry_msgs::PoseWithCovarianceStamped pose_msg;
         pose_msg.header.stamp = now;
         pose_msg.header.frame_id = "map";
         
         pose_msg.pose.pose.position.x = receiver_state_.x;
         pose_msg.pose.pose.position.y = receiver_state_.y;
         pose_msg.pose.pose.position.z = receiver_state_.z;
         
         // Identity quaternion (no rotation information in GPS)
         pose_msg.pose.pose.orientation.w = 1.0;
         pose_msg.pose.pose.orientation.x = 0.0;
         pose_msg.pose.pose.orientation.y = 0.0;
         pose_msg.pose.pose.orientation.z = 0.0;
         
         // Copy covariance
         for (int i = 0; i < 3; ++i) {
             for (int j = 0; j < 3; ++j) {
                 pose_msg.pose.covariance[6*i + j] = receiver_state_.covariance(i,j);
             }
         }
         
         position_pub_.publish(pose_msg);
         
         // Print debug information
         ROS_INFO("Position: %.3f, %.3f, %.3f (ECEF)", 
                  receiver_state_.x, receiver_state_.y, receiver_state_.z);
         ROS_INFO("Position: %.6f, %.6f, %.3f (LLH)", 
                  lat * 180.0 / PI, lon * 180.0 / PI, h);
         ROS_INFO("Clock Bias: %.3f ms", receiver_state_.clock_bias * 1000.0);
         ROS_INFO("HDOP: %.2f, VDOP: %.2f", receiver_state_.hdop, receiver_state_.vdop);
         ROS_INFO("HPL: %.2f m, VPL: %.2f m", receiver_state_.hpl, receiver_state_.vpl);
         ROS_INFO("RAIM Test Statistic: %.2f (Threshold: %.2f)", 
                  receiver_state_.test_statistic, CHI_SQUARE_THRESHOLD);
         ROS_INFO("RAIM Alert: %s", receiver_state_.raim_alert ? "YES" : "NO");
         ROS_INFO("Satellites used: %d/%d\n", 
                  std::count_if(satellites_.begin(), satellites_.end(), 
                               [](const SatelliteData& s) { return s.used; }),
                  satellites_.size());
     }
     
     // Timer callback function
     void timerCallback(const ros::TimerEvent&) {
         // Update satellite positions and pseudoranges
         updateSatellites();
         
         // Perform single point positioning
         singlePointPositioning();
         
         // Perform RAIM fault detection
         raimFaultDetection();
         
         // Calculate protection levels
         calculateProtectionLevels();
         
         // Publish the results
         publishResults();
     }
     
     // Simulate a faulty satellite
     void simulateFaultySatellite(int prn, double error) {
         for (auto& sat : satellites_) {
             if (sat.prn == prn) {
                 sat.pseudorange += error;
                 ROS_INFO("Simulated fault of %.2f meters applied to PRN %d", error, prn);
                 break;
             }
         }
     }
 };
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "gps_raim_node");
     ros::NodeHandle nh;
     
     GpsRaimNode node(nh);
     
     ros::spin();
     
     return 0;
 }