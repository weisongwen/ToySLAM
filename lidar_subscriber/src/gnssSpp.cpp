/**
 * @file gnss_spp_node.cpp
 * @brief ROS node for GNSS single point positioning using raw measurements
 * Simplified version with only SPP functionality
 */

 #include <ros/ros.h>
 #include <Eigen/Dense>
 #include <ceres/ceres.h>
 #include <sensor_msgs/NavSatFix.h>
 #include <nav_msgs/Odometry.h>
 #include <geometry_msgs/PoseWithCovarianceStamped.h>
 #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
 #include <deque>
 #include <mutex>
 #include <fstream>
 #include <iomanip>
 
 // GNSS message types from gnss_comm
 #include <gnss_comm/GnssMeasMsg.h>
 #include <gnss_comm/GnssEphemMsg.h>
 #include <gnss_comm/GnssGloEphemMsg.h>
 #include <gnss_comm/StampedFloat64Array.h>
 #include <gnss_comm/GnssObsMsg.h>
 #include <gnss_comm/GnssTimeMsg.h>
 
 // Constants
 constexpr double SPEED_OF_LIGHT = 299792458.0;  // m/s
 constexpr double GPS_L1_FREQ = 1575.42e6;       // Hz
 constexpr double GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ;
 constexpr double EARTH_ROTATION_RATE = 7.2921151467e-5;  // rad/s
 constexpr double DEFAULT_PSEUDORANGE_NOISE = 5.0;  // meters
 constexpr double WGS84_a = 6378137.0;              // WGS84 semi-major axis in meters
 constexpr double WGS84_b = 6356752.31424518;       // WGS84 semi-minor axis in meters
 constexpr double WGS84_e_sq = 1 - (WGS84_b * WGS84_b) / (WGS84_a * WGS84_a);  // WGS84 eccentricity squared
 constexpr double MU_GPS = 3.9860050e14;            // GPS value for Earth's gravitational constant [m^3/s^2]
 constexpr double PI = 3.1415926535897932;
 constexpr double MAX_EPH_AGE = 7200.0;             // Maximum ephemeris age in seconds
 constexpr double GPS_SECONDS_PER_WEEK = 604800.0;  // Seconds in a GPS week
 constexpr double CURRENT_GPS_LEAP_SECONDS = 18.0;  // Current leap seconds (GPS - UTC) as of 2023
 
 // GNSS System identifiers
 enum GnssSystem {
     GPS = 0,     // 'G'
     GLONASS = 1, // 'R'
     GALILEO = 2, // 'E'
     BEIDOU = 3,  // 'C'
     QZSS = 4,    // 'J'
     SBAS = 5,    // 'S'
     UNKNOWN = -1
 };
 
 // GPS ephemeris data for a single satellite
 struct GpsEphemeris {
     uint32_t sat;              // Satellite ID/PRN
     double toe_sec;            // Time of ephemeris (seconds in GPS week)
     double toc_sec;            // Time of clock (seconds in GPS week)
     uint32_t week;             // GPS week
     double af0, af1, af2;      // Clock correction parameters
     double crs, crc, cus, cuc, cis, cic;  // Orbital correction terms
     double delta_n;            // Mean motion difference
     double m0;                 // Mean anomaly at reference time
     double e;                  // Eccentricity
     double sqrta;              // Square root of semi-major axis
     double omg;                // Longitude of ascending node
     double omg_dot;            // Rate of right ascension
     double omega;              // Argument of perigee
     double i0;                 // Inclination angle
     double i_dot;              // Rate of inclination angle
     double tgd0;               // Group delay differential
     double health;             // Health indicator
     uint32_t iode;             // Issue of data ephemeris
     bool valid;                // Valid flag
     ros::Time last_update;     // When this ephemeris was last updated
     double ura;                // User Range Accuracy (for weighting)
 };
 
 // Ionospheric parameters (Klobuchar model)
 struct IonoParams {
     double alpha0, alpha1, alpha2, alpha3;
     double beta0, beta1, beta2, beta3;
     bool valid;
     ros::Time last_update;     // When these parameters were last updated
 };
 
 // Structure to store satellite information
 struct SatelliteInfo {
     uint32_t sat_id;        // Satellite ID/PRN
     GnssSystem system;      // GNSS system type
     double pseudorange;     // meters
     double elevation;       // radians
     double azimuth;         // radians
     double weight;          // weight for measurement (based on elevation angle & CN0)
     double cn0;             // carrier-to-noise density ratio (dB-Hz)
     double sat_pos_x;       // ECEF X position (m)
     double sat_pos_y;       // ECEF Y position (m)
     double sat_pos_z;       // ECEF Z position (m)
     double sat_clock_bias;  // seconds
     double iono_delay;      // ionospheric delay (m)
     double trop_delay;      // tropospheric delay (m)
     double tgd;             // total group delay (s)
     double psr_std;         // pseudorange standard deviation
     double ura;             // user range accuracy
     double ephemeris_time;  // Time of ephemeris (toe)
 
     // Default constructor with initialization
     SatelliteInfo() : 
         sat_id(0), system(UNKNOWN),
         pseudorange(0.0), elevation(0.0), azimuth(0.0), weight(1.0),
         cn0(0.0), sat_pos_x(0.0), sat_pos_y(0.0), sat_pos_z(0.0),
         sat_clock_bias(0.0), iono_delay(0.0), trop_delay(0.0), tgd(0.0),
         psr_std(DEFAULT_PSEUDORANGE_NOISE), ura(5.0), ephemeris_time(0.0)
     {}
 };
 
 // Structure for GNSS solution
 struct GnssSolution {
     double x;               // ECEF X (m)
     double y;               // ECEF Y (m)
     double z;               // ECEF Z (m)
     double clock_bias;      // receiver clock bias (m)
     double latitude;        // geodetic latitude (rad)
     double longitude;       // geodetic longitude (rad)
     double altitude;        // geodetic altitude (m)
     double gdop;            // Geometric dilution of precision
     double pdop;            // Position dilution of precision
     double hdop;            // Horizontal dilution of precision
     double vdop;            // Vertical dilution of precision
     double tdop;            // Time dilution of precision
     double timestamp;       // solution timestamp
     int num_satellites;     // number of satellites used
     Eigen::Matrix<double, 4, 4> covariance;  // position-clock covariance
 
     GnssSolution() :
         x(0.0), y(0.0), z(0.0), clock_bias(0.0),
         latitude(0.0), longitude(0.0), altitude(0.0),
         gdop(99.9), pdop(99.9), hdop(99.9), vdop(99.9), tdop(99.9),
         timestamp(0.0), num_satellites(0)
     {
         covariance = Eigen::Matrix4d::Identity();
     }
 };
 
 // Helper function to convert time
 double gnss_time_to_sec(const gnss_comm::GnssTimeMsg& time_msg) {
     return time_msg.tow;  // Use time of week in seconds
 }
 
 // Handle GPS time of week crossing
 double adjustTimeWithinWeek(double time1, double time2) {
     // Handle week crossovers
     double dt = time1 - time2;
     if (dt > 302400.0) dt -= 604800.0;
     else if (dt < -302400.0) dt += 604800.0;
     return dt;
 }
 
 // Helper class for ECEF to LLA conversion
 class CoordinateConverter {
 public:
     // Convert ECEF to Geodetic (LLA)
     static void ecefToLla(double x, double y, double z, double& lat, double& lon, double& alt) {
         // Calculate longitude (simple)
         lon = atan2(y, x);
         
         // Iterative calculation for latitude and altitude
         double p = sqrt(x*x + y*y);
         double lat_prev = atan2(z, p * (1.0 - WGS84_e_sq));
         double N = WGS84_a;
         double h = 0.0;
         
         const int max_iterations = 5;
         const double tolerance = 1e-12;
         
         for (int i = 0; i < max_iterations; i++) {
             N = WGS84_a / sqrt(1.0 - WGS84_e_sq * sin(lat_prev) * sin(lat_prev));
             h = p / cos(lat_prev) - N;
             double lat_next = atan2(z, p * (1.0 - WGS84_e_sq * N / (N + h)));
             
             if (fabs(lat_next - lat_prev) < tolerance)
                 break;
                 
             lat_prev = lat_next;
         }
         
         lat = lat_prev;
         alt = h;
     }
     
     // Convert LLA to ECEF
     static void llaToEcef(double lat, double lon, double alt, double& x, double& y, double& z) {
         double N = WGS84_a / sqrt(1.0 - WGS84_e_sq * sin(lat) * sin(lat));
         
         x = (N + alt) * cos(lat) * cos(lon);
         y = (N + alt) * cos(lat) * sin(lon);
         z = (N * (1.0 - WGS84_e_sq) + alt) * sin(lat);
     }
 
     // Convert ECEF to local geodetic (ENU)
     static void ecefToEnu(double ref_lat, double ref_lon, double ref_alt,
                          double ref_x, double ref_y, double ref_z,
                          double point_x, double point_y, double point_z,
                          double& e, double& n, double& u) {
         // Calculate offset in ECEF
         double dx = point_x - ref_x;
         double dy = point_y - ref_y;
         double dz = point_z - ref_z;
         
         // Compute rotation matrix from ECEF to ENU
         double sin_lat = sin(ref_lat);
         double cos_lat = cos(ref_lat);
         double sin_lon = sin(ref_lon);
         double cos_lon = cos(ref_lon);
         
         // Apply rotation
         e = -sin_lon*dx + cos_lon*dy;
         n = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz;
         u = cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz;
     }
 };
 
 // Kepler's equation solution for eccentric anomaly
 static double Kepler(double M, double e) {
     double E = M;  // Initial guess
     const int MAX_ITER = 30;
     const double EPSILON = 1.0e-12;
     
     for (int i = 0; i < MAX_ITER; i++) {
         double E_new = M + e * sin(E);
         if (fabs(E_new - E) < EPSILON) {
             E = E_new;
             break;
         }
         E = E_new;
     }
     
     return E;
 }
 
 // GPS satellite position and clock calculator
 class GpsEphemerisCalculator {
 public:
     static bool computeSatPosVel(const GpsEphemeris& eph, double transmit_time, 
                                 double& x, double& y, double& z, 
                                 double& clock_bias,
                                 bool force_use_ephemeris = false) {
         if (!eph.valid) {
             ROS_DEBUG("GPS ephemeris not valid for satellite %d", eph.sat);
             return false;
         }
         
         // Add bounds checking
         if (eph.sqrta <= 0) {
             ROS_WARN("Invalid semi-major axis (sqrta = %.2f) for GPS satellite %d", 
                     eph.sqrta, eph.sat);
             return false;
         }
         
         // Constants
         const double mu = MU_GPS;        // Earth's gravitational constant for GPS
         const double omega_e = 7.2921151467e-5; // Earth's rotation rate
         
         // Time from ephemeris reference epoch (toe)
         double tk = adjustTimeWithinWeek(transmit_time, eph.toe_sec);
         
         // Check if ephemeris is valid for this time
         if (!force_use_ephemeris && std::abs(tk) > MAX_EPH_AGE) {
             ROS_DEBUG("Ephemeris age (%.1f s) exceeds limit for GPS satellite %d (PRN %d). toe=%.1f, tx_time=%.1f", 
                     std::abs(tk), eph.sat, eph.sat, eph.toe_sec, transmit_time);
             return false;
         }
         
         ROS_DEBUG("Using ephemeris for GPS PRN %d with age %.1f seconds", eph.sat, tk);
         
         // Compute mean motion
         double a = eph.sqrta * eph.sqrta;      // Semi-major axis
         double n0 = sqrt(mu / (a * a * a));    // Computed mean motion
         double n = n0 + eph.delta_n;          // Corrected mean motion
         
         // Mean anomaly
         double M = eph.m0 + n * tk;
         
         // Solve Kepler's equation for eccentric anomaly
         double E = Kepler(M, eph.e);
         
         // Calculate satellite clock correction
         double dt = adjustTimeWithinWeek(transmit_time, eph.toc_sec);
                 
         // Clock correction calculation
         double sin_E = sin(E);
         double cos_E = cos(E);
         
         clock_bias = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt;
         
         // Relativistic correction
         double E_corr = -2.0 * sqrt(mu) * eph.e * eph.sqrta * sin_E / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
         clock_bias += E_corr;
         
         // Log detailed clock parameters for debugging
         ROS_DEBUG("GPS PRN %d clock calculation:", eph.sat);
         ROS_DEBUG("  TOC: %.3f, transmit time: %.3f, dt: %.3f seconds", 
                  eph.toc_sec, transmit_time, dt);
         ROS_DEBUG("  Clock params: af0=%.12e, af1=%.12e, af2=%.12e", 
                  eph.af0, eph.af1, eph.af2);
         ROS_DEBUG("  Polynomial term: %.12f s", eph.af0 + eph.af1 * dt + eph.af2 * dt * dt);
         ROS_DEBUG("  Relativistic correction: %.12f s", E_corr);
         ROS_DEBUG("  Final clock bias: %.12f s (%.3f m)", clock_bias, clock_bias * SPEED_OF_LIGHT);
         
         // True anomaly
         double nu = atan2(sqrt(1.0 - eph.e * eph.e) * sin_E, cos_E - eph.e);
         
         // Argument of latitude
         double phi = nu + eph.omega;
         
         // Second harmonic perturbations
         double sin_2phi = sin(2.0 * phi);
         double cos_2phi = cos(2.0 * phi);
         
         double du = eph.cus * sin_2phi + eph.cuc * cos_2phi;  // Argument of latitude correction
         double dr = eph.crs * sin_2phi + eph.crc * cos_2phi;  // Radius correction
         double di = eph.cis * sin_2phi + eph.cic * cos_2phi;  // Inclination correction
         
         // Corrected argument of latitude, radius, and inclination
         double u = phi + du;
         double r = a * (1.0 - eph.e * cos_E) + dr;
         double i = eph.i0 + di + eph.i_dot * tk;
         
         // Position in orbital plane
         double x_op = r * cos(u);
         double y_op = r * sin(u);
         
         // Corrected longitude of ascending node
         double Omega = eph.omg + (eph.omg_dot - omega_e) * tk - omega_e * eph.toe_sec;
         
         // Earth-fixed coordinates
         double sin_i = sin(i);
         double cos_i = cos(i);
         double sin_Omega = sin(Omega);
         double cos_Omega = cos(Omega);
         
         x = x_op * cos_Omega - y_op * cos_i * sin_Omega;
         y = x_op * sin_Omega + y_op * cos_i * cos_Omega;
         z = y_op * sin_i;
         
         return true;
     }
 };
 
 // Klobuchar ionospheric model
 class KlobucharIonoModel {
 public:
     static double computeIonoDelay(const IonoParams& params, double time, double lat, double lon, 
                                   double elevation, double azimuth) {
         if (!params.valid) {
             return 0.0;
         }
         
         // Use absolute value of elevation to handle negative elevations
         double elevation_abs = fabs(elevation);
         if (elevation_abs < 0.05) {
             elevation_abs = 0.05;
         }
         
         // Convert to semi-circles
         double lat_sc = lat / PI;
         double lon_sc = lon / PI;
         
         // Elevation in semi-circles (positive)
         double el_sc = elevation_abs / PI;
         
         // Earth-centered angle (semi-circles)
         double psi = 0.0137 / (el_sc + 0.11) - 0.022;
         
         // Subionospheric latitude (semi-circles)
         double phi_i = lat_sc + psi * cos(azimuth);
         phi_i = std::max(-0.416, std::min(phi_i, 0.416));
         
         // Subionospheric longitude (semi-circles)
         double lambda_i = lon_sc + psi * sin(azimuth) / cos(phi_i * PI);
         
         // Geomagnetic latitude (semi-circles)
         double phi_m = phi_i + 0.064 * cos((lambda_i - 1.617) * PI);
         
         // Local time (sec)
         double t = 43200.0 * lambda_i + time;
         t = fmod(t, 86400.0);
         if (t < 0.0) t += 86400.0;
         
         // Slant factor
         double f = 1.0 + 16.0 * pow(0.53 - el_sc, 3);
         
         // Period of ionospheric model
         double amp = params.alpha0 + params.alpha1 * phi_m + params.alpha2 * phi_m * phi_m + params.alpha3 * phi_m * phi_m * phi_m;
         amp = std::max(0.0, amp);
         
         // If all alphas are zero, use default
         if (params.alpha0 == 0.0 && params.alpha1 == 0.0 && params.alpha2 == 0.0 && params.alpha3 == 0.0) {
             amp = 5.0e-9; // Default value
         }
         
         double per = params.beta0 + params.beta1 * phi_m + params.beta2 * phi_m * phi_m + params.beta3 * phi_m * phi_m * phi_m;
         per = std::max(72000.0, per);
         
         // Phase of ionospheric model (radians)
         double x = 2.0 * PI * (t - 50400.0) / per;
         
         // Ionospheric delay
         double delay;
         if (fabs(x) < 1.57) {
             delay = f * (5.0e-9 + amp * (1.0 - x*x/2.0 + x*x*x*x/24.0));
         } else {
             delay = f * 5.0e-9;
         }
         
         // Convert to meters
         return delay * SPEED_OF_LIGHT;
     }
 };
 
 // GPS-only pseudorange residual for Ceres solver (simpler model for stability)
 struct GpsPseudorangeResidual {
     GpsPseudorangeResidual(const SatelliteInfo& sat_info)
         : sat_info_(sat_info) {}
     
     template <typename T>
     bool operator()(const T* const receiver_state, T* residual) const {
         // Extract receiver position and clock bias
         T rx = receiver_state[0];
         T ry = receiver_state[1];
         T rz = receiver_state[2];
         T clock_bias = receiver_state[3];
         
         // Satellite position
         T sx = T(sat_info_.sat_pos_x);
         T sy = T(sat_info_.sat_pos_y);
         T sz = T(sat_info_.sat_pos_z);
         
         // Compute geometric range
         T dx = sx - rx;  // Changed sign for better geometric interpretation
         T dy = sy - ry;
         T dz = sz - rz;
         T geometric_range = ceres::sqrt(dx*dx + dy*dy + dz*dz);
         
         // Compute Sagnac effect correction
         T sagnac_correction = -T(EARTH_ROTATION_RATE) * (rx * sy - ry * sx) / T(SPEED_OF_LIGHT);
         
         // Satellite clock correction
         T sat_clock_correction = T(sat_info_.sat_clock_bias) * T(SPEED_OF_LIGHT);
         
         // Compute expected pseudorange
         T expected_pseudorange = geometric_range + clock_bias + sagnac_correction + 
                                T(sat_info_.iono_delay) + T(sat_info_.trop_delay) -
                                T(sat_info_.tgd) * T(SPEED_OF_LIGHT) - sat_clock_correction;
         
         // Residual
         residual[0] = (T(sat_info_.pseudorange) - expected_pseudorange) / T(sat_info_.psr_std);
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info) {
         return new ceres::AutoDiffCostFunction<GpsPseudorangeResidual, 1, 4>(
             new GpsPseudorangeResidual(sat_info));
     }
     
 private:
     const SatelliteInfo sat_info_;
 };
 
 class GnssWlsNode {
 public:
     GnssWlsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : initialized_(false) {
         // Load parameters
         pnh.param<std::string>("frame_id", frame_id_, "gnss");
         pnh.param<double>("pseudorange_noise", pseudorange_noise_, DEFAULT_PSEUDORANGE_NOISE);
         pnh.param<int>("min_satellites", min_satellites_, 4);  // Minimum 4 satellites needed
         pnh.param<double>("initial_latitude", initial_latitude_, 22.3193);  // Default to Hong Kong
         pnh.param<double>("initial_longitude", initial_longitude_, 114.1694);  // Default to Hong Kong
         pnh.param<double>("initial_altitude", initial_altitude_, 100.0);  // Default to Hong Kong elevation
         pnh.param<bool>("apply_iono_correction", apply_iono_correction_, true);
         pnh.param<double>("min_cn0", min_cn0_, 10.0);
         pnh.param<std::string>("output_csv_path", output_csv_path_, "spp_results.csv");
         
         // Configurable elevation cutoff angle
         pnh.param<double>("cut_off_degree", cut_off_degree_, 10.0);
         
         // Debug mode options
         pnh.param<bool>("disable_cn0_filter", disable_cn0_filter_, false);
         pnh.param<bool>("disable_elevation_filter", disable_elevation_filter_, false);
         pnh.param<bool>("force_use_ephemeris", force_use_ephemeris_, true);  // Force using any available ephemeris
         
         // Fixed GPS week for time conversion
         pnh.param<int>("current_gps_week", current_gps_week_, 2134);  // Default for GVINS dataset
         
         // Get current GPS leap seconds
         pnh.param<double>("current_leap_seconds", current_leap_seconds_, CURRENT_GPS_LEAP_SECONDS);
         
         // Position correction offset for datum alignment
         pnh.param<bool>("apply_position_offset", apply_position_offset_, false);
         pnh.param<double>("position_offset_east", position_offset_east_, 0.0);
         pnh.param<double>("position_offset_north", position_offset_north_, 0.0);
         pnh.param<double>("position_offset_up", position_offset_up_, 0.0);
         
         // Calculate ECEF coordinates using proper WGS84 model
         double lat_rad = initial_latitude_ * M_PI / 180.0;
         double lon_rad = initial_longitude_ * M_PI / 180.0;
         double N = WGS84_a / sqrt(1.0 - WGS84_e_sq * sin(lat_rad) * sin(lat_rad));
         double init_x = (N + initial_altitude_) * cos(lat_rad) * cos(lon_rad);
         double init_y = (N + initial_altitude_) * cos(lat_rad) * sin(lon_rad);
         double init_z = (N * (1.0 - WGS84_e_sq) + initial_altitude_) * sin(lat_rad);
         
         current_solution_.x = init_x;
         current_solution_.y = init_y;
         current_solution_.z = init_z;
         current_solution_.clock_bias = 0.0;  // Initialize to 0
         current_solution_.timestamp = 0.0;
         
         // Initialize ionospheric parameters with defaults
         iono_params_.alpha0 = 0.1118E-07;
         iono_params_.alpha1 = 0.2235E-07;
         iono_params_.alpha2 = -0.4172E-06;
         iono_params_.alpha3 = 0.6557E-06;
         iono_params_.beta0 = 0.1249E+06;
         iono_params_.beta1 = -0.4424E+06;
         iono_params_.beta2 = 0.1507E+07;
         iono_params_.beta3 = -0.2621E+06;
         iono_params_.valid = true;  // Use default parameters
         iono_params_.last_update = ros::Time::now();
         
         ROS_INFO("Using default ionospheric parameters");
         
         // Publishers
         navsatfix_pub_ = nh.advertise<sensor_msgs::NavSatFix>("gnss_fix", 10);
         odom_pub_ = nh.advertise<nav_msgs::Odometry>("gnss_odom", 10);
         pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("gnss_pose", 10);
         
         // Subscribers for GNSS data
         ephem_sub_ = nh.subscribe("/ublox_driver/ephem", 10, &GnssWlsNode::ephemCallback, this);
         iono_params_sub_ = nh.subscribe("/ublox_driver/iono_params", 10, &GnssWlsNode::ionoParamsCallback, this);
         raw_meas_sub_ = nh.subscribe("/ublox_driver/range_meas", 10, &GnssWlsNode::rawMeasCallback, this);
         
         // Open CSV file for saving results
         if (!output_csv_path_.empty()) {
             csv_file_.open(output_csv_path_, std::ios::out);
             if (csv_file_.is_open()) {
                 // Write CSV header with column names
                 csv_file_ << "Timestamp,GPSWeek,TOW,Latitude,Longitude,Altitude,ECEF_X,ECEF_Y,ECEF_Z,ENU_E,ENU_N,ENU_U,"
                           << "ReceiverClockBias,NumSatellites,PDOP,HDOP,VDOP,TDOP" 
                           << std::endl;
                 ROS_INFO("Opened CSV file for saving results: %s", output_csv_path_.c_str());
             } else {
                 ROS_WARN("Could not open CSV file at: %s", output_csv_path_.c_str());
             }
         }
         
         ROS_INFO("GNSS SPP node initialized:");
         ROS_INFO(" - Initial position: Lat=%.6f°, Lon=%.6f°, Alt=%.1fm", 
                 initial_latitude_, initial_longitude_, initial_altitude_);
         ROS_INFO(" - ECEF position: [%.1f, %.1f, %.1f]", init_x, init_y, init_z);
         ROS_INFO(" - Minimum satellites: %d", min_satellites_);
         ROS_INFO(" - Elevation cutoff: %.1f degrees", cut_off_degree_);
         ROS_INFO(" - Minimum CN0: %.1f dB-Hz", min_cn0_);
         ROS_INFO(" - Apply ionospheric correction: %s", apply_iono_correction_ ? "true" : "false");
         ROS_INFO(" - Force use ephemeris: %s", force_use_ephemeris_ ? "true" : "false");
         ROS_INFO(" - Current GPS week: %d", current_gps_week_);
         ROS_INFO(" - Current GPS-UTC leap seconds: %.1f", current_leap_seconds_);
         
         if (apply_position_offset_) {
             ROS_INFO(" - Applying position offset: E=%.2f, N=%.2f, U=%.2f m", 
                     position_offset_east_, position_offset_north_, position_offset_up_);
         }
     }
     
     ~GnssWlsNode() {
         if (csv_file_.is_open()) {
             csv_file_.close();
         }
     }
     
     // Process GPS ephemeris data
     void ephemCallback(const gnss_comm::GnssEphemMsg::ConstPtr& msg) {
         static uint32_t ephem_count = 0;
         
         std::lock_guard<std::mutex> lock(gps_ephem_mutex_);
         
         // Check if we already have an ephemeris for this satellite
         bool update_existing = (gps_ephemeris_.find(msg->sat) != gps_ephemeris_.end());
         
         GpsEphemeris eph;
         eph.sat = msg->sat;
         eph.toe_sec = gnss_time_to_sec(msg->toe);
         eph.toc_sec = gnss_time_to_sec(msg->toc);
         eph.week = msg->week;
         eph.af0 = msg->af0;
         eph.af1 = msg->af1;
         eph.af2 = msg->af2;
         eph.crs = msg->crs;
         eph.crc = msg->crc;
         eph.cus = msg->cus;
         eph.cuc = msg->cuc;
         eph.cis = msg->cis;
         eph.cic = msg->cic;
         eph.delta_n = msg->delta_n;
         eph.m0 = msg->M0;
         eph.e = msg->e;
         eph.sqrta = sqrt(msg->A);  // A is the semi-major axis squared, we need sqrt(A)
         eph.omg = msg->OMG0;  // Longitude of ascending node
         eph.omg_dot = msg->OMG_dot;
         eph.omega = msg->omg;  // Argument of perigee
         eph.i0 = msg->i0;
         eph.i_dot = msg->i_dot;
         eph.tgd0 = msg->tgd0;
         eph.health = msg->health;
         eph.iode = msg->iode;
         eph.valid = true;
         eph.last_update = ros::Time::now();
         eph.ura = msg->ura;
         
         // Store/update ephemeris
         gps_ephemeris_[msg->sat] = eph;
         
         ephem_count++;
         if (update_existing) {
             ROS_INFO("Updated GPS ephemeris for PRN %d, toe=%.0f, valid=%d", 
                    msg->sat, gnss_time_to_sec(msg->toe), eph.valid);
         } else {
             ROS_INFO("Received new GPS ephemeris for PRN %d, toe=%.0f, valid=%d", 
                    msg->sat, gnss_time_to_sec(msg->toe), eph.valid);
         }
         
         if (ephem_count % 10 == 0 || ephem_count < 10) {
             ROS_INFO("Received total %u GPS ephemeris messages, have data for %zu satellites", 
                    ephem_count, gps_ephemeris_.size());
         }
     }
     
     // Process ionospheric parameters
     void ionoParamsCallback(const gnss_comm::StampedFloat64Array::ConstPtr& msg) {
         if (msg->data.size() != 8) {
             ROS_WARN("Invalid ionospheric parameters array size: %zu (expected 8)", msg->data.size());
             return;
         }
         
         std::lock_guard<std::mutex> lock(iono_params_mutex_);
         
         // Alpha parameters (amplitude of vertical delay)
         iono_params_.alpha0 = msg->data[0];
         iono_params_.alpha1 = msg->data[1];
         iono_params_.alpha2 = msg->data[2];
         iono_params_.alpha3 = msg->data[3];
         
         // Beta parameters (period of model)
         iono_params_.beta0 = msg->data[4];
         iono_params_.beta1 = msg->data[5];
         iono_params_.beta2 = msg->data[6];
         iono_params_.beta3 = msg->data[7];
         
         iono_params_.valid = true;
         iono_params_.last_update = ros::Time::now();
         
         ROS_INFO("Updated ionospheric parameters: alpha=[%.2e, %.2e, %.2e, %.2e], beta=[%.2e, %.2e, %.2e, %.2e]",
                 iono_params_.alpha0, iono_params_.alpha1, iono_params_.alpha2, iono_params_.alpha3,
                 iono_params_.beta0, iono_params_.beta1, iono_params_.beta2, iono_params_.beta3);
     }
     
     // Process raw GNSS measurements
     void rawMeasCallback(const gnss_comm::GnssMeasMsg::ConstPtr& msg) {
         if (msg->meas.empty()) {
             ROS_WARN("Received GNSS measurement message with no measurements");
             return;
         }
         
         std::vector<SatelliteInfo> satellites;
         
         // Get GPS time of week
         double gps_tow = 0.0;
         uint32_t gps_week = current_gps_week_;
         
         if (msg->meas.size() > 0 && msg->meas[0].time.week > 0) {
             // Use the time tag from the measurements
             gps_week = msg->meas[0].time.week;
             gps_tow = msg->meas[0].time.tow;
             ROS_INFO("Using measurement time: Week %d, TOW %.3f", gps_week, gps_tow);
         } else if (!gps_ephemeris_.empty()) {
             // Use most recent ephemeris time
             auto it = gps_ephemeris_.begin();
             gps_tow = it->second.toe_sec;
             ROS_INFO("Using ephemeris time: toe=%.3f", gps_tow);
         } else {
             gps_tow = fmod(ros::Time::now().toSec() + current_leap_seconds_, GPS_SECONDS_PER_WEEK);
             ROS_WARN("Using system time as fallback: GPS TOW = %.3f", gps_tow);
         }
         
         // Store current time for use in other functions
         current_time_ = gps_tow;
         
         ROS_INFO("Processing %zu GNSS observations at GPS TOW %.3f", msg->meas.size(), gps_tow);
         
         // Print ephemeris status periodically
         static ros::Time last_ephem_status_time = ros::Time::now();
         if ((ros::Time::now() - last_ephem_status_time).toSec() > 10.0) {
             ROS_INFO("GNSS Ephemeris Status: GPS=%zu", gps_ephemeris_.size());
             last_ephem_status_time = ros::Time::now();
         }
         
         // Diagnostic counters
         int count_empty_psr = 0;
         int count_invalid_psr = 0;
         int count_low_cn0 = 0;
         int count_no_ephemeris = 0;
         int count_ephemeris_error = 0;
         int count_below_elevation = 0;
         
         // Loop through all measurements and calculate satellite positions
         for (const auto& obs : msg->meas) {
             // Create a satellite info structure
             SatelliteInfo sat_info;
             sat_info.sat_id = obs.sat;
             
             // Only process GPS satellites
             if (obs.sat < 1 || obs.sat > 32) {
                 // Skip non-GPS satellites
                 continue;
             }
             
             sat_info.system = GPS;
             
             // Skip if no valid measurements
             if (obs.psr.empty()) {
                 count_empty_psr++;
                 continue;
             }
             
             // Check pseudorange validity
             if (obs.psr[0] <= 0 || std::isnan(obs.psr[0])) {
                 count_invalid_psr++;
                 continue;
             }
             
             // Store pseudorange
             sat_info.pseudorange = obs.psr[0];
             
             // Store CN0 if available
             if (!obs.CN0.empty() && !std::isnan(obs.CN0[0])) {
                 sat_info.cn0 = obs.CN0[0];
             } else {
                 sat_info.cn0 = 0.0;
             }
             
             // Skip measurements with low signal strength (unless disabled)
             if (!disable_cn0_filter_ && sat_info.cn0 < min_cn0_) {
                 count_low_cn0++;
                 continue;
             }
             
             // Calculate satellite position and clock bias
             {
                 std::lock_guard<std::mutex> lock(gps_ephem_mutex_);
                 
                 // Check if we have ephemeris for this satellite
                 if (gps_ephemeris_.find(sat_info.sat_id) == gps_ephemeris_.end()) {
                     count_no_ephemeris++;
                     continue;
                 }
                 
                 // Calculate exact transmission time by correcting pseudorange for light travel time
                 double transmission_time = gps_tow - sat_info.pseudorange / SPEED_OF_LIGHT;
                 
                 // Compute satellite position and clock
                 double clock_bias;
                 bool success = GpsEphemerisCalculator::computeSatPosVel(
                     gps_ephemeris_[sat_info.sat_id], 
                     transmission_time, 
                     sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                     clock_bias,
                     force_use_ephemeris_);
                 
                 if (success) {
                     sat_info.sat_clock_bias = clock_bias;
                     sat_info.tgd = gps_ephemeris_[sat_info.sat_id].tgd0;
                     sat_info.ura = gps_ephemeris_[sat_info.sat_id].ura;
                     sat_info.ephemeris_time = gps_ephemeris_[sat_info.sat_id].toe_sec;
                 } else {
                     count_ephemeris_error++;
                     continue;
                 }
             }
             
             // Calculate elevation and azimuth angles
             calculateElevationAzimuth(
                 current_solution_.x, current_solution_.y, current_solution_.z,
                 sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                 sat_info.elevation, sat_info.azimuth);
             
             // Skip satellites below elevation mask
             if (!disable_elevation_filter_ && sat_info.elevation * 180.0/M_PI < cut_off_degree_) {
                 count_below_elevation++;
                 continue;
             }
             
             // Calculate weight based on elevation and CN0
             calculateMeasurementWeight(sat_info);
             
             // Compute ionospheric delay if enabled
             sat_info.iono_delay = 0.0;
             if (apply_iono_correction_ && iono_params_.valid) {
                 // Convert ECEF to geodetic for current position
                 double lat, lon, alt;
                 CoordinateConverter::ecefToLla(current_solution_.x, current_solution_.y, current_solution_.z, lat, lon, alt);
                 
                 sat_info.iono_delay = KlobucharIonoModel::computeIonoDelay(
                     iono_params_, gps_tow, lat, lon, sat_info.elevation, sat_info.azimuth);
                 
                 ROS_DEBUG("Ionospheric delay for PRN %d: %.2f m", sat_info.sat_id, sat_info.iono_delay);
             }
             
             // Compute tropospheric delay (simplified model)
             sat_info.trop_delay = 2.3 / std::max(sin(std::abs(sat_info.elevation)), 0.1);
             
             // Add satellite to the list
             satellites.push_back(sat_info);
         }
         
         // Make sure we have enough satellites
         if (satellites.size() < min_satellites_) {
             ROS_WARN("Not enough valid satellites: %zu (need %d)", satellites.size(), min_satellites_);
             ROS_INFO("Counters: empty_psr=%d, invalid_psr=%d, low_cn0=%d, no_ephemeris=%d, ephemeris_error=%d, below_elevation=%d",
                      count_empty_psr, count_invalid_psr, count_low_cn0, count_no_ephemeris, count_ephemeris_error, count_below_elevation);
             return;
         }
         
         // Create header from the message timestamp
         std_msgs::Header header;
         header.stamp = ros::Time(gps_tow);
         header.frame_id = frame_id_;
         
         // Run WLS solver
         GnssSolution solution;
         solution.timestamp = gps_tow;
         
         // Compute position
         bool position_success = solveGpsOnlyWLS(satellites, solution);
         
         if (position_success) {
             // If we just initialized, save the reference position
             if (!initialized_) {
                 ROS_INFO("Initial position found: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm, Sats=%d", 
                         solution.latitude * 180.0 / M_PI, 
                         solution.longitude * 180.0 / M_PI, 
                         solution.altitude, 
                         solution.num_satellites);
                 
                 // Mark as initialized
                 initialized_ = true;
                 
                 // Set ENU reference point
                 ref_lat_ = solution.latitude;
                 ref_lon_ = solution.longitude;
                 ref_alt_ = solution.altitude;
                 ref_ecef_x_ = solution.x;
                 ref_ecef_y_ = solution.y;
                 ref_ecef_z_ = solution.z;
                 enu_reference_set_ = true;
                 
                 ROS_INFO("Set ENU reference: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm", 
                         ref_lat_ * 180.0 / M_PI, ref_lon_ * 180.0 / M_PI, ref_alt_);
             }
             
             // Apply position offset if enabled
             if (apply_position_offset_ && enu_reference_set_) {
                 // Convert the offset from ENU to ECEF
                 double e = position_offset_east_;
                 double n = position_offset_north_;
                 double u = position_offset_up_;
                 
                 // Compute rotation matrix from ENU to ECEF
                 double sin_lat = sin(ref_lat_);
                 double cos_lat = cos(ref_lat_);
                 double sin_lon = sin(ref_lon_);
                 double cos_lon = cos(ref_lon_);
                 
                 // ENU to ECEF transformation
                 double dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u;
                 double dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u;
                 double dz = cos_lat * n + sin_lat * u;
                 
                 // Apply offset to ECEF coordinates
                 solution.x += dx;
                 solution.y += dy;
                 solution.z += dz;
                 
                 // Recompute latitude, longitude, altitude
                 CoordinateConverter::ecefToLla(solution.x, solution.y, solution.z, 
                                               solution.latitude, solution.longitude, solution.altitude);
                 
                 ROS_DEBUG("Applied position offset: E=%.2f, N=%.2f, U=%.2f m", e, n, u);
             }
             
             // Store solution
             current_solution_ = solution;
             
             // Publish results
             publishResults(header, solution);
             
             // Calculate ENU coordinates
             double e = 0.0, n = 0.0, u = 0.0;
             if (enu_reference_set_) {
                 CoordinateConverter::ecefToEnu(ref_lat_, ref_lon_, ref_alt_,
                                               ref_ecef_x_, ref_ecef_y_, ref_ecef_z_,
                                               solution.x, solution.y, solution.z,
                                               e, n, u);
             }
             
             // Save results to CSV file
             if (csv_file_.is_open()) {
                 ros::Time ros_time = ros::Time::now();
                 csv_file_ << std::fixed << std::setprecision(6)
                           << ros_time.toSec() << "," 
                           << gps_week << "," 
                           << gps_tow << ","
                           << solution.latitude * 180.0 / M_PI << "," // Convert to degrees
                           << solution.longitude * 180.0 / M_PI << ","
                           << solution.altitude << ","
                           << solution.x << ","
                           << solution.y << ","
                           << solution.z << ","
                           << e << ","
                           << n << ","
                           << u << ","
                           << solution.clock_bias << ","
                           << solution.num_satellites << ","
                           << solution.pdop << ","
                           << solution.hdop << ","
                           << solution.vdop << ","
                           << solution.tdop << std::endl;
             }
             
             ROS_INFO("GNSS solution: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm, Sats=%d, HDOP=%.2f", 
                     solution.latitude * 180.0 / M_PI, 
                     solution.longitude * 180.0 / M_PI, 
                     solution.altitude, 
                     solution.num_satellites,
                     solution.hdop);
                     
             ROS_INFO("GPS clock bias: %.2f m", solution.clock_bias);
             
             if (enu_reference_set_) {
                 ROS_INFO("ENU position: E=%.2f, N=%.2f, U=%.2f m", e, n, u);
             }
         } else {
             ROS_WARN("Failed to compute GNSS solution with %zu satellites", satellites.size());
         }
     }
     
 private:
     // Node parameters
     std::string frame_id_;
     double pseudorange_noise_;
     int min_satellites_;
     double initial_latitude_;
     double initial_longitude_;
     double initial_altitude_;
     bool apply_iono_correction_;
     double cut_off_degree_;     // Configurable elevation cutoff angle
     double min_cn0_;
     bool initialized_;
     std::string output_csv_path_;
     std::ofstream csv_file_;
     int current_gps_week_;     // Current GPS week
     double current_leap_seconds_; // Current GPS-UTC leap seconds
     double current_time_;      // Store current time for use in other functions
     
     // Position correction parameters
     bool apply_position_offset_;
     double position_offset_east_;
     double position_offset_north_;
     double position_offset_up_;
     
     // ENU reference point (first fix)
     bool enu_reference_set_ = false;
     double ref_lat_ = 0.0;
     double ref_lon_ = 0.0;
     double ref_alt_ = 0.0;
     double ref_ecef_x_ = 0.0;
     double ref_ecef_y_ = 0.0;
     double ref_ecef_z_ = 0.0;
     
     // Debug options
     bool disable_cn0_filter_;
     bool disable_elevation_filter_;
     bool force_use_ephemeris_;
     
     // Current solution
     GnssSolution current_solution_;
     
     // ROS interfaces
     ros::Subscriber ephem_sub_;
     ros::Subscriber iono_params_sub_;
     ros::Subscriber raw_meas_sub_;
     ros::Publisher navsatfix_pub_;
     ros::Publisher odom_pub_;
     ros::Publisher pose_pub_;
     
     // Data storage
     std::map<uint32_t, GpsEphemeris> gps_ephemeris_;
     IonoParams iono_params_;
     std::mutex gps_ephem_mutex_;
     std::mutex iono_params_mutex_;
     
     void calculateElevationAzimuth(
         double rx, double ry, double rz,
         double sx, double sy, double sz,
         double& elevation, double& azimuth) {
         
         // Compute vector from receiver to satellite
         double dx = sx - rx;
         double dy = sy - ry;
         double dz = sz - rz;
         
         // Length of the vector (range)
         double range = sqrt(dx*dx + dy*dy + dz*dz);
         
         // Check for valid range
         if (range < 1e-6) {
             ROS_WARN("Invalid range in elevation calculation");
             elevation = 0.0;
             azimuth = 0.0;
             return;
         }
         
         // Convert receiver ECEF to LLA
         double lat, lon, alt;
         CoordinateConverter::ecefToLla(rx, ry, rz, lat, lon, alt);
         
         // Compute ENU vector from receiver to satellite
         // Rotation matrix from ECEF to ENU
         double sin_lat = sin(lat);
         double cos_lat = cos(lat);
         double sin_lon = sin(lon);
         double cos_lon = cos(lon);
         
         // Rotate ECEF vector to ENU
         double e = -sin_lon * dx + cos_lon * dy;
         double n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
         double u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
         
         // Compute elevation and azimuth
         double horizontal_distance = sqrt(e*e + n*n);
         
         elevation = atan2(u, horizontal_distance);
         azimuth = atan2(e, n);
         
         // Normalize azimuth to [0, 2π)
         if (azimuth < 0) {
             azimuth += 2 * M_PI;
         }
     }
     
     void calculateMeasurementWeight(SatelliteInfo& sat_info) {
         // Weight based on sin²(elevation)
         // Use absolute elevation angle for weighting to handle negative elevations
         double abs_elevation = fabs(sat_info.elevation);
         double sin_el = sin(abs_elevation);
         double elevation_weight = std::max(0.1, sin_el * sin_el);
         
         // Calculate weight based on CN0 (carrier-to-noise density ratio)
         double cn0_weight = 1.0;
         if (sat_info.cn0 > 0) {
             // Use a more gradual scaling with CN0
             cn0_weight = std::min(1.0, std::max(0.2, (sat_info.cn0 - min_cn0_) / 30.0));
         }
         
         // URA-based weight
         double ura_weight = 1.0;
         if (sat_info.ura > 0) {
             // Adjust based on URA - lower URA means higher weight
             ura_weight = 1.0 / sat_info.ura;
         }
         
         // Combined weight
         sat_info.weight = elevation_weight * cn0_weight * ura_weight;
         
         // Set pseudorange standard deviation based on weight
         sat_info.psr_std = pseudorange_noise_ / sqrt(sat_info.weight);
     }
     
     // Solve for position using GPS satellites only
     bool solveGpsOnlyWLS(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
         ROS_INFO("Starting GPS-only WLS solver with %zu total satellites", satellites.size());
         
         // Use all satellites now that we've filtered by elevation in measurement processing
         std::vector<uint32_t> gps_idx;
         for (uint32_t i = 0; i < satellites.size(); ++i) {
             gps_idx.push_back(i);
         }
         
         if (gps_idx.size() < 4) {
             ROS_WARN("Too few GPS satellites for positioning: %zu (need at least 4)", gps_idx.size());
             return false;
         }
         
         ROS_INFO("Using %zu GPS satellites for position fix", gps_idx.size());
         
         // 4-parameter state for GPS-only: [x, y, z, clock_bias]
         double state[4] = {0.0, 0.0, 0.0, 0.0};
         
         // Use current position as starting point
         state[0] = current_solution_.x;
         state[1] = current_solution_.y;
         state[2] = current_solution_.z;
         state[3] = current_solution_.clock_bias;
         
         ROS_INFO("Using current position as initial state: [%.1f, %.1f, %.1f], clock=%.1f",
                 state[0], state[1], state[2], state[3]);
         
         // Set up the Ceres problem
         ceres::Problem problem;
         
         // Vector to store loss functions to manage their memory
         std::vector<ceres::LossFunction*> loss_functions;
         
         // Add residual blocks for each GPS satellite
         for (const auto& idx : gps_idx) {
             const auto& sat = satellites[idx];
             
             ceres::CostFunction* cost_function = GpsPseudorangeResidual::Create(sat);
             ceres::LossFunction* loss_function = new ceres::HuberLoss(5.0);
             loss_functions.push_back(loss_function);
             
             problem.AddResidualBlock(
                 cost_function,
                 loss_function,
                 state);
         }
         
         // Configure the solver - standard options for stability
         ceres::Solver::Options options;
         options.linear_solver_type = ceres::DENSE_QR;
         options.minimizer_progress_to_stdout = false;
         options.max_num_iterations = 15;  // Increase iterations for better convergence
         options.function_tolerance = 1e-8;
         options.gradient_tolerance = 1e-10;
         options.parameter_tolerance = 1e-10;
         
         // Run the solver
         ceres::Solver::Summary summary;
         ceres::Solve(options, &problem, &summary);
         
         if (!summary.IsSolutionUsable()) {
             ROS_WARN("GPS-only solver failed: %s", summary.BriefReport().c_str());
             return false;
         }
         
         // Log final solution
         ROS_INFO("Iteration 0 solution: [%.2f, %.2f, %.2f], clock=%.2f, cost=%.6f", 
                state[0], state[1], state[2], state[3], summary.final_cost);
         
         // Extract solution
         ROS_INFO("GPS-only solution state: [%.2f, %.2f, %.2f], clock=%.2f", 
                 state[0], state[1], state[2], state[3]);
         
         solution.x = state[0];
         solution.y = state[1];
         solution.z = state[2];
         solution.clock_bias = state[3];
         solution.num_satellites = gps_idx.size();
         
         // Convert to geodetic coordinates
         CoordinateConverter::ecefToLla(
             solution.x, solution.y, solution.z,
             solution.latitude, solution.longitude, solution.altitude);
         
         ROS_INFO("GPS-only solution: ECEF [%.2f, %.2f, %.2f] m, clock bias %.2f m", 
                 solution.x, solution.y, solution.z, solution.clock_bias);
         ROS_INFO("             lat/lon [%.6f, %.6f], alt %.2f m", 
                 solution.latitude * 180.0/M_PI, solution.longitude * 180.0/M_PI, solution.altitude);
         
         // Calculate DOP values
         calculateGpsDOP(satellites, gps_idx, solution);
         
         return true;
     }
    
     // Calculate DOP values for GPS-only solutions
     void calculateGpsDOP(const std::vector<SatelliteInfo>& satellites, 
                        const std::vector<uint32_t>& good_idx,
                        GnssSolution& solution) {
         // We need the geometry matrix for DOP calculation
         Eigen::MatrixXd G(good_idx.size(), 4);  // 4 params for GPS-only
         
         // Fill the geometry matrix
         for (size_t i = 0; i < good_idx.size(); i++) {
             const auto& sat = satellites[good_idx[i]];
             
             // Unit vector from receiver to satellite
             double dx = sat.sat_pos_x - solution.x;
             double dy = sat.sat_pos_y - solution.y;
             double dz = sat.sat_pos_z - solution.z;
             double range = sqrt(dx*dx + dy*dy + dz*dz);
             
             // Normalize to unit vector
             if (range > 1e-10) {
                 dx /= range;
                 dy /= range;
                 dz /= range;
             }
             
             // Fill the geometry matrix
             G(i, 0) = -dx;  // x component
             G(i, 1) = -dy;  // y component
             G(i, 2) = -dz;  // z component
             G(i, 3) = 1.0;  // clock bias
         }
         
         // Create weight matrix (diagonal) based on satellite weights
         Eigen::MatrixXd W = Eigen::MatrixXd::Identity(good_idx.size(), good_idx.size());
         for (size_t i = 0; i < good_idx.size(); i++) {
             W(i, i) = satellites[good_idx[i]].weight;
         }
         
         // Compute the covariance matrix
         Eigen::MatrixXd cov_matrix;
         try {
             cov_matrix = (G.transpose() * W * G).inverse();
             
             // Calculate DOP values
             solution.gdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1) + cov_matrix(2,2) + cov_matrix(3,3));
             solution.pdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1) + cov_matrix(2,2));
             solution.hdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1));
             solution.vdop = sqrt(cov_matrix(2,2));
             solution.tdop = sqrt(cov_matrix(3,3));
             
             // Store the covariance
             solution.covariance = cov_matrix;
             
             ROS_INFO("DOP values: GDOP=%.2f, PDOP=%.2f, HDOP=%.2f, VDOP=%.2f, TDOP=%.2f",
                     solution.gdop, solution.pdop, solution.hdop, solution.vdop, solution.tdop);
         } catch (const std::exception& e) {
             ROS_WARN("Exception in GPS-only DOP calculation: %s", e.what());
             // Set default values
             solution.gdop = 99.9;
             solution.pdop = 99.9;
             solution.hdop = 99.9;
             solution.vdop = 99.9;
             solution.tdop = 99.9;
             
             // Set identity covariance as fallback
             solution.covariance = Eigen::Matrix<double, 4, 4>::Identity();
         }
     }
     
     void publishResults(const std_msgs::Header& header, const GnssSolution& solution) {
         // 1. Publish NavSatFix message
         sensor_msgs::NavSatFix navsatfix;
         navsatfix.header = header;
         navsatfix.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
         navsatfix.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
         navsatfix.latitude = solution.latitude * 180.0 / M_PI;  // Convert to degrees
         navsatfix.longitude = solution.longitude * 180.0 / M_PI;
         navsatfix.altitude = solution.altitude;
         
         // Fill in covariance (position only)
         navsatfix.position_covariance[0] = solution.covariance(0, 0);
         navsatfix.position_covariance[4] = solution.covariance(1, 1);
         navsatfix.position_covariance[8] = solution.covariance(2, 2);
         navsatfix.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
         
         // 2. Publish Odometry message (simplified without velocity)
         nav_msgs::Odometry odom;
         odom.header = header;
         odom.header.frame_id = "ecef";
         odom.child_frame_id = frame_id_;
         
         // Set position in ECEF
         odom.pose.pose.position.x = solution.x;
         odom.pose.pose.position.y = solution.y;
         odom.pose.pose.position.z = solution.z;
         odom.pose.pose.orientation.w = 1.0;  // Identity quaternion
         
         // Set position covariance
         for (int i = 0; i < 3; ++i) {
             for (int j = 0; j < 3; ++j) {
                 odom.pose.covariance[i * 6 + j] = solution.covariance(i, j);
             }
         }
         
         // 3. Publish PoseWithCovarianceStamped message (ENU format)
         geometry_msgs::PoseWithCovarianceStamped pose;
         pose.header = header;
         pose.header.frame_id = "map";  // Local ENU frame
         
         // Calculate ENU position relative to reference
         double e = 0.0, n = 0.0, u = 0.0;
         if (enu_reference_set_) {
             CoordinateConverter::ecefToEnu(ref_lat_, ref_lon_, ref_alt_,
                                           ref_ecef_x_, ref_ecef_y_, ref_ecef_z_,
                                           solution.x, solution.y, solution.z,
                                           e, n, u);
             
             pose.pose.pose.position.x = e;
             pose.pose.pose.position.y = n;
             pose.pose.pose.position.z = u;
             pose.pose.pose.orientation.w = 1.0;
             
             // Simplified covariance transformation
             for (int i = 0; i < 3; ++i) {
                 for (int j = 0; j < 3; ++j) {
                     pose.pose.covariance[i * 6 + j] = solution.covariance(i, j);
                 }
             }
         } else {
             // Until we have a reference, just set zeros
             pose.pose.pose.position.x = 0.0;
             pose.pose.pose.position.y = 0.0;
             pose.pose.pose.position.z = 0.0;
             pose.pose.pose.orientation.w = 1.0;
         }
         
         // Publish all messages
         navsatfix_pub_.publish(navsatfix);
         odom_pub_.publish(odom);
         pose_pub_.publish(pose);
     }
 };
 
 int main(int argc, char** argv) {
     ros::init(argc, argv, "gnss_spp_node");
     ros::NodeHandle nh;
     ros::NodeHandle pnh("~");
     
     // Log startup parameters
     ROS_INFO("Starting GNSS SPP Node...");
     
     // Load and display key parameters
     double min_cn0, cut_off_degree;
     int min_sats;
     bool force_use_ephemeris;
     double initial_lat, initial_lon;
     std::string output_csv_path;
     
     pnh.param<double>("min_cn0", min_cn0, 10.0);
     pnh.param<double>("cut_off_degree", cut_off_degree, 10.0);
     pnh.param<int>("min_satellites", min_sats, 4);
     pnh.param<bool>("force_use_ephemeris", force_use_ephemeris, true);
     pnh.param<double>("initial_latitude", initial_lat, 22.3193);  // Default to Hong Kong
     pnh.param<double>("initial_longitude", initial_lon, 114.1694);  // Default to Hong Kong
     pnh.param<std::string>("output_csv_path", output_csv_path, "spp_results.csv");
     
     // Position offset parameters for datum correction
     bool apply_position_offset;
     double pos_offset_e, pos_offset_n, pos_offset_u;
     pnh.param<bool>("apply_position_offset", apply_position_offset, false);
     pnh.param<double>("position_offset_east", pos_offset_e, 0.0);
     pnh.param<double>("position_offset_north", pos_offset_n, 0.0);
     pnh.param<double>("position_offset_up", pos_offset_u, 0.0);
     
     ROS_INFO("Configuration:");
     ROS_INFO(" - Initial position: (%.6f°, %.6f°)", initial_lat, initial_lon);
     ROS_INFO(" - Minimum CN0: %.1f dB-Hz", min_cn0);
     ROS_INFO(" - Elevation cutoff: %.1f degrees", cut_off_degree);
     ROS_INFO(" - Minimum satellites: %d", min_sats);
     ROS_INFO(" - Force use ephemeris: %s", force_use_ephemeris ? "true" : "false");
     ROS_INFO(" - Output CSV file: %s", output_csv_path.c_str());
     
     if (apply_position_offset) {
         ROS_INFO(" - Position offset: E=%.2f, N=%.2f, U=%.2f m", 
                 pos_offset_e, pos_offset_n, pos_offset_u);
     }
     
     GnssWlsNode node(nh, pnh);
     
     ros::spin();
     
     return 0;
 }