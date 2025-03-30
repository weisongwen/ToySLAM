/**
 * @file gnss_spp_node.cpp
 * @brief ROS node for GNSS single point positioning using raw GNSS measurements
 * Optimized for the ZED-F9P receiver in the GVINS dataset
 */

 #include <ros/ros.h>
 #include <Eigen/Dense>
 #include <ceres/ceres.h>
 #include <sensor_msgs/NavSatFix.h>
 #include <nav_msgs/Odometry.h>
 #include <geometry_msgs/PoseWithCovarianceStamped.h>
 #include <tf2/LinearMath/Quaternion.h>
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
 constexpr double GPS_L2_FREQ = 1227.60e6;       // Hz
 constexpr double GPS_L2_WAVELENGTH = SPEED_OF_LIGHT / GPS_L2_FREQ;
 constexpr double GLONASS_L1_BASE_FREQ = 1602.0e6;  // Hz
 constexpr double GLONASS_L1_DELTA_FREQ = 0.5625e6; // Hz
 constexpr double GLONASS_FREQ_NUM_MIN = -7;
 constexpr double GLONASS_FREQ_NUM_MAX = 13;
 constexpr double EARTH_ROTATION_RATE = 7.2921151467e-5;  // rad/s
 constexpr double DEFAULT_PSEUDORANGE_NOISE = 5.0;  // meters
 constexpr double WGS84_a = 6378137.0;              // WGS84 semi-major axis in meters
 constexpr double WGS84_b = 6356752.31424518;       // WGS84 semi-minor axis in meters
 constexpr double WGS84_e_sq = 1 - (WGS84_b * WGS84_b) / (WGS84_a * WGS84_a);  // WGS84 eccentricity squared
 constexpr double MU_EARTH = 3.986005e14;           // Earth's gravitational parameter [m^3/s^2]
 constexpr double MU_GPS = 3.9860050e14;            // GPS value for Earth's gravitational constant [m^3/s^2]
 constexpr double OMEGA_EARTH = 7.2921151467e-5;    // Earth's rotation rate [rad/s]
 constexpr double PI = 3.1415926535897932;
 constexpr double MAX_EPH_AGE = 7200.0;   // Maximum ephemeris age in seconds (2 hours)
 constexpr double MAX_GLO_EPH_AGE = 1800.0;  // Maximum GLONASS ephemeris age in seconds (0.5 hours)
 constexpr double GPS_SECONDS_PER_WEEK = 604800.0;  // Seconds in a GPS week
 constexpr double CURRENT_GPS_LEAP_SECONDS = 18.0;  // Current leap seconds (GPS - UTC) as of 2023
 constexpr double MAX_VELOCITY = 40.0;     // Maximum vehicle velocity in m/s (144 km/h) - appropriate for GVINS dataset
 constexpr double DOPPLER_SCALE_FACTOR = 1.0;     // Default scale factor for Doppler measurements in ZED-F9P
 
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
 
 // GLONASS ephemeris data for a single satellite
 struct GlonassEphemeris {
     uint32_t sat;              // Satellite ID/slot number
     int freq_slot;             // Frequency slot (-7 to 13)
     double toe_sec;            // Reference epoch time
     double tb_sec;             // Time of ephemeris (seconds of day)
     double tk_sec;             // Message frame time
     double pos_x, pos_y, pos_z;   // Position in PZ-90 (km)
     double vel_x, vel_y, vel_z;   // Velocity in PZ-90 (km/s)
     double acc_x, acc_y, acc_z;   // Acceleration in PZ-90 (km/s²)
     double gamma;              // Relative frequency bias
     double tau_n;              // Clock bias (seconds)
     double dtau;               // Time difference between L1 and L2
     double health;             // Health flag (0=OK)
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
     int freq_num;           // For GLONASS
     double pseudorange;     // meters
     double carrier_phase;   // cycles
     double doppler;         // Hz
     double cn0;             // carrier-to-noise density ratio (dB-Hz)
     double sat_pos_x;       // ECEF X position (m)
     double sat_pos_y;       // ECEF Y position (m)
     double sat_pos_z;       // ECEF Z position (m)
     double sat_vel_x;       // ECEF X velocity (m/s)
     double sat_vel_y;       // ECEF Y velocity (m/s)
     double sat_vel_z;       // ECEF Z velocity (m/s)
     double sat_clock_bias;  // seconds
     double sat_clock_drift; // seconds/second
     double elevation;       // radians
     double azimuth;         // radians
     double weight;          // weight for measurement (based on elevation angle & CN0)
     double iono_delay;      // ionospheric delay (m)
     double trop_delay;      // tropospheric delay (m)
     double tgd;             // total group delay (s)
     double psr_std;         // pseudorange standard deviation
     double dopp_std;        // doppler standard deviation
     double ura;             // user range accuracy for weighting
     double raw_time_tow;    // original TOW from message
     double rx_pos_x;        // Receiver ECEF X for velocity estimation (m)
     double rx_pos_y;        // Receiver ECEF Y for velocity estimation (m) 
     double rx_pos_z;        // Receiver ECEF Z for velocity estimation (m)
     double ephemeris_time;  // Time of ephemeris (toe)
 
     // Default constructor with initialization
     SatelliteInfo() : 
         sat_id(0), system(UNKNOWN), freq_num(0),
         pseudorange(0.0), carrier_phase(0.0), doppler(0.0), cn0(0.0),
         sat_pos_x(0.0), sat_pos_y(0.0), sat_pos_z(0.0),
         sat_vel_x(0.0), sat_vel_y(0.0), sat_vel_z(0.0),
         sat_clock_bias(0.0), sat_clock_drift(0.0),
         elevation(0.0), azimuth(0.0), weight(1.0),
         iono_delay(0.0), trop_delay(0.0), tgd(0.0),
         psr_std(DEFAULT_PSEUDORANGE_NOISE), dopp_std(0.3),
         ura(5.0), raw_time_tow(0.0),
         rx_pos_x(0.0), rx_pos_y(0.0), rx_pos_z(0.0),
         ephemeris_time(0.0)
     {}
 };
 
 // Structure for GNSS solution
 struct GnssSolution {
     double x;               // ECEF X (m)
     double y;               // ECEF Y (m)
     double z;               // ECEF Z (m)
     double vx;              // ECEF X velocity (m/s)
     double vy;              // ECEF Y velocity (m/s)
     double vz;              // ECEF Z velocity (m/s)
     double clock_bias;      // receiver clock bias (m)
     double clock_drift;     // receiver clock drift (m/s)
     double glonass_clock_bias; // GLONASS system time offset (m)
     double glonass_clock_drift; // GLONASS system clock drift (m/s)
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
     int num_vel_satellites; // number of satellites used for velocity
     Eigen::Matrix<double, 4, 4> covariance;  // position-clock covariance
     Eigen::Matrix<double, 4, 4> vel_covariance;  // velocity-clock drift covariance
 
     GnssSolution() :
         x(0.0), y(0.0), z(0.0),
         vx(0.0), vy(0.0), vz(0.0),
         clock_bias(0.0), clock_drift(0.0),
         glonass_clock_bias(0.0), glonass_clock_drift(0.0),
         latitude(0.0), longitude(0.0), altitude(0.0),
         gdop(99.9), pdop(99.9), hdop(99.9), vdop(99.9), tdop(99.9),
         timestamp(0.0), num_satellites(0), num_vel_satellites(0)
     {
         covariance = Eigen::Matrix4d::Identity();
         vel_covariance = Eigen::Matrix4d::Identity();
     }
 };
 
 // GNSS Time structure from gnss_comm
 struct gtime_t {
     time_t time;            // time (s) expressed by standard time_t
     double sec;             // fraction of second under 1 s
 };
 
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
     static void ecefToEnu(const Eigen::Vector3d& ref_lla, 
                          const Eigen::Vector3d& point_ecef, 
                          Eigen::Vector3d& point_enu) {
         // Convert reference point to ECEF
         double ref_x, ref_y, ref_z;
         llaToEcef(ref_lla(0), ref_lla(1), ref_lla(2), ref_x, ref_y, ref_z);
         
         // Calculate offset in ECEF
         double dx = point_ecef(0) - ref_x;
         double dy = point_ecef(1) - ref_y;
         double dz = point_ecef(2) - ref_z;
         
         // Compute rotation matrix from ECEF to ENU
         double sin_lat = sin(ref_lla(0));
         double cos_lat = cos(ref_lla(0));
         double sin_lon = sin(ref_lla(1));
         double cos_lon = cos(ref_lla(1));
         
         // Apply rotation
         point_enu(0) = -sin_lon*dx + cos_lon*dy;  // East
         point_enu(1) = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz;  // North
         point_enu(2) = cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz;  // Up
     }
     
     // Convert ECEF velocity to local ENU velocity
     static void ecefVelToEnu(const Eigen::Vector3d& ref_lla, 
                            const Eigen::Vector3d& vel_ecef, 
                            Eigen::Vector3d& vel_enu) {
         // Compute rotation matrix from ECEF to ENU
         double sin_lat = sin(ref_lla(0));
         double cos_lat = cos(ref_lla(0));
         double sin_lon = sin(ref_lla(1));
         double cos_lon = cos(ref_lla(1));
         
         // Apply rotation to velocity vector
         vel_enu(0) = -sin_lon*vel_ecef(0) + cos_lon*vel_ecef(1);  // East
         vel_enu(1) = -sin_lat*cos_lon*vel_ecef(0) - sin_lat*sin_lon*vel_ecef(1) + cos_lat*vel_ecef(2);  // North
         vel_enu(2) = cos_lat*cos_lon*vel_ecef(0) + cos_lat*sin_lon*vel_ecef(1) + sin_lat*vel_ecef(2);  // Up
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
         T dx = rx - sx;
         T dy = ry - sy;
         T dz = rz - sz;
         T geometric_range = ceres::sqrt(dx*dx + dy*dy + dz*dz);
         
         // Compute Sagnac effect correction
         T sagnac_correction = T(EARTH_ROTATION_RATE) * (rx * sy - ry * sx) / T(SPEED_OF_LIGHT);
         
         // Satellite clock correction
         T sat_clock_correction = T(sat_info_.sat_clock_bias) * T(SPEED_OF_LIGHT);
         
         // Simple tropospheric delay based on elevation
         T elevation;
         {
             // Unit vector from receiver to satellite
             T norm_rx = ceres::sqrt(rx*rx + ry*ry + rz*rz);
             T norm_dx = ceres::sqrt(dx*dx + dy*dy + dz*dz);
             
             // Dot product for cosine of angle
             T cos_angle = (dx*rx + dy*ry + dz*rz) / (norm_rx * norm_dx);
             
             // Safe acos - clamp cos_angle to [-1, 1]
             if (cos_angle > T(1.0)) cos_angle = T(1.0);
             if (cos_angle < T(-1.0)) cos_angle = T(-1.0);
             
             T angle = ceres::acos(cos_angle);
             
             // Elevation is π/2 - angle
             elevation = T(M_PI/2.0) - angle;
         }
         
         // Map function (simple 1/sin(elevation))
         // Use minimum elevation of 5 degrees
         T min_elevation = T(5.0 * M_PI/180.0);
         T safe_elevation = elevation > min_elevation ? elevation : min_elevation;
         T sin_elev = ceres::sin(safe_elevation);
         T trop_delay = T(2.3) / sin_elev; // 2.3m is typical zenith delay
         
         // TGD correction
         T tgd_correction = T(sat_info_.tgd) * T(SPEED_OF_LIGHT);
         
         // Compute expected pseudorange
         T expected_pseudorange = geometric_range + clock_bias - sat_clock_correction + 
                                sagnac_correction + T(sat_info_.iono_delay) + trop_delay + tgd_correction;
         
         // Residual
         residual[0] = (expected_pseudorange - T(sat_info_.pseudorange)) / T(sat_info_.psr_std);
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info) {
         return new ceres::AutoDiffCostFunction<GpsPseudorangeResidual, 1, 4>(
             new GpsPseudorangeResidual(sat_info));
     }
     
 private:
     const SatelliteInfo sat_info_;
 };
 
 // Revised Doppler residual specifically for ZED-F9P in GVINS dataset
 struct GpsDopplerResidual {
     GpsDopplerResidual(const SatelliteInfo& sat_info, const double doppler_scale)
         : sat_info_(sat_info), doppler_scale_(doppler_scale) {}
     
     template <typename T>
     bool operator()(const T* const velocity_state, T* residual) const {
         // Extract receiver velocity and clock drift
         T vx = velocity_state[0];
         T vy = velocity_state[1];
         T vz = velocity_state[2];
         T clock_drift = velocity_state[3];
         
         // Satellite position and velocity
         T sat_pos_x = T(sat_info_.sat_pos_x);
         T sat_pos_y = T(sat_info_.sat_pos_y);
         T sat_pos_z = T(sat_info_.sat_pos_z);
         T sat_vel_x = T(sat_info_.sat_vel_x);
         T sat_vel_y = T(sat_info_.sat_vel_y);
         T sat_vel_z = T(sat_info_.sat_vel_z);
         
         // Receiver position
         T rx = T(sat_info_.rx_pos_x);
         T ry = T(sat_info_.rx_pos_y);
         T rz = T(sat_info_.rx_pos_z);
         
         // Compute vector from receiver to satellite
         T dx = sat_pos_x - rx;
         T dy = sat_pos_y - ry;
         T dz = sat_pos_z - rz;
         T range = ceres::sqrt(dx*dx + dy*dy + dz*dz);
         
         // Line-of-sight unit vector from receiver to satellite
         T e_x = dx / range;
         T e_y = dy / range;
         T e_z = dz / range;
         
         // Wavelength
         T wavelength = T(SPEED_OF_LIGHT / GPS_L1_FREQ);
         
         // For GVINS dataset, try interpreting Doppler values with much smaller scale
         // This handles potential Hz vs kHz issues in the dataset
         T doppler = T(sat_info_.doppler) * T(doppler_scale_);
         
         // Satellite velocity along line of sight (positive = moving away)
         T sat_vel_los = e_x * sat_vel_x + e_y * sat_vel_y + e_z * sat_vel_z;
         
         // Receiver velocity along line of sight (positive = moving toward satellite)
         T rcv_vel_los = e_x * vx + e_y * vy + e_z * vz;
         
         // For ZED-F9P: Doppler is positive when satellite is moving away
         // (opposite of the conventional definition)
         T expected_doppler = (sat_vel_los - rcv_vel_los + clock_drift) / wavelength;
         
         // Residual
         residual[0] = (doppler - expected_doppler) / T(0.5);  // 0.5 Hz std dev
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info, const double doppler_scale) {
         return new ceres::AutoDiffCostFunction<GpsDopplerResidual, 1, 4>(
             new GpsDopplerResidual(sat_info, doppler_scale));
     }
     
 private:
     const SatelliteInfo sat_info_;
     const double doppler_scale_;
 };
 
 // Multi-system pseudorange residual for Ceres solver
 struct MultiSystemPseudorangeResidual {
     MultiSystemPseudorangeResidual(const SatelliteInfo& sat_info)
         : sat_info_(sat_info) {}
     
     template <typename T>
     bool operator()(const T* const receiver_state, T* residual) const {
         // Extract receiver position
         T rx = receiver_state[0];
         T ry = receiver_state[1];
         T rz = receiver_state[2];
         
         // Select the appropriate clock bias based on satellite system
         T clock_bias;
         if (sat_info_.system == GPS) {
             clock_bias = receiver_state[3];  // GPS clock bias
         } else if (sat_info_.system == GLONASS) {
             clock_bias = receiver_state[4];  // GLONASS clock bias
         } else {
             // Default to GPS clock bias for other systems
             clock_bias = receiver_state[3];
         }
         
         // Satellite position
         T sx = T(sat_info_.sat_pos_x);
         T sy = T(sat_info_.sat_pos_y);
         T sz = T(sat_info_.sat_pos_z);
         
         // Compute geometric range
         T dx = rx - sx;
         T dy = ry - sy;
         T dz = rz - sz;
         T geometric_range = ceres::sqrt(dx*dx + dy*dy + dz*dz);
         
         // Compute Sagnac effect correction
         T sagnac_correction = T(EARTH_ROTATION_RATE) * (rx * sy - ry * sx) / T(SPEED_OF_LIGHT);
         
         // Satellite clock correction
         T sat_clock_correction = T(sat_info_.sat_clock_bias) * T(SPEED_OF_LIGHT);
         
         // Simple tropospheric delay based on elevation
         T elevation;
         {
             // Unit vector from receiver to satellite
             T norm_rx = ceres::sqrt(rx*rx + ry*ry + rz*rz);
             T norm_dx = ceres::sqrt(dx*dx + dy*dy + dz*dz);
             
             // Dot product for cosine of angle
             T cos_angle = (dx*rx + dy*ry + dz*rz) / (norm_rx * norm_dx);
             
             // Safe acos - clamp cos_angle to [-1, 1]
             if (cos_angle > T(1.0)) cos_angle = T(1.0);
             if (cos_angle < T(-1.0)) cos_angle = T(-1.0);
             
             T angle = ceres::acos(cos_angle);
             
             // Elevation is π/2 - angle
             elevation = T(M_PI/2.0) - angle;
         }
         
         // Map function (simple 1/sin(elevation))
         // Use minimum elevation of 5 degrees
         T min_elevation = T(5.0 * M_PI/180.0);
         T safe_elevation = elevation > min_elevation ? elevation : min_elevation;
         T sin_elev = ceres::sin(safe_elevation);
         T trop_delay = T(2.3) / sin_elev; // 2.3m is typical zenith delay
         
         // TGD correction
         T tgd_correction = T(sat_info_.tgd) * T(SPEED_OF_LIGHT);
         
         // Compute expected pseudorange
         T expected_pseudorange = geometric_range + clock_bias - sat_clock_correction + 
                                sagnac_correction + T(sat_info_.iono_delay) + trop_delay + tgd_correction;
         
         // Residual
         residual[0] = (expected_pseudorange - T(sat_info_.pseudorange)) / T(sat_info_.psr_std);
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info) {
         return new ceres::AutoDiffCostFunction<MultiSystemPseudorangeResidual, 1, 5>(
             new MultiSystemPseudorangeResidual(sat_info));
     }
     
 private:
     const SatelliteInfo sat_info_;
 };
 
 // Time difference between two gtime_t structs
 static double time_diff(const gtime_t& t1, const gtime_t& t2) {
     return difftime(t1.time, t2.time) + t1.sec - t2.sec;
 }
 
 // Convert date/time array to gtime_t
 static gtime_t epoch2time(const double* ep) {
     const int doy[] = {1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
     gtime_t time = {0};
     int days, sec, year = (int)ep[0], mon = (int)ep[1], day = (int)ep[2];
     
     if (year < 1970 || year > 2099 || mon < 1 || mon > 12) return time;
     
     // leap year if year%4==0 in 1901-2099
     days = (year-1970)*365 + (year-1969)/4 + doy[mon-1] + day-2 + (year%4==0&&mon>=3?1:0);
     sec = (int)floor(ep[5]);
     time.time = (time_t)days*86400 + (int)ep[3]*3600 + (int)ep[4]*60 + sec;
     time.sec = ep[5] - sec;
     return time;
 }
 
 // Convert week and tow in GPS time to gtime_t struct
 static gtime_t gpst2time(uint32_t week, double tow) {
     static const double gpst0[] = {1980, 1, 6, 0, 0, 0};
     gtime_t t = epoch2time(gpst0);
     
     if (tow < -1E9 || tow > 1E9) tow = 0.0;
     t.time += 86400*7*week + (int)tow;
     t.sec = tow - (int)tow;
     return t;
 }
 
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
 
 // GpsSatellitePosition - Compute satellite position from ephemeris
 class GpsEphemerisCalculator {
 public:
     static bool computeSatPosVel(const GpsEphemeris& eph, double transmit_time, 
                                 double& x, double& y, double& z, 
                                 double& vx, double& vy, double& vz, 
                                 double& clock_bias, double& clock_drift,
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
         double tk = transmit_time - eph.toe_sec;
         
         // Handle week crossovers for orbit calculation
         if (tk > 302400.0) tk -= 604800.0;
         else if (tk < -302400.0) tk += 604800.0;
         
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
         double dt = transmit_time - eph.toc_sec;
                 
         // Handle week crossovers for clock calculation
         if (dt > 302400.0) dt -= 604800.0;
         else if (dt < -302400.0) dt += 604800.0;
                 
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
         
         // Clock drift
         clock_drift = eph.af1 + 2.0 * eph.af2 * dt;
         
         // Add relativistic correction to clock drift
         double relativistic_drift = -2.0 * sqrt(mu) * eph.e * eph.sqrta * cos_E * n / 
                                     ((1.0 - eph.e * cos_E) * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
         clock_drift += relativistic_drift;
         
         ROS_DEBUG("  Clock drift: %.12f s/s (%.3f m/s)", clock_drift, clock_drift * SPEED_OF_LIGHT);
         
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
         
         // Improved velocity calculation using analytical derivatives
         
         // Eccentric anomaly rate of change
         double E_dot = n / (1.0 - eph.e * cos_E);
         
         // True anomaly rate of change
         double nu_dot = E_dot * sqrt(1.0 - eph.e * eph.e) / (1.0 - eph.e * cos_E);
         
         // Argument of latitude rate of change
         double phi_dot = nu_dot;
         
         // Correction terms derivatives
         double du_dot = 2.0 * phi_dot * (eph.cus * cos_2phi - eph.cuc * sin_2phi);
         double dr_dot = 2.0 * phi_dot * (eph.crs * cos_2phi - eph.crc * sin_2phi);
         double di_dot = eph.i_dot + 2.0 * phi_dot * (eph.cis * cos_2phi - eph.cic * sin_2phi);
         
         // Rate of change for corrected terms
         double u_dot = phi_dot + du_dot;
         double r_dot = a * eph.e * sin_E * E_dot + dr_dot;
         
         // Rate of change for orbital plane coordinates
         double x_op_dot = r_dot * cos(u) - r * sin(u) * u_dot;
         double y_op_dot = r_dot * sin(u) + r * cos(u) * u_dot;
         
         // Rate of change for the corrected longitude of ascending node
         double Omega_dot = eph.omg_dot - omega_e;
         
         // Velocity in ECEF - improved calculation
         vx = -omega_e * y + 
              x_op_dot * cos_Omega - y_op_dot * cos_i * sin_Omega -
              y_op * sin_i * sin_Omega * di_dot -
              (x_op * sin_Omega + y_op * cos_i * cos_Omega) * Omega_dot;
         
         vy = omega_e * x + 
              x_op_dot * sin_Omega + y_op_dot * cos_i * cos_Omega -
              y_op * sin_i * cos_Omega * di_dot +
              (x_op * cos_Omega - y_op * cos_i * sin_Omega) * Omega_dot;
         
         vz = y_op_dot * sin_i + y_op * cos_i * di_dot;
         
         // Debug output for satellite velocity 
         double vel_mag = sqrt(vx*vx + vy*vy + vz*vz);
         ROS_DEBUG("GPS PRN %d velocity calculation:", eph.sat);
         ROS_DEBUG("  ECEF velocity: [%.2f, %.2f, %.2f] m/s", vx, vy, vz);
         ROS_DEBUG("  Velocity magnitude: %.2f m/s", vel_mag);
         
         return true;
     }
 };
 
 // GLONASS satellite position calculation
 class GlonassEphemerisCalculator {
 public:
     static bool computeSatPosVel(const GlonassEphemeris& eph, double transmit_time, 
                                 double& x, double& y, double& z, 
                                 double& vx, double& vy, double& vz, 
                                 double& clock_bias, double& clock_drift,
                                 bool force_use_ephemeris = false) {
         if (!eph.valid) {
             ROS_DEBUG("GLONASS ephemeris not valid for satellite %d", eph.sat);
             return false;
         }
         
         // Add some basic validation
         if (eph.pos_x == 0 && eph.pos_y == 0 && eph.pos_z == 0) {
             ROS_WARN("Invalid position (all zeros) for GLONASS satellite %d", eph.sat);
             return false;
         }
         
         // Convert transmit time to seconds of day in GLONASS time scale
         double seconds_of_day = fmod(transmit_time, 86400.0);
         
         // Time difference from ephemeris epoch (tb)
         double dt = seconds_of_day - eph.tb_sec;
         
         // Adjust for day boundary
         if (dt > 43200.0) dt -= 86400.0;
         else if (dt < -43200.0) dt += 86400.0;
         
         // Check if ephemeris is valid for this time
         if (!force_use_ephemeris && std::abs(dt) > MAX_GLO_EPH_AGE) {
             ROS_DEBUG("GLONASS ephemeris age (%.1f s) exceeds limit for satellite %d. tb=%.1f, tx_time=%.1f", 
                     std::abs(dt), eph.sat, eph.tb_sec, seconds_of_day);
             return false;
         }
         
         ROS_DEBUG("Using ephemeris for GLONASS slot %d with age %.1f seconds", eph.sat, dt);
         
         // State vector at reference time (convert from km to m)
         double pos[3] = {eph.pos_x * 1000.0, eph.pos_y * 1000.0, eph.pos_z * 1000.0};
         double vel[3] = {eph.vel_x * 1000.0, eph.vel_y * 1000.0, eph.vel_z * 1000.0};
         double acc[3] = {eph.acc_x * 1000.0, eph.acc_y * 1000.0, eph.acc_z * 1000.0};
         
         // PZ-90 constants
         const double mu = 398600.44e9;        // Earth's gravitational constant (m³/s²)
         const double J2 = 1.0826257e-3;       // Second zonal harmonic of the Earth
         const double omega = 7.292115e-5;     // Earth's rotation rate (rad/s)
         const double ae = 6378136.0;          // Earth's radius (m)
         
         // Fourth-order Runge-Kutta integration
         const int steps = std::min(abs(static_cast<int>(dt)) + 1, 10);
         const double h = dt / steps;
         
         double curr_pos[3] = {pos[0], pos[1], pos[2]};
         double curr_vel[3] = {vel[0], vel[1], vel[2]};
         
         for (int i = 0; i < steps; i++) {
             // Compute acceleration due to Earth's gravity and J2 perturbation
             double r = sqrt(curr_pos[0] * curr_pos[0] + curr_pos[1] * curr_pos[1] + curr_pos[2] * curr_pos[2]);
             double r2 = r * r;
             double r3 = r * r2;
             double r5 = r3 * r2;
             
             // Acceleration due to Earth's gravity
             double ax = -mu * curr_pos[0] / r3;
             double ay = -mu * curr_pos[1] / r3;
             double az = -mu * curr_pos[2] / r3;
             
             // J2 perturbation
             double z2 = curr_pos[2] * curr_pos[2];
             double factor = 1.5 * J2 * mu * ae * ae / r5;
             ax += factor * (5 * z2 / r2 - 1) * curr_pos[0];
             ay += factor * (5 * z2 / r2 - 1) * curr_pos[1];
             az += factor * (5 * z2 / r2 - 3) * curr_pos[2];
             
             // Coriolis acceleration
             ax += 2 * omega * curr_vel[1];
             ay += -2 * omega * curr_vel[0];
             
             // Additional acceleration from ephemeris
             ax += acc[0];
             ay += acc[1];
             az += acc[2];
             
             // RK4 for position
             double k1[6], k2[6], k3[6], k4[6];
             
             // k1 = f(t, y)
             k1[0] = curr_vel[0];
             k1[1] = curr_vel[1];
             k1[2] = curr_vel[2];
             k1[3] = ax;
             k1[4] = ay;
             k1[5] = az;
             
             // k2 = f(t + h/2, y + h*k1/2)
             double pos_tmp[3], vel_tmp[3];
             for (int j = 0; j < 3; j++) {
                 pos_tmp[j] = curr_pos[j] + h/2 * k1[j];
                 vel_tmp[j] = curr_vel[j] + h/2 * k1[j+3];
             }
             
             // Compute accelerations at new state
             r = sqrt(pos_tmp[0] * pos_tmp[0] + pos_tmp[1] * pos_tmp[1] + pos_tmp[2] * pos_tmp[2]);
             r2 = r * r;
             r3 = r * r2;
             r5 = r3 * r2;
             
             ax = -mu * pos_tmp[0] / r3;
             ay = -mu * pos_tmp[1] / r3;
             az = -mu * pos_tmp[2] / r3;
             
             z2 = pos_tmp[2] * pos_tmp[2];
             factor = 1.5 * J2 * mu * ae * ae / r5;
             ax += factor * (5 * z2 / r2 - 1) * pos_tmp[0];
             ay += factor * (5 * z2 / r2 - 1) * pos_tmp[1];
             az += factor * (5 * z2 / r2 - 3) * pos_tmp[2];
             
             ax += 2 * omega * vel_tmp[1];
             ay += -2 * omega * vel_tmp[0];
             
             ax += acc[0];
             ay += acc[1];
             az += acc[2];
             
             k2[0] = vel_tmp[0];
             k2[1] = vel_tmp[1];
             k2[2] = vel_tmp[2];
             k2[3] = ax;
             k2[4] = ay;
             k2[5] = az;
             
             // k3 = f(t + h/2, y + h*k2/2)
             for (int j = 0; j < 3; j++) {
                 pos_tmp[j] = curr_pos[j] + h/2 * k2[j];
                 vel_tmp[j] = curr_vel[j] + h/2 * k2[j+3];
             }
             
             // Compute accelerations at new state
             r = sqrt(pos_tmp[0] * pos_tmp[0] + pos_tmp[1] * pos_tmp[1] + pos_tmp[2] * pos_tmp[2]);
             r2 = r * r;
             r3 = r * r2;
             r5 = r3 * r2;
             
             ax = -mu * pos_tmp[0] / r3;
             ay = -mu * pos_tmp[1] / r3;
             az = -mu * pos_tmp[2] / r3;
             
             z2 = pos_tmp[2] * pos_tmp[2];
             factor = 1.5 * J2 * mu * ae * ae / r5;
             ax += factor * (5 * z2 / r2 - 1) * pos_tmp[0];
             ay += factor * (5 * z2 / r2 - 1) * pos_tmp[1];
             az += factor * (5 * z2 / r2 - 3) * pos_tmp[2];
             
             ax += 2 * omega * vel_tmp[1];
             ay += -2 * omega * vel_tmp[0];
             
             ax += acc[0];
             ay += acc[1];
             az += acc[2];
             
             k3[0] = vel_tmp[0];
             k3[1] = vel_tmp[1];
             k3[2] = vel_tmp[2];
             k3[3] = ax;
             k3[4] = ay;
             k3[5] = az;
             
             // k4 = f(t + h, y + h*k3)
             for (int j = 0; j < 3; j++) {
                 pos_tmp[j] = curr_pos[j] + h * k3[j];
                 vel_tmp[j] = curr_vel[j] + h * k3[j+3];
             }
             
             // Compute accelerations at new state
             r = sqrt(pos_tmp[0] * pos_tmp[0] + pos_tmp[1] * pos_tmp[1] + pos_tmp[2] * pos_tmp[2]);
             r2 = r * r;
             r3 = r * r2;
             r5 = r3 * r2;
             
             ax = -mu * pos_tmp[0] / r3;
             ay = -mu * pos_tmp[1] / r3;
             az = -mu * pos_tmp[2] / r3;
             
             z2 = pos_tmp[2] * pos_tmp[2];
             factor = 1.5 * J2 * mu * ae * ae / r5;
             ax += factor * (5 * z2 / r2 - 1) * pos_tmp[0];
             ay += factor * (5 * z2 / r2 - 1) * pos_tmp[1];
             az += factor * (5 * z2 / r2 - 3) * pos_tmp[2];
             
             ax += 2 * omega * vel_tmp[1];
             ay += -2 * omega * vel_tmp[0];
             
             ax += acc[0];
             ay += acc[1];
             az += acc[2];
             
             k4[0] = vel_tmp[0];
             k4[1] = vel_tmp[1];
             k4[2] = vel_tmp[2];
             k4[3] = ax;
             k4[4] = ay;
             k4[5] = az;
             
             // Compute new state using weighted average
             for (int j = 0; j < 3; j++) {
                 curr_pos[j] += h/6 * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
                 curr_vel[j] += h/6 * (k1[j+3] + 2*k2[j+3] + 2*k3[j+3] + k4[j+3]);
             }
         }
         
         // Set position and velocity
         x = curr_pos[0];
         y = curr_pos[1];
         z = curr_pos[2];
         vx = curr_vel[0];
         vy = curr_vel[1];
         vz = curr_vel[2];
         
         // Compute clock bias and drift
         // GLONASS time is UTC + 3 hours, and uses tau_n for clock bias parameter
         clock_bias = -eph.tau_n + eph.gamma * dt;
         clock_drift = eph.gamma;
 
         // Log detailed clock calculations for debugging        
         ROS_DEBUG("GLONASS slot %d clock calculation:", eph.sat);
         ROS_DEBUG("  TB: %.3f, transmit time: %.3f, dt: %.3f seconds", 
                  eph.tb_sec, seconds_of_day, dt);
         ROS_DEBUG("  Clock params: tau_n=%.12e, gamma=%.12e", 
                  eph.tau_n, eph.gamma);
         ROS_DEBUG("  Final clock bias: %.12f s (%.3f m)", 
                  clock_bias, clock_bias * SPEED_OF_LIGHT);
         
         // Log velocity calculation
         double vel_mag = sqrt(vx*vx + vy*vy + vz*vz);
         ROS_DEBUG("GLONASS slot %d velocity calculation:", eph.sat);
         ROS_DEBUG("  ECEF velocity: [%.2f, %.2f, %.2f] m/s", vx, vy, vz);
         ROS_DEBUG("  Velocity magnitude: %.2f m/s", vel_mag);
         
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
         
         // Convert to semi-circles
         double lat_sc = lat / PI;
         double lon_sc = lon / PI;
         
         // Use absolute value of elevation to handle negative elevations
         double elevation_abs = fabs(elevation);
         
         // Elevation in semi-circles (positive)
         double el_sc = std::max(0.05, elevation_abs / PI);
         
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
 
 // Calculate tropospheric delay
 double calculateTroposphericDelay(double time, const Eigen::Vector3d& lla, const double* azel) {
     const double temp0 = 15.0; // temperature at sea level [°C]
     const double humi = 0.7;   // relative humidity at sea level
     double hgt = lla(2);
     
     // Compute tropospheric delay
     double pres = 1013.25 * pow(1.0 - 2.2557e-5 * hgt, 5.2568);
     double temp = temp0 - 6.5e-3 * hgt + 273.15;
     double e = 6.108 * humi * exp((17.15 * temp - 4684.0) / (temp - 38.45));
     
     // Saastamoinen model
     double zhd = 0.0022768 * pres / (1.0 - 0.00266 * cos(2.0 * lla(0)) - 0.00028 * hgt / 1000.0);
     double zwd = 0.002277 * (1255.0 / temp + 0.05) * e;
     
     // Use absolute elevation value for the mapping function
     double elevation_abs = fabs(azel[1]);
     
     // Mapping function based on sin(elev)
     double map_h = 1.0 / std::max(sin(elevation_abs), 0.07);
     double map_w = 1.0 / std::max(sin(elevation_abs), 0.07);
     
     // Combined tropospheric delay in meters
     double trop_delay = zhd * map_h + zwd * map_w;
     
     return trop_delay;
 }
 
 // Time conversion helper
 double gnss_time_to_sec(const gnss_comm::GnssTimeMsg& time_msg) {
     return time_msg.tow;  // Use time of week in seconds
 }
 
 class GnssWlsNode {
 public:
     GnssWlsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : initialized_(false) {
         // Load parameters with more permissive defaults and correct Hong Kong position
         pnh.param<std::string>("frame_id", frame_id_, "gnss");
         pnh.param<double>("pseudorange_noise", pseudorange_noise_, DEFAULT_PSEUDORANGE_NOISE);
         pnh.param<int>("min_satellites", min_satellites_, 4);  // Minimum 4 satellites needed
         pnh.param<bool>("use_doppler", use_doppler_, true);    // Enable velocity estimation by default
         pnh.param<double>("initial_latitude", initial_latitude_, 22.3193);  // Default to Hong Kong
         pnh.param<double>("initial_longitude", initial_longitude_, 114.1694);  // Default to Hong Kong
         pnh.param<double>("initial_altitude", initial_altitude_, 100.0);  // Default to Hong Kong elevation
         pnh.param<bool>("apply_iono_correction", apply_iono_correction_, true);
         pnh.param<double>("elevation_mask_deg", elevation_mask_deg_, 0.0);  // Set to 0 to accept all satellites
         pnh.param<double>("min_cn0", min_cn0_, 5.0);  // Reduced from 10.0 to 5.0
         pnh.param<std::string>("output_csv_path", output_csv_path_, "spp_results.csv");
         
         // Raw measurement logging
         pnh.param<bool>("log_raw_data", log_raw_data_, false);
         pnh.param<std::string>("raw_data_path", raw_data_path_, "");
         
         // Configurable elevation cutoff angle (new)
         pnh.param<double>("cut_off_degree", cut_off_degree_, 0.0);  // Reduced to 0 to accept all satellites initially
         
         // Maximum expected velocity for GVINS dataset (ground vehicle)
         pnh.param<double>("max_velocity", max_velocity_, MAX_VELOCITY);
         
         // Doppler scale factor (for potential unit conversions in the receiver)
         pnh.param<double>("doppler_scale_factor", doppler_scale_factor_, DOPPLER_SCALE_FACTOR);
         
         // Debug mode options
         pnh.param<bool>("disable_cn0_filter", disable_cn0_filter_, false);
         pnh.param<bool>("disable_elevation_filter", disable_elevation_filter_, false);
         pnh.param<bool>("use_initialization_mode", use_initialization_mode_, true);
         pnh.param<bool>("force_use_ephemeris", force_use_ephemeris_, true);  // Force using any available ephemeris
         
         // New option for GPS-only mode
         pnh.param<bool>("gps_only_mode", gps_only_mode_, true);  // Default to GPS-only for stability
         
         // Fixed GPS week for time conversion
         pnh.param<int>("current_gps_week", current_gps_week_, 2134);  // Default for GVINS dataset
         
         // Get current GPS leap seconds
         pnh.param<double>("current_leap_seconds", current_leap_seconds_, CURRENT_GPS_LEAP_SECONDS);
         
         // Options for adaptive Doppler scaling
         pnh.param<bool>("adaptive_doppler_scaling", adaptive_doppler_scaling_, true);
         pnh.param<bool>("use_gvins_doppler_model", use_gvins_doppler_model_, true);
         
         if (disable_cn0_filter_) {
             ROS_WARN("CN0 filtering is disabled for debugging");
         }
         if (disable_elevation_filter_) {
             ROS_WARN("Elevation filtering is disabled for debugging");
         }
         if (force_use_ephemeris_) {
             ROS_WARN("Using ephemeris regardless of age - this may affect positioning accuracy");
         }
         if (gps_only_mode_) {
             ROS_INFO("GPS-only mode enabled - GLONASS satellites will be ignored");
         } else {
             ROS_INFO("Multi-system mode enabled - both GPS and GLONASS will be used");
         }
         
         // Set elevation mask for position calculation
         ROS_INFO("Using configurable elevation cutoff of %.1f degrees", cut_off_degree_);
         elevation_mask_ = 0.0;  // No filtering during satellite position calculation
         
         // Initialize with a default position if user didn't provide one
         if (initial_latitude_ == 0.0 && initial_longitude_ == 0.0 && initial_altitude_ == 0.0) {
             // Default to a reasonable location if none provided (e.g., middle of continental US)
             initial_latitude_ = 22.8333333;
             initial_longitude_ = 114.585522;
             initial_altitude_ = 50.0;
             
             ROS_WARN("No initial position provided, using default: %.6f, %.6f, %.2f", 
                     initial_latitude_, initial_longitude_, initial_altitude_);
         }
         
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
         current_solution_.vx = 0.0;
         current_solution_.vy = 0.0;
         current_solution_.vz = 0.0;
         current_solution_.clock_bias = 0.0;  // Initialize to 0 instead of unpredictable value
         current_solution_.clock_drift = 0.0;
         current_solution_.glonass_clock_bias = 0.0;  // Initialize GLONASS clock bias to 0
         current_solution_.glonass_clock_drift = 0.0;
         current_solution_.timestamp = 0.0;
         
         // Initialize ionospheric parameters with GVINS defaults
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
         
         ROS_INFO("Using default ionospheric parameters from GVINS");
         
         // Publishers
         navsatfix_pub_ = nh.advertise<sensor_msgs::NavSatFix>("gnss_fix", 10);
         odom_pub_ = nh.advertise<nav_msgs::Odometry>("gnss_odom", 10);
         pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("gnss_pose", 10);
         
         // Subscribers for GNSS data
         ephem_sub_ = nh.subscribe("/ublox_driver/ephem", 10, &GnssWlsNode::ephemCallback, this);
         glo_ephem_sub_ = nh.subscribe("/ublox_driver/glo_ephem", 10, &GnssWlsNode::gloEphemCallback, this);
         iono_params_sub_ = nh.subscribe("/ublox_driver/iono_params", 10, &GnssWlsNode::ionoParamsCallback, this);
         raw_meas_sub_ = nh.subscribe("/ublox_driver/range_meas", 10, &GnssWlsNode::rawMeasCallback, this);
         
         // Open CSV file for saving results
         if (!output_csv_path_.empty()) {
             csv_file_.open(output_csv_path_, std::ios::out);
             if (csv_file_.is_open()) {
                 // Write CSV header with column names
                 csv_file_ << "Timestamp,GPSWeek,TOW,Latitude,Longitude,Altitude,ECEF_X,ECEF_Y,ECEF_Z,ENU_E,ENU_N,ENU_U,"
                           << "VelocityX,VelocityY,VelocityZ,VelocityE,VelocityN,VelocityU,Speed,"
                           << "ReceiverClockBias,ReceiverClockDrift,NumSatellites,NumVelSatellites,PDOP,HDOP,VDOP,TDOP" 
                           << std::endl;
                 ROS_INFO("Opened CSV file for saving results: %s", output_csv_path_.c_str());
             } else {
                 ROS_WARN("Could not open CSV file at: %s", output_csv_path_.c_str());
             }
         }
         
         // Open raw data log file if enabled
         if (log_raw_data_ && !raw_data_path_.empty()) {
             raw_data_file_.open(raw_data_path_, std::ios::out);
             if (raw_data_file_.is_open()) {
                 // Create CSV header
                 raw_data_file_ << "Timestamp,SatID,System,Pseudorange,CarrierPhase,Doppler,CN0,"
                                << "SatPosX,SatPosY,SatPosZ,SatVelX,SatVelY,SatVelZ,"
                                << "SatClockBias,SatClockDrift,Elevation,Azimuth,IonoDelay,TropDelay" 
                                << std::endl;
             } else {
                 ROS_WARN("Could not open raw data log file at: %s", raw_data_path_.c_str());
             }
         }
         
         ROS_INFO("GNSS SPP node initialized:");
         ROS_INFO(" - Initial position: Lat=%.6f°, Lon=%.6f°, Alt=%.1fm", 
                 initial_latitude_, initial_longitude_, initial_altitude_);
         ROS_INFO(" - ECEF position: [%.1f, %.1f, %.1f]", init_x, init_y, init_z);
         ROS_INFO(" - Minimum satellites: %d", min_satellites_);
         ROS_INFO(" - Elevation cutoff: %.1f degrees", cut_off_degree_);
         ROS_INFO(" - Minimum CN0: %.1f dB-Hz", min_cn0_);
         ROS_INFO(" - Use Doppler: %s", use_doppler_ ? "true" : "false");
         ROS_INFO(" - Apply ionospheric correction: %s", apply_iono_correction_ ? "true" : "false");
         ROS_INFO(" - Force use ephemeris: %s", force_use_ephemeris_ ? "true" : "false");
         ROS_INFO(" - Current GPS week: %d", current_gps_week_);
         ROS_INFO(" - Current GPS-UTC leap seconds: %.1f", current_leap_seconds_);
         ROS_INFO(" - Max velocity: %.1f m/s (%.1f km/h)", max_velocity_, max_velocity_ * 3.6);
         ROS_INFO(" - Doppler scale factor: %.3f", doppler_scale_factor_);
         ROS_INFO(" - Adaptive Doppler scaling: %s", adaptive_doppler_scaling_ ? "true" : "false");
         ROS_INFO(" - Use GVINS Doppler model: %s", use_gvins_doppler_model_ ? "true" : "false");
     }
     
     ~GnssWlsNode() {
         if (csv_file_.is_open()) {
             csv_file_.close();
         }
         
         if (raw_data_file_.is_open()) {
             raw_data_file_.close();
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
     
     // Process GLONASS ephemeris data
     void gloEphemCallback(const gnss_comm::GnssGloEphemMsg::ConstPtr& msg) {
         static uint32_t glo_ephem_count = 0;
         
         std::lock_guard<std::mutex> lock(glo_ephem_mutex_);
         
         // Check if we already have an ephemeris for this satellite
         bool update_existing = (glo_ephemeris_.find(msg->sat) != glo_ephemeris_.end());
         
         GlonassEphemeris eph;
         eph.sat = msg->sat;  // Correct field name is 'sat'
         eph.freq_slot = msg->freqo;  // Using freqo as frequency slot number
         
         // GnssTimeMsg is a complex type with time fields
         // We need to extract the time from the GnssTimeMsg correctly
         eph.toe_sec = msg->toe.tow;  // Time of week in seconds
         eph.tb_sec = msg->toe.tow;   // Using toe as tb
         eph.tk_sec = msg->toe.tow;   // Using toe as time of frame
         
         eph.pos_x = msg->pos_x;
         eph.pos_y = msg->pos_y;
         eph.pos_z = msg->pos_z;
         eph.vel_x = msg->vel_x;
         eph.vel_y = msg->vel_y;
         eph.vel_z = msg->vel_z;
         eph.acc_x = msg->acc_x;
         eph.acc_y = msg->acc_y;
         eph.acc_z = msg->acc_z;
         eph.gamma = msg->gamma;
         eph.tau_n = msg->tau_n;
         eph.dtau = msg->delta_tau_n;  // Using delta_tau_n from the message
         eph.health = msg->health;
         eph.valid = true;
         eph.last_update = ros::Time::now();
         eph.ura = 4.0;  // GLONASS typically has lower accuracy
         
         // Store/update ephemeris
         glo_ephemeris_[msg->sat] = eph;
         
         glo_ephem_count++;
         if (update_existing) {
             ROS_INFO("Updated GLONASS ephemeris for slot %d, toe=%.0f, freq=%d", 
                     msg->sat, msg->toe.tow, msg->freqo);
         } else {
             ROS_INFO("Received new GLONASS ephemeris for slot %d, toe=%.0f, freq=%d", 
                     msg->sat, msg->toe.tow, msg->freqo);
         }
         
         if (glo_ephem_count % 10 == 0 || glo_ephem_count < 10) {
             ROS_INFO("Received total %u GLONASS ephemeris messages, have data for %zu satellites", 
                     glo_ephem_count, glo_ephemeris_.size());
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
         
         // RTKLIB-style time handling:
         // 1. Use time from the measurements themselves if valid
         double gps_tow = 0.0;
         uint32_t gps_week = current_gps_week_;
         
         if (msg->meas.size() > 0 && msg->meas[0].time.week > 0) {
             // Use the time tag from the measurements
             gps_week = msg->meas[0].time.week;
             gps_tow = msg->meas[0].time.tow;
             ROS_INFO("Using measurement time: Week %d, TOW %.3f", gps_week, gps_tow);
         }
         // 2. If not available, use time from ephemeris
         else if (!gps_ephemeris_.empty()) {
             // Use most recent ephemeris time
             auto it = gps_ephemeris_.begin();
             gps_tow = it->second.toe_sec;
             ROS_INFO("Using ephemeris time: toe=%.3f", gps_tow);
         }
         // 3. Last resort: use system time
         else {
             gps_tow = fmod(ros::Time::now().toSec() + current_leap_seconds_, GPS_SECONDS_PER_WEEK);
             ROS_WARN("Using system time as fallback: GPS TOW = %.3f", gps_tow);
         }
         
         // Store current time for use in other functions
         current_time_ = gps_tow;
         
         ROS_INFO("Processing %zu GNSS observations at GPS TOW %.3f", msg->meas.size(), gps_tow);
         
         // Print ephemeris status periodically
         static ros::Time last_ephem_status_time = ros::Time::now();
         if ((ros::Time::now() - last_ephem_status_time).toSec() > 10.0) {
             ROS_INFO("GNSS Ephemeris Status: GPS=%zu, GLONASS=%zu", 
                     gps_ephemeris_.size(), glo_ephemeris_.size());
             
             last_ephem_status_time = ros::Time::now();
         }
         
         // Diagnostic counters
         int count_empty_psr = 0;
         int count_invalid_psr = 0;
         int count_low_cn0 = 0;
         int count_no_ephemeris = 0;
         int count_ephemeris_error = 0;
         int count_below_elevation = 0;
         int count_negative_elevation = 0;
         int count_gps_skipped = 0;
         int count_glonass_skipped = 0;
         
         // Loop through all measurements and calculate satellite positions
         // Without filtering based on elevation yet
         for (const auto& obs : msg->meas) {
             // Create a satellite info structure
             SatelliteInfo sat_info;
             sat_info.sat_id = obs.sat;
             sat_info.raw_time_tow = obs.time.tow;  // Store raw TOW for logging
             
             ROS_DEBUG("Processing satellite %d", obs.sat);
             
             // Set system type based on the PRN range
             if (obs.sat >= 1 && obs.sat <= 32) {
                 sat_info.system = GPS;
                 ROS_DEBUG("  System: GPS");
             } else if (obs.sat >= 38 && obs.sat <= 61) {
                 sat_info.system = GLONASS;
                 ROS_DEBUG("  System: GLONASS");
                 
                 // Skip GLONASS satellites in GPS-only mode
                 if (gps_only_mode_) {
                     count_glonass_skipped++;
                     continue;
                 }
             } else if (obs.sat >= 71 && obs.sat <= 140) {
                 sat_info.system = GALILEO;
                 ROS_DEBUG("  System: GALILEO");
             } else if (obs.sat >= 141 && obs.sat <= 210) {
                 sat_info.system = BEIDOU;
                 ROS_DEBUG("  System: BEIDOU");
             } else {
                 ROS_DEBUG("  Unknown satellite system: %d", obs.sat);
                 continue;
             }
             
             // Skip if no valid measurements
             if (obs.psr.empty()) {
                 ROS_DEBUG("  No pseudorange measurements");
                 count_empty_psr++;
                 continue;
             }
             
             // Print available data for debugging
             ROS_DEBUG("  Pseudorange count: %zu", obs.psr.size());
             ROS_DEBUG("  Carrier phase count: %zu", obs.cp.size());
             ROS_DEBUG("  Doppler count: %zu", obs.dopp.size());
             ROS_DEBUG("  CN0 count: %zu", obs.CN0.size());
             
             if (!obs.psr.empty()) {
                 ROS_DEBUG("  First pseudorange: %.2f", obs.psr[0]);
             }
             if (!obs.CN0.empty()) {
                 ROS_DEBUG("  First CN0: %.2f", obs.CN0[0]);
             }
             
             // Check pseudorange validity
             if (obs.psr[0] <= 0 || std::isnan(obs.psr[0])) {
                 ROS_DEBUG("  Invalid pseudorange");
                 count_invalid_psr++;
                 continue;
             }
             
             // Store measurement data
             sat_info.pseudorange = obs.psr[0];
             
             if (!obs.cp.empty() && !std::isnan(obs.cp[0])) {
                 sat_info.carrier_phase = obs.cp[0];
             } else {
                 sat_info.carrier_phase = 0.0;
             }
             
             if (!obs.dopp.empty() && !std::isnan(obs.dopp[0])) {
                 // Store raw Doppler value
                 sat_info.doppler = obs.dopp[0];
             } else {
                 sat_info.doppler = 0.0;
             }
             
             if (!obs.CN0.empty() && !std::isnan(obs.CN0[0])) {
                 sat_info.cn0 = obs.CN0[0];
                 ROS_DEBUG("  CN0: %.2f dB-Hz", sat_info.cn0);
             } else {
                 sat_info.cn0 = 0.0;
                 ROS_DEBUG("  CN0 not available");
             }
             
             // Skip measurements with low signal strength (unless disabled)
             if (!disable_cn0_filter_ && sat_info.cn0 < min_cn0_) {
                 ROS_DEBUG("  CN0 too low: %.2f < %.2f", sat_info.cn0, min_cn0_);
                 count_low_cn0++;
                 continue;
             }
             
             // Filter out satellites with unrealistic Doppler values
             if (std::abs(sat_info.doppler) > 15000.0) {
                 ROS_DEBUG("  Unrealistic Doppler: %.1f Hz", sat_info.doppler);
                 continue;
             }
             
             // Process based on satellite system
             bool success = false;
             
             if (sat_info.system == GPS) {
                 std::lock_guard<std::mutex> lock(gps_ephem_mutex_);
                 
                 // Check if we have ephemeris for this satellite
                 if (gps_ephemeris_.find(sat_info.sat_id) == gps_ephemeris_.end()) {
                     ROS_DEBUG("  No GPS ephemeris available for PRN %d", sat_info.sat_id);
                     count_no_ephemeris++;
                     continue;
                 }
                 
                 ROS_DEBUG("  Found GPS ephemeris for PRN %d", sat_info.sat_id);
                 
                 // Calculate exact transmission time by correcting pseudorange for light travel time
                 double transmission_time = gps_tow - sat_info.pseudorange / SPEED_OF_LIGHT;
                 
                 // Compute satellite position and clock
                 double clock_bias, clock_drift;
                 success = GpsEphemerisCalculator::computeSatPosVel(
                     gps_ephemeris_[sat_info.sat_id], 
                     transmission_time, 
                     sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                     sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z,
                     clock_bias, clock_drift,
                     force_use_ephemeris_);
                 
                 if (success) {
                     sat_info.sat_clock_bias = clock_bias;
                     sat_info.sat_clock_drift = clock_drift;
                     sat_info.tgd = gps_ephemeris_[sat_info.sat_id].tgd0;
                     sat_info.ura = gps_ephemeris_[sat_info.sat_id].ura;
                     sat_info.ephemeris_time = gps_ephemeris_[sat_info.sat_id].toe_sec;
                     
                     ROS_DEBUG("  Computed GPS satellite position: [%.2f, %.2f, %.2f]", 
                             sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z);
                     ROS_DEBUG("  GPS satellite velocity: [%.2f, %.2f, %.2f]",
                             sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z);
                     ROS_DEBUG("  GPS satellite clock bias: %.9f s (%.3f m)", 
                             sat_info.sat_clock_bias, sat_info.sat_clock_bias * SPEED_OF_LIGHT);
                     ROS_DEBUG("  GPS satellite clock drift: %.9f s/s (%.3f m/s)",
                             sat_info.sat_clock_drift, sat_info.sat_clock_drift * SPEED_OF_LIGHT);
                 } else {
                     ROS_DEBUG("  Failed to compute GPS satellite position for PRN %d", sat_info.sat_id);
                     count_ephemeris_error++;
                 }
                 
             } else if (sat_info.system == GLONASS) {
                 std::lock_guard<std::mutex> lock(glo_ephem_mutex_);
                 
                 // Check if we have ephemeris for this satellite
                 if (glo_ephemeris_.find(sat_info.sat_id) == glo_ephemeris_.end()) {
                     ROS_DEBUG("  No GLONASS ephemeris available for slot %d", sat_info.sat_id);
                     count_no_ephemeris++;
                     continue;
                 }
                 
                 ROS_DEBUG("  Found GLONASS ephemeris for slot %d", sat_info.sat_id);
                 
                 // GLONASS time (UTC + 3 hours, but without leap seconds)
                 // GPS TOW -> GLONASS time
                 double glonass_time = fmod(gps_tow - current_leap_seconds_, 86400.0);  
                 
                 // Time of transmission
                 double transmission_time = glonass_time - sat_info.pseudorange / SPEED_OF_LIGHT;
                 
                 // Compute satellite position and clock
                 double clock_bias, clock_drift;
                 success = GlonassEphemerisCalculator::computeSatPosVel(
                     glo_ephemeris_[sat_info.sat_id], 
                     transmission_time, 
                     sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                     sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z,
                     clock_bias, clock_drift,
                     force_use_ephemeris_);
                 
                 if (success) {
                     sat_info.sat_clock_bias = clock_bias;
                     sat_info.sat_clock_drift = clock_drift;
                     sat_info.freq_num = glo_ephemeris_[sat_info.sat_id].freq_slot;
                     sat_info.tgd = 0.0;  // GLONASS doesn't use TGD
                     sat_info.ura = glo_ephemeris_[sat_info.sat_id].ura;
                     sat_info.ephemeris_time = glo_ephemeris_[sat_info.sat_id].toe_sec;
                     
                     ROS_DEBUG("  Computed GLONASS satellite position: [%.2f, %.2f, %.2f]", 
                             sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z);
                     ROS_DEBUG("  GLONASS satellite velocity: [%.2f, %.2f, %.2f]",
                             sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z);
                     ROS_DEBUG("  GLONASS satellite clock bias: %.9f s (%.3f m)", 
                             sat_info.sat_clock_bias, sat_info.sat_clock_bias * SPEED_OF_LIGHT);
                     ROS_DEBUG("  GLONASS satellite clock drift: %.9f s/s (%.3f m/s)",
                             sat_info.sat_clock_drift, sat_info.sat_clock_drift * SPEED_OF_LIGHT);
                 } else {
                     ROS_DEBUG("  Failed to compute GLONASS satellite position for slot %d", sat_info.sat_id);
                     count_ephemeris_error++;
                 }
                 
             } else if (sat_info.system == GALILEO) {
                 // Skip Galileo processing since ephemeris not available
                 count_no_ephemeris++;
                 continue;
             } else if (sat_info.system == BEIDOU) {
                 // Skip BeiDou processing since ephemeris not available
                 count_no_ephemeris++;
                 continue;
             } else {
                 ROS_DEBUG("  Satellite system not supported for processing (PRN %d, system %d)", 
                         sat_info.sat_id, sat_info.system);
                 continue;
             }
             
             // Skip if error computing satellite position
             if (!success) {
                 continue;
             }
             
             // Calculate elevation and azimuth angles - no filtering yet
             calculateElevationAzimuth(
                 current_solution_.x, current_solution_.y, current_solution_.z,
                 sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                 sat_info.elevation, sat_info.azimuth);
             
             // Check for negative elevation angles - this indicates an incorrect receiver position
             if (sat_info.elevation < 0) {
                 count_negative_elevation++;
             }
             
             // Add elevation debug output
             ROS_DEBUG("  Satellite %d: Elevation: %.2f degrees, Azimuth: %.2f degrees", 
                     sat_info.sat_id,
                     sat_info.elevation * 180.0 / M_PI, 
                     sat_info.azimuth * 180.0 / M_PI);
             
             // Calculate weight - use GVINS approach with absolute elevation
             calculateMeasurementWeight(sat_info);
             
             // Compute ionospheric delay if enabled
             sat_info.iono_delay = 0.0;
             if (apply_iono_correction_ && iono_params_.valid) {
                 // Convert ECEF to geodetic for current position
                 double lat, lon, alt;
                 CoordinateConverter::ecefToLla(current_solution_.x, current_solution_.y, current_solution_.z, lat, lon, alt);
                 
                 sat_info.iono_delay = KlobucharIonoModel::computeIonoDelay(
                     iono_params_, gps_tow, lat, lon, sat_info.elevation, sat_info.azimuth);
                 
                 ROS_DEBUG("  Ionospheric delay: %.2f m", sat_info.iono_delay);
             }
             
             // This satellite passed all checks for position computation
             ROS_DEBUG("  Satellite ready for position calculation");
             satellites.push_back(sat_info);
             
             // Log raw data if enabled
             if (log_raw_data_ && raw_data_file_.is_open()) {
                 std::string system_str;
                 switch(sat_info.system) {
                     case GPS: system_str = "GPS"; break;
                     case GLONASS: system_str = "GLONASS"; break;
                     case GALILEO: system_str = "GALILEO"; break;
                     case BEIDOU: system_str = "BEIDOU"; break;
                     default: system_str = "UNKNOWN"; break;
                 }
                 
                 raw_data_file_ << std::fixed << std::setprecision(9);
                 raw_data_file_ << gps_tow << ","
                                << sat_info.sat_id << ","
                                << system_str << ","
                                << sat_info.pseudorange << ","
                                << sat_info.carrier_phase << ","
                                << sat_info.doppler << ","
                                << sat_info.cn0 << ","
                                << std::setprecision(3)
                                << sat_info.sat_pos_x << ","
                                << sat_info.sat_pos_y << ","
                                << sat_info.sat_pos_z << ","
                                << sat_info.sat_vel_x << ","
                                << sat_info.sat_vel_y << ","
                                << sat_info.sat_vel_z << ","
                                << std::setprecision(9)
                                << sat_info.sat_clock_bias << ","
                                << sat_info.sat_clock_drift << ","
                                << std::setprecision(6)
                                << sat_info.elevation * 180.0/M_PI << ","
                                << sat_info.azimuth * 180.0/M_PI << ","
                                << std::setprecision(3)
                                << sat_info.iono_delay << ","
                                << sat_info.trop_delay
                                << std::endl;
             }
         }
         
         // Make sure we have enough satellites (regardless of elevation)
         if (satellites.size() < min_satellites_) {
             ROS_WARN("Not enough valid satellites: %zu (need %d)", satellites.size(), min_satellites_);
             return;
         }
         
         // During initialization or if disable_elevation_filter is set, use ALL satellites
         bool use_all_sats = !initialized_ || disable_elevation_filter_;
         
         // Create header from the message timestamp
         std_msgs::Header header;
         header.stamp = ros::Time(gps_tow);
         header.frame_id = frame_id_;
         
         // Run WLS solver - special initialization 
         GnssSolution solution;
         solution.timestamp = gps_tow;
         
         // Use GPS-only mode for more stability
         bool position_success;
         if (gps_only_mode_) {
             position_success = solveGpsOnlyWLS(satellites, solution, use_all_sats);
         } else {
             position_success = solveMultiSystemWLS(satellites, solution, use_all_sats);
         }
         
         if (position_success) {
             // If we just initialized, log and save the position
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
             
             // Estimate velocity from Doppler measurements if enabled
             if (use_doppler_) {
                 if (gps_only_mode_) {
                     // Store receiver position in each satellite for velocity estimation
                     for (auto& sat : satellites) {
                         sat.rx_pos_x = solution.x;
                         sat.rx_pos_y = solution.y;
                         sat.rx_pos_z = solution.z;
                     }
                     
                     // Special velocity estimation for GVINS dataset
                     if (use_gvins_doppler_model_) {
                         estimateGvinsVelocity(satellites, solution);
                     } else {
                         estimateGpsVelocityWithCeres(satellites, solution);
                     }
                 } else {
                     // Multi-system velocity estimation
                     estimateMultiSystemVelocity(satellites, solution);
                 }
             }
             
             // Store solution
             current_solution_ = solution;
             
             // Publish results
             publishResults(header, solution);
             
             // Calculate ENU coordinates
             double e = 0.0, n = 0.0, u = 0.0;
             calculateENUCoordinates(solution, e, n, u);
             
             // Calculate ENU velocity
             double ve = 0.0, vn = 0.0, vu = 0.0;
             if (use_doppler_) {
                 Eigen::Vector3d ref_lla(ref_lat_, ref_lon_, ref_alt_);
                 Eigen::Vector3d vel_ecef(solution.vx, solution.vy, solution.vz);
                 Eigen::Vector3d vel_enu;
                 CoordinateConverter::ecefVelToEnu(ref_lla, vel_ecef, vel_enu);
                 ve = vel_enu(0);
                 vn = vel_enu(1);
                 vu = vel_enu(2);
             }
             
             // Calculate speed for reporting
             double speed = sqrt(solution.vx*solution.vx + solution.vy*solution.vy + solution.vz*solution.vz);
             
             // Save results to CSV file
             if (csv_file_.is_open()) {
                 // Format: Timestamp,GPSWeek,TOW,Lat,Lon,Alt,ECEF_X,ECEF_Y,ECEF_Z,ENU_E,ENU_N,ENU_U,VelocityX,VelocityY,VelocityZ,VelocityE,VelocityN,VelocityU,Speed,ClockBias,ClockDrift,NumSats,NumVelSats,PDOP,HDOP,VDOP,TDOP
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
                           << solution.vx << ","
                           << solution.vy << ","
                           << solution.vz << ","
                           << ve << ","
                           << vn << ","
                           << vu << ","
                           << speed << ","
                           << solution.clock_bias << ","
                           << solution.clock_drift << ","
                           << solution.num_satellites << ","
                           << solution.num_vel_satellites << ","
                           << solution.pdop << ","
                           << solution.hdop << ","
                           << solution.vdop << ","
                           << solution.tdop << std::endl;
             }
             
             if (gps_only_mode_) {
                 ROS_INFO("GNSS solution (GPS-only): Lat=%.7f°, Lon=%.7f°, Alt=%.2fm, Sats=%d, HDOP=%.2f", 
                         solution.latitude * 180.0 / M_PI, 
                         solution.longitude * 180.0 / M_PI, 
                         solution.altitude, 
                         solution.num_satellites,
                         solution.hdop);
                         
                 ROS_INFO("GPS clock bias: %.2f m, drift: %.2f m/s", 
                         solution.clock_bias, solution.clock_drift);
                 
                 if (use_doppler_) {
                     ROS_INFO("Velocity: [%.2f, %.2f, %.2f] m/s (ENU: [%.2f, %.2f, %.2f]), Speed: %.2f km/h", 
                             solution.vx, solution.vy, solution.vz, ve, vn, vu, speed * 3.6);
                 }
                 
                 ROS_INFO("ENU position: E=%.2f, N=%.2f, U=%.2f m", e, n, u);
             } else {
                 ROS_INFO("GNSS solution: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm, Sats=%d, HDOP=%.2f", 
                         solution.latitude * 180.0 / M_PI, 
                         solution.longitude * 180.0 / M_PI, 
                         solution.altitude, 
                         solution.num_satellites,
                         solution.hdop);
                         
                 ROS_INFO("GNSS clocks: GPS=%.2fm, GLONASS=%.2fm", 
                         solution.clock_bias, solution.glonass_clock_bias);
                 
                 if (use_doppler_) {
                     ROS_INFO("Velocity: [%.2f, %.2f, %.2f] m/s (ENU: [%.2f, %.2f, %.2f]), Speed: %.2f km/h", 
                             solution.vx, solution.vy, solution.vz, ve, vn, vu, speed * 3.6);
                 }
                 
                 ROS_INFO("ENU position: E=%.2f, N=%.2f, U=%.2f m", e, n, u);
             }
         } else {
             ROS_WARN("Failed to compute GNSS solution with %zu satellites", satellites.size());
         }
     }
     
     // GVINS-specific velocity estimation method that handles the special Doppler format
     bool estimateGvinsVelocity(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
         std::vector<SatelliteInfo> doppler_sats;
         
         // Log satellite velocity and Doppler range rate analysis for debugging
         ROS_INFO("Satellite velocity and Doppler analysis:");
         for (const auto& sat : satellites) {
             if (sat.system == GPS && std::isfinite(sat.doppler)) {
                 // Calculate line-of-sight vector
                 double dx = sat.sat_pos_x - solution.x;
                 double dy = sat.sat_pos_y - solution.y;
                 double dz = sat.sat_pos_z - solution.z;
                 double range = sqrt(dx*dx + dy*dy + dz*dz);
                 
                 double ex = dx / range;
                 double ey = dy / range;
                 double ez = dz / range;
                 
                 // Calculate satellite velocity along line of sight
                 double sat_vel_los = ex * sat.sat_vel_x + ey * sat.sat_vel_y + ez * sat.sat_vel_z;
                 
                 // Convert Doppler to range rate (standard RINEX convention)
                 double lambda = SPEED_OF_LIGHT / GPS_L1_FREQ;
                 double range_rate = -sat.doppler * lambda; // Negative sign for RINEX convention
                 
                 ROS_INFO("  PRN %d: Doppler = %.1f Hz, Range rate = %.2f m/s, Sat LOS vel = %.2f m/s, Diff = %.2f m/s",
                        sat.sat_id, sat.doppler, range_rate, sat_vel_los, sat_vel_los - range_rate);
                 
                 // Check ephemeris age if that field is available
                 if (sat.ephemeris_time > 0) {
                     double eph_age = current_time_ - sat.ephemeris_time;
                     ROS_INFO("  PRN %d: Ephemeris age = %.1f seconds", sat.sat_id, eph_age);
                     if (eph_age > 7200.0) {
                         ROS_WARN("  PRN %d: Ephemeris too old! Consider updating ephemeris.", sat.sat_id);
                     }
                 }
                 
                 // Check satellite velocity magnitude for reasonableness
                 double sat_vel_mag = sqrt(sat.sat_vel_x*sat.sat_vel_x + 
                                          sat.sat_vel_y*sat.sat_vel_y + 
                                          sat.sat_vel_z*sat.sat_vel_z);
                 
                 ROS_INFO("  PRN %d: Satellite velocity magnitude = %.2f m/s", sat.sat_id, sat_vel_mag);
                 if (sat_vel_mag < 1.0 || sat_vel_mag > 5000.0) {
                     ROS_WARN("  PRN %d: Suspicious satellite velocity magnitude!", sat.sat_id);
                 }
                 
                 // Add valid satellites to the vector for velocity estimation
                 if (sat.elevation * 180.0/M_PI >= cut_off_degree_ && sat.cn0 >= min_cn0_) {
                     SatelliteInfo sat_with_rx = sat;
                     sat_with_rx.rx_pos_x = solution.x;
                     sat_with_rx.rx_pos_y = solution.y;
                     sat_with_rx.rx_pos_z = solution.z;
                     doppler_sats.push_back(sat_with_rx);
                 }
             }
         }
         
         // Check if we have enough satellites for velocity determination
         if (doppler_sats.size() < 4) {
             ROS_WARN("Not enough GPS satellites for velocity: %zu (need at least 4)", 
                     doppler_sats.size());
             return false;
         }
         
         // Set up linear system for velocity estimation
         Eigen::MatrixXd A(doppler_sats.size(), 4);  // 4 unknowns: vx, vy, vz, clock_drift
         Eigen::VectorXd b(doppler_sats.size());
         Eigen::VectorXd weights(doppler_sats.size());
         
         for (size_t i = 0; i < doppler_sats.size(); i++) {
             const auto& sat = doppler_sats[i];
             
             // Calculate line-of-sight unit vector
             double dx = sat.sat_pos_x - solution.x;
             double dy = sat.sat_pos_y - solution.y;
             double dz = sat.sat_pos_z - solution.z;
             double range = sqrt(dx*dx + dy*dy + dz*dz);
             
             double ex = dx / range;
             double ey = dy / range;
             double ez = dz / range;
             
             // Calculate satellite velocity in line-of-sight direction
             double sat_vel_los = ex * sat.sat_vel_x + ey * sat.sat_vel_y + ez * sat.sat_vel_z;
             
             // RINEX Doppler to range rate conversion
             // In RINEX, positive Doppler means satellite is moving away from receiver
             double lambda = SPEED_OF_LIGHT / GPS_L1_FREQ;
             double range_rate = -sat.doppler * lambda;  // Convert Hz to m/s with correct sign
             
             // The correct Doppler equation is:
             // range_rate = sat_vel_los - rcv_vel_los + clock_drift
             // where rcv_vel_los = ex*vx + ey*vy + ez*vz
             
             // Rearranging to solve for receiver velocity:
             // ex*vx + ey*vy + ez*vz + (-1)*clock_drift = sat_vel_los - range_rate
             
             A(i, 0) = ex;      // vx coefficient
             A(i, 1) = ey;      // vy coefficient
             A(i, 2) = ez;      // vz coefficient
             A(i, 3) = -1.0;    // clock_drift coefficient (note: negative)
             
             b(i) = sat_vel_los - range_rate;
             
             // Weight by signal strength (CN0)
             weights(i) = std::min(std::max(sat.cn0 - 30.0, 5.0), 20.0) / 20.0;
         }
         
         // Weighted least squares solution
         Eigen::MatrixXd W = weights.asDiagonal();
         Eigen::VectorXd x;
         
         try {
             // Solve the system Ax = b using weighted least squares
             x = (A.transpose() * W * A).ldlt().solve(A.transpose() * W * b);
             
             double vx = x(0);
             double vy = x(1);
             double vz = x(2);
             double clock_drift = x(3);
             
             double speed = sqrt(vx*vx + vy*vy + vz*vz);
             
             // Check if the solution is reasonable for a ground vehicle
             if (speed < max_velocity_) {
                 // Convert to local ENU coordinates for verification
                 Eigen::Vector3d ecef_vel(vx, vy, vz);
                 Eigen::Vector3d enu_vel;
                 
                 // Check if we have a valid reference for ENU conversion
                 if (enu_reference_set_) {
                     Eigen::Vector3d ref_lla(ref_lat_, ref_lon_, ref_alt_);
                     CoordinateConverter::ecefVelToEnu(ref_lla, ecef_vel, enu_vel);
                 } else {
                     // Use current position as temporary reference
                     double lat = solution.latitude;
                     double lon = solution.longitude;
                     double alt = solution.altitude;
                     Eigen::Vector3d curr_lla(lat, lon, alt);
                     CoordinateConverter::ecefVelToEnu(curr_lla, ecef_vel, enu_vel);
                 }
                 
                 ROS_INFO("Doppler velocity solution: ECEF [%.2f, %.2f, %.2f] m/s, Speed = %.2f km/h",
                        vx, vy, vz, speed * 3.6);
                 ROS_INFO("                         ENU: [%.2f, %.2f, %.2f] m/s", 
                        enu_vel(0), enu_vel(1), enu_vel(2));
                 ROS_INFO("Clock drift: %.2f m/s (%.3e s/s)", 
                        clock_drift, clock_drift / SPEED_OF_LIGHT);
                 
                 // Calculate residuals to check solution quality
                 Eigen::VectorXd residuals = A * x - b;
                 double rms_residual = sqrt((residuals.transpose() * W * residuals)[0] / doppler_sats.size());
                 ROS_INFO("RMS residual: %.3f m/s", rms_residual);
                 
                 // Store the solution
                 solution.vx = vx;
                 solution.vy = vy;
                 solution.vz = vz;
                 solution.clock_drift = clock_drift;
                 solution.num_vel_satellites = doppler_sats.size();
                 
                 // Calculate covariance
                 try {
                     solution.vel_covariance = (A.transpose() * W * A).inverse();
                 } catch (const std::exception& e) {
                     ROS_WARN("Failed to calculate velocity covariance: %s", e.what());
                     solution.vel_covariance = Eigen::Matrix4d::Identity();
                 }
                 
                 return true;
             } else {
                 ROS_WARN("Doppler velocity too high: %.2f m/s, falling back to position difference method", speed);
             }
         } catch (const std::exception& e) {
             ROS_WARN("Doppler velocity estimation failed: %s", e.what());
         }
         
         // Fall back to position difference method if Doppler method fails
         static double last_time = 0.0;
         static double last_x = 0.0, last_y = 0.0, last_z = 0.0;
         static double last_vx = 0.0, last_vy = 0.0, last_vz = 0.0;
         
         if (last_time > 0.0) {
             double dt = current_time_ - last_time;
             if (dt > 0.0 && dt < 1.0) {  // Valid time difference
                 double dx = solution.x - last_x;
                 double dy = solution.y - last_y;
                 double dz = solution.z - last_z;
                 
                 double vx = dx / dt;
                 double vy = dy / dt;
                 double vz = dz / dt;
                 
                 // Apply smoothing with previous velocity estimates
                 if (last_vx != 0.0 || last_vy != 0.0 || last_vz != 0.0) {
                     double alpha = 0.3;  // Smoothing factor
                     vx = alpha * vx + (1 - alpha) * last_vx;
                     vy = alpha * vy + (1 - alpha) * last_vy;
                     vz = alpha * vz + (1 - alpha) * last_vz;
                 }
                 
                 double speed = sqrt(vx*vx + vy*vy + vz*vz);
                 
                 if (speed < max_velocity_) {
                     // Convert to local ENU coordinates
                     Eigen::Vector3d ecef_vel(vx, vy, vz);
                     Eigen::Vector3d enu_vel;
                     
                     if (enu_reference_set_) {
                         Eigen::Vector3d ref_lla(ref_lat_, ref_lon_, ref_alt_);
                         CoordinateConverter::ecefVelToEnu(ref_lla, ecef_vel, enu_vel);
                     } else {
                         // Use current position as temporary reference
                         double lat = solution.latitude;
                         double lon = solution.longitude;
                         double alt = solution.altitude;
                         Eigen::Vector3d curr_lla(lat, lon, alt);
                         CoordinateConverter::ecefVelToEnu(curr_lla, ecef_vel, enu_vel);
                     }
                     
                     ROS_INFO("Position difference velocity: ECEF [%.2f, %.2f, %.2f] m/s, Speed = %.2f km/h",
                            vx, vy, vz, speed * 3.6);
                     ROS_INFO("                           ENU: [%.2f, %.2f, %.2f] m/s", 
                            enu_vel(0), enu_vel(1), enu_vel(2));
                     
                     solution.vx = vx;
                     solution.vy = vy;
                     solution.vz = vz;
                     solution.clock_drift = 0.0;  // No clock drift estimate
                     solution.num_vel_satellites = doppler_sats.size();
                     
                     // Store for next iteration
                     last_vx = vx;
                     last_vy = vy;
                     last_vz = vz;
                     
                     // Update last position and time
                     last_time = current_time_;
                     last_x = solution.x;
                     last_y = solution.y;
                     last_z = solution.z;
                     
                     return true;
                 }
             }
         }
         
         // Update last position and time
         last_time = current_time_;
         last_x = solution.x;
         last_y = solution.y;
         last_z = solution.z;
         
         // First fix or invalid velocity - return zero velocity
         ROS_INFO("First position fix or invalid velocity, returning zero velocity");
         solution.vx = 0.0;
         solution.vy = 0.0;
         solution.vz = 0.0;
         solution.clock_drift = 0.0;
         solution.num_vel_satellites = doppler_sats.size();
         
         return true;
     }
     
 private:
     // Node parameters
     std::string frame_id_;
     double pseudorange_noise_;
     int min_satellites_;
     bool use_doppler_;
     double initial_latitude_;
     double initial_longitude_;
     double initial_altitude_;
     bool apply_iono_correction_;
     double elevation_mask_deg_;
     double elevation_mask_;
     double cut_off_degree_;     // Configurable elevation cutoff angle
     double min_cn0_;
     bool initialized_;
     std::string output_csv_path_;
     std::ofstream csv_file_;
     int current_gps_week_;     // Current GPS week
     double current_leap_seconds_; // Current GPS-UTC leap seconds
     double current_time_;      // Store current time for use in other functions
     double max_velocity_;      // Maximum expected velocity in m/s
     double doppler_scale_factor_; // Scale factor for Doppler measurements
     bool adaptive_doppler_scaling_; // Whether to try different scale factors
     bool use_gvins_doppler_model_; // Whether to use GVINS-specific Doppler model
     
     // ENU reference point (first fix)
     bool enu_reference_set_ = false;
     double ref_lat_ = 0.0;
     double ref_lon_ = 0.0;
     double ref_alt_ = 0.0;
     double ref_ecef_x_ = 0.0;
     double ref_ecef_y_ = 0.0;
     double ref_ecef_z_ = 0.0;
     
     // Raw data logging
     bool log_raw_data_;
     std::string raw_data_path_;
     std::ofstream raw_data_file_;
     
     // Debug options
     bool disable_cn0_filter_;
     bool disable_elevation_filter_;
     bool use_initialization_mode_;
     bool force_use_ephemeris_;
     bool gps_only_mode_;       // GPS-only mode for stability
     
     // Current solution
     GnssSolution current_solution_;
     
     // ROS interfaces
     ros::Subscriber ephem_sub_;
     ros::Subscriber glo_ephem_sub_;
     ros::Subscriber iono_params_sub_;
     ros::Subscriber raw_meas_sub_;
     ros::Publisher navsatfix_pub_;
     ros::Publisher odom_pub_;
     ros::Publisher pose_pub_;
     
     // Data storage
     std::map<uint32_t, GpsEphemeris> gps_ephemeris_;
     std::map<uint32_t, GlonassEphemeris> glo_ephemeris_;
     IonoParams iono_params_;
     std::mutex gps_ephem_mutex_;
     std::mutex glo_ephem_mutex_;
     std::mutex iono_params_mutex_;
     
     // Calculate ENU coordinates relative to reference point
     void calculateENUCoordinates(const GnssSolution& solution, double& e, double& n, double& u) {
         if (!enu_reference_set_) {
             e = 0.0;
             n = 0.0;
             u = 0.0;
             return;
         }
         
         // Calculate offset in ECEF
        double dx = solution.x - ref_ecef_x_;
        double dy = solution.y - ref_ecef_y_;
        double dz = solution.z - ref_ecef_z_;
        
        // Rotation matrix from ECEF to ENU
        double sin_lat = sin(ref_lat_);
        double cos_lat = cos(ref_lat_);
        double sin_lon = sin(ref_lon_);
        double cos_lon = cos(ref_lon_);
        
        // Rotate ECEF displacement to ENU
        e = -sin_lon * dx + cos_lon * dy;
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
    }
    
    void calculateElevationAzimuth(
        double rx, double ry, double rz,
        double sx, double sy, double sz,
        double& elevation, double& azimuth) {
        
        // Print input values for debugging elevation issues
        ROS_DEBUG("Elevation calculation inputs:");
        ROS_DEBUG("  Receiver: [%.2f, %.2f, %.2f]", rx, ry, rz);
        ROS_DEBUG("  Satellite: [%.2f, %.2f, %.2f]", sx, sy, sz);
        
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
        
        // Debug LLA conversion
        ROS_DEBUG("  Receiver LLA: [%.6f°, %.6f°, %.2fm]", 
                  lat * 180.0/M_PI, lon * 180.0/M_PI, alt);
        
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
        
        // Debug ENU vector
        ROS_DEBUG("  ENU vector: [%.2f, %.2f, %.2f]", e, n, u);
        
        // Compute elevation and azimuth
        double horizontal_distance = sqrt(e*e + n*n);
        
        elevation = atan2(u, horizontal_distance);
        azimuth = atan2(e, n);
        
        // Normalize azimuth to [0, 2π)
        if (azimuth < 0) {
            azimuth += 2 * M_PI;
        }
        
        // Debug final values
        ROS_DEBUG("  Computed: elev = %.2f° (%.6f rad), azim = %.2f° (%.6f rad)", 
                  elevation * 180.0/M_PI, elevation, 
                  azimuth * 180.0/M_PI, azimuth);
    }
    
    void calculateMeasurementWeight(SatelliteInfo& sat_info) {
        // GVINS approach - weight based on sin²(elevation)
        // Use absolute elevation angle for weighting to handle negative elevations
        double elevation_deg = sat_info.elevation * 180.0 / M_PI;
        double abs_elevation = fabs(sat_info.elevation);
        double sin_el = sin(abs_elevation);
        double elevation_weight = std::max(0.01, sin_el * sin_el);
        
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
        
        // Combined weight (product of elevation and CN0 weights)
        sat_info.weight = elevation_weight * cn0_weight * ura_weight;
        
        // Set pseudorange standard deviation based on weight (for whitening)
        sat_info.psr_std = pseudorange_noise_ / sqrt(sat_info.weight);
        
        // Set Doppler standard deviation - this affects velocity estimate quality
        sat_info.dopp_std = 0.3 / sqrt(sat_info.weight);
        
        ROS_DEBUG("Satellite %d: elevation=%.1f°, CN0=%.1f dB-Hz, URA=%.1f, weight=%.3f", 
                  sat_info.sat_id, elevation_deg, sat_info.cn0, sat_info.ura, sat_info.weight);
    }
    
    // Solve for position using GPS satellites only
    bool solveGpsOnlyWLS(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution, bool use_all_satellites = false) {
        ROS_INFO("Starting GPS-only WLS solver with %zu total satellites", satellites.size());
        
        // Extract GPS satellites
        std::vector<uint32_t> gps_idx;
        for (uint32_t i = 0; i < satellites.size(); ++i) {
            if (satellites[i].system == GPS && 
                (use_all_satellites || satellites[i].elevation * 180.0/M_PI >= cut_off_degree_)) {
                gps_idx.push_back(i);
            }
        }
        
        if (gps_idx.size() < 4) {
            ROS_WARN("Too few GPS satellites for positioning: %zu (need at least 4)", gps_idx.size());
            return false;
        }
        
        ROS_INFO("Using %zu GPS satellites for position fix", gps_idx.size());
        
        // 4-parameter state for GPS-only: [x, y, z, clock_bias]
        double state[4] = {0.0, 0.0, 0.0, 0.0};
        
        // Use the Earth surface as a starting point if not initialized
        if (!initialized_) {
            // Start at a point on Earth's surface using proper WGS84 model
            double lat_rad = initial_latitude_ * M_PI / 180.0;
            double lon_rad = initial_longitude_ * M_PI / 180.0;
            double N = WGS84_a / sqrt(1.0 - WGS84_e_sq * sin(lat_rad) * sin(lat_rad));
            double x0 = (N + initial_altitude_) * cos(lat_rad) * cos(lon_rad);
            double y0 = (N + initial_altitude_) * cos(lat_rad) * sin(lon_rad);
            double z0 = (N * (1.0 - WGS84_e_sq) + initial_altitude_) * sin(lat_rad);
            
            state[0] = x0;
            state[1] = y0;
            state[2] = z0;
            state[3] = 0.0;  // Initial clock bias
            
            ROS_INFO("Starting with Earth surface position: [%.1f, %.1f, %.1f]", x0, y0, z0);
        } else {
            // Use current solution
            state[0] = current_solution_.x;
            state[1] = current_solution_.y;
            state[2] = current_solution_.z;
            state[3] = current_solution_.clock_bias;
            
            ROS_INFO("Using current position as initial state: [%.1f, %.1f, %.1f], clock=%.1f",
                   state[0], state[1], state[2], state[3]);
        }
        
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
        options.max_num_iterations = 10;  // Fewer iterations per outer loop
        options.function_tolerance = 1e-6;
        
        // Run the solver
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        if (!summary.IsSolutionUsable()) {
            ROS_WARN("GPS-only solver failed: %s", summary.BriefReport().c_str());
            return false;
        }
        
        // Log intermediate solution
        ROS_INFO("Iteration 0 solution: [%.2f, %.2f, %.2f], clock=%.2f, cost=%.6f", 
               state[0], state[1], state[2], state[3], summary.final_cost);
        
        // Extract solution
        ROS_INFO("GPS-only solution state: [%.2f, %.2f, %.2f], clock=%.2f", 
                state[0], state[1], state[2], state[3]);
        
        solution.x = state[0];
        solution.y = state[1];
        solution.z = state[2];
        solution.clock_bias = state[3];
        solution.glonass_clock_bias = 0.0;  // Not used in GPS-only mode
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
    
    // Solve for position using multiple GNSS systems
    bool solveMultiSystemWLS(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution, bool use_all_satellites = false) {
        ROS_INFO("Starting multi-system WLS solver with %zu total satellites", satellites.size());
        
        // Create vectors of indices for satellites by system
        std::vector<uint32_t> gps_idx, glo_idx;
        
        // Apply elevation filter unless in initialization mode
        for (uint32_t i = 0; i < satellites.size(); ++i) {
            if (use_all_satellites || satellites[i].elevation * 180.0/M_PI >= cut_off_degree_) {
                if (satellites[i].system == GPS) {
                    gps_idx.push_back(i);
                } else if (satellites[i].system == GLONASS) {
                    glo_idx.push_back(i);
                }
            }
        }
        
        ROS_INFO("Satellites by system: GPS=%zu, GLONASS=%zu", gps_idx.size(), glo_idx.size());
        
        // Need at least 5 satellites for multi-system (5 parameters)
        size_t total_usable = gps_idx.size() + glo_idx.size();
        if (total_usable < 5) {
            ROS_WARN("Too few satellites for multi-system solution: %zu (need at least 5)", total_usable);
            
            // Try GPS-only as fallback if we have enough GPS satellites
            if (gps_idx.size() >= 4) {
                ROS_INFO("Falling back to GPS-only solution with %zu satellites", gps_idx.size());
                return solveGpsOnlyWLS(satellites, solution, use_all_satellites);
            }
            
            return false;
        }
        
        // Need at least 1 satellite from each system
        if (gps_idx.size() == 0 || glo_idx.size() == 0) {
            ROS_WARN("Need at least 1 satellite from each system for multi-system solution");
            
            // Try GPS-only as fallback if we have enough GPS satellites
            if (gps_idx.size() >= 4) {
                ROS_INFO("Falling back to GPS-only solution with %zu satellites", gps_idx.size());
                return solveGpsOnlyWLS(satellites, solution, use_all_satellites);
            }
            
            return false;
        }
        
        // 5-parameter state: [x, y, z, gps_clock_bias, glonass_clock_bias]
        double state[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        
        // Use the Earth surface as a starting point if not initialized
       if (!initialized_) {
           // Start at a point on Earth's surface using proper WGS84 model
           double lat_rad = initial_latitude_ * M_PI / 180.0;
           double lon_rad = initial_longitude_ * M_PI / 180.0;
           double N = WGS84_a / sqrt(1.0 - WGS84_e_sq * sin(lat_rad) * sin(lat_rad));
           double x0 = (N + initial_altitude_) * cos(lat_rad) * cos(lon_rad);
           double y0 = (N + initial_altitude_) * cos(lat_rad) * sin(lon_rad);
           double z0 = (N * (1.0 - WGS84_e_sq) + initial_altitude_) * sin(lat_rad);
           
           state[0] = x0;
           state[1] = y0;
           state[2] = z0;
           state[3] = 0.0;  // GPS clock bias
           state[4] = 0.0;  // GLONASS clock bias
           
           ROS_INFO("Starting with Earth surface position: [%.1f, %.1f, %.1f]", x0, y0, z0);
       } else {
           // Use current solution
           state[0] = current_solution_.x;
           state[1] = current_solution_.y;
           state[2] = current_solution_.z;
           state[3] = current_solution_.clock_bias;
           state[4] = current_solution_.glonass_clock_bias;
           
           ROS_INFO("Using current position as initial state: [%.1f, %.1f, %.1f], GPS clock=%.1f, GLO clock=%.1f",
                  state[0], state[1], state[2], state[3], state[4]);
       }
       
       // Set up the Ceres problem
       ceres::Problem problem;
       
       // Vector to store loss functions to manage their memory
       std::vector<ceres::LossFunction*> loss_functions;
       
       // Add residual blocks for each satellite
       for (uint32_t i = 0; i < satellites.size(); ++i) {
           const auto& sat = satellites[i];
           
           // Skip satellites that don't pass elevation filter unless in initialization mode
           if (!use_all_satellites && sat.elevation * 180.0/M_PI < cut_off_degree_) {
               continue;
           }
           
           ceres::CostFunction* cost_function = MultiSystemPseudorangeResidual::Create(sat);
           ceres::LossFunction* loss_function = new ceres::HuberLoss(5.0);
           loss_functions.push_back(loss_function);
           
           problem.AddResidualBlock(
               cost_function,
               loss_function,
               state);
       }
       
       // Configure the solver
       ceres::Solver::Options options;
       options.linear_solver_type = ceres::DENSE_QR;
       options.minimizer_progress_to_stdout = false;
       options.max_num_iterations = 10;
       options.function_tolerance = 1e-6;
       
       // Run the solver
       ceres::Solver::Summary summary;
       ceres::Solve(options, &problem, &summary);
       
       if (!summary.IsSolutionUsable()) {
           ROS_WARN("Multi-system solver failed: %s", summary.BriefReport().c_str());
           
           // Try GPS-only as fallback if we have enough GPS satellites
           if (gps_idx.size() >= 4) {
               ROS_INFO("Falling back to GPS-only solution with %zu satellites", gps_idx.size());
               return solveGpsOnlyWLS(satellites, solution, use_all_satellites);
           }
           
           return false;
       }
       
       // Log solution
       ROS_INFO("Multi-system solution: [%.2f, %.2f, %.2f], GPS=%.2f, GLO=%.2f, cost=%.6f", 
               state[0], state[1], state[2], state[3], state[4], summary.final_cost);
       
       // Extract solution
       solution.x = state[0];
       solution.y = state[1];
       solution.z = state[2];
       solution.clock_bias = state[3];  // GPS clock bias
       solution.glonass_clock_bias = state[4];  // GLONASS clock bias
       solution.num_satellites = total_usable;
       
       // Convert to geodetic coordinates
       CoordinateConverter::ecefToLla(
           solution.x, solution.y, solution.z,
           solution.latitude, solution.longitude, solution.altitude);
       
       ROS_INFO("Multi-system solution: ECEF [%.2f, %.2f, %.2f] m", 
               solution.x, solution.y, solution.z);
       ROS_INFO("                 lat/lon [%.6f, %.6f], alt %.2f m", 
               solution.latitude * 180.0/M_PI, solution.longitude * 180.0/M_PI, solution.altitude);
       ROS_INFO("                 clocks: GPS=%.2f m, GLONASS=%.2f m", 
               solution.clock_bias, solution.glonass_clock_bias);
       
       // Calculate DOP values
       std::vector<uint32_t> all_idx;
       all_idx.insert(all_idx.end(), gps_idx.begin(), gps_idx.end());
       all_idx.insert(all_idx.end(), glo_idx.begin(), glo_idx.end());
       calculateMultiSystemDOP(satellites, all_idx, solution);
       
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
   
   // Calculate DOP values for multi-system solutions
   void calculateMultiSystemDOP(const std::vector<SatelliteInfo>& satellites, 
                             const std::vector<uint32_t>& good_idx,
                             GnssSolution& solution) {
       // For multi-system DOP, we'll use a modified G matrix that includes both clock bias terms
       Eigen::MatrixXd G(good_idx.size(), 5);  // 5 params for multi-system
       
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
           
           // Set clock bias column based on system
           if (sat.system == GPS) {
               G(i, 3) = 1.0;  // GPS clock bias
               G(i, 4) = 0.0;  // No GLONASS contribution
           } else if (sat.system == GLONASS) {
               G(i, 3) = 0.0;  // No GPS contribution 
               G(i, 4) = 1.0;  // GLONASS clock bias
           } else {
               // Default to GPS for other systems
               G(i, 3) = 1.0;
               G(i, 4) = 0.0;
           }
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
           
           // Calculate DOP values (focusing on position)
           solution.gdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1) + cov_matrix(2,2) + 
                             cov_matrix(3,3) + cov_matrix(4,4));
           solution.pdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1) + cov_matrix(2,2));
           solution.hdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1));
           solution.vdop = sqrt(cov_matrix(2,2));
           solution.tdop = sqrt(cov_matrix(3,3) + cov_matrix(4,4));  // Combined time DOP
           
           // Store the covariance, but since we're using a larger matrix,
           // we need to extract the 4x4 part for compatibility
           Eigen::Matrix<double, 4, 4> reduced_cov = Eigen::Matrix<double, 4, 4>::Zero();
           for (int i = 0; i < 3; i++) {
               for (int j = 0; j < 3; j++) {
                   reduced_cov(i, j) = cov_matrix(i, j);
               }
           }
           // Use the GPS clock bias for the 4th component
           reduced_cov(3, 3) = cov_matrix(3, 3);
           solution.covariance = reduced_cov;
           
           ROS_INFO("DOP values: GDOP=%.2f, PDOP=%.2f, HDOP=%.2f, VDOP=%.2f, TDOP=%.2f",
                   solution.gdop, solution.pdop, solution.hdop, solution.vdop, solution.tdop);
       } catch (const std::exception& e) {
           ROS_WARN("Exception in multi-system DOP calculation: %s", e.what());
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
   
   // Estimate velocity using Ceres solver for GPS-only mode
   bool estimateGpsVelocityWithCeres(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
       // Select GPS satellites with valid Doppler measurements
       std::vector<SatelliteInfo> doppler_sats;
       
       // Print raw Doppler values to identify the scale
       ROS_INFO("Raw Doppler values from satellites:");
       for (const auto& sat : satellites) {
           if (sat.system == GPS && std::isfinite(sat.doppler)) {
               ROS_INFO("  PRN %d: Doppler = %.3f Hz, CN0 = %.1f dB-Hz", 
                       sat.sat_id, sat.doppler, sat.cn0);
           }
       }
       
       for (const auto& sat : satellites) {
           // Only use GPS satellites that have valid Doppler and are above elevation mask
           if (sat.system == GPS && 
               std::isfinite(sat.doppler) && 
               sat.doppler != 0.0 &&
               sat.elevation * 180.0/M_PI >= cut_off_degree_) {
                   
               // Skip satellites with unrealistic Doppler values
               if (std::abs(sat.doppler) > 15000.0) {
                   ROS_DEBUG("Skipping satellite %d with unrealistic Doppler: %.1f Hz", 
                           sat.sat_id, sat.doppler);
                   continue;
               }
               
               // Use satellites with good CN0 for velocity
               if (sat.cn0 >= min_cn0_) {
                   doppler_sats.push_back(sat);
               }
           }
       }
       
       // Check if we have enough satellites
       if (doppler_sats.size() < 4) {
           ROS_WARN("Not enough GPS satellites with valid Doppler: %zu (need at least 4)", 
                   doppler_sats.size());
           return false;
       }
       
       ROS_INFO("Using %zu GPS satellites for velocity estimation with scale factor %.3f", 
               doppler_sats.size(), doppler_scale_factor_);
       
       // 4-parameter state for velocity: [vx, vy, vz, clock_drift]
       double velocity_state[4] = {0.0, 0.0, 0.0, 0.0};
       
       // Set up the Ceres problem
       ceres::Problem problem;
       
       // Vector to store loss functions to manage their memory
       std::vector<ceres::LossFunction*> loss_functions;
       
       // Add residual blocks for each GPS satellite with Doppler
       for (const auto& sat : doppler_sats) {
           ceres::CostFunction* cost_function = GpsDopplerResidual::Create(sat, doppler_scale_factor_);
           ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
           loss_functions.push_back(loss_function);
           
           problem.AddResidualBlock(
               cost_function,
               loss_function,
               velocity_state);
       }
       
       // Configure the solver
       ceres::Solver::Options options;
       options.linear_solver_type = ceres::DENSE_QR;
       options.minimizer_progress_to_stdout = false;
       options.max_num_iterations = 10;
       options.function_tolerance = 1e-6;
       
       // Run the solver
       ceres::Solver::Summary summary;
       ceres::Solve(options, &problem, &summary);
       
       if (!summary.IsSolutionUsable()) {
           ROS_WARN("GPS velocity solver failed: %s", summary.BriefReport().c_str());
           return false;
       }
       
       // Extract velocity and clock drift
       double vx = velocity_state[0];
       double vy = velocity_state[1];
       double vz = velocity_state[2];
       double clock_drift = velocity_state[3];
       
       // Calculate velocity magnitude
       double speed = sqrt(vx*vx + vy*vy + vz*vz);
       
       // Check for unrealistic velocity
       if (speed > max_velocity_) {
           ROS_WARN("Estimated velocity (%.2f m/s) exceeds maximum (%.2f m/s), scaling down", 
                   speed, max_velocity_);
           
           // Scale velocity to max_velocity
           double scale = max_velocity_ / speed;
           vx *= scale;
           vy *= scale;
           vz *= scale;
           speed = max_velocity_;
       }
       
       // Store results in solution
       solution.vx = vx;
       solution.vy = vy;
       solution.vz = vz;
       solution.clock_drift = clock_drift;
       solution.num_vel_satellites = doppler_sats.size();
       
       // Log results
       ROS_INFO("Velocity solution: [%.2f, %.2f, %.2f] m/s (%.2f km/h)",
               vx, vy, vz, speed * 3.6);
       ROS_INFO("Clock drift: %.2f m/s (%.3e s/s)", 
               clock_drift, clock_drift / SPEED_OF_LIGHT);
       
       // Calculate covariance matrix
       Eigen::MatrixXd G(doppler_sats.size(), 4);
       Eigen::MatrixXd W = Eigen::MatrixXd::Identity(doppler_sats.size(), doppler_sats.size());
       
       // Fill the design matrix
       for (size_t i = 0; i < doppler_sats.size(); i++) {
           const auto& sat = doppler_sats[i];
           
           // Range vector from receiver to satellite
           double dx = sat.sat_pos_x - solution.x;
           double dy = sat.sat_pos_y - solution.y;
           double dz = sat.sat_pos_z - solution.z;
           double range = sqrt(dx*dx + dy*dy + dz*dz);
           
           // Normalize to line-of-sight unit vector
           double ex = dx / range;
           double ey = dy / range;
           double ez = dz / range;
           
           // Fill design matrix with line-of-sight vector components
           G(i, 0) = -ex;  // Negative because velocity is from receiver to satellite
           G(i, 1) = -ey;
           G(i, 2) = -ez;
           G(i, 3) = 1.0;  // Coefficient for clock drift
           
           // Set weight
           W(i, i) = sat.weight;
       }
       
       try {
           solution.vel_covariance = (G.transpose() * W * G).inverse();
       } catch (const std::exception& e) {
           ROS_WARN("Failed to calculate velocity covariance: %s", e.what());
           solution.vel_covariance = Eigen::Matrix4d::Identity();
       }
       
       return true;
   }
   
   // Estimate velocity from Doppler measurements for multi-system mode
   bool estimateMultiSystemVelocity(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
       // Select satellites with valid Doppler measurements
       std::vector<SatelliteInfo> doppler_sats;
       int gps_count = 0, glonass_count = 0;
       
       for (const auto& sat : satellites) {
           // Only use satellites that have valid Doppler and are above elevation mask
           if (std::isfinite(sat.doppler) && 
             sat.doppler != 0.0 &&
             sat.elevation * 180.0/M_PI >= cut_off_degree_) {
                 
             // Skip satellites with unrealistic Doppler values
             if (std::abs(sat.doppler) > 15000.0) {
                 ROS_DEBUG("Skipping satellite %d with unrealistic Doppler: %.1f Hz", 
                         sat.sat_id, sat.doppler);
                 continue;
             }
             
             // Use satellites with good CN0 for velocity
             if (sat.cn0 >= min_cn0_) {
                 doppler_sats.push_back(sat);
                 if (sat.system == GPS) {
                     gps_count++;
                 } else if (sat.system == GLONASS) {
                     glonass_count++;
                 }
             }
         }
       }
       
       // Check if we have enough satellites
       if (doppler_sats.size() < 5 || gps_count < 1 || glonass_count < 1) {
           ROS_WARN("Not enough satellites with valid Doppler for multi-system solution: %zu (%d GPS, %d GLONASS)", 
                   doppler_sats.size(), gps_count, glonass_count);
           
           // Try GPS-only as fallback if we have enough GPS satellites
           if (gps_count >= 4) {
               ROS_INFO("Falling back to GPS-only velocity estimation with %d satellites", gps_count);
               for (auto& sat : doppler_sats) {
                   sat.rx_pos_x = solution.x;
                   sat.rx_pos_y = solution.y;
                   sat.rx_pos_z = solution.z;
               }
               return estimateGpsVelocityWithCeres(satellites, solution);
           }
           
           return false;
       }
       
       ROS_INFO("Using %zu satellites (%d GPS, %d GLONASS) for velocity estimation", 
               doppler_sats.size(), gps_count, glonass_count);
       
       // Debug satellite velocities to help identify issues
       ROS_DEBUG("Satellite velocity information:");
       for (const auto& sat : doppler_sats) {
           const char* sys_name = (sat.system == GPS) ? "GPS" : ((sat.system == GLONASS) ? "GLONASS" : "Other");
           double vel_mag = sqrt(sat.sat_vel_x*sat.sat_vel_x + 
                              sat.sat_vel_y*sat.sat_vel_y + 
                              sat.sat_vel_z*sat.sat_vel_z);
           ROS_DEBUG("  %s %d: ECEF Velocity [%.2f, %.2f, %.2f], Magnitude: %.2f m/s, Doppler: %.2f Hz",
                   sys_name, sat.sat_id, sat.sat_vel_x, sat.sat_vel_y, sat.sat_vel_z, vel_mag, sat.doppler);
       }
       
       // Create matrices for weighted least squares
       int n = doppler_sats.size();
       Eigen::MatrixXd H(n, 5);  // Design matrix: 3 for velocity, 2 for clock drifts (GPS+GLO)
       Eigen::VectorXd z(n);     // Measurement vector
       Eigen::VectorXd w(n);     // Weight vector
       
       // Fill the matrices
       for (int i = 0; i < n; i++) {
           const auto& sat = doppler_sats[i];
           
           // Range vector from receiver to satellite
           double dx = sat.sat_pos_x - solution.x;
           double dy = sat.sat_pos_y - solution.y;
           double dz = sat.sat_pos_z - solution.z;
           double range = sqrt(dx*dx + dy*dy + dz*dz);
           
           // Normalize to line-of-sight unit vector
           double ex = dx / range;
           double ey = dy / range;
           double ez = dz / range;
           
           // Satellite velocity projected onto the line-of-sight
           double sat_vel_los = ex * sat.sat_vel_x + ey * sat.sat_vel_y + ez * sat.sat_vel_z;
           
           // Fill design matrix with line-of-sight vector components
           H(i, 0) = -ex;  // Negative because velocity is from receiver to satellite
           H(i, 1) = -ey;
           H(i, 2) = -ez;
           
           // Set clock drift columns based on satellite system
           if (sat.system == GPS) {
               H(i, 3) = 1.0;  // GPS clock drift
               H(i, 4) = 0.0;  // No GLONASS contribution
           } else if (sat.system == GLONASS) {
               H(i, 3) = 0.0;  // No GPS contribution
               H(i, 4) = 1.0;  // GLONASS clock drift
           } else {
               // Default to GPS for other systems
               H(i, 3) = 1.0;
               H(i, 4) = 0.0;
           }
           
           // Compute wavelength based on system and frequency
           double wavelength;
           if (sat.system == GPS) {
               wavelength = SPEED_OF_LIGHT / GPS_L1_FREQ;
           } else if (sat.system == GLONASS) {
               // GLONASS uses FDMA - frequency depends on frequency number
               int k = sat.freq_num;
               if (k < GLONASS_FREQ_NUM_MIN || k > GLONASS_FREQ_NUM_MAX) {
                   k = 0;  // Default to channel 0 if invalid
               }
               double frequency = GLONASS_L1_BASE_FREQ + k * GLONASS_L1_DELTA_FREQ;
               wavelength = SPEED_OF_LIGHT / frequency;
           } else {
               // Default to GPS L1 for other systems
               wavelength = SPEED_OF_LIGHT / GPS_L1_FREQ;
           }
           
           // Convert Doppler to range rate with special scaling for ZED-F9P
           double doppler = sat.doppler * doppler_scale_factor_;
           double range_rate = -doppler * wavelength;
           
           // Adjust for satellite velocity and clock drift
           double adjusted_range_rate = range_rate - sat_vel_los + sat.sat_clock_drift * SPEED_OF_LIGHT;
           
           // Set the measurement
           z(i) = adjusted_range_rate;
           
           // Set the weight based on satellite elevation and CN0
           // Use the same weights as for position
           w(i) = sat.weight;
           
           const char* sys_name = (sat.system == GPS) ? "GPS" : ((sat.system == GLONASS) ? "GLONASS" : "Other");
           ROS_DEBUG("  %s %d: Doppler=%.1f Hz, Range Rate=%.2f m/s, Sat Vel (LOS)=%.2f m/s, Weight=%.3f",
                   sys_name, sat.sat_id, sat.doppler, range_rate, sat_vel_los, w(i));
       }
       
       // Create weight matrix
       Eigen::DiagonalMatrix<double, Eigen::Dynamic> W = w.asDiagonal();
       
       // Solve weighted least squares
       Eigen::VectorXd x;
       try {
           // x = (H^T * W * H)^-1 * H^T * W * z
           x = (H.transpose() * W * H).ldlt().solve(H.transpose() * W * z);
       } catch (const std::exception& e) {
           ROS_ERROR("Matrix inversion failed in multi-system velocity estimation: %s", e.what());
           return false;
       }
       
       // Extract velocity and clock drifts
       double vx = x(0);
       double vy = x(1);
       double vz = x(2);
       double gps_clock_drift = x(3);
       double glonass_clock_drift = x(4);
       
       // Calculate velocity magnitude
       double speed = sqrt(vx*vx + vy*vy + vz*vz);
       
       // Calculate residuals for quality assessment
       Eigen::VectorXd residuals = H * x - z;
       double rms_residual = sqrt((residuals.transpose() * W * residuals)(0, 0) / n);
       
       // Calculate covariance matrix - use simplified 4x4 for compatibility
       Eigen::Matrix<double, 4, 4> covariance = Eigen::Matrix<double, 4, 4>::Identity();
       try {
           Eigen::Matrix<double, 5, 5> full_cov = (H.transpose() * W * H).inverse();
           // Just copy the velocity part and GPS clock drift
           for (int i = 0; i < 3; i++) {
               for (int j = 0; j < 3; j++) {
                   covariance(i, j) = full_cov(i, j);
               }
           }
           covariance(3, 3) = full_cov(3, 3);  // GPS clock drift variance
           
           solution.vel_covariance = covariance;
       } catch (const std::exception& e) {
           ROS_WARN("Failed to calculate velocity covariance: %s", e.what());
           solution.vel_covariance = Eigen::Matrix4d::Identity();
       }
       
       // Check for unrealistic velocity (use more strict limit for pedestrian applications)
       if (speed > max_velocity_) {
           ROS_WARN("Estimated velocity (%.2f m/s) exceeds pedestrian maximum (%.2f m/s), scaling down", 
                   speed, max_velocity_);
           
           // Scale velocity to max_velocity
           double scale = max_velocity_ / speed;
           vx *= scale;
           vy *= scale;
           vz *= scale;
           speed = max_velocity_;
       }
       
       // Log results
       ROS_INFO("Multi-system velocity solution: [%.2f, %.2f, %.2f] m/s (%.2f km/h)",
              vx, vy, vz, speed * 3.6);
       ROS_INFO("Clock drifts: GPS=%.2f m/s, GLONASS=%.2f m/s", 
              gps_clock_drift, glonass_clock_drift);
       ROS_INFO("RMS residual: %.3f m/s", rms_residual);
       
       // Store results in solution
       solution.vx = vx;
       solution.vy = vy;
       solution.vz = vz;
       solution.clock_drift = gps_clock_drift;
       solution.glonass_clock_drift = glonass_clock_drift;
       solution.num_vel_satellites = doppler_sats.size();
       
       return true;
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
       // Map from ECEF covariance to lat/lon/alt (simplified)
       navsatfix.position_covariance[0] = solution.covariance(0, 0);
       navsatfix.position_covariance[4] = solution.covariance(1, 1);
       navsatfix.position_covariance[8] = solution.covariance(2, 2);
       navsatfix.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
       
       // 2. Publish Odometry message
       nav_msgs::Odometry odom;
       odom.header = header;
       odom.header.frame_id = "ecef";
       odom.child_frame_id = frame_id_;
       
       // Set position in ECEF
       odom.pose.pose.position.x = solution.x;
       odom.pose.pose.position.y = solution.y;
       odom.pose.pose.position.z = solution.z;
       odom.pose.pose.orientation.w = 1.0;  // Identity quaternion
       
       // Set velocity in ECEF
       odom.twist.twist.linear.x = solution.vx;
       odom.twist.twist.linear.y = solution.vy;
       odom.twist.twist.linear.z = solution.vz;
       
       // Set position covariance
       for (int i = 0; i < 3; ++i) {
           for (int j = 0; j < 3; ++j) {
               odom.pose.covariance[i * 6 + j] = solution.covariance(i, j);
           }
       }
       
       // Set velocity covariance if available
       if (use_doppler_) {
           for (int i = 0; i < 3; ++i) {
               for (int j = 0; j < 3; ++j) {
                   odom.twist.covariance[i * 6 + j] = solution.vel_covariance(i, j);
               }
           }
       }
       
       // 3. Publish PoseWithCovarianceStamped message (ENU format)
       geometry_msgs::PoseWithCovarianceStamped pose;
       pose.header = header;
       pose.header.frame_id = "map";  // Local ENU frame
       
       // Calculate ENU position relative to reference
       double e = 0.0, n = 0.0, u = 0.0;
       calculateENUCoordinates(solution, e, n, u);
       
       if (enu_reference_set_) {
           pose.pose.pose.position.x = e;
           pose.pose.pose.position.y = n;
           pose.pose.pose.position.z = u;
           pose.pose.pose.orientation.w = 1.0;  // Identity quaternion
           
           // Transform covariance from ECEF to ENU (simplified)
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
    double min_cn0, cut_off_degree, max_velocity, doppler_scale_factor;
    int min_sats;
    bool use_doppler, force_use_ephemeris, gps_only_mode;
    double initial_lat, initial_lon;
    std::string output_csv_path;
    
    pnh.param<double>("min_cn0", min_cn0, 5.0);
    pnh.param<double>("cut_off_degree", cut_off_degree, 0.0);
    pnh.param<int>("min_satellites", min_sats, 4);
    pnh.param<bool>("use_doppler", use_doppler, true);
    pnh.param<bool>("force_use_ephemeris", force_use_ephemeris, true);
    pnh.param<bool>("gps_only_mode", gps_only_mode, true);
    pnh.param<double>("initial_latitude", initial_lat, 22.3193);  // Default to Hong Kong
    pnh.param<double>("initial_longitude", initial_lon, 114.1694);  // Default to Hong Kong
    pnh.param<double>("max_velocity", max_velocity, MAX_VELOCITY);
    pnh.param<double>("doppler_scale_factor", doppler_scale_factor, DOPPLER_SCALE_FACTOR);
    pnh.param<std::string>("output_csv_path", output_csv_path, "spp_results.csv");
    
    ROS_INFO("Configuration:");
    ROS_INFO(" - Initial position: (%.6f°, %.6f°)", initial_lat, initial_lon);
    ROS_INFO(" - Minimum CN0: %.1f dB-Hz", min_cn0);
    ROS_INFO(" - Elevation cutoff: %.1f degrees", cut_off_degree);
    ROS_INFO(" - Minimum satellites: %d", min_sats);
    ROS_INFO(" - Use Doppler: %s", use_doppler ? "true" : "false");
    ROS_INFO(" - Force use ephemeris: %s", force_use_ephemeris ? "true" : "false");
    ROS_INFO(" - GPS-only mode: %s", gps_only_mode ? "true" : "false");
    ROS_INFO(" - Max velocity: %.1f m/s (%.1f km/h)", max_velocity, max_velocity * 3.6);
    ROS_INFO(" - Doppler scale factor: %.3f", doppler_scale_factor);
    ROS_INFO(" - Output CSV file: %s", output_csv_path.c_str());
    
    GnssWlsNode node(nh, pnh);
    
    ros::spin();
    
    return 0;
}