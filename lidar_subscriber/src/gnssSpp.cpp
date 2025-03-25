/**
 * @file gnss_spp_node.cpp
 * @brief ROS node for GNSS single point positioning using raw GNSS measurements
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
 constexpr double THRESHOLD_ELEVATION = 10.0 * M_PI / 180.0;  // 10 degrees in radians
 constexpr double DEFAULT_PSEUDORANGE_NOISE = 5.0;  // meters
 constexpr double WGS84_a = 6378137.0;              // WGS84 semi-major axis in meters
 constexpr double WGS84_b = 6356752.31424518;       // WGS84 semi-minor axis in meters
 constexpr double WGS84_e_sq = 1 - (WGS84_b * WGS84_b) / (WGS84_a * WGS84_a);  // WGS84 eccentricity squared
 constexpr double MU_EARTH = 3.986005e14;           // Earth's gravitational parameter [m^3/s^2]
 constexpr double OMEGA_EARTH = 7.2921151467e-5;    // Earth's rotation rate [rad/s]
 constexpr double PI = 3.1415926535897932;
 
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
 };
 
 // Ionospheric parameters (Klobuchar model)
 struct IonoParams {
     double alpha0, alpha1, alpha2, alpha3;
     double beta0, beta1, beta2, beta3;
     bool valid;
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
     
     // Receiver position (needed for velocity estimation via Doppler)
     double rx_pos_x;        // Receiver estimated ECEF X (m)
     double rx_pos_y;        // Receiver estimated ECEF Y (m)
     double rx_pos_z;        // Receiver estimated ECEF Z (m)
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
 };
 
 // Pseudorange residual for Ceres solver
 struct PseudorangeResidual {
     PseudorangeResidual(const SatelliteInfo& sat_info)
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
         
         // Correct for satellite clock bias
         T sat_clock_correction = T(sat_info_.sat_clock_bias) * T(SPEED_OF_LIGHT);
         
         // Ionospheric delay
         T iono_delay = T(sat_info_.iono_delay);
         
         // Compute expected pseudorange
         T expected_pseudorange = geometric_range + clock_bias - sat_clock_correction + sagnac_correction + iono_delay;
         
         // Weight based on elevation and CN0
         T weight = T(sat_info_.weight);
         
         // Residual
         residual[0] = weight * (expected_pseudorange - T(sat_info_.pseudorange));
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info) {
         return new ceres::AutoDiffCostFunction<PseudorangeResidual, 1, 4>(
             new PseudorangeResidual(sat_info));
     }
     
 private:
     const SatelliteInfo sat_info_;
 };
 
 // Doppler residual for velocity estimation
 struct DopplerResidual {
     DopplerResidual(const SatelliteInfo& sat_info, double wavelength)
         : sat_info_(sat_info), wavelength_(wavelength) {}
     
     template <typename T>
     bool operator()(const T* const velocity_state, T* residual) const {
         // Extract receiver velocity and clock drift
         T vx = velocity_state[0];
         T vy = velocity_state[1];
         T vz = velocity_state[2];
         T clock_drift = velocity_state[3];
         
         // Satellite position and velocity
         T sx = T(sat_info_.sat_pos_x);
         T sy = T(sat_info_.sat_pos_y);
         T sz = T(sat_info_.sat_pos_z);
         T svx = T(sat_info_.sat_vel_x);
         T svy = T(sat_info_.sat_vel_y);
         T svz = T(sat_info_.sat_vel_z);
         
         // Compute LOS vector (unit vector from receiver to satellite)
         T dx = sx - T(sat_info_.rx_pos_x);
         T dy = sy - T(sat_info_.rx_pos_y);
         T dz = sz - T(sat_info_.rx_pos_z);
         T range = ceres::sqrt(dx*dx + dy*dy + dz*dz);
         
         // Check for zero range
         if (range < T(1e-6)) {
             residual[0] = T(0.0);
             return true;
         }
         
         T los_x = dx / range;
         T los_y = dy / range;
         T los_z = dz / range;
         
         // Compute relative velocity along LOS
         T vel_rel = (svx - vx) * los_x + (svy - vy) * los_y + (svz - vz) * los_z;
         
         // Convert to Doppler shift
         T expected_doppler = -vel_rel / T(wavelength_) - clock_drift;
         
         // Weight based on elevation and CN0
         T weight = T(sat_info_.weight);
         
         // Residual
         residual[0] = weight * (expected_doppler - T(sat_info_.doppler));
         
         return true;
     }
     
     static ceres::CostFunction* Create(const SatelliteInfo& sat_info, double wavelength) {
         return new ceres::AutoDiffCostFunction<DopplerResidual, 1, 4>(
             new DopplerResidual(sat_info, wavelength));
     }
     
 private:
     const SatelliteInfo sat_info_;
     const double wavelength_;
 };
 
 // GpsSatellitePosition - Compute satellite position from ephemeris
 class GpsEphemerisCalculator {
 public:
     static bool computeSatPosVel(const GpsEphemeris& eph, double transmit_time, 
                                 double& x, double& y, double& z, 
                                 double& vx, double& vy, double& vz, 
                                 double& clock_bias, double& clock_drift) {
         if (!eph.valid) {
             return false;
         }
         
         // Constants
         const double mu = 3.986005e14;          // Earth's gravitational constant
         const double omega_e = 7.2921151467e-5; // Earth's rotation rate
         
         // Time from ephemeris reference epoch (toe)
         double tk = transmit_time - eph.toe_sec;
         
         // Handle week crossovers
         if (tk > 302400.0) tk -= 604800.0;
         else if (tk < -302400.0) tk += 604800.0;
         
         // Compute mean motion
         double a = eph.sqrta * eph.sqrta;      // Semi-major axis
         double n0 = sqrt(mu / (a * a * a));    // Computed mean motion
         double n = n0 + eph.delta_n;          // Corrected mean motion
         
         // Mean anomaly
         double M = eph.m0 + n * tk;
         
         // Solve Kepler's equation for eccentric anomaly (iterative)
         double E = M;
         double E_old;
         int iter = 0;
         do {
             E_old = E;
             E = M + eph.e * sin(E);
             iter++;
         } while (fabs(E - E_old) > 1e-12 && iter < 10);
         
         // True anomaly
         double sin_E = sin(E);
         double cos_E = cos(E);
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
         
         // Compute velocity using derivative of position
         // Time derivative of eccentric anomaly
         double E_dot = n / (1.0 - eph.e * cos_E);
         
         // Time derivative of argument of latitude
         double phi_dot = sqrt(1.0 - eph.e * eph.e) * E_dot / (1.0 - eph.e * cos_E);
         
         // Time derivatives of correction terms
         double du_dot = 2.0 * phi_dot * (eph.cus * cos_2phi - eph.cuc * sin_2phi);
         double dr_dot = 2.0 * phi_dot * (eph.crs * cos_2phi - eph.crc * sin_2phi);
         double di_dot = eph.i_dot + 2.0 * phi_dot * (eph.cis * cos_2phi - eph.cic * sin_2phi);
         
         // Time derivative of corrected argument of latitude
         double u_dot = phi_dot + du_dot;
         
         // Time derivative of radius
         double r_dot = a * eph.e * sin_E * E_dot + dr_dot;
         
         // Time derivative of orbital position
         double x_op_dot = r_dot * cos(u) - r * u_dot * sin(u);
         double y_op_dot = r_dot * sin(u) + r * u_dot * cos(u);
         
         // Time derivative of corrected longitude of ascending node
         double Omega_dot = eph.omg_dot - omega_e;
         
         // Velocity
         vx = x_op_dot * cos_Omega - y_op_dot * cos_i * sin_Omega -
              (x_op * sin_Omega + y_op * cos_i * cos_Omega) * Omega_dot +
              y_op * sin_i * sin_Omega * di_dot;
              
         vy = x_op_dot * sin_Omega + y_op_dot * cos_i * cos_Omega +
              (x_op * cos_Omega - y_op * cos_i * sin_Omega) * Omega_dot -
              y_op * sin_i * cos_Omega * di_dot;
              
         vz = y_op_dot * sin_i + y_op * cos_i * di_dot;
         
         // Calculate satellite clock bias and drift
         double dt = transmit_time - eph.toc_sec;
         
         // Handle week crossovers for clock
         if (dt > 302400.0) dt -= 604800.0;
         else if (dt < -302400.0) dt += 604800.0;
         
         // Clock correction
         clock_bias = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt;
         
         // Relativistic correction
         double E_corr = -4.442807633e-10 * eph.e * eph.sqrta * sin_E;
         clock_bias += E_corr;
         
         // Clock drift
         clock_drift = eph.af1 + 2.0 * eph.af2 * dt;
         
         return true;
     }
 };
 
 // GLONASS satellite position calculation
 class GlonassEphemerisCalculator {
 public:
     static bool computeSatPosVel(const GlonassEphemeris& eph, double transmit_time, 
                                 double& x, double& y, double& z, 
                                 double& vx, double& vy, double& vz, 
                                 double& clock_bias, double& clock_drift) {
         if (!eph.valid) {
             return false;
         }
         
         // Convert transmit time to seconds of day in GLONASS time scale
         double seconds_of_day = fmod(transmit_time, 86400.0);
         
         // Time difference from ephemeris epoch (tb)
         double dt = seconds_of_day - eph.tb_sec;
         
         // Adjust for day boundary
         if (dt > 43200.0) dt -= 86400.0;
         else if (dt < -43200.0) dt += 86400.0;
         
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
         clock_bias = -eph.tau_n + eph.gamma * dt;
         clock_drift = eph.gamma;
         
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
         
         // Elevation in semi-circles (positive)
         double el_sc = std::max(0.05, elevation / PI);
         
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
 
 // Time conversion helper
 double gnss_time_to_sec(const gnss_comm::GnssTimeMsg& time_msg) {
     return time_msg.tow;  // Use time of week in seconds
 }
 
 class GnssWlsNode {
 public:
     GnssWlsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : initialized_(false) {
         // Load parameters
         pnh.param<std::string>("frame_id", frame_id_, "gnss");
         pnh.param<double>("pseudorange_noise", pseudorange_noise_, DEFAULT_PSEUDORANGE_NOISE);
         pnh.param<int>("min_satellites", min_satellites_, 5);
         pnh.param<bool>("use_doppler", use_doppler_, true);
         pnh.param<double>("initial_latitude", initial_latitude_, 0.0);
         pnh.param<double>("initial_longitude", initial_longitude_, 0.0);
         pnh.param<double>("initial_altitude", initial_altitude_, 0.0);
         pnh.param<bool>("apply_iono_correction", apply_iono_correction_, true);
         pnh.param<double>("elevation_mask", elevation_mask_deg_, 10.0);
         pnh.param<double>("min_cn0", min_cn0_, 10.0);
         pnh.param<bool>("output_debug_info", output_debug_info_, false);
         pnh.param<std::string>("debug_output_path", debug_output_path_, "");
         
         // Initialize elevation mask in radians
         elevation_mask_ = elevation_mask_deg_ * M_PI / 180.0;
         
         // Convert initial position to ECEF for solver
         double init_x, init_y, init_z;
         CoordinateConverter::llaToEcef(
             initial_latitude_ * M_PI / 180.0,
             initial_longitude_ * M_PI / 180.0,
             initial_altitude_,
             init_x, init_y, init_z);
         
         current_solution_.x = init_x;
         current_solution_.y = init_y;
         current_solution_.z = init_z;
         current_solution_.vx = 0.0;
         current_solution_.vy = 0.0;
         current_solution_.vz = 0.0;
         current_solution_.clock_bias = 0.0;
         current_solution_.clock_drift = 0.0;
         current_solution_.timestamp = 0.0;
         
         // Initialize ionospheric parameters
         iono_params_.alpha0 = 0.0;
         iono_params_.alpha1 = 0.0;
         iono_params_.alpha2 = 0.0;
         iono_params_.alpha3 = 0.0;
         iono_params_.beta0 = 0.0;
         iono_params_.beta1 = 0.0;
         iono_params_.beta2 = 0.0;
         iono_params_.beta3 = 0.0;
         iono_params_.valid = false;
         
         // Publishers
         navsatfix_pub_ = nh.advertise<sensor_msgs::NavSatFix>("gnss_fix", 10);
         odom_pub_ = nh.advertise<nav_msgs::Odometry>("gnss_odom", 10);
         pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("gnss_pose", 10);
         
         // Subscribers for GNSS data
         ephem_sub_ = nh.subscribe("/ublox_driver/ephem", 10, &GnssWlsNode::ephemCallback, this);
         glo_ephem_sub_ = nh.subscribe("/ublox_driver/glo_ephem", 10, &GnssWlsNode::gloEphemCallback, this);
         iono_params_sub_ = nh.subscribe("/ublox_driver/iono_params", 10, &GnssWlsNode::ionoParamsCallback, this);
         raw_meas_sub_ = nh.subscribe("/ublox_driver/range_meas", 10, &GnssWlsNode::rawMeasCallback, this);
         
         // Open debug file if enabled
         if (output_debug_info_ && !debug_output_path_.empty()) {
             debug_file_.open(debug_output_path_, std::ios::out);
             if (debug_file_.is_open()) {
                 debug_file_ << "Timestamp,NumSats,Latitude,Longitude,Altitude,PDOP,HDOP,VDOP,TDOP,ClockBias" << std::endl;
             } else {
                 ROS_WARN("Could not open debug file at: %s", debug_output_path_.c_str());
             }
         }
         
         ROS_INFO("GNSS WLS node initialized:");
         ROS_INFO(" - Minimum satellites: %d", min_satellites_);
         ROS_INFO(" - Elevation mask: %.1f degrees", elevation_mask_deg_);
         ROS_INFO(" - Minimum CN0: %.1f dB-Hz", min_cn0_);
         ROS_INFO(" - Use Doppler: %s", use_doppler_ ? "true" : "false");
         ROS_INFO(" - Apply ionospheric correction: %s", apply_iono_correction_ ? "true" : "false");
     }
     
     ~GnssWlsNode() {
         if (debug_file_.is_open()) {
             debug_file_.close();
         }
     }
     
     // Process GPS ephemeris data
     void ephemCallback(const gnss_comm::GnssEphemMsg::ConstPtr& msg) {
         std::lock_guard<std::mutex> lock(gps_ephem_mutex_);
         
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
         eph.sqrta = msg->A;  // A is the semi-major axis squared
         eph.omg = msg->omg;  // Longitude of ascending node
         eph.omg_dot = msg->OMG_dot;
         eph.omega = msg->omg;  // Argument of perigee
         eph.i0 = msg->i0;
         eph.i_dot = msg->i_dot;
         eph.tgd0 = msg->tgd0;
         eph.health = msg->health;
         eph.iode = msg->iode;
         eph.valid = true;
         
         // Store/update ephemeris
         gps_ephemeris_[msg->sat] = eph;
         
         ROS_DEBUG("Received GPS ephemeris for PRN %d, toe=%.0f", msg->sat, gnss_time_to_sec(msg->toe));
     }
     
     // Process GLONASS ephemeris data
    // Process GLONASS ephemeris data
    void gloEphemCallback(const gnss_comm::GnssGloEphemMsg::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(glo_ephem_mutex_);
        
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
        eph.dtau = msg->delta_tau_n;  // Now correctly using delta_tau_n from the message
        eph.health = msg->health;
        eph.valid = true;
        
        // Store/update ephemeris
        glo_ephemeris_[msg->sat] = eph;  // Use 'sat' which is the correct field name
        
        ROS_DEBUG("Received GLONASS ephemeris for PRN %d, toe=%.0f, freq=%d", 
                msg->sat, msg->toe.tow, msg->freqo);
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
         
         ROS_DEBUG("Received ionospheric parameters: alpha=[%.2e, %.2e, %.2e, %.2e], beta=[%.2e, %.2e, %.2e, %.2e]",
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
         
         // Set a fixed timestamp
         double gps_time = ros::Time::now().toSec();  // Using current time as fallback
         
         // Loop through all measurements
         for (const auto& obs : msg->meas) {
             // Skip if no measurements
             if (obs.psr.empty()) continue;
             
             // Create a satellite info structure
             SatelliteInfo sat_info;
             sat_info.sat_id = obs.sat;
             
             // Set system type based on the PRN range
             if (obs.sat >= 1 && obs.sat <= 32) {
                 sat_info.system = GPS;
             } else if (obs.sat >= 38 && obs.sat <= 63) {
                 sat_info.system = GLONASS;
             } else {
                 // Skip other satellite systems for now
                 continue;
             }
             
             // Use the first frequency's measurements (L1 for GPS/GLONASS)
             if (obs.psr.empty() || obs.psr[0] <= 0) {
                 continue;  // Skip if no valid pseudorange
             }
             
             // Store measurement data
             sat_info.pseudorange = obs.psr[0];  // First pseudorange
             
             if (!obs.cp.empty()) {
                 sat_info.carrier_phase = obs.cp[0];  // First carrier phase
             } else {
                 sat_info.carrier_phase = 0.0;
             }
             
             if (!obs.dopp.empty()) {
                 sat_info.doppler = obs.dopp[0];  // First Doppler
             } else {
                 sat_info.doppler = 0.0;
             }
             
             if (!obs.CN0.empty()) {
                 sat_info.cn0 = obs.CN0[0];  // First CN0
             } else {
                 sat_info.cn0 = 0.0;
             }
             
             // Skip measurements with low signal strength
             if (sat_info.cn0 < min_cn0_) {
                 continue;
             }
             
             // Process based on satellite system
             if (sat_info.system == GPS) {
                 std::lock_guard<std::mutex> lock(gps_ephem_mutex_);
                 
                 // Check if we have ephemeris for this satellite
                 if (gps_ephemeris_.find(sat_info.sat_id) == gps_ephemeris_.end()) {
                     continue;  // No ephemeris available
                 }
                 
                 // Time of signal transmission
                 double transmission_time = gps_time - sat_info.pseudorange / SPEED_OF_LIGHT;
                 
                 // Compute satellite position and clock
                 double clock_bias, clock_drift;
                 bool success = GpsEphemerisCalculator::computeSatPosVel(
                     gps_ephemeris_[sat_info.sat_id], 
                     transmission_time, 
                     sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                     sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z,
                     clock_bias, clock_drift);
                 
                 if (!success) {
                     continue;  // Failed to compute satellite position
                 }
                 
                 sat_info.sat_clock_bias = clock_bias;
                 sat_info.sat_clock_drift = clock_drift;
             } 
             else if (sat_info.system == GLONASS) {
                 std::lock_guard<std::mutex> lock(glo_ephem_mutex_);
                 
                 // Check if we have ephemeris for this satellite
                 if (glo_ephemeris_.find(sat_info.sat_id) == glo_ephemeris_.end()) {
                     continue;  // No ephemeris available
                 }
                 
                 // Time of signal transmission
                 double transmission_time = gps_time - sat_info.pseudorange / SPEED_OF_LIGHT;
                 
                 // Compute satellite position and clock
                 double clock_bias, clock_drift;
                 bool success = GlonassEphemerisCalculator::computeSatPosVel(
                     glo_ephemeris_[sat_info.sat_id], 
                     transmission_time, 
                     sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                     sat_info.sat_vel_x, sat_info.sat_vel_y, sat_info.sat_vel_z,
                     clock_bias, clock_drift);
                 
                 if (!success) {
                     continue;  // Failed to compute satellite position
                 }
                 
                 sat_info.sat_clock_bias = clock_bias;
                 sat_info.sat_clock_drift = clock_drift;
                 sat_info.freq_num = glo_ephemeris_[sat_info.sat_id].freq_slot;
             }
             
             // Set receiver estimated position for Doppler residual calculation
             sat_info.rx_pos_x = current_solution_.x;
             sat_info.rx_pos_y = current_solution_.y;
             sat_info.rx_pos_z = current_solution_.z;
             
             // Calculate elevation and azimuth angles
             calculateElevationAzimuth(
                 current_solution_.x, current_solution_.y, current_solution_.z,
                 sat_info.sat_pos_x, sat_info.sat_pos_y, sat_info.sat_pos_z,
                 sat_info.elevation, sat_info.azimuth);
             
             // Apply elevation mask
             if (sat_info.elevation < elevation_mask_) {
                 continue;
             }
             
             // Calculate weight based on elevation angle and CN0
             calculateMeasurementWeight(sat_info);
             
             // Compute ionospheric delay if enabled
             sat_info.iono_delay = 0.0;
             if (apply_iono_correction_ && iono_params_.valid) {
                 // Convert ECEF to geodetic for current position
                 double lat, lon, alt;
                 CoordinateConverter::ecefToLla(current_solution_.x, current_solution_.y, current_solution_.z, lat, lon, alt);
                 
                 sat_info.iono_delay = KlobucharIonoModel::computeIonoDelay(
                     iono_params_, gps_time, lat, lon, sat_info.elevation, sat_info.azimuth);
             }
             
             satellites.push_back(sat_info);
         }
         
         // Check if we have enough satellites
         if (satellites.size() < min_satellites_) {
             ROS_WARN("Not enough valid satellites for positioning: %zu (need %d)", 
                      satellites.size(), min_satellites_);
             return;
         }
         
         // Run WLS solver
         GnssSolution solution;
         solution.timestamp = gps_time;
         
         if (solvePositionWLS(satellites, solution)) {
             // If Doppler measurements are available, estimate velocity
             if (use_doppler_) {
                 if (solveVelocityWLS(satellites, solution)) {
                     ROS_DEBUG("Estimated velocity: [%.2f, %.2f, %.2f] m/s", 
                              solution.vx, solution.vy, solution.vz);
                 }
             }
             
             // Store solution
             current_solution_ = solution;
             
             // Create a header from the message timestamp
             std_msgs::Header header;
             header.stamp = ros::Time(solution.timestamp);
             header.frame_id = frame_id_;
             
             // Publish results
             publishResults(header, solution);
             
             // Write debug info if enabled
             if (output_debug_info_ && debug_file_.is_open()) {
                 debug_file_ << solution.timestamp << ","
                            << solution.num_satellites << ","
                            << solution.latitude * 180.0 / M_PI << ","  // Convert to degrees
                            << solution.longitude * 180.0 / M_PI << ","
                            << solution.altitude << ","
                            << solution.pdop << ","
                            << solution.hdop << ","
                            << solution.vdop << ","
                            << solution.tdop << ","
                            << solution.clock_bias << std::endl;
             }
             
             // Set initialization flag
             if (!initialized_) {
                 initialized_ = true;
                 ROS_INFO("GNSS WLS node initialized successfully");
             }
             
             ROS_INFO("GNSS solution: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm, Sats=%d, HDOP=%.2f", 
                      solution.latitude * 180.0 / M_PI, 
                      solution.longitude * 180.0 / M_PI, 
                      solution.altitude, 
                      solution.num_satellites,
                      solution.hdop);
         } else {
             ROS_WARN("Failed to compute GNSS solution with %zu satellites", satellites.size());
         }
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
     double min_cn0_;
     bool initialized_;
     bool output_debug_info_;
     std::string debug_output_path_;
     std::ofstream debug_file_;
     
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
     
     void calculateElevationAzimuth(
         double rx, double ry, double rz,
         double sx, double sy, double sz,
         double& elevation, double& azimuth) {
         
         // Convert receiver ECEF to LLA
         double lat, lon, alt;
         CoordinateConverter::ecefToLla(rx, ry, rz, lat, lon, alt);
         
         // Compute ENU vector from receiver to satellite
         double dx = sx - rx;
         double dy = sy - ry;
         double dz = sz - rz;
         
         // Convert ECEF vector to ENU
         double sin_lat = sin(lat);
         double cos_lat = cos(lat);
         double sin_lon = sin(lon);
         double cos_lon = cos(lon);
         
         // Rotation matrix from ECEF to ENU
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
         // Calculate weight based on elevation angle: sin²(elevation)
         double elevation_weight = sin(sat_info.elevation) * sin(sat_info.elevation);
         
         // Calculate weight based on CN0 (carrier-to-noise density ratio)
         double cn0_weight = 1.0;
         if (sat_info.cn0 > 0) {
             // Normalize CN0 weight between 0.1 and 1.0
             // Typical CN0 range: 30-50 dB-Hz
             cn0_weight = std::min(1.0, std::max(0.1, (sat_info.cn0 - 30.0) / 20.0));
         }
         
         // Combined weight (product of elevation and CN0 weights)
         sat_info.weight = elevation_weight * cn0_weight;
     }
     
     bool solvePositionWLS(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
         // We need at least 4 satellites
         if (satellites.size() < 4) {
             return false;
         }
         
         // Initial state vector [x, y, z, clock_bias]
         double state[4] = {
             current_solution_.x,
             current_solution_.y,
             current_solution_.z,
             current_solution_.clock_bias
         };
         
         // Set up the Ceres problem
         ceres::Problem problem;
         
         // Add residual blocks for each satellite
         for (const auto& sat : satellites) {
             ceres::CostFunction* cost_function = PseudorangeResidual::Create(sat);
             
             problem.AddResidualBlock(
                 cost_function,
                 new ceres::HuberLoss(1.0),  // Huber loss for robustness
                 state);
         }
         
         // Configure the solver
         ceres::Solver::Options options;
         options.linear_solver_type = ceres::DENSE_QR;
         options.minimizer_progress_to_stdout = false;
         options.max_num_iterations = 25;
         options.function_tolerance = 1e-8;
         
         // Run the solver
         ceres::Solver::Summary summary;
         ceres::Solve(options, &problem, &summary);
         
         if (!summary.IsSolutionUsable()) {
             ROS_WARN("Ceres solver failed: %s", summary.BriefReport().c_str());
             return false;
         }
         
         // Extract solution
         solution.x = state[0];
         solution.y = state[1];
         solution.z = state[2];
         solution.clock_bias = state[3];
         solution.num_satellites = satellites.size();
         
         // Convert ECEF to geodetic (LLA)
         CoordinateConverter::ecefToLla(
             solution.x, solution.y, solution.z,
             solution.latitude, solution.longitude, solution.altitude);
         
         // Calculate DOP values
         calculateDOP(satellites, solution);
         
         // Extract covariance from the solver
         extractCovariance(problem, solution.covariance);
         
         return true;
     }
     
     bool solveVelocityWLS(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
         // Need at least 4 satellites with Doppler
         int valid_count = 0;
         for (const auto& sat : satellites) {
             if (sat.doppler != 0.0) valid_count++;
         }
         
         if (valid_count < 4) {
             return false;
         }
         
         // Initial velocity state [vx, vy, vz, clock_drift]
         double velocity_state[4] = {
             current_solution_.vx,
             current_solution_.vy,
             current_solution_.vz,
             current_solution_.clock_drift
         };
         
         // Set up the Ceres problem
         ceres::Problem problem;
         
         // Add residual blocks for each satellite with valid Doppler
         for (const auto& sat : satellites) {
             if (sat.doppler == 0.0) continue;  // Skip satellites without Doppler
             
             // Determine wavelength based on satellite system
             double wavelength = GPS_L1_WAVELENGTH;  // Default to GPS L1
             
             if (sat.system == GLONASS) {
                 // GLONASS uses FDMA, so wavelength depends on frequency number
                 double freq = GLONASS_L1_BASE_FREQ + sat.freq_num * GLONASS_L1_DELTA_FREQ;
                 wavelength = SPEED_OF_LIGHT / freq;
             }
             
             ceres::CostFunction* cost_function = DopplerResidual::Create(sat, wavelength);
             
             problem.AddResidualBlock(
                 cost_function,
                 new ceres::HuberLoss(0.5),  // Huber loss for robustness
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
             ROS_WARN("Velocity solver failed: %s", summary.BriefReport().c_str());
             return false;
         }
         
         // Extract solution
         solution.vx = velocity_state[0];
         solution.vy = velocity_state[1];
         solution.vz = velocity_state[2];
         solution.clock_drift = velocity_state[3];
         
         return true;
     }
     
     void calculateDOP(const std::vector<SatelliteInfo>& satellites, GnssSolution& solution) {
         // We need the geometry matrix for DOP calculation
         Eigen::MatrixXd G(satellites.size(), 4);
         
         // Fill the geometry matrix
         for (size_t i = 0; i < satellites.size(); i++) {
             const auto& sat = satellites[i];
             
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
             G(i, 3) = 1.0;  // receiver clock bias
         }
         
         // Compute the covariance matrix
         Eigen::MatrixXd cov_matrix = (G.transpose() * G).inverse();
         
         // Calculate DOP values
         solution.gdop = sqrt(cov_matrix.trace());
         solution.pdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1) + cov_matrix(2,2));
         solution.hdop = sqrt(cov_matrix(0,0) + cov_matrix(1,1));
         solution.vdop = sqrt(cov_matrix(2,2));
         solution.tdop = sqrt(cov_matrix(3,3));
     }
     
     void extractCovariance(const ceres::Problem& problem, Eigen::Matrix<double, 4, 4>& covariance) {
         // Initialize covariance matrix
         covariance = Eigen::Matrix<double, 4, 4>::Identity() * 100.0;  // Default high uncertainty
         
         // Get the covariance from Ceres
         ceres::Covariance::Options cov_options;
         ceres::Covariance covariance_estimator(cov_options);
         
         double* state_ptr = new double[4];
         state_ptr[0] = current_solution_.x;
         state_ptr[1] = current_solution_.y;
         state_ptr[2] = current_solution_.z;
         state_ptr[3] = current_solution_.clock_bias;
         
         std::vector<std::pair<const double*, const double*>> covariance_blocks;
         covariance_blocks.push_back(std::make_pair(state_ptr, state_ptr));
         
         // Create a mutable copy of the problem for covariance computation
         ceres::Problem* mutable_problem = const_cast<ceres::Problem*>(&problem);
         
         // Compute the covariance
         if (covariance_estimator.Compute(covariance_blocks, mutable_problem)) {
             double cov_values[16] = {0};
             covariance_estimator.GetCovarianceBlock(state_ptr, state_ptr, cov_values);
             
             // Fill the covariance matrix
             for (int i = 0; i < 4; ++i) {
                 for (int j = 0; j < 4; ++j) {
                     covariance(i, j) = cov_values[i * 4 + j];
                 }
             }
         } else {
             ROS_WARN("Failed to compute covariance");
         }
         
         delete[] state_ptr;
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
         
         // Set covariance
         for (int i = 0; i < 3; ++i) {
             for (int j = 0; j < 3; ++j) {
                 odom.pose.covariance[i * 6 + j] = solution.covariance(i, j);
             }
         }
         
         // 3. Publish PoseWithCovarianceStamped message (ENU format)
         geometry_msgs::PoseWithCovarianceStamped pose;
         pose.header = header;
         pose.header.frame_id = "map";  // Local ENU frame
         
         // Convert ECEF to local ENU
         // Need a reference point for ENU - using the first fix
         static bool reference_set = false;
         static double ref_lat, ref_lon, ref_alt;
         static double ref_ecef_x, ref_ecef_y, ref_ecef_z;
         
         if (!reference_set && initialized_) {
             ref_lat = solution.latitude;
             ref_lon = solution.longitude;
             ref_alt = solution.altitude;
             ref_ecef_x = solution.x;
             ref_ecef_y = solution.y;
             ref_ecef_z = solution.z;
             reference_set = true;
             
             ROS_INFO("Set ENU reference: Lat=%.7f°, Lon=%.7f°, Alt=%.2fm", 
                     ref_lat * 180.0 / M_PI, ref_lon * 180.0 / M_PI, ref_alt);
         }
         
         if (reference_set) {
             // Calculate ENU position relative to reference
             double dx = solution.x - ref_ecef_x;
             double dy = solution.y - ref_ecef_y;
             double dz = solution.z - ref_ecef_z;
             
             // Rotation matrix from ECEF to ENU
             double sin_lat = sin(ref_lat);
             double cos_lat = cos(ref_lat);
             double sin_lon = sin(ref_lon);
             double cos_lon = cos(ref_lon);
             
             // Rotate ECEF displacement to ENU
             double e = -sin_lon * dx + cos_lon * dy;
             double n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
             double u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
             
             pose.pose.pose.position.x = e;
             pose.pose.pose.position.y = n;
             pose.pose.pose.position.z = u;
             pose.pose.pose.orientation.w = 1.0;  // Identity quaternion
             
             // Transform velocity from ECEF to ENU
             double ve = -sin_lon * solution.vx + cos_lon * solution.vy;
             double vn = -sin_lat * cos_lon * solution.vx - sin_lat * sin_lon * solution.vy + cos_lat * solution.vz;
             double vu = cos_lat * cos_lon * solution.vx + cos_lat * sin_lon * solution.vy + sin_lat * solution.vz;
             
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
     ros::init(argc, argv, "gnss_wls_node");
     ros::NodeHandle nh;
     ros::NodeHandle pnh("~");
     
     GnssWlsNode node(nh, pnh);
     
     ros::spin();
     
     return 0;
 }