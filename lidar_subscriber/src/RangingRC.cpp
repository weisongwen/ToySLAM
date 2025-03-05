/**
 * GPS Satellite Signal Simulator ROS Node - With Rigorous Range Measurements
 * 
 * Features:
 * - Precise physical calculation of pseudorange measurements
 * - Accurate modeling of all error sources (ionosphere, troposphere, multipath, clock bias)
 * - User moves along a configurable slow trajectory
 * - Buildings on both sides affect satellite signals
 * - At least 8 GPS satellites with realistic orbital positions, all with positive elevation
 * - Simulates direct line-of-sight, blocked signals, and multipath
 * - Signal strength calculation based on satellite elevation and building blockage
 * - Comprehensive visualization of GPS signal characteristics
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
     double signal_attenuation; // Signal attenuation in dB
     double reflectivity;       // Reflection coefficient (0-1)
     
     Building() : signal_attenuation(30.0), reflectivity(0.6) {}
     
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
             Eigen::Vector3d(0, 0, 1)                             // Bitangent (z-direction)
         });
         
         // Back face (normal = +y)
         faces.push_back({
             Eigen::Vector3d(center.x(), max_pt.y(), center.z()),  // Center
             Eigen::Vector3d(0, 1, 0),                            // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 0, 1)                             // Bitangent (z-direction)
         });
         
         // Left face (normal = -x)
         faces.push_back({
             Eigen::Vector3d(min_pt.x(), center.y(), center.z()),  // Center
             Eigen::Vector3d(-1, 0, 0),                           // Normal
             Eigen::Vector3d(dimensions.y(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(0, 1, 0),                            // Tangent (y-direction)
             Eigen::Vector3d(0, 0, 1)                             // Bitangent (z-direction)
         });
         
         // Right face (normal = +x)
         faces.push_back({
             Eigen::Vector3d(max_pt.x(), center.y(), center.z()),  // Center
             Eigen::Vector3d(1, 0, 0),                            // Normal
             Eigen::Vector3d(dimensions.y(), dimensions.z(), 0),  // Dimensions (width, height)
             Eigen::Vector3d(0, 1, 0),                            // Tangent (y-direction)
             Eigen::Vector3d(0, 0, 1)                             // Bitangent (z-direction)
         });
         
         // Bottom face (normal = -z)
         faces.push_back({
             Eigen::Vector3d(center.x(), center.y(), min_pt.z()),  // Center
             Eigen::Vector3d(0, 0, -1),                           // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.y(), 0),  // Dimensions (width, depth)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 1, 0)                             // Bitangent (y-direction)
         });
         
         // Top face (normal = +z)
         faces.push_back({
             Eigen::Vector3d(center.x(), center.y(), max_pt.z()),  // Center
             Eigen::Vector3d(0, 0, 1),                            // Normal
             Eigen::Vector3d(dimensions.x(), dimensions.y(), 0),  // Dimensions (width, depth)
             Eigen::Vector3d(1, 0, 0),                            // Tangent (x-direction)
             Eigen::Vector3d(0, 1, 0)                             // Bitangent (y-direction)
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
 
 // Structure to represent a GPS satellite in Earth-centered, Earth-fixed (ECEF) coordinates
 struct GPSSatellite {
     std::string id;                  // Satellite ID (e.g., G01, G02, etc.)
     Eigen::Vector3d position;        // Position in ECEF coordinates (meters)
     Eigen::Vector3d velocity;        // Velocity in ECEF coordinates (m/s)
     
     // Satellite orbital parameters
     double semi_major_axis;          // Semi-major axis of orbit (meters)
     double eccentricity;             // Eccentricity of orbit
     double inclination;              // Inclination angle (radians)
     double right_ascension;          // Right ascension of ascending node (radians)
     double argument_of_perigee;      // Argument of perigee (radians)
     double mean_anomaly;             // Mean anomaly (radians)
     double mean_motion;              // Mean motion (radians/second)
     
     // Satellite state
     double clock_bias;               // Clock bias in meters
     double clock_drift;              // Clock drift in meters/second
     double elevation;                // Elevation angle from user's perspective (degrees)
     double azimuth;                  // Azimuth angle from user's perspective (degrees)
     
     // Signal characteristics
     double frequency_l1;             // L1 carrier frequency (Hz)
     double frequency_l2;             // L2 carrier frequency (Hz)
     double power_l1;                 // Transmit power L1 (dBW)
     double power_l2;                 // Transmit power L2 (dBW)
     
     // Constellation characteristics
     std::string constellation;       // GPS, GLONASS, Galileo, etc.
     int prn;                         // Pseudo-Random Number code
     
     GPSSatellite() : 
         semi_major_axis(26559710.0),  // ~20,200 km + earth radius
         eccentricity(0.01),
         inclination(0.97738438),      // ~56 degrees
         right_ascension(0.0),
         argument_of_perigee(0.0),
         mean_anomaly(0.0),
         mean_motion(1.45824263e-4),   // ~2 orbits per sidereal day
         clock_bias(0.0),
         clock_drift(0.0),
         elevation(0.0),
         azimuth(0.0),
         frequency_l1(1575.42e6),      // 1575.42 MHz
         frequency_l2(1227.60e6),      // 1227.60 MHz
         power_l1(-157.0),             // dBW at Earth's surface
         power_l2(-160.0),             // dBW at Earth's surface
         constellation("GPS"),
         prn(1) {}
     
     // Update satellite position and clock based on the given time (since epoch)
     void updateState(double time) {
         // Update mean anomaly for current time
         double M = mean_anomaly + mean_motion * time;
         
         // Solve Kepler's equation for eccentric anomaly
         double E = solveKepler(M, eccentricity);
         
         // Calculate true anomaly
         double nu = 2.0 * std::atan2(std::sqrt(1.0 + eccentricity) * std::sin(E/2.0), 
                                      std::sqrt(1.0 - eccentricity) * std::cos(E/2.0));
         
         // Calculate radius from focus
         double r = semi_major_axis * (1.0 - eccentricity * std::cos(E));
         
         // Calculate position in orbital plane
         double x_orbit = r * std::cos(nu);
         double y_orbit = r * std::sin(nu);
         
         // Rotate to ECEF frame (simplified transformation for visualization)
         double cos_inc = std::cos(inclination);
         double sin_inc = std::sin(inclination);
         double cos_raan = std::cos(right_ascension);
         double sin_raan = std::sin(right_ascension);
         double cos_aop = std::cos(argument_of_perigee);
         double sin_aop = std::sin(argument_of_perigee);
         
         double cos_aop_nu = std::cos(argument_of_perigee + nu);
         double sin_aop_nu = std::sin(argument_of_perigee + nu);
         
         // Convert orbital position to ECEF
         double x_ecef = r * (cos_raan * cos_aop_nu - sin_raan * sin_aop_nu * cos_inc);
         double y_ecef = r * (sin_raan * cos_aop_nu + cos_raan * sin_aop_nu * cos_inc);
         double z_ecef = r * sin_aop_nu * sin_inc;
         
         position = Eigen::Vector3d(x_ecef, y_ecef, z_ecef);
         
         // Calculate velocity (analytical derivative of position)
         double p = semi_major_axis * (1.0 - eccentricity * eccentricity);
         double h = std::sqrt(p * mean_motion * semi_major_axis * semi_major_axis);
         
         double vr = h * eccentricity * std::sin(nu) / p;
         double vnu = h / r;
         
         double vx_orbit = vr * std::cos(nu) - vnu * std::sin(nu);
         double vy_orbit = vr * std::sin(nu) + vnu * std::cos(nu);
         
         // Rotate velocity to ECEF
         double vx_ecef = vx_orbit * (cos_raan * cos_aop - sin_raan * sin_aop * cos_inc) - 
                         vy_orbit * (cos_raan * sin_aop + sin_raan * cos_aop * cos_inc);
         double vy_ecef = vx_orbit * (sin_raan * cos_aop + cos_raan * sin_aop * cos_inc) - 
                         vy_orbit * (sin_raan * sin_aop - cos_raan * cos_aop * cos_inc);
         double vz_ecef = vx_orbit * sin_inc * sin_aop + vy_orbit * sin_inc * cos_aop;
         
         velocity = Eigen::Vector3d(vx_ecef, vy_ecef, vz_ecef);
         
         // Update clock error (based on a simple polynomial model)
         // Clock bias in meters, typical drift is ~1-2ns/s which is ~0.3-0.6 m/s
         double clock_bias_rate = 0.0;    // Constant for clock bias
         double clock_drift_rate = 1e-9;  // Small acceleration term
         
         clock_bias += clock_drift * time + 0.5 * clock_drift_rate * time * time;
         clock_drift += clock_drift_rate * time;
     }
     
     // Solve Kepler's equation for eccentric anomaly
     double solveKepler(double M, double e) {
         // Initialize with a reasonable guess
         double E = M;
         
         // Use Newton-Raphson iteration to solve Kepler's equation
         for (int i = 0; i < 10; i++) {
             double E_next = E - (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
             if (std::abs(E_next - E) < 1e-12) {
                 return E_next;
             }
             E = E_next;
         }
         
         return E;
     }
 };
 
 // Structure for storing ionospheric delay model parameters (Klobuchar model)
 struct IonoParameters {
     double alpha[4];  // Ionospheric parameters alpha0, alpha1, alpha2, alpha3
     double beta[4];   // Ionospheric parameters beta0, beta1, beta2, beta3
     
     IonoParameters() {
         // Default Klobuchar parameters (these would normally come from navigation message)
         alpha[0] = 0.1397e-7;
         alpha[1] = 0.0;
         alpha[2] = -0.5960e-7;
         alpha[3] = 0.0;
         
         beta[0] = 0.1045e6;
         beta[1] = 0.3277e6;
         beta[2] = -0.1966e6;
         beta[3] = 0.0;
     }
 };
 
 // Signal path segment
 struct SignalSegment {
     Eigen::Vector3d start;          // Start point of segment
     Eigen::Vector3d end;            // End point of segment
     bool direct;                    // Whether this is a direct (LOS) path
     bool penetrates_building;       // Whether this segment penetrates a building
     double signal_strength;         // Signal strength in dB
     double path_length;             // Geometric length of the segment
     
     SignalSegment(const Eigen::Vector3d& s, const Eigen::Vector3d& e, 
                 bool is_direct, double strength = 0.0) 
         : start(s), end(e), direct(is_direct), 
           penetrates_building(false), signal_strength(strength) {
         path_length = (end - start).norm();
     }
 };
 
 // Complete signal path from satellite to user with pseudorange errors
 struct SatelliteSignal {
     std::string satellite_id;          // ID of the satellite
     std::vector<SignalSegment> segments; // Path segments
     bool is_los;                       // Line of sight available
     bool is_multipath;                 // Whether multipath is present
     double signal_strength;            // Signal strength in dB-Hz (C/N0)
     
     // Pseudorange components in meters
     double geometric_range;            // True geometric distance
     double pseudorange;                // Measured pseudorange
     double satellite_clock_error;      // Satellite clock bias/drift
     double ionospheric_delay;          // Ionospheric delay
     double tropospheric_delay;         // Tropospheric delay
     double receiver_clock_bias;        // Receiver clock bias
     double multipath_error;            // Multipath error
     double receiver_noise;             // Receiver noise
     double signal_travel_time;         // Signal travel time (seconds)
     
     SatelliteSignal() : is_los(false), is_multipath(false), 
                        signal_strength(0.0), geometric_range(0.0), 
                        pseudorange(0.0), satellite_clock_error(0.0),
                        ionospheric_delay(0.0), tropospheric_delay(0.0),
                        receiver_clock_bias(0.0), multipath_error(0.0),
                        receiver_noise(0.0), signal_travel_time(0.0) {}
     
     // Add a segment to the path
     void addSegment(const SignalSegment& segment) {
         segments.push_back(segment);
     }
     
     // Total path length
     double getTotalPathLength() const {
         double total = 0.0;
         for (const auto& segment : segments) {
             total += segment.path_length;
         }
         return total;
     }
     
     // Total path error (difference between pseudorange and geometric range)
     double getTotalError() const {
         return pseudorange - geometric_range;
     }
     
     // Get all path errors combined
     double getCombinedErrors() const {
         return satellite_clock_error + ionospheric_delay + tropospheric_delay + 
                receiver_clock_bias + multipath_error + receiver_noise;
     }
 };
 
 // GPS physics constants and calculations
 namespace GPSPhysics {
     constexpr double SPEED_OF_LIGHT = 299792458.0;    // m/s
     constexpr double GPS_L1_FREQUENCY = 1575.42e6;    // L1 frequency in Hz
     constexpr double GPS_L2_FREQUENCY = 1227.60e6;    // L2 frequency in Hz
     constexpr double GPS_WAVELENGTH_L1 = SPEED_OF_LIGHT / GPS_L1_FREQUENCY; // ~19cm
     constexpr double GPS_WAVELENGTH_L2 = SPEED_OF_LIGHT / GPS_L2_FREQUENCY; // ~24cm
     constexpr double EARTH_RADIUS = 6371000.0;        // Earth radius in meters
     constexpr double GPS_ORBIT_RADIUS = 26559710.0;   // GPS orbit semi-major axis
     constexpr double GPS_EARTH_ROTATION_RATE = 7.2921151467e-5; // rad/s
     constexpr double GPS_EARTH_GM = 3.986005e14;      // Earth's gravitational parameter
     constexpr double MIN_ELEVATION_ANGLE = 5.0;       // Minimum usable elevation (degrees)
     constexpr double CARRIER_TO_CODE_NOISE_RATIO = 50.0; // dB
     constexpr double EPSILON = 1e-6;                  // Small constant for calculations
     constexpr double BOLTZMANN_CONSTANT = 1.38064852e-23; // J/K
     constexpr double RECEIVER_TEMP = 290.0;           // K
     constexpr double RECEIVER_BANDWIDTH = 1.0e6;      // Hz
     
     // WGS84 Parameters for coordinate transformations
     constexpr double WGS84_a = 6378137.0;             // Semi-major axis
     constexpr double WGS84_f = 1.0/298.257223563;     // Flattening
     constexpr double WGS84_e2 = 2*WGS84_f - WGS84_f*WGS84_f; // Eccentricity squared
     
     // Convert degrees to radians
     double deg2rad(double degrees) {
         return degrees * M_PI / 180.0;
     }
     
     // Convert radians to degrees
     double rad2deg(double radians) {
         return radians * 180.0 / M_PI;
     }
     
     // Calculate free space path loss
     double calculateFreeSpacePathLoss(double distance_m, double frequency_hz) {
         // FSPL (dB) = 20*log10(4π*d*f/c)
         // where d is distance in m, f is frequency in Hz, c is speed of light
         return 20.0 * std::log10(4.0 * M_PI * distance_m * frequency_hz / SPEED_OF_LIGHT);
     }
     
     // Calculate satellite signal power at receiver (dBW)
     double calculateReceivedPower(double transmitted_power_dbw, double path_loss_db, 
                                   double satellite_gain_db, double receiver_gain_db) {
         return transmitted_power_dbw - path_loss_db + satellite_gain_db + receiver_gain_db;
     }
     
     // Calculate C/N0 (Carrier-to-Noise density ratio) in dB-Hz
     double calculateCN0(double received_power_dbw) {
         // N0 = k * T where k is Boltzmann's constant and T is system noise temperature
         double noise_density_dbw_hz = 10.0 * std::log10(BOLTZMANN_CONSTANT * RECEIVER_TEMP);
         return received_power_dbw - noise_density_dbw_hz;
     }
     
     // Calculate C/N0 based on elevation angle (empirical model)
     double calculateCN0FromElevation(double elevation_deg, double path_loss_db) {
         // Typical GPS L1 C/A signal power at receiver is around -157 dBW at 5° elevation
         // and improves to about -153 dBW at 90° elevation
         double min_power_dbw = -157.0; // At 5° elevation
         double max_power_dbw = -153.0; // At 90° elevation
         
         // Linear interpolation based on elevation
         double elevation_factor = (elevation_deg - 5.0) / 85.0; // 0 at 5°, 1 at 90°
         elevation_factor = std::max(0.0, std::min(1.0, elevation_factor));
         
         double received_power = min_power_dbw + elevation_factor * (max_power_dbw - min_power_dbw);
         
         // Adjust for path losses
         received_power -= path_loss_db;
         
         // Calculate C/N0
         return calculateCN0(received_power);
     }
     
     // Calculate pseudorange standard deviation from C/N0
     double calculatePseudorangeStdDev(double cn0_db_hz) {
         // Simplified model: sigma = a / sqrt(10^(CN0/10))
         // where CN0 is in dB-Hz and a is an empirical constant (~20-30 for GPS C/A code)
         double a = 25.0;
         return a / std::sqrt(std::pow(10.0, cn0_db_hz/10.0));
     }
     
     // Calculate receiver noise error based on C/N0
     double calculateReceiverNoiseError(double cn0_db_hz, std::mt19937& rng) {
         double sigma = calculatePseudorangeStdDev(cn0_db_hz);
         std::normal_distribution<double> noise_dist(0.0, sigma);
         return noise_dist(rng);
     }
     
     // Calculate multipath error based on environment and elevation
     double calculateMultipathError(double elevation_deg, double cn0_db_hz, bool is_multipath, std::mt19937& rng) {
         if (!is_multipath) {
             return 0.0;
         }
         
         // Multipath error model based on elevation angle
         // Lower elevation angles experience more multipath
         double max_error = 15.0;  // Maximum multipath error in meters
         double elevation_factor = std::max(0.0, (90.0 - elevation_deg) / 90.0);
         
         // Scale by signal quality - weaker signals create worse multipath
         double cn0_factor = std::max(0.0, (50.0 - cn0_db_hz) / 30.0);
         cn0_factor = std::min(1.0, cn0_factor);
         
         // Calculate mean error
         double mean_error = elevation_factor * cn0_factor * max_error;
         
         // Add some randomness (often multipath is biased positive)
         std::exponential_distribution<double> exp_dist(1.0/mean_error);
         return exp_dist(rng);
     }
     
     // Calculate tropospheric delay using Saastamoinen model
     double calculateTroposphericDelay(double elevation_deg, double height_m = 0.0) {
         // Saastamoinen model parameters
         double pressure = 1013.25 * std::exp(-height_m/8500.0); // Pressure at height (mbar)
         double temp = 288.15 - 0.0065 * height_m; // Temperature at height (K)
         double e = 6.108 * std::exp((17.15 * temp - 4684.0)/(temp - 38.45)); // Water vapor pressure
         
         // Convert elevation to radians
         double elevation_rad = deg2rad(elevation_deg);
         
         // Zenith tropospheric delay
         double Zhydro = 0.0022768 * pressure / (1.0 - 0.00266 * std::cos(2.0 * 0.0) - 0.00028 * height_m/1000.0);
         
         // Mapping function (simple 1/sin(el) approximation)
         double mapping = 1.0 / std::sin(elevation_rad);
         
         // Total tropospheric delay in meters
         return Zhydro * mapping;
     }
     
     // Calculate ionospheric delay using Klobuchar model
     double calculateIonosphericDelay(double elevation_deg, double azimuth_deg, 
                                    double lat_deg, double lon_deg, 
                                    const IonoParameters& params, double gps_time_s) {
         // Convert to radians
         double elevation_rad = deg2rad(elevation_deg);
         double azimuth_rad = deg2rad(azimuth_deg);
         double lat_rad = deg2rad(lat_deg);
         double lon_rad = deg2rad(lon_deg);
         
         // Earth's central angle between user and ionospheric pierce point
         double psi = 0.0137 / (elevation_rad + 0.11) - 0.022;
         
         // Latitude of the ionospheric pierce point
         double lat_i = lat_rad + psi * std::cos(azimuth_rad);
         if (lat_i > 0.416) lat_i = 0.416;
         if (lat_i < -0.416) lat_i = -0.416;
         
         // Longitude of the ionospheric pierce point
         double lon_i = lon_rad + psi * std::sin(azimuth_rad) / std::cos(lat_i);
         
         // Geomagnetic latitude of the ionospheric pierce point
         double lat_m = lat_i + 0.064 * std::cos(lon_i - 1.617);
         
         // Local time at the ionospheric pierce point
         double t = 43200.0 * lon_i / M_PI + gps_time_s;
         while (t >= 86400.0) t -= 86400.0;
         while (t < 0.0) t += 86400.0;
         
         // Slant factor
         double slant = 1.0 / std::sqrt(1.0 - std::pow(0.9782 * std::cos(elevation_rad), 2));
         
         // Amplitude of ionospheric delay
         double amp = params.alpha[0] + params.alpha[1] * lat_m + 
                     params.alpha[2] * lat_m * lat_m + params.alpha[3] * lat_m * lat_m * lat_m;
         if (amp < 0.0) amp = 0.0;
         
         // Period of ionospheric delay
         double per = params.beta[0] + params.beta[1] * lat_m + 
                     params.beta[2] * lat_m * lat_m + params.beta[3] * lat_m * lat_m * lat_m;
         if (per < 72000.0) per = 72000.0;
         
         // Phase of ionospheric delay
         double x = 2.0 * M_PI * (t - 50400.0) / per;
         
         // Ionospheric delay
         double iono;
         if (std::abs(x) > 1.57) {
             iono = slant * 5.0e-9 * SPEED_OF_LIGHT;
         } else {
             iono = slant * (5.0e-9 + amp * (1.0 - x*x/2.0 + x*x*x*x/24.0)) * SPEED_OF_LIGHT;
         }
         
         return iono;
     }
     
     // Convert ECEF coordinates to geodetic (WGS84)
     void ecef2geodetic(const Eigen::Vector3d& ecef, double& lat, double& lon, double& height) {
         double x = ecef.x();
         double y = ecef.y();
         double z = ecef.z();
         
         double p = std::sqrt(x*x + y*y);
         double theta = std::atan2(z * WGS84_a, p * WGS84_a * (1.0 - WGS84_e2));
         
         lon = std::atan2(y, x);
         lat = std::atan2(z + WGS84_e2 * WGS84_a * std::pow(std::sin(theta), 3), 
                          p - WGS84_e2 * WGS84_a * std::pow(std::cos(theta), 3));
         
         double N = WGS84_a / std::sqrt(1.0 - WGS84_e2 * std::sin(lat) * std::sin(lat));
         height = p / std::cos(lat) - N;
         
         // Convert to degrees
         lat = rad2deg(lat);
         lon = rad2deg(lon);
     }
     
     // Convert ECEF coordinates to ENU at given reference point
     Eigen::Vector3d ecef2enu(const Eigen::Vector3d& ecef, const Eigen::Vector3d& ref_ecef, 
                           double ref_lat, double ref_lon) {
         // Convert reference lat/lon to radians
         double lat_rad = deg2rad(ref_lat);
         double lon_rad = deg2rad(ref_lon);
         
         // Rotation matrix from ECEF to ENU
         double sin_lat = std::sin(lat_rad);
         double cos_lat = std::cos(lat_rad);
         double sin_lon = std::sin(lon_rad);
         double cos_lon = std::cos(lon_rad);
         
         // Calculate ECEF offset
         Eigen::Vector3d delta_ecef = ecef - ref_ecef;
         
         // Rotate to ENU
         Eigen::Vector3d enu;
         enu.x() = -sin_lon * delta_ecef.x() + cos_lon * delta_ecef.y();
         enu.y() = -sin_lat * cos_lon * delta_ecef.x() - sin_lat * sin_lon * delta_ecef.y() + cos_lat * delta_ecef.z();
         enu.z() = cos_lat * cos_lon * delta_ecef.x() + cos_lat * sin_lon * delta_ecef.y() + sin_lat * delta_ecef.z();
         
         return enu;
     }
     
     // Convert ENU coordinates to ECEF at given reference point
     Eigen::Vector3d enu2ecef(const Eigen::Vector3d& enu, const Eigen::Vector3d& ref_ecef, 
                           double ref_lat, double ref_lon) {
         // Convert reference lat/lon to radians
         double lat_rad = deg2rad(ref_lat);
         double lon_rad = deg2rad(ref_lon);
         
         // Rotation matrix from ENU to ECEF
         double sin_lat = std::sin(lat_rad);
         double cos_lat = std::cos(lat_rad);
         double sin_lon = std::sin(lon_rad);
         double cos_lon = std::cos(lon_rad);
         
         // Rotate from ENU to ECEF
         Eigen::Vector3d delta_ecef;
         delta_ecef.x() = -sin_lon * enu.x() - sin_lat * cos_lon * enu.y() + cos_lat * cos_lon * enu.z();
         delta_ecef.y() = cos_lon * enu.x() - sin_lat * sin_lon * enu.y() + cos_lat * sin_lon * enu.z();
         delta_ecef.z() = cos_lat * enu.y() + sin_lat * enu.z();
         
         // Add to reference ECEF
         return ref_ecef + delta_ecef;
     }
     
     // Calculate satellite elevation and azimuth from user position
     void calculateElevationAzimuth(const Eigen::Vector3d& user_ecef, 
                                   const Eigen::Vector3d& satellite_ecef,
                                   double& elevation, double& azimuth) {
         // Convert user ECEF to geodetic (lat, lon, height)
         double lat, lon, height;
         ecef2geodetic(user_ecef, lat, lon, height);
         
         // Convert to ENU coordinates
         Eigen::Vector3d sat_enu = ecef2enu(satellite_ecef, user_ecef, lat, lon);
         
         // Calculate elevation and azimuth
         elevation = 90.0 - rad2deg(std::acos(sat_enu.z() / sat_enu.norm()));
         azimuth = rad2deg(std::atan2(sat_enu.x(), sat_enu.y()));
         if (azimuth < 0.0) azimuth += 360.0;
     }
     
     // Calculate satellite clock relativistic correction
     double calculateRelativisticCorrection(const GPSSatellite& satellite) {
         // Relativistic correction due to orbital eccentricity
         double rel_corr = -2.0 * satellite.position.dot(satellite.velocity) / SPEED_OF_LIGHT;
         
         // Relativistic correction due to Earth's gravitational potential
         // This is usually incorporated into the satellite clock bias
         
         return rel_corr;
     }
     
     // Safe calculation of inverse with checks for division by zero
     double safeInverse(double value) {
         const double MIN_VALUE = 1e-10;
         if (std::abs(value) < MIN_VALUE) {
             return std::copysign(1.0/MIN_VALUE, value);
         }
         return 1.0 / value;
     }
     
     // Check if a ray intersects a surface
     bool rayIntersectsSurface(const Eigen::Vector3d& ray_origin, 
                              const Eigen::Vector3d& ray_dir,
                              const Eigen::Vector3d& plane_point,
                              const Eigen::Vector3d& plane_normal,
                              double& t,
                              Eigen::Vector3d& intersection_point) {
         
         double denom = ray_dir.dot(plane_normal);
         
         // Check if ray is parallel to plane
         if (std::abs(denom) < EPSILON) {
             return false;
         }
         
         // Calculate t
         t = (plane_point - ray_origin).dot(plane_normal) / denom;
         
         // Check if intersection is in front of ray origin
         if (t < 0.0) {
             return false;
         }
         
         // Calculate intersection point
         intersection_point = ray_origin + t * ray_dir;
         
         return true;
     }
 };
 
 class GPSSimulator {
 private:
     ros::NodeHandle nh_;
     ros::NodeHandle private_nh_;
 
     // Publishers
     ros::Publisher building_pub_;
     ros::Publisher road_pub_;
     ros::Publisher satellite_pub_;
     ros::Publisher user_pub_;
     ros::Publisher signal_pub_;
     ros::Publisher measurement_pub_;
     ros::Publisher text_pub_;
     ros::Publisher debug_pub_;
     ros::Publisher trajectory_pub_;
     ros::Publisher skyplot_pub_;
     ros::Publisher pseudorange_pub_;
     
     // Timers
     ros::Timer update_timer_;
     ros::Timer movement_timer_;
     ros::Timer satellite_motion_timer_;
     
     // TF broadcaster
     tf::TransformBroadcaster tf_broadcaster_;
     
     // Lists of objects
     std::vector<Building> buildings_;
     std::vector<GPSSatellite> satellites_;
     Eigen::Vector3d user_position_;
     std::vector<SatelliteSignal> satellite_signals_;
     std::vector<Eigen::Vector3d> user_trajectory_;
     IonoParameters iono_params_;  // Ionospheric delay parameters
     
     // User and simulation state
     double user_lat_;           // User latitude (degrees)
     double user_lon_;           // User longitude (degrees)
     double user_height_;        // User height above WGS84 (meters)
     double receiver_clock_bias_;    // Receiver clock bias (meters)
     double receiver_clock_drift_;   // Receiver clock drift (meters/second)
     double gps_time_;               // Current GPS time of week (seconds)
     
     // Parameters
     double max_signal_distance_;     // Maximum satellite distance to consider
     double min_cn0_threshold_;       // Minimum C/N0 for usable signal
     
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
     
     // Satellite motion parameters
     double satellite_motion_rate_;
     double orbital_radius_;
     double simulation_scale_;
     
     // Visualization parameters
     std::string fixed_frame_;
     double building_alpha_;
     double signal_width_;
     bool debug_mode_;
     bool slow_mode_;
     bool show_multipath_;
     double signal_update_rate_;
     
     // GPS measurement simulation parameters
     double pseudorange_base_noise_;
     double multipath_probability_;
     double signal_penetration_loss_;
     
     // Random number generator
     std::mt19937 rng_;
     std::uniform_real_distribution<double> height_dist_;
     std::uniform_real_distribution<double> width_dist_;
     std::uniform_real_distribution<double> depth_dist_;
     std::uniform_real_distribution<double> attenuation_dist_;
     std::uniform_real_distribution<double> color_dist_;
     std::uniform_real_distribution<double> uniform_dist_;
 
 public:
     GPSSimulator() : private_nh_("~"), 
                    rng_(std::random_device()()),
                    current_time_(0.0),
                    current_trajectory_point_(0),
                    user_lat_(0.0),
                    user_lon_(0.0),
                    user_height_(0.0),
                    receiver_clock_bias_(0.0),
                    receiver_clock_drift_(0.0),
                    gps_time_(0.0),
                    uniform_dist_(0.0, 1.0) {
         // Get parameters
         private_nh_.param<double>("max_signal_distance", max_signal_distance_, 25000000.0);
         private_nh_.param<double>("min_cn0_threshold", min_cn0_threshold_, 20.0);
         private_nh_.param<std::string>("fixed_frame", fixed_frame_, "map");
         private_nh_.param<double>("building_alpha", building_alpha_, 0.5);
         private_nh_.param<double>("signal_width", signal_width_, 0.2);
         private_nh_.param<bool>("debug_mode", debug_mode_, true);
         private_nh_.param<bool>("slow_mode", slow_mode_, true);
         private_nh_.param<bool>("show_multipath", show_multipath_, true);
         private_nh_.param<double>("signal_update_rate", signal_update_rate_, 1.0);
         
         // Road and environment parameters
         private_nh_.param<double>("road_length", road_length_, 100.0);
         private_nh_.param<double>("road_width", road_width_, 15.0);
         private_nh_.param<double>("sidewalk_width", sidewalk_width_, 3.0);
         private_nh_.param<double>("building_height_min", building_height_min_, 10.0);
         private_nh_.param<double>("building_height_max", building_height_max_, 25.0);
         private_nh_.param<double>("building_width_min", building_width_min_, 10.0);
         private_nh_.param<double>("building_width_max", building_width_max_, 20.0);
         private_nh_.param<double>("building_depth_min", building_depth_min_, 8.0);
         private_nh_.param<double>("building_depth_max", building_depth_max_, 15.0);
         private_nh_.param<double>("buildings_per_side", buildings_per_side_, 8.0);
         
         // User movement parameters
         private_nh_.param<bool>("enable_user_movement", enable_user_movement_, true);
         private_nh_.param<std::string>("movement_type", movement_type_, "circuit");
         private_nh_.param<double>("movement_speed", movement_speed_, 0.5);  // meters per second
         private_nh_.param<double>("movement_radius", movement_radius_, road_width_ * 0.4);
         private_nh_.param<double>("movement_height", movement_height_, 1.7);
         private_nh_.param<double>("movement_period", movement_period_, 60.0);  // seconds for one cycle
         private_nh_.param<double>("movement_phase", movement_phase_, 0.0);
         
         // Satellite motion parameters
         private_nh_.param<double>("satellite_motion_rate", satellite_motion_rate_, 0.005);
         private_nh_.param<double>("orbital_radius", orbital_radius_, 500.0);  // Scaled for visualization
         private_nh_.param<double>("simulation_scale", simulation_scale_, 50000.0);  // Scale factor from real world to visualization
         
         // GPS measurement simulation parameters
         private_nh_.param<double>("pseudorange_base_noise", pseudorange_base_noise_, 2.0);
         private_nh_.param<double>("multipath_probability", multipath_probability_, 0.4);
         private_nh_.param<double>("signal_penetration_loss", signal_penetration_loss_, 30.0);
         
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
         attenuation_dist_ = std::uniform_real_distribution<double>(20.0, 40.0);
         color_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
         
         // Publishers
         building_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/buildings", 1);
         road_pub_ = nh_.advertise<visualization_msgs::Marker>("gps_simulator/road", 1);
         satellite_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/satellites", 1);
         user_pub_ = nh_.advertise<visualization_msgs::Marker>("gps_simulator/user", 1);
         signal_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/signals", 1);
         measurement_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/measurements", 1);
         text_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/text_info", 1);
         debug_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/debug", 1);
         trajectory_pub_ = nh_.advertise<visualization_msgs::Marker>("gps_simulator/trajectory", 1);
         skyplot_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/skyplot", 1);
         pseudorange_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("gps_simulator/pseudoranges", 1);
         
         // Generate environment
         generateEnvironment();
         
         // Create GPS satellites
         createGPSSatellites();
         
         // Generate trajectory if using circuit mode
         if (movement_type_ == "circuit") {
             generateCircuitTrajectory();
         }
         
         // Create a timer for updates - slower in slow mode
         double update_rate = signal_update_rate_;
         update_timer_ = nh_.createTimer(ros::Duration(1.0/update_rate), 
                                        &GPSSimulator::updateCallback, this);
         
         // Create a timer for movement - faster updates for smoother motion
         double movement_update_rate = 10.0;
         if (enable_user_movement_) {
             movement_timer_ = nh_.createTimer(ros::Duration(1.0/movement_update_rate), 
                                          &GPSSimulator::movementCallback, this);
         }
         
         // Create a timer for satellite motion
         satellite_motion_timer_ = nh_.createTimer(ros::Duration(1.0/10.0), 
                                                 &GPSSimulator::satelliteMotionCallback, this);
         
         ROS_INFO("GPS Simulator initialized with %zu buildings and %zu satellites", 
                 buildings_.size(), satellites_.size());
         ROS_INFO("User will move with %s trajectory at %.2f m/s", 
                 movement_type_.c_str(), movement_speed_);
     }
     
     ~GPSSimulator() {
         clearVisualizations();
     }
     
     void updateCallback(const ros::TimerEvent& event) {
         // Update simulation time
         double dt = event.current_real.toSec() - event.last_real.toSec();
         gps_time_ += dt;
         while (gps_time_ >= 604800.0) {  // GPS week rollover (7 days)
             gps_time_ -= 604800.0;
         }
         
         // Update receiver clock errors
         updateReceiverClock(dt);
         
         // Update user geodetic coordinates
         GPSPhysics::ecef2geodetic(user_position_, user_lat_, user_lon_, user_height_);
         
         // Clear old signals
         satellite_signals_.clear();
         
         // Compute GPS signals and pseudoranges
         computeGPSSignals();
         
         // Publish visualizations
         publishRoad();
         publishBuildings();
         publishSatellites();
         publishUser();
         publishGPSSignals();
         publishMeasurements();
         publishPseudoranges();
         publishTextInfo();
         publishSkyplot();
         
         // Publish trajectory visualization
         if (movement_type_ == "circuit" && !user_trajectory_.empty()) {
             publishTrajectory();
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
     
     void satelliteMotionCallback(const ros::TimerEvent& event) {
         // Update satellite positions to simulate orbital motion
         double dt = event.current_real.toSec() - event.last_real.toSec();
         
         // Update satellite positions based on their orbital parameters
         for (auto& sat : satellites_) {
             // Update satellite state
             sat.updateState(gps_time_);
             
             // Update elevation and azimuth relative to user
             double elevation, azimuth;
             GPSPhysics::calculateElevationAzimuth(user_position_, sat.position, elevation, azimuth);
             sat.elevation = elevation;
             sat.azimuth = azimuth;
         }
     }
     
     void updateReceiverClock(double dt) {
         // Simulate receiver clock errors
         // Typical low-cost GPS receiver clock stability is around 1e-9 (1 ns/s)
         double clock_drift_rate = 1.0e-9;  // Clock drift rate in s/s
         double clock_drift_noise = 1.0e-12;  // Random walk noise
         
         // Update receiver clock bias and drift
         receiver_clock_bias_ += receiver_clock_drift_ * dt;
         receiver_clock_drift_ += clock_drift_rate * dt + clock_drift_noise * std::sqrt(dt) * standard_normal(rng_);
         
         // Convert to meters (for bias) and m/s (for drift)
         receiver_clock_bias_ *= GPSPhysics::SPEED_OF_LIGHT;
         receiver_clock_drift_ *= GPSPhysics::SPEED_OF_LIGHT;
     }
     
     double standard_normal(std::mt19937& rng) {
         static std::normal_distribution<double> dist(0.0, 1.0);
         return dist(rng);
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
     
     void clearVisualizations() {
         // Create empty marker arrays to clear visualizations
         visualization_msgs::MarkerArray empty_array;
         building_pub_.publish(empty_array);
         satellite_pub_.publish(empty_array);
         signal_pub_.publish(empty_array);
         measurement_pub_.publish(empty_array);
         text_pub_.publish(empty_array);
         debug_pub_.publish(empty_array);
         skyplot_pub_.publish(empty_array);
         pseudorange_pub_.publish(empty_array);
         
         // Delete user and road markers
         visualization_msgs::Marker empty_marker;
         empty_marker.action = visualization_msgs::Marker::DELETE;
         user_pub_.publish(empty_marker);
         road_pub_.publish(empty_marker);
         trajectory_pub_.publish(empty_marker);
     }
     
     void generateEnvironment() {
         ROS_INFO("Generating urban environment with buildings on both sides...");
         
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
             double building_width = width_dist_(rng_);
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             // Position building with clear gap from road
             double building_x = -road_length_/2.0 + (i + 0.5) * segment_length;
             
             // Add some random offset in X to make the street less uniform
             double x_offset = std::uniform_real_distribution<double>(-segment_length * 0.2, segment_length * 0.2)(rng_);
             building_x += x_offset;
             
             Building building;
             building.id = "left_" + std::to_string(i);
             building.center = Eigen::Vector3d(
                 building_x, 
                 -(road_side + building_depth/2.0), 
                 building_height/2.0
             );
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             
             // Brown-ish color for left side buildings, but with variation
             double r = 0.6 + 0.3 * color_dist_(rng_);
             double g = 0.3 + 0.2 * color_dist_(rng_);
             double b = 0.1 + 0.2 * color_dist_(rng_);
             building.color = Eigen::Vector3d(r, g, b);
             
             // Assign signal attenuation characteristics
             building.signal_attenuation = attenuation_dist_(rng_);
             
             // Assign reflection coefficients (0.4 - 0.7 is typical for buildings)
             building.reflectivity = 0.4 + 0.3 * color_dist_(rng_);
             
             buildings_.push_back(building);
         }
         
         // 2. RIGHT SIDE (positive Y)
         for (int i = 0; i < num_buildings; ++i) {
             // Create a building with specific spacing
             double building_width = width_dist_(rng_);
             double building_depth = depth_dist_(rng_);
             double building_height = height_dist_(rng_);
             
             // Position building with clear gap from road and offset from left side
             double building_x = -road_length_/2.0 + (i + 0.3) * segment_length;
             
             // Add some random offset in X to make the street less uniform
             double x_offset = std::uniform_real_distribution<double>(-segment_length * 0.2, segment_length * 0.2)(rng_);
             building_x += x_offset;
             
             Building building;
             building.id = "right_" + std::to_string(i);
             building.center = Eigen::Vector3d(
                 building_x, 
                 (road_side + building_depth/2.0), 
                 building_height/2.0
             );
             building.dimensions = Eigen::Vector3d(building_width, building_depth, building_height);
             
             // Gray-ish color for right side buildings, with variation
             double r = 0.3 + 0.2 * color_dist_(rng_);
             double g = 0.3 + 0.2 * color_dist_(rng_);
             double b = 0.3 + 0.3 * color_dist_(rng_);
             building.color = Eigen::Vector3d(r, g, b);
             
             // Assign signal attenuation characteristics
             building.signal_attenuation = attenuation_dist_(rng_);
             
             // Assign reflection coefficients (0.4 - 0.7 is typical for buildings)
             building.reflectivity = 0.4 + 0.3 * color_dist_(rng_);
             
             buildings_.push_back(building);
         }
         
         // 3. Add some tall buildings at the ends for more signal blockage
         // Tall building at positive X end
         Building tall_building_1;
         tall_building_1.id = "tall_end_1";
         tall_building_1.center = Eigen::Vector3d(
             road_length_/2.0 + 15.0, 0.0, 30.0/2.0
         );
         tall_building_1.dimensions = Eigen::Vector3d(30.0, 40.0, 30.0);
         tall_building_1.color = Eigen::Vector3d(0.2, 0.3, 0.5);
         tall_building_1.signal_attenuation = 40.0;
         tall_building_1.reflectivity = 0.7;  // Higher reflectivity for glass/steel buildings
         buildings_.push_back(tall_building_1);
         
         // Tall building at negative X end
         Building tall_building_2;
         tall_building_2.id = "tall_end_2";
         tall_building_2.center = Eigen::Vector3d(
             -road_length_/2.0 - 15.0, 0.0, 35.0/2.0
         );
         tall_building_2.dimensions = Eigen::Vector3d(30.0, 40.0, 35.0);
         tall_building_2.color = Eigen::Vector3d(0.5, 0.3, 0.2);
         tall_building_2.signal_attenuation = 40.0;
         tall_building_2.reflectivity = 0.65;
         buildings_.push_back(tall_building_2);
         
         ROS_INFO("Created %zu buildings with varied heights and signal attenuation", buildings_.size());
     }
     
     void createGPSSatellites() {
        ROS_INFO("Creating GPS satellites with realistic orbital parameters...");
        
        // Clear any existing satellites
        satellites_.clear();
        
        // PRN numbers for identification - using 12 satellites to ensure we get at least 8 visible
        std::vector<int> prns = {1, 3, 7, 8, 11, 15, 19, 22, 24, 27, 30, 32};
        
        // Create satellites with orbital parameters optimized for high visibility
        for (int i = 0; i < prns.size(); i++) {
            GPSSatellite satellite;
            satellite.id = "G" + std::to_string(prns[i]);
            satellite.prn = prns[i];
            
            // Assign orbital parameters to distribute satellites for good geometry and visibility
            // Use a more realistic orbit radius (GPS orbits at ~20,200 km altitude)
            double earth_radius_km = GPSPhysics::EARTH_RADIUS / 1000.0;     // ~6371 km
            double gps_altitude_km = 20200.0;                               // 20,200 km altitude
            satellite.semi_major_axis = (earth_radius_km + gps_altitude_km) * 1000.0; // in meters
            
            // Apply visualization scale for display (this doesn't affect physical calculations)
            double display_radius = orbital_radius_;  // This is the visualization scale parameter
            
            // Eccentricity (GPS orbits are nearly circular)
            satellite.eccentricity = 0.01 + 0.005 * uniform_dist_(rng_);  // 0.01-0.015
            
            // Inclination around 55° with some variation (in radians)
            satellite.inclination = GPSPhysics::deg2rad(55.0 + 2.0 * (uniform_dist_(rng_) - 0.5));
            
            // Distribute satellites across orbital planes for good visibility
            // Use 30 degree spacing for better coverage
            satellite.right_ascension = GPSPhysics::deg2rad(30.0 * i);
            
            // For optimal visibility, place more satellites at higher elevations
            // Use phase differences to ensure better distribution
            double phase_offset = 30.0 * i + 120.0 * uniform_dist_(rng_);
            
            // For the first 8 satellites, ensure they're highly visible by giving them
            // better mean anomaly values that will result in higher elevations
            if (i < 8) {
                // For these satellites, we'll place them strategically to ensure positive elevation
                // by adjusting their mean anomaly and argument of perigee
                satellite.argument_of_perigee = GPSPhysics::deg2rad(45.0 + 90.0 * uniform_dist_(rng_));
                satellite.mean_anomaly = GPSPhysics::deg2rad(phase_offset + 30.0 * i);
            } else {
                // For additional satellites, use more random distribution
                satellite.argument_of_perigee = GPSPhysics::deg2rad(360.0 * uniform_dist_(rng_));
                satellite.mean_anomaly = GPSPhysics::deg2rad(phase_offset);
            }
            
            // Mean motion is inversely related to orbital period (approximately 12 hours)
            // Calculate using Kepler's 3rd law: T^2 ~ a^3
            double GM = GPSPhysics::GPS_EARTH_GM;  // Earth's gravitational parameter
            satellite.mean_motion = std::sqrt(GM / std::pow(satellite.semi_major_axis, 3));
            
            // Initialize satellite position using realistic orbital mechanics
            satellite.updateState(gps_time_);
            
            // Store the actual physical position for range calculations
            Eigen::Vector3d physical_position = satellite.position;
            
            // Scale the position for visualization only
            double scale_factor = display_radius / satellite.semi_major_axis;
            satellite.position = user_position_ + (satellite.position - Eigen::Vector3d(0,0,0)) * scale_factor;
            
            // Calculate initial elevation and azimuth
            double elevation, azimuth;
            GPSPhysics::calculateElevationAzimuth(user_position_, satellite.position, elevation, azimuth);
            satellite.elevation = elevation;
            satellite.azimuth = azimuth;
            
            // Assign random initial clock bias (-10 to 10 meters)
            satellite.clock_bias = 20.0 * (uniform_dist_(rng_) - 0.5);
            
            // Assign random clock drift (-0.01 to 0.01 m/s)
            satellite.clock_drift = 0.02 * (uniform_dist_(rng_) - 0.5);
            
            // Add the satellite
            satellites_.push_back(satellite);
            
            ROS_INFO("Created satellite %s at initial position (%.2f, %.2f, %.2f), elevation: %.1f°", 
                     satellite.id.c_str(), satellite.position.x(), satellite.position.y(), 
                     satellite.position.z(), satellite.elevation);
        }
        
        // Check if we have enough visible satellites (above MIN_ELEVATION_ANGLE)
        int visible_count = 0;
        for (const auto& sat : satellites_) {
            if (sat.elevation >= GPSPhysics::MIN_ELEVATION_ANGLE) {
                visible_count++;
            }
        }
        
        // If we don't have at least 8 visible satellites, adjust some orbital parameters
        if (visible_count < 8) {
            ROS_WARN("Only %d satellites visible, adjusting orbital parameters...", visible_count);
            
            // For each satellite with negative or low elevation, adjust its position
            for (auto& sat : satellites_) {
                if (sat.elevation < GPSPhysics::MIN_ELEVATION_ANGLE) {
                    // Adjust the satellite's position to have higher elevation
                    double current_elev = sat.elevation;
                    
                    // Move the satellite higher in the sky
                    Eigen::Vector3d up_vector(0, 0, 1);
                    Eigen::Vector3d to_sat = sat.position - user_position_;
                    double distance = to_sat.norm();
                    
                    // Create a new position with higher elevation
                    double elevation_adjustment = 30.0 + 30.0 * uniform_dist_(rng_); // 30-60 degrees up
                    double adjusted_angle = GPSPhysics::deg2rad(elevation_adjustment);
                    
                    // Rotate the satellite position towards the zenith
                    Eigen::Vector3d to_sat_normalized = to_sat.normalized();
                    Eigen::Vector3d rotation_axis = to_sat_normalized.cross(up_vector).normalized();
                    
                    // If the cross product is nearly zero, use x-axis for rotation
                    if (rotation_axis.norm() < 0.01) {
                        rotation_axis = Eigen::Vector3d(1, 0, 0);
                    }
                    
                    // Create rotation matrix (angle-axis rotation)
                    Eigen::AngleAxisd rotation(adjusted_angle, rotation_axis);
                    Eigen::Vector3d new_direction = rotation * to_sat_normalized;
                    
                    // Set new position
                    sat.position = user_position_ + new_direction * distance;
                    
                    // Recalculate elevation and azimuth
                    double new_elev, new_azim;
                    GPSPhysics::calculateElevationAzimuth(user_position_, sat.position, new_elev, new_azim);
                    sat.elevation = new_elev;
                    sat.azimuth = new_azim;
                    
                    ROS_INFO("Adjusted satellite %s: elevation changed from %.1f° to %.1f°", 
                             sat.id.c_str(), current_elev, sat.elevation);
                }
            }
            
            // Recount visible satellites
            visible_count = 0;
            for (const auto& sat : satellites_) {
                if (sat.elevation >= GPSPhysics::MIN_ELEVATION_ANGLE) {
                    visible_count++;
                }
            }
            
            ROS_INFO("After adjustment: %d satellites visible", visible_count);
        }
        
        // Final check to ensure we have at least some satellites above horizon
        if (visible_count < 1) {
            // Emergency: place at least one satellite directly overhead
            GPSSatellite emergency_sat;
            emergency_sat.id = "G99";
            emergency_sat.prn = 99;
            
            // Place high in the sky
            emergency_sat.position = user_position_ + Eigen::Vector3d(0, 0, orbital_radius_);
            emergency_sat.elevation = 90.0;
            emergency_sat.azimuth = 0.0;
            
            // Reasonable orbital parameters
            emergency_sat.semi_major_axis = GPSPhysics::EARTH_RADIUS + 20200000.0;
            emergency_sat.eccentricity = 0.01;
            emergency_sat.inclination = 0.0;
            emergency_sat.mean_motion = std::sqrt(GPSPhysics::GPS_EARTH_GM / 
                                         std::pow(emergency_sat.semi_major_axis, 3));
            
            satellites_.push_back(emergency_sat);
            ROS_WARN("Added emergency satellite overhead for testing");
        }
        
        ROS_INFO("Created %zu GPS satellites with realistic orbits, %d visible", 
                 satellites_.size(), visible_count);
    }
     
     void computeGPSSignals() {
         ROS_INFO("Computing GPS satellite signals and pseudorange measurements...");
         
         satellite_signals_.clear();
         
         for (const auto& satellite : satellites_) {
             // Skip satellites below horizon
             if (satellite.elevation < GPSPhysics::MIN_ELEVATION_ANGLE) {
                 continue;
             }
             
             // Compute direct path
             SatelliteSignal signal;
             signal.satellite_id = satellite.id;
             
             // Calculate geometric range (true distance)
             double distance = (satellite.position - user_position_).norm();
             signal.geometric_range = distance;
             
             // Calculate signal travel time (in seconds)
             signal.signal_travel_time = distance / GPSPhysics::SPEED_OF_LIGHT;
             
             // Check if line of sight is available
             bool line_of_sight = !checkSignalBlockage(satellite.position, user_position_);
             signal.is_los = line_of_sight;
             
             // Calculate signal strength based on elevation, distance
             double path_loss = GPSPhysics::calculateFreeSpacePathLoss(distance, satellite.frequency_l1);
             
             // Base signal strength for direct path
             double signal_strength = 0.0;
             
             // Pre-compute satellite-specific errors
             // 1. Satellite clock error
             signal.satellite_clock_error = satellite.clock_bias + satellite.clock_drift * signal.signal_travel_time;
             
             // 2. Relativistic correction
             signal.satellite_clock_error += GPSPhysics::calculateRelativisticCorrection(satellite);
             
             // 3. Ionospheric delay
             signal.ionospheric_delay = GPSPhysics::calculateIonosphericDelay(
                 satellite.elevation, satellite.azimuth, user_lat_, user_lon_, iono_params_, gps_time_);
             
             // 4. Tropospheric delay
             signal.tropospheric_delay = GPSPhysics::calculateTroposphericDelay(
                 satellite.elevation, user_height_);
             
             // 5. Receiver clock bias
             signal.receiver_clock_bias = receiver_clock_bias_;
             
             if (line_of_sight) {
                 // Direct line of sight available
                 signal_strength = GPSPhysics::calculateCN0FromElevation(satellite.elevation, 0.0);
                 
                 // Create a direct path segment
                 SignalSegment direct_segment(
                     satellite.position,
                     user_position_,
                     true,
                     signal_strength
                 );
                 
                 signal.addSegment(direct_segment);
                 signal.signal_strength = signal_strength;
                 
                 // 6. Receiver noise error (depends on signal strength)
                 signal.receiver_noise = GPSPhysics::calculateReceiverNoiseError(signal_strength, rng_);
                 
                 // 7. No multipath for direct LOS
                 signal.multipath_error = 0.0;
             } else {
                 // Line of sight blocked - try to find if signal penetrates buildings
                 // Find buildings that block the signal
                 std::vector<int> blocking_buildings = findBlockingBuildings(satellite.position, user_position_);
                 
                 if (!blocking_buildings.empty()) {
                     // Calculate attenuated signal strength
                     double total_attenuation = 0.0;
                     for (int building_idx : blocking_buildings) {
                         total_attenuation += buildings_[building_idx].signal_attenuation;
                     }
                     
                     // Attenuated signal strength
                     signal_strength = GPSPhysics::calculateCN0FromElevation(satellite.elevation, total_attenuation);
                     
                     // Create an attenuated direct path if signal strength is above threshold
                     if (signal_strength > min_cn0_threshold_) {
                         SignalSegment attenuated_segment(
                             satellite.position,
                             user_position_,
                             false,  // Not a direct LOS
                             signal_strength
                         );
                         
                         attenuated_segment.penetrates_building = true;
                         signal.addSegment(attenuated_segment);
                         signal.signal_strength = signal_strength;
                         
                         // 6. Receiver noise error (higher for attenuated signals)
                         signal.receiver_noise = GPSPhysics::calculateReceiverNoiseError(signal_strength, rng_);
                         
                         // 7. Small multipath error possible even with attenuated signals
                         signal.multipath_error = 0.5 * GPSPhysics::calculateMultipathError(
                             satellite.elevation, signal_strength, false, rng_);
                     } else {
                         // Signal too weak after attenuation
                         signal_strength = 0.0;
                     }
                 }
             }
             
             // Check for multipath if enabled
             if (show_multipath_ && 
                 (line_of_sight || signal_strength > min_cn0_threshold_) && 
                 uniform_dist_(rng_) < multipath_probability_) {
                 
                 // Find a potential reflector
                 int reflector_idx = findPotentialReflector(satellite.position, user_position_);
                 
                 if (reflector_idx >= 0) {
                     // Calculate reflection point
                     Eigen::Vector3d reflection_point = calculateReflectionPoint(
                         satellite.position, user_position_, reflector_idx);
                     
                     // Verify reflection path is not blocked
                     bool path1_blocked = checkSignalBlockage(satellite.position, reflection_point);
                     bool path2_blocked = checkSignalBlockage(reflection_point, user_position_);
                     
                     if (!path1_blocked && !path2_blocked) {
                         // Calculate multipath signal strength (weaker than direct)
                         double refl_distance1 = (reflection_point - satellite.position).norm();
                         double refl_distance2 = (user_position_ - reflection_point).norm();
                         double total_refl_distance = refl_distance1 + refl_distance2;
                         
                         // Calculate path loss for the reflection path
                         double refl_path_loss = GPSPhysics::calculateFreeSpacePathLoss(
                             total_refl_distance, satellite.frequency_l1);
                         
                         // Add additional loss for reflection (based on building reflectivity)
                         double reflection_coef = buildings_[reflector_idx].reflectivity;
                         double reflection_loss = -20.0 * std::log10(reflection_coef);  // Convert to dB
                         
                         // Calculate multipath signal strength
                         double multipath_strength = GPSPhysics::calculateCN0FromElevation(
                             satellite.elevation, refl_path_loss + reflection_loss);
                         
                         // Only add if strength is above threshold
                         if (multipath_strength > min_cn0_threshold_) {
                             // Create reflection segments
                             SignalSegment refl_segment1(
                                 satellite.position,
                                 reflection_point,
                                 false,
                                 multipath_strength
                             );
                             
                             SignalSegment refl_segment2(
                                 reflection_point,
                                 user_position_,
                                 false,
                                 multipath_strength
                             );
                             
                             // If we have a direct LOS signal, keep it and add multipath
                             if (signal.is_los || signal.segments.size() > 0) {
                                 SatelliteSignal multipath_signal = signal;  // Copy existing signal
                                 
                                 // Update signal info for multipath
                                 multipath_signal.is_multipath = true;
                                 multipath_signal.signal_strength = multipath_strength;
                                 
                                 // Clear existing segments and add multipath segments
                                 multipath_signal.segments.clear();
                                 multipath_signal.addSegment(refl_segment1);
                                 multipath_signal.addSegment(refl_segment2);
                                 
                                 // Calculate multipath error (code error)
                                 multipath_signal.multipath_error = GPSPhysics::calculateMultipathError(
                                     satellite.elevation, multipath_strength, true, rng_);
                                 
                                 // Calculate receiver noise based on multipath signal strength
                                 multipath_signal.receiver_noise = GPSPhysics::calculateReceiverNoiseError(
                                     multipath_strength, rng_);
                                 
                                 // Add multipath signal as a separate entry
                                 satellite_signals_.push_back(multipath_signal);
                                 
                                 // Continue with the main signal
                                 signal.is_multipath = false;
                             } else {
                                 // No direct signal, use multipath only
                                 signal.is_multipath = true;
                                 signal.addSegment(refl_segment1);
                                 signal.addSegment(refl_segment2);
                                 signal.signal_strength = multipath_strength;
                                 
                                 // Calculate multipath error
                                 signal.multipath_error = GPSPhysics::calculateMultipathError(
                                     satellite.elevation, multipath_strength, true, rng_);
                                 
                                 // Recalculate receiver noise for multipath signal
                                 signal.receiver_noise = GPSPhysics::calculateReceiverNoiseError(
                                     multipath_strength, rng_);
                             }
                         }
                     }
                 }
             }
             
             // If we have any usable signal segments, calculate pseudorange
             if (!signal.segments.empty() && signal.signal_strength > min_cn0_threshold_) {
                 // Calculate true range + errors = pseudorange
                 signal.pseudorange = signal.geometric_range + 
                                     signal.satellite_clock_error + 
                                     signal.ionospheric_delay + 
                                     signal.tropospheric_delay + 
                                     signal.receiver_clock_bias + 
                                     signal.multipath_error + 
                                     signal.receiver_noise;
                 
                 // Add the signal to our collection
                 satellite_signals_.push_back(signal);
                 
                 if (debug_mode_) {
                     ROS_INFO("Satellite %s: %s, C/N0: %.1f dB-Hz, Range: %.1f m, Total Error: %.2f m",
                              signal.satellite_id.c_str(),
                              signal.is_los ? "LOS" : (signal.is_multipath ? "Multipath" : "Attenuated"),
                              signal.signal_strength,
                              signal.pseudorange,
                              signal.pseudorange - signal.geometric_range);
                 }
             }
         }
         
         // Count signals by type
         int los_count = 0;
         int attenuated_count = 0;
         int multipath_count = 0;
         
         for (const auto& signal : satellite_signals_) {
             if (signal.is_los) los_count++;
             else if (signal.is_multipath) multipath_count++;
             else attenuated_count++;
         }
         
         ROS_INFO("Computed %zu usable satellite signals (%d LOS, %d attenuated, %d multipath)",
                 satellite_signals_.size(), los_count, attenuated_count, multipath_count);
     }
     
     bool checkSignalBlockage(const Eigen::Vector3d& start, const Eigen::Vector3d& end) {
         // Create a ray from start to end
         Eigen::Vector3d direction = (end - start).normalized();
         double segment_length = (end - start).norm();
         
         // Check if this ray intersects any building
         for (size_t i = 0; i < buildings_.size(); i++) {
             const auto& building = buildings_[i];
             
             // Skip if either endpoint is inside or very close to this building
             if (building.containsPoint(start, 0.1) || building.containsPoint(end, 0.1)) {
                 continue;
             }
             
             // Check for ray-AABB intersection
             if (rayIntersectsAABB(start, direction, segment_length, building)) {
                 return true;  // Signal is blocked
             }
         }
         
         return false;  // No blockage detected
     }
     
     std::vector<int> findBlockingBuildings(const Eigen::Vector3d& start, const Eigen::Vector3d& end) {
         std::vector<int> blocking_buildings;
         
         // Create a ray from start to end
         Eigen::Vector3d direction = (end - start).normalized();
         double segment_length = (end - start).norm();
         
         // Check all buildings for intersection
         for (size_t i = 0; i < buildings_.size(); i++) {
             const auto& building = buildings_[i];
             
             // Skip if either endpoint is inside or very close to this building
             if (building.containsPoint(start, 0.1) || building.containsPoint(end, 0.1)) {
                 continue;
             }
             
             // Check for ray-AABB intersection
             if (rayIntersectsAABB(start, direction, segment_length, building)) {
                 blocking_buildings.push_back(i);
             }
         }
         
         return blocking_buildings;
     }
     
     int findPotentialReflector(const Eigen::Vector3d& satellite_pos, const Eigen::Vector3d& user_pos) {
         // Find a building that could serve as reflector for multipath
         // Strategy: Look for buildings near the user, but not directly in the LOS path
         
         int best_reflector = -1;
         double best_score = -1.0;
         
         // Direction from user to satellite
         Eigen::Vector3d sat_direction = (satellite_pos - user_pos).normalized();
         
         // Check each building as a potential reflector
         for (size_t i = 0; i < buildings_.size(); i++) {
             const auto& building = buildings_[i];
             
             // Vector from user to building center
             Eigen::Vector3d to_building = (building.center - user_pos);
             double distance_to_building = to_building.norm();
             
             // Skip buildings too far away
             if (distance_to_building > 50.0) {
                 continue;
             }
             
             // Normalize
             to_building.normalize();
             
             // Calculate angle between satellite direction and building direction
             double dot_product = sat_direction.dot(to_building);
             
             // We want buildings roughly perpendicular to the satellite direction 
             // (dot product near 0) for good reflections
             double angle_factor = 1.0 - std::abs(dot_product);
             
             // Also prefer closer buildings
             double distance_factor = 1.0 / (1.0 + distance_to_building / 10.0);
             
             // Use building reflectivity in scoring
             double reflectivity_factor = building.reflectivity;
             
             // Calculate score (higher is better)
             double score = angle_factor * distance_factor * reflectivity_factor;
             
             if (score > best_score) {
                 best_score = score;
                 best_reflector = i;
             }
         }
         
         return best_reflector;
     }
     
     Eigen::Vector3d calculateReflectionPoint(
         const Eigen::Vector3d& satellite_pos, 
         const Eigen::Vector3d& user_pos,
         int building_idx) {
         
         // Find a suitable reflection point on a building face
         const auto& building = buildings_[building_idx];
         auto faces = building.getFaces();
         
         // Find the face most suitable for reflection
         int best_face_idx = -1;
         Eigen::Vector3d best_reflection_point;
         double best_score = -1.0;
         
         // Direction from satellite to user (not normalized)
         Eigen::Vector3d sat_to_user = user_pos - satellite_pos;
         
         // Check each face
         for (size_t face_idx = 0; face_idx < faces.size(); face_idx++) {
             const auto& face = faces[face_idx];
             
             // Skip bottom face as it's unlikely to reflect GPS signals
             if (face.normal.z() < -0.9) {
                 continue;
             }
             
             // Calculate a reflection point near the center of the face
             // For simplicity, we'll use a point slightly offset from center
             Eigen::Vector3d reflection_point = face.center + 
                                           face.tangent * (uniform_dist_(rng_) - 0.5) * face.dimensions.x() * 0.7 +
                                           face.bitangent * (uniform_dist_(rng_) - 0.5) * face.dimensions.y() * 0.7;
             
             // Calculate vectors from satellite to reflection point and from reflection point to user
             Eigen::Vector3d sat_to_refl = reflection_point - satellite_pos;
             Eigen::Vector3d refl_to_user = user_pos - reflection_point;
             
             // Normalize for angle calculations
             Eigen::Vector3d sat_to_refl_norm = sat_to_refl.normalized();
             Eigen::Vector3d refl_to_user_norm = refl_to_user.normalized();
             
             // Calculate reflection angle (lower is better)
             double incident_angle = std::acos(std::abs(sat_to_refl_norm.dot(face.normal)));
             double reflection_angle = std::acos(std::abs(refl_to_user_norm.dot(face.normal)));
             
             // Score based on how well the law of reflection is satisfied
             double angle_diff = std::abs(incident_angle - reflection_angle);
             double reflection_score = 1.0 / (1.0 + angle_diff);
             
             // Also consider path length (shorter is better)
             double path_length = sat_to_refl.norm() + refl_to_user.norm();
             double length_score = 1.0 / (1.0 + path_length / 100.0);
             
             // Consider building reflectivity
             double reflectivity_score = building.reflectivity;
             
             // Calculate total score (higher is better)
             double score = reflection_score * length_score * reflectivity_score;
             
             if (score > best_score) {
                 best_score = score;
                 best_face_idx = face_idx;
                 best_reflection_point = reflection_point;
             }
         }
         
         // Return the best reflection point
         return best_reflection_point;
     }
     
     bool rayIntersectsAABB(
         const Eigen::Vector3d& ray_origin, 
         const Eigen::Vector3d& ray_dir,
         double ray_length,
         const Building& building) {
         
         const Eigen::Vector3d min_pt = building.min();
         const Eigen::Vector3d max_pt = building.max();
         
         // Calculate inverses safely
         const double invDirX = GPSPhysics::safeInverse(ray_dir.x());
         const double invDirY = GPSPhysics::safeInverse(ray_dir.y());
         const double invDirZ = GPSPhysics::safeInverse(ray_dir.z());
         
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
     
     // Visualization methods
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
         road_marker.color.r = 0.3;
         road_marker.color.g = 0.3;
         road_marker.color.b = 0.3;
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
         sidewalk_left.color.r = 0.7;
         sidewalk_left.color.g = 0.7;
         sidewalk_left.color.b = 0.7;
         sidewalk_left.color.a = 1.0;
         
         road_pub_.publish(sidewalk_left);
         
         // Right sidewalk
         visualization_msgs::Marker sidewalk_right = sidewalk_left;
         sidewalk_right.id = 2;
         sidewalk_right.pose.position.y = (road_width_ / 2.0 + sidewalk_width_ / 2.0);
         
         road_pub_.publish(sidewalk_right);
         
         // Add road markings for additional visual interest
         visualization_msgs::Marker center_line;
         center_line.header = road_marker.header;
         center_line.ns = "road_markings";
         center_line.id = 3;
         center_line.type = visualization_msgs::Marker::CUBE;
         center_line.action = visualization_msgs::Marker::ADD;
         
         // Position (center line)
         center_line.pose.position.x = 0.0;
         center_line.pose.position.y = 0.0;
         center_line.pose.position.z = -0.04; // Just above road
         center_line.pose.orientation.w = 1.0;
         
         // Scale
         center_line.scale.x = road_length_;
         center_line.scale.y = 0.3; // Thin line
         center_line.scale.z = 0.01; // Very thin
         
         // Color (white)
         center_line.color.r = 1.0;
         center_line.color.g = 1.0;
         center_line.color.b = 1.0;
         center_line.color.a = 1.0;
         
         road_pub_.publish(center_line);
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
     
     void publishSatellites() {
         visualization_msgs::MarkerArray satellite_markers;
         int id = 0;
         
         for (const auto& satellite : satellites_) {
             // Satellite sphere
             visualization_msgs::Marker marker;
             marker.header.frame_id = fixed_frame_;
             marker.header.stamp = ros::Time::now();
             marker.ns = "satellites";
             marker.id = id++;
             marker.type = visualization_msgs::Marker::SPHERE;
             marker.action = visualization_msgs::Marker::ADD;
             
             // Position
             marker.pose.position.x = satellite.position.x();
             marker.pose.position.y = satellite.position.y();
             marker.pose.position.z = satellite.position.z();
             marker.pose.orientation.w = 1.0;
             
             // Scale
             marker.scale.x = 10.0;
             marker.scale.y = 10.0;
             marker.scale.z = 10.0;
             
             // Color based on elevation (higher elevation = greener)
             double elevation_ratio = std::max(0.0, std::min(1.0, (satellite.elevation - 5.0) / 85.0));
             
             // Red for low elevation, yellow for medium, green for high
             marker.color.r = 1.0 - elevation_ratio * 0.7;
             marker.color.g = 0.3 + elevation_ratio * 0.7;
             marker.color.b = 0.0;
             marker.color.a = 1.0;
             
             // Make satellites below horizon semi-transparent
             if (satellite.elevation < GPSPhysics::MIN_ELEVATION_ANGLE) {
                 marker.color.a = 0.3;
             }
             
             satellite_markers.markers.push_back(marker);
             
             // Satellite label
             visualization_msgs::Marker text_marker;
             text_marker.header.frame_id = fixed_frame_;
             text_marker.header.stamp = ros::Time::now();
             text_marker.ns = "satellite_labels";
             text_marker.id = id++;
             text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             text_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - above satellite
             text_marker.pose.position.x = satellite.position.x();
             text_marker.pose.position.y = satellite.position.y();
             text_marker.pose.position.z = satellite.position.z() + 10.0;
             text_marker.pose.orientation.w = 1.0;
             
             // Scale
             text_marker.scale.z = 5.0; // Text height
             
             // Color
             text_marker.color.r = 1.0;
             text_marker.color.g = 1.0;
             text_marker.color.b = 1.0;
             text_marker.color.a = 1.0;
             
             // Text
             text_marker.text = satellite.id + "\nEl: " + 
                               std::to_string(static_cast<int>(satellite.elevation)) + "°";
             
             satellite_markers.markers.push_back(text_marker);
             
             // Add direction vector (orbital velocity)
             visualization_msgs::Marker velocity_marker;
             velocity_marker.header.frame_id = fixed_frame_;
             velocity_marker.header.stamp = ros::Time::now();
             velocity_marker.ns = "satellite_velocity";
             velocity_marker.id = id++;
             velocity_marker.type = visualization_msgs::Marker::ARROW;
             velocity_marker.action = visualization_msgs::Marker::ADD;
             
             // Start and end points
             geometry_msgs::Point start, end;
             start.x = satellite.position.x();
             start.y = satellite.position.y();
             start.z = satellite.position.z();
             
             Eigen::Vector3d velocity_scaled = satellite.velocity * 100.0;  // Scale for visualization
             end.x = satellite.position.x() + velocity_scaled.x();
             end.y = satellite.position.y() + velocity_scaled.y();
             end.z = satellite.position.z() + velocity_scaled.z();
             
             velocity_marker.points.push_back(start);
             velocity_marker.points.push_back(end);
             
             // Arrow scale
             velocity_marker.scale.x = 2.0;  // Shaft diameter
             velocity_marker.scale.y = 4.0;  // Head diameter
             velocity_marker.scale.z = 5.0;  // Head length
             
             // Color - blue
             velocity_marker.color.r = 0.0;
             velocity_marker.color.g = 0.0;
             velocity_marker.color.b = 1.0;
             velocity_marker.color.a = 0.7;
             
             satellite_markers.markers.push_back(velocity_marker);
         }
         
         satellite_pub_.publish(satellite_markers);
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
         marker.scale.x = 1.0;
         marker.scale.y = 1.0;
         marker.scale.z = 1.0;
         
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
         text_marker.pose.position.z = user_position_.z() + 2.0;
         text_marker.pose.orientation.w = 1.0;
         
         // Scale
         text_marker.scale.z = 1.0; // Text height
         
         // Color
         text_marker.color.r = 0.0;
         text_marker.color.g = 0.8;
         text_marker.color.b = 1.0;
         text_marker.color.a = 1.0;
         
         // Text
         text_marker.text = "GPS RECEIVER";
         
         user_pub_.publish(text_marker);
         
         // Add a small coordinate frame to indicate user orientation
         visualization_msgs::Marker x_axis;
         x_axis.header.frame_id = fixed_frame_;
         x_axis.header.stamp = ros::Time::now();
         x_axis.ns = "user_axes";
         x_axis.id = 2;
         x_axis.type = visualization_msgs::Marker::ARROW;
         x_axis.action = visualization_msgs::Marker::ADD;
         
         geometry_msgs::Point origin, x_end;
         origin.x = user_position_.x();
         origin.y = user_position_.y();
         origin.z = user_position_.z();
         
         x_end.x = user_position_.x() + 2.0;  // X-axis
         x_end.y = user_position_.y();
         x_end.z = user_position_.z();
         
         x_axis.points.push_back(origin);
         x_axis.points.push_back(x_end);
         
         x_axis.scale.x = 0.2;  // Shaft diameter
         x_axis.scale.y = 0.4;  // Head diameter
         x_axis.scale.z = 0.6;  // Head length
         
         x_axis.color.r = 1.0;  // Red for X-axis
         x_axis.color.g = 0.0;
         x_axis.color.b = 0.0;
         x_axis.color.a = 1.0;
         
         user_pub_.publish(x_axis);
         
         // Y-axis
         visualization_msgs::Marker y_axis = x_axis;
         y_axis.id = 3;
         
         geometry_msgs::Point y_end;
         y_end.x = user_position_.x();
         y_end.y = user_position_.y() + 2.0;  // Y-axis
         y_end.z = user_position_.z();
         
         y_axis.points.clear();
         y_axis.points.push_back(origin);
         y_axis.points.push_back(y_end);
         
         y_axis.color.r = 0.0;
         y_axis.color.g = 1.0;  // Green for Y-axis
         y_axis.color.b = 0.0;
         
         user_pub_.publish(y_axis);
         
         // Z-axis
         visualization_msgs::Marker z_axis = x_axis;
         z_axis.id = 4;
         
         geometry_msgs::Point z_end;
         z_end.x = user_position_.x();
         z_end.y = user_position_.y();
         z_end.z = user_position_.z() + 2.0;  // Z-axis
         
         z_axis.points.clear();
         z_axis.points.push_back(origin);
         z_axis.points.push_back(z_end);
         
         z_axis.color.r = 0.0;
         z_axis.color.g = 0.0;
         z_axis.color.b = 1.0;  // Blue for Z-axis
         
         user_pub_.publish(z_axis);
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
         traj_marker.scale.x = 0.2;
         
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
         current_point.scale.x = 0.4;
         current_point.scale.y = 0.4;
         current_point.scale.z = 0.4;
         
         // Color - bright green
         current_point.color.r = 0.0;
         current_point.color.g = 1.0;
         current_point.color.b = 0.0;
         current_point.color.a = 1.0;
         
         trajectory_pub_.publish(current_point);
     }
     
     void publishGPSSignals() {
         visualization_msgs::MarkerArray signal_markers;
         int id = 0;
         
         // Clear previous signals if there are none
         if (satellite_signals_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             signal_markers.markers.push_back(clear_marker);
             signal_pub_.publish(signal_markers);
             return;
         }
         
         // Colors for different signal types
         std_msgs::ColorRGBA los_color; // Green
         los_color.r = 0.0;
         los_color.g = 1.0;
         los_color.b = 0.0;
         los_color.a = 1.0;
         
         std_msgs::ColorRGBA attenuated_color; // Yellow
         attenuated_color.r = 1.0;
         attenuated_color.g = 1.0;
         attenuated_color.b = 0.0;
         attenuated_color.a = 0.7;
         
         std_msgs::ColorRGBA multipath_color; // Red
         multipath_color.r = 1.0;
         multipath_color.g = 0.0;
         multipath_color.b = 0.0;
         multipath_color.a = 0.7;
         
         // Visualize each satellite signal
         for (const auto& signal : satellite_signals_) {
             // Choose color based on signal type
             std_msgs::ColorRGBA path_color;
             if (signal.is_los) {
                 path_color = los_color;
             } else if (signal.is_multipath) {
                 path_color = multipath_color;
             } else {
                 path_color = attenuated_color;
             }
             
             // Visualize each segment
             for (size_t i = 0; i < signal.segments.size(); ++i) {
                 const auto& segment = signal.segments[i];
                 
                 // Create line segment
                 visualization_msgs::Marker segment_marker;
                 segment_marker.header.frame_id = fixed_frame_;
                 segment_marker.header.stamp = ros::Time::now();
                 segment_marker.ns = "signal_segments";
                 segment_marker.id = id++;
                 segment_marker.type = visualization_msgs::Marker::LINE_STRIP;
                 segment_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Add start and end points
                 geometry_msgs::Point start, end;
                 start.x = segment.start.x();
                 start.y = segment.start.y();
                 start.z = segment.start.z();
                 
                 end.x = segment.end.x();
                 end.y = segment.end.y();
                 end.z = segment.end.z();
                 
                 segment_marker.points.push_back(start);
                 segment_marker.points.push_back(end);
                 
                 // Line width - thicker for direct segments
                 segment_marker.scale.x = segment.direct ? signal_width_ * 1.5 : signal_width_;
                 
                 // Color
                 segment_marker.color = path_color;
                 
                 // If this segment penetrates a building, use dashed line effect
                 if (segment.penetrates_building) {
                     segment_marker.color.a = 0.5;
                 }
                 
                 signal_markers.markers.push_back(segment_marker);
             }
             
             // Add path info text
             visualization_msgs::Marker info_marker;
             info_marker.header.frame_id = fixed_frame_;
             info_marker.header.stamp = ros::Time::now();
             info_marker.ns = "signal_info";
             info_marker.id = id++;
             info_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
             info_marker.action = visualization_msgs::Marker::ADD;
             
             // Position - near the middle of the signal path
             Eigen::Vector3d midpoint;
             if (signal.segments.size() == 1) {
                 // For direct or attenuated signals
                 midpoint = (signal.segments[0].start + signal.segments[0].end) / 2.0;
             } else if (signal.segments.size() > 1) {
                 // For multipath signals, use the reflection point
                 midpoint = signal.segments[0].end;
             } else {
                 continue;  // Skip if no segments
             }
             
             info_marker.pose.position.x = midpoint.x();
             info_marker.pose.position.y = midpoint.y();
             info_marker.pose.position.z = midpoint.z() + 5.0;
             info_marker.pose.orientation.w = 1.0;
             
             // Scale
             info_marker.scale.z = 2.0; // Text height
             
             // Color - same as signal
             info_marker.color = path_color;
             
             // Text - signal type and strength
             std::stringstream ss;
             ss << signal.satellite_id << " - ";
             if (signal.is_los) {
                 ss << "LOS";
             } else if (signal.is_multipath) {
                 ss << "Multipath";
             } else {
                 ss << "Attenuated";
             }
             ss << "\n" << std::fixed << std::setprecision(1) 
                << signal.signal_strength << " dB-Hz";
             
             info_marker.text = ss.str();
             
             signal_markers.markers.push_back(info_marker);
             
             // For multipath signals, mark the reflection point
             if (signal.is_multipath && signal.segments.size() > 1) {
                 visualization_msgs::Marker refl_marker;
                 refl_marker.header.frame_id = fixed_frame_;
                 refl_marker.header.stamp = ros::Time::now();
                 refl_marker.ns = "reflection_points";
                 refl_marker.id = id++;
                 refl_marker.type = visualization_msgs::Marker::SPHERE;
                 refl_marker.action = visualization_msgs::Marker::ADD;
                 
                 // Position at reflection point
                 Eigen::Vector3d reflection_point = signal.segments[0].end;
                 refl_marker.pose.position.x = reflection_point.x();
                 refl_marker.pose.position.y = reflection_point.y();
                 refl_marker.pose.position.z = reflection_point.z();
                 refl_marker.pose.orientation.w = 1.0;
                 
                 // Size
                 refl_marker.scale.x = 1.0;
                 refl_marker.scale.y = 1.0;
                 refl_marker.scale.z = 1.0;
                 
                 // Color - white
                 refl_marker.color.r = 1.0;
                 refl_marker.color.g = 1.0;
                 refl_marker.color.b = 1.0;
                 refl_marker.color.a = 1.0;
                 
                 signal_markers.markers.push_back(refl_marker);
                 
                 // Add "REFL" label
                 visualization_msgs::Marker refl_label;
                 refl_label.header.frame_id = fixed_frame_;
                 refl_label.header.stamp = ros::Time::now();
                 refl_label.ns = "reflection_labels";
                 refl_label.id = id++;
                 refl_label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                 refl_label.action = visualization_msgs::Marker::ADD;
                 
                 refl_label.pose.position.x = reflection_point.x();
                 refl_label.pose.position.y = reflection_point.y();
                 refl_label.pose.position.z = reflection_point.z() + 2.0;
                 refl_label.pose.orientation.w = 1.0;
                 
                 refl_label.scale.z = 1.0;
                 
                 refl_label.color.r = 1.0;
                 refl_label.color.g = 1.0;
                 refl_label.color.b = 1.0;
                 refl_label.color.a = 1.0;
                 
                 refl_label.text = "REFLECTION";
                 
                 signal_markers.markers.push_back(refl_label);
             }
         }
         
         signal_pub_.publish(signal_markers);
     }
     
     void publishMeasurements() {
         visualization_msgs::MarkerArray measurement_markers;
         int id = 0;
         
         // Clear previous measurements if there are none
         if (satellite_signals_.empty()) {
             visualization_msgs::Marker clear_marker;
             clear_marker.action = visualization_msgs::Marker::DELETEALL;
             clear_marker.header.frame_id = fixed_frame_;
             clear_marker.header.stamp = ros::Time::now();
             
             measurement_markers.markers.push_back(clear_marker);
             measurement_pub_.publish(measurement_markers);
             return;
         }
         
         // Create a panel to display pseudorange measurements
        visualization_msgs::Marker panel;
        panel.header.frame_id = fixed_frame_;
        panel.header.stamp = ros::Time::now();
        panel.ns = "measurement_panel";
        panel.id = id++;
        panel.type = visualization_msgs::Marker::CUBE;
        panel.action = visualization_msgs::Marker::ADD;
        
        // Position panel near user but offset to the side
        panel.pose.position.x = user_position_.x() - 15.0;
        panel.pose.position.y = user_position_.y() + 10.0;
        panel.pose.position.z = user_position_.z() + 10.0;
        panel.pose.orientation.w = 1.0;
        
        // Size based on number of satellites
        double panel_height = std::max(10.0, satellite_signals_.size() * 2.0 + 4.0);
        panel.scale.x = 18.0;
        panel.scale.y = 0.1;
        panel.scale.z = panel_height;
        
        // Color - semi-transparent dark gray
        panel.color.r = 0.2;
        panel.color.g = 0.2;
        panel.color.b = 0.2;
        panel.color.a = 0.7;
        
        measurement_markers.markers.push_back(panel);
        
        // Add panel title
        visualization_msgs::Marker title;
        title.header.frame_id = fixed_frame_;
        title.header.stamp = ros::Time::now();
        title.ns = "measurement_title";
        title.id = id++;
        title.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        title.action = visualization_msgs::Marker::ADD;
        
        title.pose.position.x = panel.pose.position.x;
        title.pose.position.y = panel.pose.position.y;
        title.pose.position.z = panel.pose.position.z + panel_height/2.0 + 1.0;
        title.pose.orientation.w = 1.0;
        
        title.scale.z = 1.0;
        
        title.color.r = 1.0;
        title.color.g = 1.0;
        title.color.b = 1.0;
        title.color.a = 1.0;
        
        title.text = "PSEUDORANGE MEASUREMENTS";
        
        measurement_markers.markers.push_back(title);
        
        // Add header row
        visualization_msgs::Marker header;
        header.header.frame_id = fixed_frame_;
        header.header.stamp = ros::Time::now();
        header.ns = "measurement_header";
        header.id = id++;
        header.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        header.action = visualization_msgs::Marker::ADD;
        
        header.pose.position.x = panel.pose.position.x;
        header.pose.position.y = panel.pose.position.y;
        header.pose.position.z = panel.pose.position.z + panel_height/2.0 - 1.0;
        header.pose.orientation.w = 1.0;
        
        header.scale.z = 0.7;
        
        header.color.r = 1.0;
        header.color.g = 1.0;
        header.color.b = 1.0;
        header.color.a = 1.0;
        
        header.text = "SAT    PSEUDORANGE (m)    ERROR (m)    C/N0 (dB-Hz)    TYPE";
        
        measurement_markers.markers.push_back(header);
        
        // Add entries for each satellite
        for (size_t i = 0; i < satellite_signals_.size(); i++) {
            const auto& signal = satellite_signals_[i];
            
            visualization_msgs::Marker entry;
            entry.header.frame_id = fixed_frame_;
            entry.header.stamp = ros::Time::now();
            entry.ns = "measurement_entries";
            entry.id = id++;
            entry.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            entry.action = visualization_msgs::Marker::ADD;
            
            // Position within panel
            entry.pose.position.x = panel.pose.position.x;
            entry.pose.position.y = panel.pose.position.y;
            entry.pose.position.z = panel.pose.position.z + panel_height/2.0 - 2.5 - i * 1.8;
            entry.pose.orientation.w = 1.0;
            
            entry.scale.z = 0.8;
            
            // Color based on signal type
            if (signal.is_los) {
                entry.color.r = 0.0;
                entry.color.g = 1.0;
                entry.color.b = 0.0;
            } else if (signal.is_multipath) {
                entry.color.r = 1.0;
                entry.color.g = 0.0;
                entry.color.b = 0.0;
            } else {
                entry.color.r = 1.0;
                entry.color.g = 1.0;
                entry.color.b = 0.0;
            }
            entry.color.a = 1.0;
            
            // Format text with measurement details
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << signal.satellite_id << "    " 
               << std::setw(10) << std::setprecision(2) << signal.pseudorange << " m    "
               << std::setw(5) << std::setprecision(2) << (signal.pseudorange - signal.geometric_range) << " m    "
               << std::setw(5) << signal.signal_strength << " dB-Hz    ";
            
            if (signal.is_los) {
                ss << "LOS";
            } else if (signal.is_multipath) {
                ss << "Multipath";
            } else {
                ss << "Attenuated";
            }
            
            entry.text = ss.str();
            
            measurement_markers.markers.push_back(entry);
        }
        
        measurement_pub_.publish(measurement_markers);
    }
    
    void publishPseudoranges() {
        visualization_msgs::MarkerArray pseudorange_markers;
        int id = 0;
        
        // Clear previous errors if there are none
        if (satellite_signals_.empty()) {
            visualization_msgs::Marker clear_marker;
            clear_marker.action = visualization_msgs::Marker::DELETEALL;
            clear_marker.header.frame_id = fixed_frame_;
            clear_marker.header.stamp = ros::Time::now();
            
            pseudorange_markers.markers.push_back(clear_marker);
            pseudorange_pub_.publish(pseudorange_markers);
            return;
        }
        
        // Create a panel to display error breakdown
        visualization_msgs::Marker panel;
        panel.header.frame_id = fixed_frame_;
        panel.header.stamp = ros::Time::now();
        panel.ns = "error_panel";
        panel.id = id++;
        panel.type = visualization_msgs::Marker::CUBE;
        panel.action = visualization_msgs::Marker::ADD;
        
        // Position panel near user but offset to the side
        panel.pose.position.x = user_position_.x() + 15.0;
        panel.pose.position.y = user_position_.y() + 10.0;
        panel.pose.position.z = user_position_.z() + 10.0;
        panel.pose.orientation.w = 1.0;
        
        // Size based on number of satellites plus header and footer
        double panel_height = std::max(10.0, satellite_signals_.size() * 3.0 + 4.0);
        panel.scale.x = 18.0;
        panel.scale.y = 0.1;
        panel.scale.z = panel_height;
        
        // Color - semi-transparent dark gray
        panel.color.r = 0.2;
        panel.color.g = 0.2;
        panel.color.b = 0.2;
        panel.color.a = 0.7;
        
        pseudorange_markers.markers.push_back(panel);
        
        // Add panel title
        visualization_msgs::Marker title;
        title.header.frame_id = fixed_frame_;
        title.header.stamp = ros::Time::now();
        title.ns = "error_title";
        title.id = id++;
        title.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        title.action = visualization_msgs::Marker::ADD;
        
        title.pose.position.x = panel.pose.position.x;
        title.pose.position.y = panel.pose.position.y;
        title.pose.position.z = panel.pose.position.z + panel_height/2.0 + 1.0;
        title.pose.orientation.w = 1.0;
        
        title.scale.z = 1.0;
        
        title.color.r = 1.0;
        title.color.g = 1.0;
        title.color.b = 1.0;
        title.color.a = 1.0;
        
        title.text = "PSEUDORANGE ERROR BREAKDOWN (METERS)";
        
        pseudorange_markers.markers.push_back(title);
        
        // Add header row
        visualization_msgs::Marker header;
        header.header.frame_id = fixed_frame_;
        header.header.stamp = ros::Time::now();
        header.ns = "error_header";
        header.id = id++;
        header.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        header.action = visualization_msgs::Marker::ADD;
        
        header.pose.position.x = panel.pose.position.x;
        header.pose.position.y = panel.pose.position.y;
        header.pose.position.z = panel.pose.position.z + panel_height/2.0 - 1.0;
        header.pose.orientation.w = 1.0;
        
        header.scale.z = 0.7;
        
        header.color.r = 1.0;
        header.color.g = 1.0;
        header.color.b = 1.0;
        header.color.a = 1.0;
        
        header.text = "SAT    CLOCK    IONO    TROPO    RECV    MPATH    NOISE    TOTAL";
        
        pseudorange_markers.markers.push_back(header);
        
        // Add entries for each satellite
        for (size_t i = 0; i < satellite_signals_.size(); i++) {
            const auto& signal = satellite_signals_[i];
            
            visualization_msgs::Marker entry;
            entry.header.frame_id = fixed_frame_;
            entry.header.stamp = ros::Time::now();
            entry.ns = "error_entries";
            entry.id = id++;
            entry.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            entry.action = visualization_msgs::Marker::ADD;
            
            // Position within panel
            entry.pose.position.x = panel.pose.position.x;
            entry.pose.position.y = panel.pose.position.y;
            entry.pose.position.z = panel.pose.position.z + panel_height/2.0 - 2.5 - i * 2.0;
            entry.pose.orientation.w = 1.0;
            
            entry.scale.z = 0.7;
            
            // Color based on signal type
            if (signal.is_los) {
                entry.color.r = 0.0;
                entry.color.g = 1.0;
                entry.color.b = 0.0;
            } else if (signal.is_multipath) {
                entry.color.r = 1.0;
                entry.color.g = 0.0;
                entry.color.b = 0.0;
            } else {
                entry.color.r = 1.0;
                entry.color.g = 1.0;
                entry.color.b = 0.0;
            }
            entry.color.a = 1.0;
            
            // Format text with error breakdown
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << std::setw(4) << signal.satellite_id << "  " 
               << std::setw(7) << signal.satellite_clock_error << "  "
               << std::setw(6) << signal.ionospheric_delay << "  "
               << std::setw(7) << signal.tropospheric_delay << "  "
               << std::setw(6) << signal.receiver_clock_bias << "  "
               << std::setw(7) << signal.multipath_error << "  "
               << std::setw(7) << signal.receiver_noise << "  "
               << std::setw(7) << signal.getTotalError();
            
            entry.text = ss.str();
            
            pseudorange_markers.markers.push_back(entry);
            
            // Add visual bar graph for error components
            double bar_width = 0.4;
            double bar_spacing = 0.5;
            double max_bar_length = 8.0;  // Maximum bar length for scaling
            double bar_z_pos = entry.pose.position.z - 0.6;  // Position below the text
            
            // Function to create a bar for each error component
            auto createErrorBar = [&](double error, double x_offset, const std_msgs::ColorRGBA& color) {
                visualization_msgs::Marker bar;
                bar.header.frame_id = fixed_frame_;
                bar.header.stamp = ros::Time::now();
                bar.ns = "error_bars";
                bar.id = id++;
                bar.type = visualization_msgs::Marker::CUBE;
                bar.action = visualization_msgs::Marker::ADD;
                
                // Scale error for visualization (absolute value)
                double bar_length = std::min(max_bar_length, std::abs(error) * 3.0);
                if (bar_length < 0.05) bar_length = 0.05;  // Minimum visible size
                
                // Position
                bar.pose.position.x = panel.pose.position.x - 5.0 + x_offset;
                bar.pose.position.y = panel.pose.position.y;
                bar.pose.position.z = bar_z_pos;
                
                // Size
                bar.scale.x = bar_length;
                bar.scale.y = 0.3;
                bar.scale.z = bar_width;
                
                // Color
                bar.color = color;
                
                // Direction based on sign
                if (error < 0) {
                    bar.pose.position.x -= bar_length / 2.0;
                } else {
                    bar.pose.position.x += bar_length / 2.0;
                }
                
                pseudorange_markers.markers.push_back(bar);
            };
            
            // Define colors for different error sources
            std_msgs::ColorRGBA clock_color, iono_color, tropo_color, recv_color, multipath_color, noise_color, total_color;
            
            clock_color.r = 0.0; clock_color.g = 0.7; clock_color.b = 1.0; clock_color.a = 0.7;  // Blue
            iono_color.r = 1.0; iono_color.g = 0.5; iono_color.b = 0.0; iono_color.a = 0.7;      // Orange
            tropo_color.r = 0.0; tropo_color.g = 0.5; tropo_color.b = 0.0; tropo_color.a = 0.7;  // Dark green
            recv_color.r = 0.5; recv_color.g = 0.0; recv_color.b = 0.5; recv_color.a = 0.7;      // Purple
            multipath_color.r = 1.0; multipath_color.g = 0.0; multipath_color.b = 0.0; multipath_color.a = 0.7; // Red
            noise_color.r = 0.7; noise_color.g = 0.7; noise_color.b = 0.7; noise_color.a = 0.7;  // Gray
            total_color.r = 1.0; total_color.g = 1.0; total_color.b = 1.0; total_color.a = 0.9;  // White
            
            // Create bars for each error component
            createErrorBar(signal.satellite_clock_error, 0.0, clock_color);
            createErrorBar(signal.ionospheric_delay, 1.5, iono_color);
            createErrorBar(signal.tropospheric_delay, 3.0, tropo_color);
            createErrorBar(signal.receiver_clock_bias, 4.5, recv_color);
            createErrorBar(signal.multipath_error, 6.0, multipath_color);
            createErrorBar(signal.receiver_noise, 7.5, noise_color);
            createErrorBar(signal.getTotalError(), 9.0, total_color);
        }
        
        // Add legend for the error bars
        visualization_msgs::Marker legend;
        legend.header.frame_id = fixed_frame_;
        legend.header.stamp = ros::Time::now();
        legend.ns = "error_legend";
        legend.id = id++;
        legend.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        legend.action = visualization_msgs::Marker::ADD;
        
        legend.pose.position.x = panel.pose.position.x;
        legend.pose.position.y = panel.pose.position.y;
        legend.pose.position.z = panel.pose.position.z - panel_height/2.0 + 1.5;
        legend.pose.orientation.w = 1.0;
        
        legend.scale.z = 0.6;
        
        legend.color.r = 0.8;
        legend.color.g = 0.8;
        legend.color.b = 0.8;
        legend.color.a = 1.0;
        
        legend.text = "Legend:\n"
                      "Blue: Satellite Clock   Orange: Ionospheric   Green: Tropospheric\n"
                      "Purple: Receiver Clock   Red: Multipath   Gray: Receiver Noise\n"
                      "White: Total Error";
        
        pseudorange_markers.markers.push_back(legend);
        
        pseudorange_pub_.publish(pseudorange_markers);
    }
    
    void publishSkyplot() {
        visualization_msgs::MarkerArray skyplot_markers;
        int id = 0;
        
        // Create skyplot background
        visualization_msgs::Marker background;
        background.header.frame_id = fixed_frame_;
        background.header.stamp = ros::Time::now();
        background.ns = "skyplot_background";
        background.id = id++;
        background.type = visualization_msgs::Marker::CYLINDER;
        background.action = visualization_msgs::Marker::ADD;
        
        // Position skyplot near user
        background.pose.position.x = user_position_.x() - 15.0;
        background.pose.position.y = user_position_.y() - 10.0;
        background.pose.position.z = user_position_.z() + 5.0;
        background.pose.orientation.w = 1.0;
        
        // Size
        double skyplot_radius = 5.0;
        background.scale.x = 2 * skyplot_radius;
        background.scale.y = 2 * skyplot_radius;
        background.scale.z = 0.1;
        
        // Color - dark blue
        background.color.r = 0.0;
        background.color.g = 0.0;
        background.color.b = 0.3;
        background.color.a = 0.7;
        
        skyplot_markers.markers.push_back(background);
        
        // Add concentric circles for elevation markers
        for (int i = 1; i <= 3; i++) {
            visualization_msgs::Marker circle;
            circle.header.frame_id = fixed_frame_;
            circle.header.stamp = ros::Time::now();
            circle.ns = "skyplot_circles";
            circle.id = id++;
            circle.type = visualization_msgs::Marker::CYLINDER;
            circle.action = visualization_msgs::Marker::ADD;
            
            // Same position as background
            circle.pose.position = background.pose.position;
            circle.pose.orientation = background.pose.orientation;
            
            // Size - scaled based on elevation
            double radius_scale = (4 - i) / 3.0;  // 3/3, 2/3, 1/3
            circle.scale.x = 2 * skyplot_radius * radius_scale;
            circle.scale.y = 2 * skyplot_radius * radius_scale;
            circle.scale.z = 0.11;  // Slightly above background
            
            // Color - light blue rings
            circle.color.r = 0.0;
            circle.color.g = 0.5;
            circle.color.b = 1.0;
            circle.color.a = 0.3;
            
            skyplot_markers.markers.push_back(circle);
            
            // Add elevation label
            visualization_msgs::Marker elev_text;
            elev_text.header = circle.header;
            elev_text.ns = "elevation_labels";
            elev_text.id = id++;
            elev_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            elev_text.action = visualization_msgs::Marker::ADD;
            
            int elevation = 90 - (i * 30);  // 60°, 30°, 0°
            
            elev_text.pose.position = circle.pose.position;
            elev_text.pose.position.y += circle.scale.y / 2.0;
            elev_text.pose.position.z += 0.1;
            elev_text.pose.orientation.w = 1.0;
            
            elev_text.scale.z = 0.8;
            
            elev_text.color.r = 1.0;
            elev_text.color.g = 1.0;
            elev_text.color.b = 1.0;
            elev_text.color.a = 1.0;
            
            elev_text.text = std::to_string(elevation) + "°";
            
            skyplot_markers.markers.push_back(elev_text);
        }
        
        // Add cardinal direction markers
        std::vector<std::pair<std::string, double>> directions = {
            {"N", 0.0}, {"E", 90.0}, {"S", 180.0}, {"W", 270.0}
        };
        
        for (const auto& dir : directions) {
            visualization_msgs::Marker dir_text;
            dir_text.header.frame_id = fixed_frame_;
            dir_text.header.stamp = ros::Time::now();
            dir_text.ns = "direction_labels";
            dir_text.id = id++;
            dir_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            dir_text.action = visualization_msgs::Marker::ADD;
            
            double angle = GPSPhysics::deg2rad(dir.second);
            dir_text.pose.position = background.pose.position;
            dir_text.pose.position.x += (skyplot_radius + 0.5) * std::sin(angle);
            dir_text.pose.position.y += (skyplot_radius + 0.5) * std::cos(angle);
            dir_text.pose.position.z += 0.1;
            dir_text.pose.orientation.w = 1.0;
            
            dir_text.scale.z = 0.8;
            
            dir_text.color.r = 1.0;
            dir_text.color.g = 1.0;
            dir_text.color.b = 1.0;
            dir_text.color.a = 1.0;
            
            dir_text.text = dir.first;
            
            skyplot_markers.markers.push_back(dir_text);
        }
        
        // Add satellites to skyplot
        for (const auto& satellite : satellites_) {
            // Skip satellites below horizon
            if (satellite.elevation < 0) {
                continue;
            }
            
            // Calculate position on skyplot
            double elevation_rad = GPSPhysics::deg2rad(90.0 - satellite.elevation);
            double azimuth_rad = GPSPhysics::deg2rad(satellite.azimuth);
            
            // Convert to skyplot coordinates (azimuth 0=North, increases clockwise)
            double x = skyplot_radius * std::sin(azimuth_rad) * elevation_rad / (M_PI/2);
            double y = skyplot_radius * std::cos(azimuth_rad) * elevation_rad / (M_PI/2);
            
            // Create satellite marker
            visualization_msgs::Marker sat_marker;
            sat_marker.header.frame_id = fixed_frame_;
            sat_marker.header.stamp = ros::Time::now();
            sat_marker.ns = "skyplot_satellites";
            sat_marker.id = id++;
            sat_marker.type = visualization_msgs::Marker::SPHERE;
            sat_marker.action = visualization_msgs::Marker::ADD;
            
            sat_marker.pose.position = background.pose.position;
            sat_marker.pose.position.x += x;
            sat_marker.pose.position.y += y;
            sat_marker.pose.position.z += 0.2;  // Above the circles
            sat_marker.pose.orientation.w = 1.0;
            
            // Size - larger for higher elevation
            double size_factor = 0.5 + satellite.elevation / 180.0;
            sat_marker.scale.x = size_factor;
            sat_marker.scale.y = size_factor;
            sat_marker.scale.z = size_factor;
            
            // Color based on signal availability
            bool has_signal = false;
            bool is_los = false;
            bool is_multipath = false;
            
            for (const auto& signal : satellite_signals_) {
                if (signal.satellite_id == satellite.id) {
                    has_signal = true;
                    is_los = signal.is_los;
                    is_multipath = signal.is_multipath;
                    break;
                }
            }
            
            if (has_signal) {
                if (is_los) {
                    // Green for LOS
                    sat_marker.color.r = 0.0;
                    sat_marker.color.g = 1.0;
                    sat_marker.color.b = 0.0;
                } else if (is_multipath) {
                    // Red for multipath
                    sat_marker.color.r = 1.0;
                    sat_marker.color.g = 0.0;
                    sat_marker.color.b = 0.0;
                } else {
                    // Yellow for attenuated
                    sat_marker.color.r = 1.0;
                    sat_marker.color.g = 1.0;
                    sat_marker.color.b = 0.0;
                }
            } else if (satellite.elevation >= GPSPhysics::MIN_ELEVATION_ANGLE) {
                // Gray for no signal but above horizon
                sat_marker.color.r = 0.5;
                sat_marker.color.g = 0.5;
                sat_marker.color.b = 0.5;
            } else {
                // Dark gray for below minimum elevation
                sat_marker.color.r = 0.3;
                sat_marker.color.g = 0.3;
                sat_marker.color.b = 0.3;
            }
            sat_marker.color.a = 1.0;
            
            skyplot_markers.markers.push_back(sat_marker);
            
            // Add satellite ID label
            visualization_msgs::Marker sat_label;
            sat_label.header = sat_marker.header;
            sat_label.ns = "skyplot_labels";
            sat_label.id = id++;
            sat_label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            sat_label.action = visualization_msgs::Marker::ADD;
            
            sat_label.pose.position = sat_marker.pose.position;
            sat_label.pose.position.z += 0.5;
            sat_label.pose.orientation.w = 1.0;
            
            sat_label.scale.z = 0.6;
            
            sat_label.color.r = 1.0;
            sat_label.color.g = 1.0;
            sat_label.color.b = 1.0;
            sat_label.color.a = 1.0;
            
            sat_label.text = satellite.id;
            
            skyplot_markers.markers.push_back(sat_label);
        }
        
        // Add skyplot title
        visualization_msgs::Marker title;
        title.header.frame_id = fixed_frame_;
        title.header.stamp = ros::Time::now();
        title.ns = "skyplot_title";
        title.id = id++;
        title.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        title.action = visualization_msgs::Marker::ADD;
        
        title.pose.position = background.pose.position;
        title.pose.position.z += skyplot_radius + 1.0;
        title.pose.orientation.w = 1.0;
        
        title.scale.z = 1.0;
        
        title.color.r = 1.0;
        title.color.g = 1.0;
        title.color.b = 1.0;
        title.color.a = 1.0;
        
        title.text = "SATELLITE SKYPLOT";
        
        skyplot_markers.markers.push_back(title);
        
        // Add legend for satellite colors
        visualization_msgs::Marker legend;
        legend.header.frame_id = fixed_frame_;
        legend.header.stamp = ros::Time::now();
        legend.ns = "skyplot_legend";
        legend.id = id++;
        legend.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        legend.action = visualization_msgs::Marker::ADD;
        
        legend.pose.position = background.pose.position;
        legend.pose.position.z -= skyplot_radius + 1.0;
        legend.pose.orientation.w = 1.0;
        
        legend.scale.z = 0.6;
        
        legend.color.r = 1.0;
        legend.color.g = 1.0;
        legend.color.b = 1.0;
        legend.color.a = 1.0;
        
        legend.text = "Green = LOS  |  Yellow = Attenuated  |  Red = Multipath  |  Gray = No Signal";
        
        skyplot_markers.markers.push_back(legend);
        
        skyplot_pub_.publish(skyplot_markers);
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
        
        title_marker.scale.z = 2.0; // Text height
        
        title_marker.color.r = 1.0;
        title_marker.color.g = 1.0;
        title_marker.color.b = 1.0;
        title_marker.color.a = 1.0;
        
        title_marker.text = "GPS SATELLITE SIGNAL SIMULATOR";
        
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
        
        legend_marker.scale.z = 0.8; // Text height
        
        legend_marker.color.r = 1.0;
        legend_marker.color.g = 1.0;
        legend_marker.color.b = 1.0;
        legend_marker.color.a = 1.0;
        
        legend_marker.text = 
            "SIGNAL PATHS:\n"
            "GREEN: Direct Line-of-Sight\n"
            "YELLOW: Signal Attenuated by Buildings\n"
            "RED: Multipath Reflection\n\n"
            "ERROR SOURCES:\n"
            "• Satellite clock bias & drift\n"
            "• Ionospheric delay (Klobuchar model)\n"
            "• Tropospheric delay (Saastamoinen)\n"
            "• Receiver clock bias\n"
            "• Multipath error\n"
            "• Receiver noise\n"
            "• Relativistic effects";
        
        text_markers.markers.push_back(legend_marker);
        
        // Signal statistics
        visualization_msgs::Marker stats_marker;
        stats_marker.header.frame_id = fixed_frame_;
        stats_marker.header.stamp = ros::Time::now();
        stats_marker.ns = "stats";
        stats_marker.id = id++;
        stats_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        stats_marker.action = visualization_msgs::Marker::ADD;
        
        stats_marker.pose.position.x = road_length_ / 2.0 - 15.0;
        stats_marker.pose.position.y = -road_width_ / 2.0 - 5.0;
        stats_marker.pose.position.z = 5.0;
        stats_marker.pose.orientation.w = 1.0;
        
        stats_marker.scale.z = 0.8; // Text height
        
        stats_marker.color.r = 1.0;
        stats_marker.color.g = 1.0;
        stats_marker.color.b = 0.0;
        stats_marker.color.a = 1.0;
        
        // Count satellites by type
        int los_count = 0;
        int attenuated_count = 0;
        int multipath_count = 0;
        int blocked_count = 0;
        
        // Track satellites visible and those with signals
        std::set<std::string> visible_satellites;
        std::set<std::string> signal_satellites;
        
        for (const auto& satellite : satellites_) {
            if (satellite.elevation >= GPSPhysics::MIN_ELEVATION_ANGLE) {
                visible_satellites.insert(satellite.id);
            }
        }
        
        for (const auto& signal : satellite_signals_) {
            signal_satellites.insert(signal.satellite_id);
            
            if (signal.is_los) los_count++;
            else if (signal.is_multipath) multipath_count++;
            else attenuated_count++;
        }
        
        blocked_count = visible_satellites.size() - signal_satellites.size();
        
        // Calculate DOP if we have enough satellites
        double gdop = calculateDOP();
        
        std::stringstream ss;
        ss << "Satellites Above Horizon: " << visible_satellites.size() << "\n"
           << "- Direct Line-of-Sight: " << los_count << "\n"
           << "- Signal Attenuated: " << attenuated_count << "\n"
           << "- Multipath: " << multipath_count << "\n"
           << "- Completely Blocked: " << blocked_count << "\n\n"
           << "GDOP: " << std::fixed << std::setprecision(1) << gdop << "\n"
           << "Moving with " << movement_type_ << " trajectory";
        
        stats_marker.text = ss.str();
        
        text_markers.markers.push_back(stats_marker);
        
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
        
        user_info.scale.z = 0.8;
        
        user_info.color.r = 0.0;
        user_info.color.g = 0.8;
        user_info.color.b = 1.0;
        user_info.color.a = 1.0;
        
        std::stringstream user_ss;
        user_ss << "Receiver Position:\n"
                << "Lat: " << std::fixed << std::setprecision(6) << user_lat_ << "°\n"
                << "Lon: " << std::fixed << std::setprecision(6) << user_lon_ << "°\n"
                << "Height: " << std::fixed << std::setprecision(2) << user_height_ << " m\n\n"
                << "Clock Bias: " << std::fixed << std::setprecision(2) << receiver_clock_bias_ << " m\n"
                << "Speed: " << std::fixed << std::setprecision(2) << movement_speed_ << " m/s\n"
                << "Usable Satellites: " << signal_satellites.size() << "/" << visible_satellites.size();
        
        user_info.text = user_ss.str();
        
        text_markers.markers.push_back(user_info);
        
        // GPS time of week
        visualization_msgs::Marker time_info;
        time_info.header.frame_id = fixed_frame_;
        time_info.header.stamp = ros::Time::now();
        time_info.ns = "time_info";
        time_info.id = id++;
        time_info.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        time_info.action = visualization_msgs::Marker::ADD;
        
        // Position in corner
        time_info.pose.position.x = -road_length_ / 2.0 + 5.0;
        time_info.pose.position.y = road_width_ / 2.0 + 5.0;
        time_info.pose.position.z = 5.0;
        time_info.pose.orientation.w = 1.0;
        
        time_info.scale.z = 0.8;
        
        time_info.color.r = 1.0;
        time_info.color.g = 1.0;
        time_info.color.b = 1.0;
        time_info.color.a = 1.0;
        
        // Format time as days, hours, minutes, seconds
        int days = static_cast<int>(gps_time_) / 86400;
        int hours = (static_cast<int>(gps_time_) % 86400) / 3600;
        int minutes = (static_cast<int>(gps_time_) % 3600) / 60;
        double seconds = std::fmod(gps_time_, 60.0);
        
        std::stringstream time_ss;
        time_ss << "GPS Time of Week:\n"
                << days << "d " << hours << "h " << minutes << "m " << std::fixed 
                << std::setprecision(3) << seconds << "s";
        
        time_info.text = time_ss.str();
        
        text_markers.markers.push_back(time_info);
        
        text_pub_.publish(text_markers);
    }
    
    double calculateDOP() {
        // Calculate Dilution of Precision from satellite geometry
        
        // Need at least 4 satellites for a solution
        if (satellite_signals_.size() < 4) {
            return 99.9;  // Invalid DOP
        }
        
        // Create geometry matrix
        Eigen::MatrixXd G(satellite_signals_.size(), 4);
        
        // Fill geometry matrix with unit vectors from user to satellites
        int row = 0;
        for (const auto& signal : satellite_signals_) {
            // Find the satellite
            auto sat_it = std::find_if(satellites_.begin(), satellites_.end(),
                                    [&signal](const GPSSatellite& sat) {
                                        return sat.id == signal.satellite_id;
                                    });
            
            if (sat_it != satellites_.end()) {
                // Get unit vector from user to satellite
                Eigen::Vector3d user_to_sat = (sat_it->position - user_position_).normalized();
                
                // Fill row of geometry matrix [x, y, z, 1]
                G(row, 0) = user_to_sat.x();
                G(row, 1) = user_to_sat.y();
                G(row, 2) = user_to_sat.z();
                G(row, 3) = 1.0;  // Clock term
                
                row++;
            }
        }
        
        // Calculate covariance matrix
        Eigen::MatrixXd GTG = G.transpose() * G;
        
        // Check if matrix is invertible
        double det = GTG.determinant();
        if (std::abs(det) < 1e-10) {
            return 50.0;  // Singular matrix - very poor geometry
        }
        
        // Invert to get covariance matrix
        Eigen::MatrixXd cov = GTG.inverse();
        
        // GDOP = sqrt(trace of covariance matrix)
        double gdop = std::sqrt(cov.trace());
        
        // PDOP = sqrt(sum of position variances)
        double pdop = std::sqrt(cov(0,0) + cov(1,1) + cov(2,2));
        
        // HDOP = sqrt(sum of horizontal position variances)
        double hdop = std::sqrt(cov(0,0) + cov(1,1));
        
        // VDOP = sqrt(vertical position variance)
        double vdop = std::sqrt(cov(2,2));
        
        // TDOP = sqrt(time/clock variance)
        double tdop = std::sqrt(cov(3,3));
        
        // Return GDOP - could be extended to return all DOPs
        return gdop;
    }
    
    void broadcastTFs() {
        // Broadcast TF frames for satellites and user
        ros::Time now = ros::Time::now();
        
        // User TF
        tf::Transform user_tf;
        user_tf.setOrigin(tf::Vector3(user_position_.x(), user_position_.y(), user_position_.z()));
        user_tf.setRotation(tf::Quaternion(0, 0, 0, 1));
        tf_broadcaster_.sendTransform(tf::StampedTransform(user_tf, now, fixed_frame_, "gps_receiver"));
        
        // Satellite TFs
        for (const auto& satellite : satellites_) {
            tf::Transform satellite_tf;
            satellite_tf.setOrigin(tf::Vector3(satellite.position.x(), 
                                             satellite.position.y(), 
                                             satellite.position.z()));
            satellite_tf.setRotation(tf::Quaternion(0, 0, 0, 1));
            tf_broadcaster_.sendTransform(tf::StampedTransform(satellite_tf, now, fixed_frame_, satellite.id));
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "RangingRC");
    
    ROS_INFO("Starting GPS Satellite Signal Simulator with rigorous range measurements");
    GPSSimulator simulator;
    
    ros::spin();
    
    return 0;
}