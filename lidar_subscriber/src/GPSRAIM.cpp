#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/NavSatStatus.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cmath>
#include <deque>

class GPSRAIMNode {
private:
    // ROS handles
    ros::NodeHandle nh_;
    ros::Publisher raim_results_pub_;
    ros::Publisher protection_level_pub_;
    ros::Publisher simulated_gps_pub_;
    ros::Publisher satellite_viz_pub_;
    ros::Publisher position_viz_pub_;
    ros::Publisher protection_level_viz_pub_;
    ros::Publisher pose_with_cov_pub_;
    ros::Publisher path_pub_;
    ros::Timer simulation_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // RAIM parameters
    double chi_square_threshold_;
    double alarm_limit_;
    double prob_false_alarm_; 
    double prob_missed_detection_; 
    int min_satellites_for_raim_;
    
    // Visualization parameters
    std::string fixed_frame_;
    std::string user_frame_;
    
    // Satellite information
    struct SatelliteInfo {
        Eigen::Vector3d position_ecef;
        double pseudorange;
        double elevation;
        double azimuth;
        int id;
        bool is_faulty;
    };
    std::vector<SatelliteInfo> satellites_;

    // Circular trajectory parameters
    double trajectory_radius_m_;
    double trajectory_speed_mps_;
    double trajectory_angular_velocity_rad_s_;
    double trajectory_center_lat_deg_;
    double trajectory_center_lon_deg_;
    double trajectory_center_alt_m_;
    double trajectory_angle_rad_;
    Eigen::Vector3d trajectory_center_ecef_;
    
    // Path history for visualization
    nav_msgs::Path path_;
    std::deque<geometry_msgs::PoseStamped> pose_history_;
    int max_path_history_;

    // Simulation parameters
    double simulation_rate_hz_;
    double noise_stddev_m_;
    double failure_probability_;
    double failure_magnitude_m_;
    int num_satellites_;
    int satellite_failure_idx_;
    bool force_satellite_failure_;
    
    // Constants
    const double DEG_TO_RAD = M_PI / 180.0;
    const double RAD_TO_DEG = 180.0 / M_PI;
    const double EARTH_RADIUS_M = 6371000.0;
    const double SPEED_OF_LIGHT = 299792458.0; // m/s
    
    // User position and protection levels
    Eigen::Vector3d true_position_ecef_;
    Eigen::Vector3d estimated_position_ecef_;
    double horizontal_protection_level_;
    double vertical_protection_level_;
    
    // Previous trajectory center ENU for creating consistent paths
    Eigen::Vector3d previous_center_enu_;
    bool first_update_;
    
    // Random number generators
    std::mt19937 gen_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
    
    // Student-t distribution values for different confidence levels
    std::map<double, double> t_distribution_values_;

public:
    GPSRAIMNode() : nh_("~"), tf_broadcaster_() {
        // Load parameters
        loadParameters();
        
        // Initialize t-distribution values for different confidence levels
        initializeStatisticalValues();
        
        // Initialize random number generators
        std::random_device rd;
        gen_ = std::mt19937(rd());
        uniform_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
        normal_dist_ = std::normal_distribution<double>(0.0, noise_stddev_m_);
        
        // Set up publishers
        raim_results_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("raim_results", 10);
        protection_level_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("protection_levels", 10);
        satellite_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("satellite_markers", 10);
        position_viz_pub_ = nh_.advertise<visualization_msgs::Marker>("position_marker", 10);
        protection_level_viz_pub_ = nh_.advertise<visualization_msgs::Marker>("protection_level_marker", 10);
        pose_with_cov_pub_ = nh_.advertise<geometry_msgs::PoseWithCovariance>("gps_pose_with_covariance", 10);
        path_pub_ = nh_.advertise<nav_msgs::Path>("gps_path", 10);
        simulated_gps_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("simulated_gps", 10);
        
        // Initialize trajectory angle
        trajectory_angle_rad_ = 0.0;
        
        // Calculate trajectory center in ECEF
        trajectory_center_ecef_ = geodeticToECEF(
            trajectory_center_lat_deg_ * DEG_TO_RAD, 
            trajectory_center_lon_deg_ * DEG_TO_RAD, 
            trajectory_center_alt_m_);
        
        // Initialize user position at the start of the circular trajectory
        updateUserPosition();
        estimated_position_ecef_ = true_position_ecef_;
        
        // Initialize protection levels
        horizontal_protection_level_ = 0.0;
        vertical_protection_level_ = 0.0;
        
        // Initialize path history
        path_.header.frame_id = fixed_frame_;
        max_path_history_ = 500;
        first_update_ = true;
        
        // Start simulation timer after initialization
        simulation_timer_ = nh_.createTimer(ros::Duration(1.0/simulation_rate_hz_), 
                                          &GPSRAIMNode::simulationCallback, this);
        
        ROS_INFO("GPS RAIM node initialized with circular trajectory");
        ROS_INFO("Trajectory center: Lat=%.6f°, Lon=%.6f°, Alt=%.1fm, Radius=%.1fm", 
                trajectory_center_lat_deg_, trajectory_center_lon_deg_, 
                trajectory_center_alt_m_, trajectory_radius_m_);
    }

    void loadParameters() {
        // RAIM parameters
        nh_.param<double>("chi_square_threshold", chi_square_threshold_, 5.99); 
        nh_.param<double>("alarm_limit", alarm_limit_, 100.0); // meters
        nh_.param<double>("prob_false_alarm", prob_false_alarm_, 0.05); 
        nh_.param<double>("prob_missed_detection", prob_missed_detection_, 0.001);
        nh_.param<int>("min_satellites_for_raim", min_satellites_for_raim_, 5);
        
        // Visualization parameters
        nh_.param<std::string>("fixed_frame", fixed_frame_, "world");
        nh_.param<std::string>("user_frame", user_frame_, "gps_user");
        
        // Circular trajectory parameters
        nh_.param<double>("trajectory_radius_m", trajectory_radius_m_, 100.0);
        nh_.param<double>("trajectory_speed_mps", trajectory_speed_mps_, 5.0);
        nh_.param<double>("trajectory_center_lat_deg", trajectory_center_lat_deg_, 40.0);
        nh_.param<double>("trajectory_center_lon_deg", trajectory_center_lon_deg_, -75.0);
        nh_.param<double>("trajectory_center_alt_m", trajectory_center_alt_m_, 100.0);
        
        // Calculate angular velocity from speed and radius
        trajectory_angular_velocity_rad_s_ = trajectory_speed_mps_ / trajectory_radius_m_;
        
        // Simulation parameters
        nh_.param<double>("simulation_rate_hz", simulation_rate_hz_, 10.0);
        nh_.param<double>("noise_stddev_m", noise_stddev_m_, 5.0);
        nh_.param<double>("failure_probability", failure_probability_, 0.1);
        nh_.param<double>("failure_magnitude_m", failure_magnitude_m_, 30.0);
        nh_.param<int>("num_satellites", num_satellites_, 8);
        
        // Satellite failure parameters
        nh_.param<bool>("force_satellite_failure", force_satellite_failure_, false);
        nh_.param<int>("satellite_failure_idx", satellite_failure_idx_, 0);
    }
    
    void initializeStatisticalValues() {
        // Student-t distribution values for different confidence levels
        t_distribution_values_[0.5] = 0.674;    // 50% confidence
        t_distribution_values_[0.95] = 1.96;    // 95% confidence
        t_distribution_values_[0.99] = 2.576;   // 99% confidence
        t_distribution_values_[0.999] = 3.291;  // 99.9% confidence
        t_distribution_values_[0.9999] = 3.891; // 99.99% confidence
    }
    
    void simulationCallback(const ros::TimerEvent& event) {
        // Update user position along circular trajectory
        updateUserPosition();
        
        // Generate and place satellites around user
        generateSatelliteConstellation();
        
        // Perform RAIM and calculate protection levels
        runRAIMAndCalculateProtectionLevels();
        
        // Visualize everything
        visualize();
        
        // Publish GPS data
        publishGPSData();
        
        // Update and publish path
        updatePath();
        
        // Broadcast TF transform for the user position
        broadcastTransform();
    }
    
    void updateUserPosition() {
        // Update angle based on angular velocity
        trajectory_angle_rad_ += trajectory_angular_velocity_rad_s_ / simulation_rate_hz_;
        if (trajectory_angle_rad_ > 2 * M_PI) {
            trajectory_angle_rad_ -= 2 * M_PI;
        }
        
        // Get center coordinates in geodetic for local frame calculations
        double center_lat, center_lon, center_alt;
        ECEFToGeodetic(trajectory_center_ecef_, center_lat, center_lon, center_alt);
        
        // Calculate position offset in local ENU frame
        double east = trajectory_radius_m_ * cos(trajectory_angle_rad_);
        double north = trajectory_radius_m_ * sin(trajectory_angle_rad_);
        
        // Convert ENU offset to ECEF
        Eigen::Vector3d enu_offset(east, north, 0.0);
        true_position_ecef_ = ENUToECEF(enu_offset, trajectory_center_ecef_, center_lat, center_lon);
        
        // For the first update, set previous center ENU
        if (first_update_) {
            previous_center_enu_ = Eigen::Vector3d(0, 0, 0);
            first_update_ = false;
        }
    }
    
    void generateSatelliteConstellation() {
        satellites_.clear();
        
        // Get user position in geodetic for local frame calculations
        double user_lat, user_lon, user_alt;
        ECEFToGeodetic(true_position_ecef_, user_lat, user_lon, user_alt);
        
        for (int i = 0; i < num_satellites_; i++) {
            SatelliteInfo sat;
            
            // Distribute satellites in sky with good geometry
            sat.azimuth = 360.0 * i / num_satellites_ + 10.0 * normal_dist_(gen_) / noise_stddev_m_;
            sat.elevation = 20.0 + 60.0 * uniform_dist_(gen_); // Elevations between 20° and 80°
            
            // Calculate satellite position in ECEF
            double sat_distance = EARTH_RADIUS_M + 20200000.0; // GPS orbit ~20,200 km
            
            // Convert satellite azimuth and elevation to local ENU coordinates
            double x = sat_distance * cos(sat.elevation * DEG_TO_RAD) * cos(sat.azimuth * DEG_TO_RAD);
            double y = sat_distance * cos(sat.elevation * DEG_TO_RAD) * sin(sat.azimuth * DEG_TO_RAD);
            double z = sat_distance * sin(sat.elevation * DEG_TO_RAD);
            
            // Convert local ENU to ECEF for satellite position
            Eigen::Vector3d sat_enu(x, y, z);
            sat.position_ecef = ENUToECEF(sat_enu, true_position_ecef_, user_lat, user_lon);
            
            // Calculate true range
            double true_range = (sat.position_ecef - true_position_ecef_).norm();
            
            // Add noise to the pseudorange
            double noise = normal_dist_(gen_);
            
            // Possibility of adding a satellite failure
            sat.is_faulty = false;
            
            // Apply forced satellite failure or random failures
            if (force_satellite_failure_ && i == satellite_failure_idx_) {
                noise += failure_magnitude_m_;
                sat.is_faulty = true;
                ROS_DEBUG("Forced failure on satellite %d with magnitude %.2f meters", i+1, failure_magnitude_m_);
            }
            else if (!force_satellite_failure_ && uniform_dist_(gen_) < failure_probability_) {
                noise += (uniform_dist_(gen_) > 0.5 ? 1 : -1) * failure_magnitude_m_;
                sat.is_faulty = true;
                ROS_DEBUG("Satellite %d has simulated failure of %.2f meters", i+1, noise);
            }
            
            sat.pseudorange = true_range + noise;
            sat.id = i + 1; // PRN numbers starting from 1
            
            satellites_.push_back(sat);
        }
    }
    
    void runRAIMAndCalculateProtectionLevels() {
        // Extract satellite data
        std::vector<Eigen::Vector3d> sat_positions;
        std::vector<double> pseudoranges;
        std::vector<int> satellite_ids;
        
        for (const auto& sat : satellites_) {
            sat_positions.push_back(sat.position_ecef);
            pseudoranges.push_back(sat.pseudorange);
            satellite_ids.push_back(sat.id);
        }
        
        // Check if we have enough satellites for RAIM
        if (sat_positions.size() < min_satellites_for_raim_) {
            ROS_WARN("Not enough satellites for RAIM: %zu (minimum %d required)",
                    sat_positions.size(), min_satellites_for_raim_);
            return;
        }
        
        // Initial position estimate (needed for linearization)
        Eigen::Vector4d user_state = Eigen::Vector4d::Zero(); // x, y, z, clock_bias
        user_state.head(3) = estimated_position_ecef_; // Use previous estimate
        
        // Weighted least-squares position estimation
        Eigen::Vector4d estimated_state;
        Eigen::MatrixXd geometry_matrix;
        Eigen::MatrixXd weight_matrix;
        bool convergence = estimatePositionWeightedLeastSquares(
            sat_positions, pseudoranges, user_state, estimated_state, geometry_matrix, weight_matrix);
        
        if (!convergence) {
            ROS_WARN("Position estimation did not converge");
            return;
        }
        
        // Extract position and clock bias
        estimated_position_ecef_ = estimated_state.head(3);
        double clock_bias = estimated_state(3);
        
        // Calculate residuals and test statistic for RAIM
        Eigen::VectorXd residuals;
        double test_statistic = calculateRAIMResiduals(
            sat_positions, pseudoranges, estimated_state, geometry_matrix, weight_matrix, residuals);
        
        // Determine if RAIM detected a failure
        bool raim_failure = test_statistic > chi_square_threshold_;
        
        // Calculate position covariance
        Eigen::MatrixXd covariance = calculatePositionCovariance(geometry_matrix, weight_matrix);
        
        // Calculate protection levels
        calculateRigorousProtectionLevels(
            geometry_matrix, covariance, weight_matrix, 
            horizontal_protection_level_, vertical_protection_level_);
        
        // Check if protection levels exceed alarm limits
        bool integrity_available = horizontal_protection_level_ <= alarm_limit_ && 
                                  vertical_protection_level_ <= alarm_limit_;
        
        // Publish RAIM results
        std_msgs::Float64MultiArray raim_msg;
        raim_msg.data.push_back(test_statistic);
        raim_msg.data.push_back(raim_failure ? 1.0 : 0.0);
        raim_msg.data.push_back(integrity_available ? 1.0 : 0.0);
        raim_results_pub_.publish(raim_msg);
        
        // Publish protection levels
        std_msgs::Float64MultiArray pl_msg;
        pl_msg.data.push_back(horizontal_protection_level_);
        pl_msg.data.push_back(vertical_protection_level_);
        protection_level_pub_.publish(pl_msg);
        
        // Publish position with covariance for RVIZ
        publishPositionWithCovariance(estimated_position_ecef_, covariance);
        
        // Calculate actual position error
        double position_error = (estimated_position_ecef_ - true_position_ecef_).norm();
        
        ROS_INFO("RAIM Results: TS=%.2f (Threshold=%.2f), Failure=%s, HPL=%.2f m, VPL=%.2f m, Error=%.2f m",
                test_statistic, chi_square_threshold_, 
                raim_failure ? "TRUE" : "FALSE",
                horizontal_protection_level_, vertical_protection_level_,
                position_error);
        
        // Perform fault exclusion if failure detected
        if (raim_failure && sat_positions.size() > min_satellites_for_raim_) {
            performFaultExclusion(sat_positions, pseudoranges, satellite_ids, estimated_state);
        }
    }
    
    bool estimatePositionWeightedLeastSquares(
            const std::vector<Eigen::Vector3d>& sat_positions,
            const std::vector<double>& pseudoranges,
            const Eigen::Vector4d& initial_state,
            Eigen::Vector4d& estimated_state,
            Eigen::MatrixXd& geometry_matrix,
            Eigen::MatrixXd& weight_matrix) {
        
        const int max_iterations = 10;
        const double convergence_threshold = 0.01; // meters
        
        estimated_state = initial_state;
        
        if (estimated_state.head(3).norm() < 1.0) {
            // If initial position is at origin, use a better initial guess
            estimated_state.head(3) = true_position_ecef_; // In simulation we know the true position
        }
        
        // Setup weight matrix based on satellite elevations
        weight_matrix.resize(sat_positions.size(), sat_positions.size());
        weight_matrix.setZero();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Current position estimate
            Eigen::Vector3d position = estimated_state.head(3);
            double clock_bias = estimated_state(3);
            
            // Setup geometry matrix (G matrix)
            geometry_matrix.resize(sat_positions.size(), 4);
            Eigen::VectorXd predicted_ranges(sat_positions.size());
            Eigen::VectorXd delta_ranges(sat_positions.size());
            
            // Re-calculate weights based on satellite elevations
            for (size_t i = 0; i < sat_positions.size(); ++i) {
                Eigen::Vector3d sat_pos = sat_positions[i];
                
                // Calculate geometric range
                Eigen::Vector3d difference = sat_pos - position;
                double range = difference.norm();
                predicted_ranges(i) = range + clock_bias;
                
                // Calculate line-of-sight vector
                Eigen::Vector3d los = difference / range;
                
                // Fill geometry matrix row
                geometry_matrix.row(i) << -los.transpose(), 1.0;
                
                // Calculate difference between measured and predicted pseudorange
                delta_ranges(i) = pseudoranges[i] - predicted_ranges(i);
                
                // Calculate elevation angle for weighting
                // Convert ECEF position to geodetic
                double lat, lon, alt;
                ECEFToGeodetic(position, lat, lon, alt);
                
                // Calculate local East-North-Up (ENU) coordinates for the satellite
                Eigen::Vector3d sat_enu = ECEFToENU(sat_pos, position, lat, lon);
                
                // Calculate elevation angle
                double elevation = atan2(sat_enu(2), sqrt(sat_enu(0)*sat_enu(0) + sat_enu(1)*sat_enu(1)));
                
                // Set weight based on elevation (sin^2(elevation) model)
                double sin_elev = sin(elevation);
                weight_matrix(i, i) = sin_elev * sin_elev;
                
                // Minimum weight for numerical stability
                if (weight_matrix(i, i) < 0.01) weight_matrix(i, i) = 0.01;
            }
            
            // Solve for state update using weighted least squares
            // (G^T * W * G)^-1 * G^T * W * delta_ranges
            Eigen::Vector4d delta_state;
            delta_state = (geometry_matrix.transpose() * weight_matrix * geometry_matrix).inverse() 
                         * geometry_matrix.transpose() * weight_matrix * delta_ranges;
            
            // Update state
            estimated_state += delta_state;
            
            // Check for convergence
            if (delta_state.head(3).norm() < convergence_threshold) {
                return true; // Converged
            }
        }
        
        ROS_WARN("Position estimation did not converge within %d iterations", max_iterations);
        return false;
    }
    
    double calculateRAIMResiduals(
            const std::vector<Eigen::Vector3d>& sat_positions,
            const std::vector<double>& pseudoranges,
            const Eigen::Vector4d& estimated_state,
            const Eigen::MatrixXd& geometry_matrix,
            const Eigen::MatrixXd& weight_matrix,
            Eigen::VectorXd& residuals) {
        
        Eigen::Vector3d position = estimated_state.head(3);
        double clock_bias = estimated_state(3);
        
        // Calculate predicted pseudoranges based on estimated position
        Eigen::VectorXd predicted_ranges(sat_positions.size());
        for (size_t i = 0; i < sat_positions.size(); ++i) {
            predicted_ranges(i) = (sat_positions[i] - position).norm() + clock_bias;
        }
        
        // Calculate residuals (measured minus predicted)
        Eigen::VectorXd delta_ranges(sat_positions.size());
        for (size_t i = 0; i < sat_positions.size(); ++i) {
            delta_ranges(i) = pseudoranges[i] - predicted_ranges(i);
        }
        
        // Calculate weighted least-squares solution
        Eigen::MatrixXd G_W_Ginv = (geometry_matrix.transpose() * weight_matrix * geometry_matrix).inverse();
        Eigen::MatrixXd hat_matrix = geometry_matrix * G_W_Ginv * geometry_matrix.transpose() * weight_matrix;
        
        // Calculate the residual matrix S = I - hat_matrix
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(sat_positions.size(), sat_positions.size()) - hat_matrix;
        
        // Calculate the residuals vector
        residuals = S * delta_ranges;
        
        // Calculate the test statistic (weighted sum of squared residuals)
        // Normalized by degrees of freedom
        double weighted_ssr = residuals.transpose() * weight_matrix * residuals;
        int dof = sat_positions.size() - 4; // 4 unknowns (x, y, z, clock bias)
        double test_statistic = weighted_ssr / dof;
        
        return test_statistic;
    }
    
    Eigen::MatrixXd calculatePositionCovariance(
            const Eigen::MatrixXd& geometry_matrix,
            const Eigen::MatrixXd& weight_matrix) {
        
        // Calculate covariance matrix using weighted least squares formula
        // Cov = (G^T * W * G)^-1
        Eigen::MatrixXd covariance = (geometry_matrix.transpose() * weight_matrix * geometry_matrix).inverse();
        
        // Scale by estimated measurement variance 
        // In a real implementation, you would estimate this from the residuals
        double estimated_variance = noise_stddev_m_ * noise_stddev_m_;
        covariance *= estimated_variance;
        
        return covariance;
    }
    
    void calculateRigorousProtectionLevels(
            const Eigen::MatrixXd& geometry_matrix,
            const Eigen::MatrixXd& covariance,
            const Eigen::MatrixXd& weight_matrix,
            double& horizontal_protection_level,
            double& vertical_protection_level) {
        
        // Convert current ECEF position to geodetic for local frame rotation
        double user_lat, user_lon, user_alt;
        ECEFToGeodetic(estimated_position_ecef_, user_lat, user_lon, user_alt);
        
        // Create rotation matrix from ECEF to ENU
        Eigen::Matrix3d R_ecef_to_enu = createRotationMatrix(user_lat, user_lon);
        
        // Transform position covariance to local ENU frame
        Eigen::Matrix3d position_cov_ecef = covariance.block<3,3>(0,0);
        Eigen::Matrix3d position_cov_enu = R_ecef_to_enu * position_cov_ecef * R_ecef_to_enu.transpose();
        
        // Extract horizontal (east-north) covariance
        Eigen::Matrix2d horizontal_cov = position_cov_enu.block<2,2>(0,0);
        
        // Compute eigenvalues for semi-major and semi-minor axes
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(horizontal_cov);
        Eigen::Vector2d eigenvalues = eigen_solver.eigenvalues();
        double semi_major_axis = sqrt(eigenvalues(1)); // Largest eigenvalue
        double semi_minor_axis = sqrt(eigenvalues(0)); // Smallest eigenvalue
        
        // Vertical component (height error)
        double vertical_std = sqrt(position_cov_enu(2,2));
        
        // Get multiplier based on probability of missed detection
        double k_md = getMultiplierForProbability(1.0 - prob_missed_detection_);
        double k_fa = getMultiplierForProbability(1.0 - prob_false_alarm_);
        
        // Calculate basic protection levels
        horizontal_protection_level = k_md * semi_major_axis;
        vertical_protection_level = k_md * vertical_std;
        
        // Account for potential undetected biases using slope parameters
        std::vector<double> horizontal_slopes;
        std::vector<double> vertical_slopes;
        
        int num_sats = geometry_matrix.rows();
        for (int i = 0; i < num_sats; ++i) {
            // Create the sensitivity vector for this satellite
            Eigen::Vector4d sensitivity = G_inverse_row(geometry_matrix, weight_matrix, i);
            
            // Transform the position part to ENU
            Eigen::Vector3d sensitivity_pos_ecef = sensitivity.head(3);
            Eigen::Vector3d sensitivity_pos_enu = R_ecef_to_enu * sensitivity_pos_ecef;
            
            // For horizontal, use the east-north components
            double h_slope = sqrt(sensitivity_pos_enu(0)*sensitivity_pos_enu(0) + 
                                 sensitivity_pos_enu(1)*sensitivity_pos_enu(1));
            horizontal_slopes.push_back(h_slope);
            
            // For vertical, use the up component
            double v_slope = fabs(sensitivity_pos_enu(2));
            vertical_slopes.push_back(v_slope);
        }
        
        // Find the maximum slope values
        double max_h_slope = *std::max_element(horizontal_slopes.begin(), horizontal_slopes.end());
        double max_v_slope = *std::max_element(vertical_slopes.begin(), vertical_slopes.end());
        
        // Calculate the minimum detectable bias
        int dof = num_sats - 4;
        double min_detectable_bias = k_fa * noise_stddev_m_ * sqrt(weight_matrix.diagonal().maxCoeff());
        
        // Update protection levels with slope-based calculations
        horizontal_protection_level = std::max(horizontal_protection_level, max_h_slope * min_detectable_bias);
        vertical_protection_level = std::max(vertical_protection_level, max_v_slope * min_detectable_bias);
    }
    
    Eigen::Matrix3d createRotationMatrix(double lat, double lon) {
        double sin_lat = sin(lat);
        double cos_lat = cos(lat);
        double sin_lon = sin(lon);
        double cos_lon = cos(lon);
        
        Eigen::Matrix3d R;
        // ECEF to ENU rotation matrix
        R << -sin_lon, cos_lon, 0,
             -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat,
              cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat;
        
        return R;
    }
    
    Eigen::Vector4d G_inverse_row(
            const Eigen::MatrixXd& geometry_matrix,
            const Eigen::MatrixXd& weight_matrix,
            int satellite_index) {
        
        // Calculate sensitivity vector for fault detection
        Eigen::MatrixXd G_W_Ginv = (geometry_matrix.transpose() * weight_matrix * geometry_matrix).inverse();
        
        // Create unit vector for the satellite
        Eigen::VectorXd e_i = Eigen::VectorXd::Zero(geometry_matrix.rows());
        e_i(satellite_index) = 1.0;
        
        // Calculate sensitivity vector
        Eigen::Vector4d sensitivity = G_W_Ginv * geometry_matrix.transpose() * weight_matrix * e_i;
        
        return sensitivity;
    }
    
    double getMultiplierForProbability(double probability) {
        // Find the closest value in the lookup table
        double result = 0.0;
        double min_diff = std::numeric_limits<double>::max();
        
        for (const auto& pair : t_distribution_values_) {
            double diff = std::abs(probability - pair.first);
            if (diff < min_diff) {
                min_diff = diff;
                result = pair.second;
            }
        }
        
        return result;
    }
    
    void performFaultExclusion(
            const std::vector<Eigen::Vector3d>& sat_positions,
            const std::vector<double>& pseudoranges,
            const std::vector<int>& satellite_ids,
            const Eigen::Vector4d& initial_state) {
        
        ROS_INFO("Performing fault exclusion...");
        
        double min_test_statistic = std::numeric_limits<double>::max();
        int excluded_sat_idx = -1;
        
        // Try removing each satellite and recalculate the test statistic
        for (size_t i = 0; i < sat_positions.size(); ++i) {
            // Create subsets without the i-th satellite
            std::vector<Eigen::Vector3d> subset_positions;
            std::vector<double> subset_ranges;
            std::vector<int> subset_ids;
            
            for (size_t j = 0; j < sat_positions.size(); ++j) {
                if (j != i) {
                    subset_positions.push_back(sat_positions[j]);
                    subset_ranges.push_back(pseudoranges[j]);
                    subset_ids.push_back(satellite_ids[j]);
                }
            }
            
            // Re-estimate position with the subset
            Eigen::Vector4d estimated_state;
            Eigen::MatrixXd geometry_matrix;
            Eigen::MatrixXd weight_matrix;
            bool convergence = estimatePositionWeightedLeastSquares(
                subset_positions, subset_ranges, initial_state, estimated_state, geometry_matrix, weight_matrix);
            
            if (!convergence) continue;
            
            // Calculate residuals and test statistic
            Eigen::VectorXd residuals;
            double test_statistic = calculateRAIMResiduals(
                subset_positions, subset_ranges, estimated_state, geometry_matrix, weight_matrix, residuals);
            
            // Check if this exclusion gives a better test statistic
            if (test_statistic < min_test_statistic) {
                min_test_statistic = test_statistic;
                excluded_sat_idx = i;
            }
        }
        
        // If exclusion improved the test statistic below threshold
        if (excluded_sat_idx >= 0 && min_test_statistic < chi_square_threshold_) {
            ROS_INFO("Satellite %d identified as faulty and excluded. New test statistic: %.2f",
                    satellite_ids[excluded_sat_idx], min_test_statistic);
            
            // Mark the satellite as faulty in our visualization
            if (excluded_sat_idx < satellites_.size()) {
                satellites_[excluded_sat_idx].is_faulty = true;
            }
        } else {
            ROS_WARN("Fault exclusion unsuccessful. Best test statistic: %.2f",
                    min_test_statistic);
        }
    }
    
    void visualize() {
        visualizeSatellites();
        visualizePosition();
        visualizeProtectionLevel();
    }
    
    void visualizeSatellites() {
        visualization_msgs::MarkerArray marker_array;
        
        // Get user position in geodetic for local frame
        double user_lat, user_lon, user_alt;
        ECEFToGeodetic(true_position_ecef_, user_lat, user_lon, user_alt);
        
        // Visualize each satellite
        for (size_t i = 0; i < satellites_.size(); ++i) {
            const auto& sat = satellites_[i];
            
            // Satellite marker
            visualization_msgs::Marker sat_marker;
            sat_marker.header.frame_id = user_frame_;  // Attach to the moving user frame
            sat_marker.header.stamp = ros::Time::now();
            sat_marker.ns = "satellites";
            sat_marker.id = sat.id;
            sat_marker.type = visualization_msgs::Marker::SPHERE;
            sat_marker.action = visualization_msgs::Marker::ADD;
            
            // Convert satellite ECEF to local ENU frame centered at the user
            Eigen::Vector3d sat_enu = ECEFToENU(sat.position_ecef, true_position_ecef_, user_lat, user_lon);
            
            // Scale for visualization (satellites are far away)
            double scale_factor = 0.05; 
            sat_marker.pose.position.x = sat_enu(0) * scale_factor;
            sat_marker.pose.position.y = sat_enu(1) * scale_factor;
            sat_marker.pose.position.z = sat_enu(2) * scale_factor;
            
            sat_marker.pose.orientation.w = 1.0;
            sat_marker.scale.x = 5.0;
            sat_marker.scale.y = 5.0;
            sat_marker.scale.z = 5.0;
            
            // Color based on faulty status
            if (sat.is_faulty) {
                sat_marker.color.r = 1.0;
                sat_marker.color.g = 0.0;
                sat_marker.color.b = 0.0;
            } else {
                sat_marker.color.r = 0.0;
                sat_marker.color.g = 1.0;
                sat_marker.color.b = 0.0;
            }
            sat_marker.color.a = 0.8;
            
            marker_array.markers.push_back(sat_marker);
            
            // Line from receiver to satellite
            visualization_msgs::Marker line_marker;
            line_marker.header.frame_id = user_frame_;  // Attach to the moving user frame
            line_marker.header.stamp = ros::Time::now();
            line_marker.ns = "sat_lines";
            line_marker.id = sat.id;
            line_marker.type = visualization_msgs::Marker::LINE_STRIP;
            line_marker.action = visualization_msgs::Marker::ADD;
            
            // Line from origin to satellite
            geometry_msgs::Point start_point;
            start_point.x = 0;
            start_point.y = 0;
            start_point.z = 0;
            
            geometry_msgs::Point end_point;
            end_point.x = sat_marker.pose.position.x;
            end_point.y = sat_marker.pose.position.y;
            end_point.z = sat_marker.pose.position.z;
            
            line_marker.points.push_back(start_point);
            line_marker.points.push_back(end_point);
            
            line_marker.scale.x = 0.5; // Line width
            
            // Line color
            if (sat.is_faulty) {
                line_marker.color.r = 1.0;
                line_marker.color.g = 0.3;
                line_marker.color.b = 0.3;
            } else {
                line_marker.color.r = 0.3;
                line_marker.color.g = 0.8;
                line_marker.color.b = 0.3;
            }
            line_marker.color.a = 0.6;
            
            marker_array.markers.push_back(line_marker);
        }
        
        satellite_viz_pub_.publish(marker_array);
    }
    
    void visualizePosition() {
        visualization_msgs::Marker position_marker;
        position_marker.header.frame_id = user_frame_;  // Attach to the moving user frame
        position_marker.header.stamp = ros::Time::now();
        position_marker.ns = "position";
        position_marker.id = 0;
        position_marker.type = visualization_msgs::Marker::SPHERE;
        position_marker.action = visualization_msgs::Marker::ADD;
        
        // Position is at the origin of the user frame
        position_marker.pose.position.x = 0;
        position_marker.pose.position.y = 0;
        position_marker.pose.position.z = 0;
        
        position_marker.pose.orientation.w = 1.0;
        position_marker.scale.x = 2.0;
        position_marker.scale.y = 2.0;
        position_marker.scale.z = 2.0;
        
        position_marker.color.r = 0.0;
        position_marker.color.g = 0.0;
        position_marker.color.b = 1.0;
        position_marker.color.a = 1.0;
        
        position_viz_pub_.publish(position_marker);
    }
    
    void visualizeProtectionLevel() {
        visualization_msgs::Marker protection_level_marker;
        protection_level_marker.header.frame_id = user_frame_;  // Attach to the moving user frame
        protection_level_marker.header.stamp = ros::Time::now();
        protection_level_marker.ns = "protection_level";
        protection_level_marker.id = 0;
        protection_level_marker.type = visualization_msgs::Marker::CYLINDER;
        protection_level_marker.action = visualization_msgs::Marker::ADD;
        
        protection_level_marker.pose.position.x = 0;
        protection_level_marker.pose.position.y = 0;
        protection_level_marker.pose.position.z = 0;
        
        protection_level_marker.pose.orientation.w = 1.0;
        
        // Scale for visualization (radius = HPL, height = 2*VPL)
        protection_level_marker.scale.x = horizontal_protection_level_ * 2.0;
        protection_level_marker.scale.y = horizontal_protection_level_ * 2.0;
        protection_level_marker.scale.z = vertical_protection_level_ * 2.0;
        
        // Color based on alarm state
        if (horizontal_protection_level_ > alarm_limit_ || vertical_protection_level_ > alarm_limit_) {
            // Red if protection level exceeds alarm limit
            protection_level_marker.color.r = 1.0;
            protection_level_marker.color.g = 0.0;
            protection_level_marker.color.b = 0.0;
        } else {
            // Blue otherwise
            protection_level_marker.color.r = 0.0;
            protection_level_marker.color.g = 0.5;
            protection_level_marker.color.b = 1.0;
        }
        protection_level_marker.color.a = 0.3;
        
        protection_level_viz_pub_.publish(protection_level_marker);
    }
    
    void publishPositionWithCovariance(
            const Eigen::Vector3d& position_ecef,
            const Eigen::MatrixXd& covariance) {
        
        // Publish as PoseWithCovariance message
        geometry_msgs::PoseWithCovariance pose_msg;
        
        // Position is at the origin of the user frame
        pose_msg.pose.position.x = 0.0;
        pose_msg.pose.position.y = 0.0;
        pose_msg.pose.position.z = 0.0;
        
        // Identity quaternion
        pose_msg.pose.orientation.w = 1.0;
        pose_msg.pose.orientation.x = 0.0;
        pose_msg.pose.orientation.y = 0.0;
        pose_msg.pose.orientation.z = 0.0;
        
        // Convert covariance to local ENU frame
        double lat, lon, alt;
        ECEFToGeodetic(position_ecef, lat, lon, alt);
        Eigen::Matrix3d R = createRotationMatrix(lat, lon);
        Eigen::Matrix3d local_cov = R * covariance.block<3,3>(0,0) * R.transpose();
        
        // Copy position covariance to ROS message
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                pose_msg.covariance[i*6 + j] = local_cov(i, j);
            }
        }
        
        pose_with_cov_pub_.publish(pose_msg);
    }
    
    void publishGPSData() {
        sensor_msgs::NavSatFix gps_msg;
        gps_msg.header.stamp = ros::Time::now();
        gps_msg.header.frame_id = user_frame_;
        
        // Convert true position ECEF to geodetic for the message
        double lat, lon, alt;
        ECEFToGeodetic(true_position_ecef_, lat, lon, alt);
        
        gps_msg.latitude = lat * RAD_TO_DEG;
        gps_msg.longitude = lon * RAD_TO_DEG;
        gps_msg.altitude = alt;
        
        // Status fields
        gps_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
        gps_msg.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
        
        // Covariance
        gps_msg.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
        
        // Diagonal covariance from protection levels
        gps_msg.position_covariance[0] = pow(horizontal_protection_level_ / 2.0, 2); // East
        gps_msg.position_covariance[4] = pow(horizontal_protection_level_ / 2.0, 2); // North
        gps_msg.position_covariance[8] = pow(vertical_protection_level_ / 2.0, 2);   // Up
        
        simulated_gps_pub_.publish(gps_msg);
    }
    
    void broadcastTransform() {
        // Get center coordinates in geodetic for local frame calculations
        double center_lat, center_lon, center_alt;
        ECEFToGeodetic(trajectory_center_ecef_, center_lat, center_lon, center_alt);
        
        // Convert true position to ENU relative to trajectory center
        Eigen::Vector3d user_enu = ECEFToENU(true_position_ecef_, trajectory_center_ecef_, 
                                            center_lat, center_lon);
        
        // Create transform message
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = ros::Time::now();
        transform_stamped.header.frame_id = fixed_frame_;
        transform_stamped.child_frame_id = user_frame_;
        
        // Set translation based on ENU coordinates
        transform_stamped.transform.translation.x = user_enu(0);
        transform_stamped.transform.translation.y = user_enu(1);
        transform_stamped.transform.translation.z = user_enu(2);
        
        // Set orientation (identity quaternion for now)
        transform_stamped.transform.rotation.w = 1.0;
        transform_stamped.transform.rotation.x = 0.0;
        transform_stamped.transform.rotation.y = 0.0;
        transform_stamped.transform.rotation.z = 0.0;
        
        // Broadcast the transform
        tf_broadcaster_.sendTransform(transform_stamped);
    }
    
    void updatePath() {
        // Create a pose for the current position
        geometry_msgs::PoseStamped current_pose;
        current_pose.header.frame_id = fixed_frame_;
        current_pose.header.stamp = ros::Time::now();
        
        // Convert true position to ENU relative to trajectory center
        double center_lat, center_lon, center_alt;
        ECEFToGeodetic(trajectory_center_ecef_, center_lat, center_lon, center_alt);
        Eigen::Vector3d user_enu = ECEFToENU(true_position_ecef_, trajectory_center_ecef_, 
                                            center_lat, center_lon);
        
        current_pose.pose.position.x = user_enu(0);
        current_pose.pose.position.y = user_enu(1);
        current_pose.pose.position.z = user_enu(2);
        
        // Identity orientation
        current_pose.pose.orientation.w = 1.0;
        
        // Add to pose history
        pose_history_.push_back(current_pose);
        
        // Limit history size
        while (pose_history_.size() > max_path_history_) {
            pose_history_.pop_front();
        }
        
        // Update path message
        path_.header.stamp = ros::Time::now();
        path_.poses = std::vector<geometry_msgs::PoseStamped>(pose_history_.begin(), pose_history_.end());
        
        // Publish path
        path_pub_.publish(path_);
    }
    
    // Coordinate transformation utilities
    
    Eigen::Vector3d geodeticToECEF(double lat, double lon, double alt) {
        // WGS84 ellipsoid parameters
        const double a = 6378137.0;               // semi-major axis
        const double f = 1.0 / 298.257223563;     // flattening
        const double e_squared = f * (2.0 - f);   // eccentricity squared
        
        double sin_lat = sin(lat);
        double cos_lat = cos(lat);
        double sin_lon = sin(lon);
        double cos_lon = cos(lon);
        
        double N = a / sqrt(1.0 - e_squared * sin_lat * sin_lat);
        
        double x = (N + alt) * cos_lat * cos_lon;
        double y = (N + alt) * cos_lat * sin_lon;
        double z = (N * (1.0 - e_squared) + alt) * sin_lat;
        
        return Eigen::Vector3d(x, y, z);
    }
    
    void ECEFToGeodetic(const Eigen::Vector3d& ecef, double& lat, double& lon, double& alt) {
        // WGS84 ellipsoid parameters
        const double a = 6378137.0;               // semi-major axis
        const double f = 1.0 / 298.257223563;     // flattening
        const double b = a * (1.0 - f);           // semi-minor axis
        const double e_squared = f * (2.0 - f);   // eccentricity squared
        
        double x = ecef(0);
        double y = ecef(1);
        double z = ecef(2);
        
        // Calculate longitude
        lon = atan2(y, x);
        
        // Distance from Z-axis
        double p = sqrt(x*x + y*y);
        
        // Initial guess for latitude
        double lat_guess = atan2(z, p * (1.0 - e_squared));
        double N, h;
        
        // Iterative calculation for latitude and height
        for (int i = 0; i < 5; i++) {
            double sin_lat = sin(lat_guess);
            N = a / sqrt(1.0 - e_squared * sin_lat * sin_lat);
            h = p / cos(lat_guess) - N;
            lat_guess = atan2(z, p * (1.0 - e_squared * N / (N + h)));
        }
        
        lat = lat_guess;
        alt = p / cos(lat) - N;
        
        // Special case for poles
        if (p < 1.0) {
            lat = z > 0 ? M_PI/2 : -M_PI/2;
            alt = fabs(z) - b;
        }
    }
    
    Eigen::Vector3d ECEFToENU(const Eigen::Vector3d& ecef, const Eigen::Vector3d& ref_ecef, double ref_lat, double ref_lon) {
        // Calculate the difference in ECEF coordinates
        Eigen::Vector3d delta_ecef = ecef - ref_ecef;
        
        // Use the rotation matrix
        Eigen::Matrix3d R = createRotationMatrix(ref_lat, ref_lon);
        
        // Apply the rotation
        return R * delta_ecef;
    }
    
    Eigen::Vector3d ENUToECEF(const Eigen::Vector3d& enu, const Eigen::Vector3d& ref_ecef, double ref_lat, double ref_lon) {
        // Get rotation matrix
        Eigen::Matrix3d R = createRotationMatrix(ref_lat, ref_lon);
        
        // Invert (actually transpose since it's orthogonal)
        Eigen::Matrix3d R_inv = R.transpose();
        
        // Apply the rotation and add reference
        return ref_ecef + R_inv * enu;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_raim_node");
    GPSRAIMNode raim_node;
    ros::spin();
    return 0;
}