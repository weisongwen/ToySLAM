#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <random>
#include <vector>
#include <cmath>

// Add this function before the GPSRaimNode class definition
double erfinv(double x) {
    // Approximation of inverse error function
    // Accurate to about 7 decimal places
    
    // Handle edge cases
    if (x >= 1.0) return std::numeric_limits<double>::infinity();
    if (x <= -1.0) return -std::numeric_limits<double>::infinity();
    if (x == 0) return 0;

    const double pi = 3.141592653589793238462643383279502884;

    // Coefficients for approximation
    const double a[4] = {
        0.886226899, -1.645349621, 0.914624893, -0.140543331
    };
    const double b[4] = {
        -2.118377725, 1.442710462, -0.329097515, 0.012229801
    };
    const double c[4] = {
        -1.970840454, -1.624906493, 3.429567803, 1.641345311
    };
    const double d[2] = {
        3.543889200, 1.637067800
    };

    double y = x;
    double result;

    if (abs(y) <= 0.7) {
        double z = y * y;
        double num = ((a[3] * z + a[2]) * z + a[1]) * z + a[0];
        double den = ((b[3] * z + b[2]) * z + b[1]) * z + b[0];
        result = y * num / den;
    }
    else {
        double z = sqrt(-log((1.0 - abs(y)) / 2.0));
        double num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
        double den = (d[1] * z + d[0]) * z + 1.0;
        result = (y > 0 ? 1 : -1) * num / den;
    }

    // One iteration of Newton's method to refine result
    result = result - (erf(result) - x) / (2.0/sqrt(pi) * exp(-result * result));

    return result;
}


class GPSRaimNode {
private:
    ros::NodeHandle nh_;
    ros::Publisher position_pub_;
    ros::Publisher raim_pub_;
    
    // Constants
    const double SPEED_OF_LIGHT = 299792458.0;  // m/s
    const int NUM_SATELLITES = 8;
    const double ALPHA = 0.05;  // False alarm probability
    const double BETA = 0.05;   // Missed detection probability
    const double T_ALPHA = 12.592; // Chi-square threshold for alpha (dof=6)
    
    // RAIM parameters
    const double PHMI = 1e-7;  // Integrity risk requirement
    const double PFA = 1e-5;   // False alert probability requirement
    const double PMD = 1e-3;   // Missed detection probability requirement
    const double HAL = 40.0;   // Horizontal Alert Limit (meters)
    const double VAL = 50.0;   // Vertical Alert Limit (meters)
    
    struct Satellite {
        Eigen::Vector3d position;
        double pseudorange;
        double elevation;
        double azimuth;
        double weight;         // Measurement weight based on elevation
        double variance;       // Measurement variance
    };
    
    std::vector<Satellite> satellites_;
    Eigen::Vector4d receiver_state_;  // [x, y, z, clock_bias]
    std::random_device rd_;
    std::default_random_engine gen_;
    std::normal_distribution<double> noise_dist_;
    ros::Timer timer;

public:
    GPSRaimNode() : 
        gen_(rd_()),
        noise_dist_(0.0, 3.0)
    {
        position_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/gps/position", 1);
        raim_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/gps/raim", 1);
        
        receiver_state_ = Eigen::Vector4d(0, 0, 0, 0);
        
        initializeSatellites();
        
        timer = nh_.createTimer(ros::Duration(1.0), 
                                         &GPSRaimNode::processCallback, this);
    }

private:

struct PseudorangeCostFunctor {
    PseudorangeCostFunctor(const Eigen::Vector3d& sat_pos, 
                          const double measured_range,
                          const double weight)
        : sat_pos_(sat_pos), measured_range_(measured_range), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const receiver_state, T* residual) const {
        Eigen::Matrix<T,3,1> receiver_pos(receiver_state[0], 
                                        receiver_state[1], 
                                        receiver_state[2]);
        Eigen::Matrix<T,3,1> sat_pos_t = sat_pos_.cast<T>();
        
        // Compute predicted range
        T predicted_range = (sat_pos_t - receiver_pos).norm();
        
        // Add clock bias
        predicted_range += receiver_state[3];
        
        // Compute weighted residual
        residual[0] = T(weight_) * (predicted_range - T(measured_range_));
        
        return true;
    }

    const Eigen::Vector3d sat_pos_;
    const double measured_range_;
    const double weight_;
};

    // Add these member variables to the class
    double last_pdop_;
    double last_hdop_;
    double last_vdop_;
    double last_tdop_;

void singlePointPosition() {
    // Initial guess is current receiver state
    double receiver_state_array[4] = {
        receiver_state_(0),
        receiver_state_(1),
        receiver_state_(2),
        receiver_state_(3)
    };

    // Build the problem using Ceres
    ceres::Problem problem;
    
    for (const auto& sat : satellites_) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<PseudorangeCostFunctor, 1, 4>(
                new PseudorangeCostFunctor(sat.position, 
                                         sat.pseudorange,
                                         sqrt(sat.weight)));
        
        problem.AddResidualBlock(cost_function, nullptr, receiver_state_array);
    }

    // Set solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Check if optimization succeeded
    if (summary.termination_type == ceres::CONVERGENCE) {
        // Update receiver state
        for (int i = 0; i < 4; ++i) {
            receiver_state_(i) = receiver_state_array[i];
        }
        std::cout<<"receiver state -> "<< receiver_state_<<"\n";
    } else {
        ROS_WARN("GPS position estimation did not converge!");
    }

    // Optional: Compute DOP values
    Eigen::MatrixXd G = computeGeometryMatrix();
    Eigen::MatrixXd GTG = G.transpose() * G;
    Eigen::MatrixXd P = GTG.inverse();

    double PDOP = sqrt(P(0,0) + P(1,1) + P(2,2));
    double HDOP = sqrt(P(0,0) + P(1,1));
    double VDOP = sqrt(P(2,2));
    double TDOP = sqrt(P(3,3));

    // Store DOP values if needed
    last_pdop_ = PDOP;
    last_hdop_ = HDOP;
    last_vdop_ = VDOP;
    last_tdop_ = TDOP;
    }

    double computeElevationWeight(double elevation_rad) {
        // Elevation-dependent weighting model
        const double a = 0.13;
        const double b = 0.53;
        double elevation_deg = elevation_rad * 180.0 / M_PI;
        return pow(a + b * sin(elevation_rad), 2);
    }

    void initializeSatellites() {
        // Similar to previous implementation but with additional parameters
        double orbit_radius = 26600000.0;
        // double orbit_radius = 26600000.0;
        satellites_.clear();
        
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            Satellite sat;
            double angle = 2.0 * M_PI * i / NUM_SATELLITES;
            double elevation = M_PI/4.0 + 0.2 * noise_dist_(gen_); // Add some variation
            // double elevation = M_PI/4.0; // Add some variation
            
            sat.position = Eigen::Vector3d(
                orbit_radius * cos(angle) * cos(elevation),
                orbit_radius * sin(angle) * cos(elevation),
                orbit_radius * sin(elevation)
            );
            
            Eigen::Vector3d receiver_pos = receiver_state_.head<3>();
            Eigen::Vector3d sat_vector = sat.position - receiver_pos;
            double true_range = sat_vector.norm();
            
            // Calculate elevation and azimuth
            sat.elevation = asin(sat_vector(2) / sat_vector.norm());
            sat.azimuth = atan2(sat_vector(1), sat_vector(0));
            
            // Compute elevation-dependent weighting
            sat.weight = computeElevationWeight(sat.elevation);
            
            // Compute measurement variance based on elevation-dependent model
            double sigma_ura = 3.0; // Base URA value
            sat.variance = pow(sigma_ura / sin(sat.elevation), 2);
            
            // Generate pseudorange with elevation-dependent noise
            double noise_scale = 1.0 / sin(sat.elevation);
            sat.pseudorange = true_range + 
                            receiver_state_(3) + 
                            noise_dist_(gen_) * noise_scale;
            
            satellites_.push_back(sat);
        }
    }

    Eigen::MatrixXd computeWeightMatrix() {
        Eigen::MatrixXd W = Eigen::MatrixXd::Zero(NUM_SATELLITES, NUM_SATELLITES);
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            W(i,i) = satellites_[i].weight;
        }
        return W;
    }

    Eigen::MatrixXd computeGeometryMatrix() {
        Eigen::MatrixXd G(NUM_SATELLITES, 4);
        Eigen::Vector3d receiver_pos = receiver_state_.head<3>();
        
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            Eigen::Vector3d sat_vector = satellites_[i].position - receiver_pos;
            double range = sat_vector.norm();
            G.block<1,3>(i,0) = -sat_vector.transpose() / range;
            G(i,3) = 1.0;
        }
        return G;
    }

    void computeProtectionLevels(const Eigen::MatrixXd& G, 
                               const Eigen::MatrixXd& W,
                               double& HPL,
                               double& VPL) {
        // Compute weighted least squares solution matrix
        Eigen::MatrixXd GT_W = G.transpose() * W;
        Eigen::MatrixXd H = (GT_W * G).inverse() * GT_W;
        
        // Compute fault-free covariance matrix
        Eigen::MatrixXd P = H * W.inverse() * H.transpose();
        
        // Transform to local East-North-Up (ENU) frame
        double lat = 0.0; // Assuming receiver position is known
        double lon = 0.0;
        
        // Rotation matrix from ECEF to ENU
        Eigen::Matrix3d R_ECEF_ENU;
        R_ECEF_ENU << -sin(lon), cos(lon), 0,
                     -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat),
                      cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat);
        
        // Extract position components of the covariance matrix
        Eigen::Matrix3d P_pos = P.block<3,3>(0,0);
        Eigen::Matrix3d P_ENU = R_ECEF_ENU * P_pos * R_ECEF_ENU.transpose();
        
        // Compute horizontal position variance
        double var_E = P_ENU(0,0);
        double var_N = P_ENU(1,1);
        double cov_EN = P_ENU(0,1);
        
        // Compute semi-major axis of horizontal error ellipse
        double d_major = 0.5 * (var_E + var_N + 
                       sqrt(pow(var_E - var_N, 2) + 4*pow(cov_EN, 2)));
        
        // Compute vertical position variance
        double var_U = P_ENU(2,2);
        
        // Slope parameters for each satellite
        std::vector<double> slopes(NUM_SATELLITES);
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(NUM_SATELLITES);
            e_i(i) = 1.0;
            
            // Compute slope for each satellite
            Eigen::VectorXd p_i = H * e_i;
            Eigen::Vector3d p_i_ENU = R_ECEF_ENU * p_i.head<3>();
            double horizontal_comp = sqrt(pow(p_i_ENU(0), 2) + pow(p_i_ENU(1), 2));
            double vertical_comp = abs(p_i_ENU(2));
            
            slopes[i] = sqrt(pow(horizontal_comp, 2) + pow(vertical_comp, 2));
        }
        
        // Find maximum slope
        double max_slope = *std::max_element(slopes.begin(), slopes.end());
        
        // Compute bias amplification factors
        double H_bias = sqrt(d_major);
        double V_bias = sqrt(var_U);
        
        // Non-centrality parameter based on PMD
        double lambda = computeNoncentralityParameter();
        
        // Compute protection levels
        HPL = max_slope * H_bias * sqrt(lambda);
        VPL = max_slope * V_bias * sqrt(lambda);
    }

    double computeNoncentralityParameter() {
        // Compute non-centrality parameter based on PMD and PFA
        // Using inverse chi-square distribution
        double dof = NUM_SATELLITES - 4; // Degrees of freedom
        double Q_md = sqrt(2) * erfinv(1 - 2*PMD);
        double Q_fa = sqrt(2) * erfinv(1 - 2*PFA);
        
        return pow(Q_md + Q_fa, 2);
    }

    void performRAIMCheck(const Eigen::VectorXd& residuals,
                         const Eigen::MatrixXd& G,
                         const Eigen::MatrixXd& W,
                         bool& raim_available,
                         int& faulty_satellite) {
        // Compute test statistic
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(NUM_SATELLITES, NUM_SATELLITES) - 
                           G * (G.transpose() * W * G).inverse() * G.transpose() * W;
        double test_statistic = residuals.transpose() * W * S * W * residuals;
        
        // Check global test
        raim_available = (test_statistic < T_ALPHA);
        
        // Local test for fault detection
        faulty_satellite = -1;
        if (!raim_available) {
            std::vector<double> normalized_residuals(NUM_SATELLITES);
            for (int i = 0; i < NUM_SATELLITES; ++i) {
                double w_i = sqrt(W(i,i));
                normalized_residuals[i] = abs(w_i * residuals(i)) / sqrt(S(i,i));
            }
            
            // Find satellite with maximum normalized residual
            auto max_it = std::max_element(normalized_residuals.begin(), 
                                         normalized_residuals.end());
            faulty_satellite = std::distance(normalized_residuals.begin(), max_it);
        }
    }

    void processCallback(const ros::TimerEvent& event) {
        // Update satellite positions and measurements
        initializeSatellites();
        
        // Perform single point positioning (using previous implementation)
        singlePointPosition();
        
        // Compute geometry and weight matrices
        Eigen::MatrixXd G = computeGeometryMatrix();
        Eigen::MatrixXd W = computeWeightMatrix();
        
        // Compute residuals
        Eigen::VectorXd residuals(NUM_SATELLITES);
        for (int i = 0; i < NUM_SATELLITES; ++i) {
            Eigen::Vector3d receiver_pos = receiver_state_.head<3>();
            double predicted_range = (satellites_[i].position - receiver_pos).norm() + 
                                   receiver_state_(3);
            residuals(i) = satellites_[i].pseudorange - predicted_range;
        }
        
        // Perform RAIM check
        bool raim_available;
        int faulty_satellite;
        performRAIMCheck(residuals, G, W, raim_available, faulty_satellite);
        
        // Compute protection levels
        double HPL, VPL;
        computeProtectionLevels(G, W, HPL, VPL);
        
        // Publish RAIM results
        sensor_msgs::NavSatFix raim_msg;
        raim_msg.header.stamp = ros::Time::now();
        raim_msg.header.frame_id = "gps";
        raim_msg.position_covariance[0] = HPL;
        raim_msg.position_covariance[4] = HPL;
        raim_msg.position_covariance[8] = VPL;
        raim_msg.status.status = raim_available ? 0 : -1;
        raim_msg.status.service = faulty_satellite + 1; // 0 means no fault
        
        raim_pub_.publish(raim_msg);
        
        // Publish position (using previous implementation)
        publishPosition();
    }

    void publishPosition() {
        sensor_msgs::NavSatFix pos_msg;
        pos_msg.header.stamp = ros::Time::now();
        pos_msg.header.frame_id = "gps";
        pos_msg.latitude = receiver_state_(0);
        pos_msg.longitude = receiver_state_(1);
        pos_msg.altitude = receiver_state_(2);
        
        position_pub_.publish(pos_msg);
    }

    // ... (keep the previous singlePointPosition implementation)
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gps_raim_node");
    GPSRaimNode raim_node;
    ros::spin();
    return 0;
}