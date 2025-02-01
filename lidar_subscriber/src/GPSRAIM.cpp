#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <ros/ros.h>

#include <string>

using namespace Eigen;
using namespace std;

const double TARGET_RISK = 1e-7; // Integrity risk requirement
const double FAULT_PROB_BASE = 1e-5; // Base satellite fault probability

struct Satellite {
    Vector3d pos_ecef;    // ECEF position (m)
    double pseudorange;   // Measured pseudorange (m)
    double variance;      // Measurement variance (m²)
    double fault_prob;    // Prior fault probability
};

class ARAIM {
public:
    ARAIM(const vector<Satellite>& sats) : satellites(sats) {}
    
    void calculateIntegrity() {
        // Full-set solution
        Vector4d full_pos;
        Matrix4d full_cov;
        if (!solvePosition(satellites, full_pos, full_cov)) {
            cerr << "Full solution failed!" << endl;
            return;
        }

        // Calculate all single-fault hypotheses
        vector<Hypothesis> hypotheses;
        for (size_t i = 0; i < satellites.size(); ++i) {
            Hypothesis h;
            h.excluded_sat = i;
            h.fault_prob = satellites[i].fault_prob;
            
            // Create subset without the excluded satellite
            vector<Satellite> subset = satellites;
            subset.erase(subset.begin() + i);
            
            if (solvePosition(subset, h.position, h.covariance)) {
                h.separation = (full_pos.head<3>() - h.position.head<3>());
                hypotheses.push_back(h);
            }
        }

        // Calculate protection levels
        calculateProtectionLevels(full_pos, full_cov, hypotheses);
    }

private:
    struct Hypothesis {
        Vector4d position;
        Matrix4d covariance;
        Vector3d separation;
        size_t excluded_sat;
        double fault_prob;
    };

    vector<Satellite> satellites;

    bool solvePosition(const vector<Satellite>& sats, Vector4d& result, Matrix4d& cov) {
        ceres::Problem problem;
        result << 0, 0, 0, 0;

        for (const auto& sat : sats) {
            ceres::CostFunction* cf = 
                new ceres::AutoDiffCostFunction<PseudorangeResidual, 1, 4>(
                    new PseudorangeResidual(sat));
            auto ID = problem.AddResidualBlock(cf, nullptr, result.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // get the residuals


        std::cout<< "result-> " << std::setprecision (10)<<result <<"\n";

        if (!summary.IsSolutionUsable()) return false;

        // Calculate covariance matrix
        MatrixXd H(sats.size(), 4);
        VectorXd W(sats.size());
        for (size_t i = 0; i < sats.size(); ++i) {
            Vector3d sat_pos = sats[i].pos_ecef;
            double dx = sat_pos[0] - result[0];
            double dy = sat_pos[1] - result[1];
            double dz = sat_pos[2] - result[2];
            double range = sqrt(dx*dx + dy*dy + dz*dz);
            
            H(i, 0) = dx / range;
            H(i, 1) = dy / range;
            H(i, 2) = dz / range;
            H(i, 3) = 1.0;
            W[i] = 1.0 / sats[i].variance;
        }

        cov = (H.transpose() * W.asDiagonal() * H).inverse();
        return true;
    }

    void calculateProtectionLevels(const Vector4d& full_pos, const Matrix4d& full_cov,
                                  const vector<Hypothesis>& hypotheses) {
        // Calculate protection levels using MHSS method
        double hpl = 0.0, vpl = 0.0;
        double total_risk_h = 0.0, total_risk_v = 0.0;

        for (const auto& h : hypotheses) {
            // Calculate combined uncertainties
            Vector3d full_std = full_cov.diagonal().head<3>().cwiseSqrt();
            Vector3d h_std = h.covariance.diagonal().head<3>().cwiseSqrt();
            Vector3d combined_std = (full_std.array().square() + h_std.array().square()).sqrt();

            // Calculate K-factor based on allocated risk
            double allocated_risk = TARGET_RISK * h.fault_prob;
            double K = inverseQFunction(allocated_risk);

            // Horizontal components
            double h_sep = h.separation.head<2>().norm();
            double h_pl = h_sep + K * combined_std.head<2>().norm();
            hpl = max(hpl, h_pl);

            // Vertical component
            double v_sep = abs(h.separation[2]);
            double v_pl = v_sep + K * combined_std[2];
            vpl = max(vpl, v_pl);

            // Calculate actual risk contributions (for verification)
            total_risk_h += h.fault_prob * QFunction((hpl - h_sep)/combined_std.head<2>().norm());
            total_risk_v += h.fault_prob * QFunction((vpl - v_sep)/combined_std[2]);
        }

        cout << "Calculated Protection Levels:" << endl;
        cout << "HPL: " << hpl << " meters" << endl;
        cout << "VPL: " << vpl << " meters" << endl;
        cout << "Achieved Horizontal Risk: " << total_risk_h << endl;
        cout << "Achieved Vertical Risk: " << total_risk_v << endl;
    }

    static double QFunction(double x) {
        return 0.5 * erfc(x / sqrt(2.0));
    }

    static double inverseQFunction(double p) {
        // Approximation of inverse Q function for small p
        if (p <= 0) return numeric_limits<double>::infinity();
        if (p >= 0.5) return 0.0;
        
        double t = sqrt(-2.0 * log(p));
        return t - (2.515517 + 0.802853*t + 0.010328*t*t) /
                   (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t);
    }

    struct PseudorangeResidual {
        Satellite sat;
        PseudorangeResidual(Satellite s) : sat(s) {}
        
        template <typename T>
        bool operator()(const T* const pos, T* residual) const {
            T dx = T(sat.pos_ecef[0]) - pos[0];
            T dy = T(sat.pos_ecef[1]) - pos[1];
            T dz = T(sat.pos_ecef[2]) - pos[2];
            T range = sqrt(dx*dx + dy*dy + dz*dz);
            residual[0] = (T(sat.pseudorange) - (range + pos[3])) / T(sqrt(sat.variance));
            return true;
        }
    };
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "GPS_RAIM_node");

    // vector<Satellite> sats = {
    //     {Vector3d(-1.6e7, 1.7e7, 2.0e7), 2.1e7, 25.0, FAULT_PROB_BASE},
    //     {Vector3d(1.8e7, -1.3e7, 2.1e7), 2.2e7, 25.0, FAULT_PROB_BASE},
    //     {Vector3d(-2.0e7, -1.4e7, 1.9e7), 2.3e7, 25.0, FAULT_PROB_BASE},
    //     {Vector3d(1.7e7, 1.6e7, 2.2e7), 2.0e7, 25.0, FAULT_PROB_BASE},
    //     {Vector3d(-1.5e7, -1.8e7, 2.1e7), 2.1e7, 25.0, FAULT_PROB_BASE}
    // };

    vector<Satellite> sats = {
        {Vector3d(-10816000, 24273000, 0), 20841005, 25.0, FAULT_PROB_BASE},
        {Vector3d(-4000000, 22630000, 13280000), 20380003, 25.0, FAULT_PROB_BASE},
        {Vector3d(-16820000, 20040000, -4610000), 21707006, 25.0, FAULT_PROB_BASE},
        {Vector3d(3262000, 18490000, 18790000), 21723004, 25.0, FAULT_PROB_BASE},
        {Vector3d(-23000000, 13280000, 0), 22160005, 25.0, FAULT_PROB_BASE}
    };

    ARAIM araim(sats);
    araim.calculateIntegrity();

    ros::spin();
    return 0;
}