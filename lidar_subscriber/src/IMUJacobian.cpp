void propagateCovariance(const Vector3d& accel, const Vector3d& gyro, double delta_t) {
    Matrix9d F = Matrix9d::Zero();
    Matrix9d G = Matrix9d::Zero();
    
    Matrix3d R = dR;
    Vector3d a = R * accel;
    Vector3d w = gyro;

    // State transition matrix
    F.block<3,3>(0,0) = Matrix3d::Identity();
    F.block<3,3>(0,3) = -delta_t * R;
    F.block<3,3>(3,3) = Matrix3d::Identity();
    F.block<3,3>(3,6) = -R * delta_t;
    F.block<3,3>(6,6) = Matrix3d::Identity();

    // Noise covariance matrix
    G.block<3,3>(0,0) = 0.5 * delta_t * delta_t * R;
    G.block<3,3>(3,3) = delta_t * R;
    G.block<3,3>(6,6) = Matrix3d::Identity() * delta_t;

    covariance = F * covariance * F.transpose() + 
                G * (ACCEL_NOISE_SIGMA * ACCEL_NOISE_SIGMA * Matrix9d::Identity()) * G.transpose();
}

struct IMUFactor : public ceres::SizedCostFunction<9, 7, 9, 7, 9> {
    IMUFactor(const IMUPreintegrator& preint) : preint_(preint) {}

    virtual bool Evaluate(double const* const* parameters, 
                         double* residuals, 
                         double** jacobians) const {
        // [Previous parameter unpacking and residual calculation remains the same]

        // Jacobian calculations
        if (jacobians) {
            Matrix3d Ri = qi.toRotationMatrix();
            Matrix3d Rj = qj.toRotationMatrix();
            
            // Pre-compute common terms
            Matrix3d DR = preint_.dR.transpose();
            Matrix3d RiT = Ri.transpose();
            Matrix3d RiT_Rj = RiT * Rj;
            Matrix3d Jr = RotationJacobian((DR * RiT_Rj).log());

            // Jacobian w.r.t. pose_i (position)
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                
                // Position residual derivatives
                J.block<3,3>(0,0) = -RiT;
                
                // Velocity residual derivatives
                J.block<3,3>(3,0) = Matrix3d::Zero();
                
                // Orientation residual derivatives
                J.block<3,3>(6,0) = Matrix3d::Zero();
                
                // Quaternion derivatives (using right perturbation)
                Matrix3d dq_dtheta = 0.5 * RiT;
                J.block<3,3>(0,3) = RiT * skewSymmetric(vi * preint_.dt - (pj - pi));
                J.block<3,3>(3,3) = RiT * skewSymmetric(vi);
                J.block<3,3>(6,3) = -Jr * RiT_Rj.transpose();
            }

            // Jacobian w.r.t. speed_bias_i
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                
                // Velocity residual derivatives
                J.block<3,3>(3,0) = -RiT;
                
                // Position residual derivatives
                J.block<3,3>(0,0) = -RiT * preint_.dt;
                
                // Accelerometer bias derivatives
                J.block<3,3>(0,3) = -preint_.dP_dba;
                J.block<3,3>(3,3) = -preint_.dV_dba;
                
                // Gyroscope bias derivatives
                J.block<3,3>(6,6) = -preint_.dR_dbg;
            }

            // Jacobian w.r.t. pose_j
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 9, 7, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                
                // Position residual derivatives
                J.block<3,3>(0,0) = RiT;
                
                // Orientation residual derivatives
                J.block<3,3>(6,3) = Jr;
            }

            // Jacobian w.r.t. speed_bias_j
            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> J(jacobians[3]);
                J.setZero();
                
                // Velocity residual derivatives
                J.block<3,3>(3,0) = RiT;
            }
        }
        return true;
    }

private:
    Matrix3d skewSymmetric(const Vector3d& v) const {
        Matrix3d S;
        S <<  0, -v.z(),  v.y(),
            v.z(),   0, -v.x(),
            -v.y(), v.x(),   0;
        return S;
    }

    Matrix3d RotationJacobian(const Vector3d& phi) const {
        double theta = phi.norm();
        Matrix3d J = Matrix3d::Identity();
        if (theta > 1e-6) {
            Matrix3d phi_hat = skewSymmetric(phi);
            J = Matrix3d::Identity() 
                + (1 - cos(theta)) / (theta * theta) * phi_hat
                + (theta - sin(theta)) / (theta * theta * theta) * phi_hat * phi_hat;
        }
        return J;
    }
};