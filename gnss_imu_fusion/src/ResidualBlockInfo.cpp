#include "ResidualBlockInfo.h"
#include <iostream> // For replacing ROS_WARN
#include <cstring>  // For memset

// A simple replacement for ROS logging macros for this example
#define ROS_WARN(msg) std::cerr << "[WARN] " << msg << std::endl

ResidualBlockInfo::ResidualBlockInfo(ceres::CostFunction* _cost_function,
                                     ceres::LossFunction* _loss_function,
                                     std::vector<double*>& _parameter_blocks,
                                     std::vector<int>& _drop_set)
    : cost_function(_cost_function), loss_function(_loss_function),
      parameter_blocks(_parameter_blocks), drop_set(_drop_set) {

    // Calculate sizes from the cost function
    num_residuals = cost_function->num_residuals();
    parameter_block_sizes.clear();
    
    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
        parameter_block_sizes.push_back(cost_function->parameter_block_sizes()[i]);
    }

    // Allocate memory for raw jacobian pointers
    raw_jacobians = new double*[parameter_blocks.size()];
    jacobians.resize(parameter_blocks.size());

    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
        jacobians[i].resize(num_residuals, parameter_block_sizes[i]);
        jacobians[i].setZero();
    }

    residuals.resize(num_residuals);
    residuals.setZero();
}

ResidualBlockInfo::~ResidualBlockInfo() {
    delete[] raw_jacobians;
    // Only delete the cost function if we own it
    if (cost_function) {
        delete cost_function;
        cost_function = nullptr;
    }
        
    // Only delete the loss function if we own it  
    if (loss_function) {
        delete loss_function;
        loss_function = nullptr;
    }
}

void ResidualBlockInfo::Evaluate() {
            // Skip evaluation if we don't have all parameters
        if (parameter_blocks_data.size() != parameter_blocks.size()) {
            ROS_WARN("Parameter blocks data size mismatch in Evaluate()");
            return;
        }
        
        // Allocate memory for parameters and residuals
        double** parameters = new double*[parameter_blocks.size()];
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            parameters[i] = parameter_blocks_data[i];
            raw_jacobians[i] = new double[num_residuals * parameter_block_sizes[i]];
            memset(raw_jacobians[i], 0, sizeof(double) * num_residuals * parameter_block_sizes[i]);
        }
        
        double* raw_residuals = new double[num_residuals];
        memset(raw_residuals, 0, sizeof(double) * num_residuals);
        
        // Evaluate the cost function
        cost_function->Evaluate(parameters, raw_residuals, raw_jacobians);
        
        // Apply loss function if needed
        if (loss_function) {
            double residual_scaling = 1.0;
            double alpha_sq_norm = 0.0;
            
            for (int i = 0; i < num_residuals; i++) {
                alpha_sq_norm += raw_residuals[i] * raw_residuals[i];
            }
            
            double sqrt_rho1 = 1.0;
            if (alpha_sq_norm > 0) {
                double rho[3];
                loss_function->Evaluate(alpha_sq_norm, rho);
                sqrt_rho1 = sqrt(rho[1]);
                
                if (sqrt_rho1 == 0) {
                    residual_scaling = 0.0;
                } else {
                    residual_scaling = sqrt_rho1 / alpha_sq_norm;
                }
            }
            
            for (int i = 0; i < num_residuals; i++) {
                raw_residuals[i] *= sqrt_rho1;
            }
            
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
                for (int j = 0; j < parameter_block_sizes[i] * num_residuals; j++) {
                    raw_jacobians[i][j] *= residual_scaling;
                }
            }
        }
        
        // Copy raw residuals to Eigen vector
        for (int i = 0; i < num_residuals; i++) {
            residuals(i) = raw_residuals[i];
        }
        
        // Copy raw jacobians to Eigen matrices
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                mat_jacobian(raw_jacobians[i], num_residuals, parameter_block_sizes[i]);
            jacobians[i] = mat_jacobian;
        }
        
        // Clean up
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            delete[] raw_jacobians[i];
        }
        delete[] parameters;
        delete[] raw_residuals;
}

int ResidualBlockInfo::localSize(int size) {
    return size == 7 ? 6 : size;
}
