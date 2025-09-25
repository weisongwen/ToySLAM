#ifndef RESIDUAL_BLOCK_INFO_H
#define RESIDUAL_BLOCK_INFO_H

#include <vector>
#include <Eigen/Dense>
#include "ceres/cost_function.h"
#include "ceres/loss_function.h"

// Structure to hold all information related to a single residual block.
struct ResidualBlockInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor: Initializes the residual block with necessary components.
    ResidualBlockInfo(ceres::CostFunction* _cost_function,
                      ceres::LossFunction* _loss_function,
                      std::vector<double*>& _parameter_blocks,
                      std::vector<int>& _drop_set);

    // Destructor: Cleans up dynamically allocated memory.
    ~ResidualBlockInfo();

    // Evaluates the cost function to compute residuals and jacobians.
    void Evaluate();

    // Utility to get the local parameterization size.
    int localSize(int size);

    // --- Members ---
    
    // Core Ceres components
    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;

    // Pointers to the actual parameter blocks
    std::vector<double*> parameter_blocks;
    
    // Parameter block sizes, inferred from cost_function
    std::vector<int> parameter_block_sizes;
    
    // Number of residuals, inferred from cost_function
    int num_residuals;

    // Indices of parameter blocks to be marginalized out from this residual
    std::vector<int> drop_set;

    // --- Data for Evaluation ---

    // Pointers to the backed-up data of parameter blocks for evaluation
    std::vector<double*> parameter_blocks_data;

    // Raw C-style arrays for interfacing with Ceres' Evaluate API
    double** raw_jacobians;

    // Eigen structures to hold the computed jacobians and residuals
    std::vector<Eigen::MatrixXd> jacobians;
    Eigen::VectorXd residuals;
};

#endif // RESIDUAL_BLOCK_INFO_H
