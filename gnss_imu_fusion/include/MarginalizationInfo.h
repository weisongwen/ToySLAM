#ifndef MARGINALIZATION_INFO_H
#define MARGINALIZATION_INFO_H

#include <vector>
#include <map>
#include <Eigen/Dense>

// Forward declaration to avoid circular dependencies
class ResidualBlockInfo;

// Marginalization information class - handles Schur complement
class MarginalizationInfo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    MarginalizationInfo();

    // Destructor
    ~MarginalizationInfo();

    // Adds a residual block to be considered in the marginalization process.
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);

    // Evaluates all residual blocks to compute jacobians and residuals.
    void preMarginalize();

    // Performs the Schur complement to marginalize out specified parameters.
    void marginalize();

    // Utility to get the local parameterization size (e.g., 7 for pose becomes 6 for update).
    int localSize(int size);
    
    // Utility to get the global size from a local size.
    int globalSize(int size);

    // --- Accessors for MarginalizationFactor ---

    const Eigen::MatrixXd& getLinearizedJacobians() const;
    const Eigen::VectorXd& getLinearizedResiduals() const;
    const std::vector<double*>& getKeepBlockData() const;
    const std::vector<int>& getKeepBlockSizes() const;
    int getKeepBlockSize() const;
    const std::vector<int>& getKeepBlockIdx() const;

private:
    // Residual blocks to be marginalized
    std::vector<ResidualBlockInfo*> residual_block_infos;

    // Parameter block information
    std::map<double*, int> parameter_block_size;
    std::map<double*, int> parameter_block_idx;
    std::map<double*, double*> parameter_block_data;

    // Kept parameter block information
    int keep_block_size;
    std::vector<int> keep_block_sizes;
    std::vector<double*> keep_block_data;
    std::vector<double*> keep_block_addr;
    std::vector<int> keep_block_idx;

    // Linearized system after marginalization (prior information)
    Eigen::MatrixXd linearized_jacobians_;
    Eigen::VectorXd linearized_residuals_;
};

#endif // MARGINALIZATION_INFO_H
