#include "MarginalizationInfo.h"
#include "ResidualBlockInfo.h" 
#include <iostream>            // For replacing ROS_WARN/ERROR with std::cerr/cout
#include <algorithm>           // For std::find
#include <cstring>             // For memcpy

// A simple replacement for ROS logging macros for this example
#define ROS_WARN(fmt, ...)   fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define ROS_ERROR(fmt, ...)  fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define ROS_INFO_STREAM(msg) std::cout << "[INFO] " << msg << std::endl

MarginalizationInfo::MarginalizationInfo() {
    keep_block_size = 0;
    keep_block_idx.clear();
    keep_block_data.clear();
    keep_block_addr.clear();
}

MarginalizationInfo::~MarginalizationInfo() {
    // Clean up parameter block data
    for (auto& it : parameter_block_data) {
        if (it.second) {
            delete[] it.second;
            it.second = nullptr;
        }
    }
    parameter_block_data.clear();

    // Clean up residual blocks
    for (auto& it : residual_block_infos) {
        delete it;
    }
    residual_block_infos.clear();
}

int MarginalizationInfo::localSize(int size) {
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) {
    return size == 6 ? 7 : size;
}

void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo* residual_block_info) {
    if (!residual_block_info) {
            ROS_WARN("Trying to add null ResidualBlockInfo");
            return;
        }
        
        residual_block_infos.emplace_back(residual_block_info);
        
        // Add all parameter blocks to our tracking
        for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
            double* addr = residual_block_info->parameter_blocks[i];
            // Skip null parameter blocks
            if (!addr) {
                ROS_WARN("Null parameter block address in ResidualBlockInfo");
                continue;
            }
            
            int size = residual_block_info->parameter_block_sizes[i];
            if (size <= 0) {
                ROS_WARN("Invalid parameter block size: %d", size);
                continue;
            }
            
            parameter_block_size[addr] = size;
            
            // If this is a new parameter block, make a copy of its data
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                double* data = new double[size];
                memcpy(data, addr, sizeof(double) * size);
                parameter_block_data[addr] = data;
                parameter_block_idx[addr] = 0;
            }
        }
}

void MarginalizationInfo::preMarginalize() {
    // Evaluate all residual blocks (compute Jacobians and residuals)
    for (auto it : residual_block_infos) {
        if (!it) continue;
        
        it->parameter_blocks_data.clear();
        
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            double* addr = it->parameter_blocks[i];
            // Skip null parameter blocks
            if (!addr) continue;
            
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                ROS_ERROR("Parameter block %p not found in marginalization info", addr);
                continue;
            }
            
            it->parameter_blocks_data.push_back(parameter_block_data[addr]);
        }
        
        // Only evaluate if we have all parameter blocks
        if (it->parameter_blocks_data.size() == it->parameter_blocks.size()) {
            it->Evaluate();
        }
    }
}

void MarginalizationInfo::marginalize() {
    // Count total parameters size and index them
    int total_block_size = 0;
    int total_block_size_local = 0;
    for (const auto& it : parameter_block_size) {
        total_block_size += it.second;
        total_block_size_local += localSize(it.second);
        // if (it.second == 7) {total_block_size -=1;}
    }
    
    // Map parameters to indices
    int idx = 0;
    for (auto& it : parameter_block_idx) {
        it.second = idx;
        idx += parameter_block_size[it.first];
    }
    
    // Get parameters to keep (not in any drop set)
    keep_block_size = 0;
    keep_block_sizes.clear();
    keep_block_idx.clear();
    keep_block_data.clear();
    keep_block_addr.clear();
    
    for (const auto& it : parameter_block_idx) {
        double* addr = it.first;
        
        if (!addr) continue; // Skip null addresses
        
        int size = parameter_block_size[addr];
        if (size <= 0) continue; // Skip invalid sizes
        
        // Check if this parameter should be dropped (marginalized)
        bool is_dropped = false;
        for (const auto& rbi : residual_block_infos) {
            if (!rbi) continue;
            
            for (int i = 0; i < static_cast<int>(rbi->parameter_blocks.size()); i++) {
                if (rbi->parameter_blocks[i] == addr && 
                    std::find(rbi->drop_set.begin(), rbi->drop_set.end(), i) != rbi->drop_set.end()) {
                    is_dropped = true;
                    break;
                }
            }
            if (is_dropped) break;
        }
        
        if (!is_dropped) {
            // This parameter is kept
            keep_block_size += size;
            keep_block_sizes.push_back(size);
            
            if (parameter_block_data.find(addr) != parameter_block_data.end()) {
                keep_block_data.push_back(parameter_block_data[addr]);
                keep_block_addr.push_back(addr);
                keep_block_idx.push_back(parameter_block_idx[addr]);
            }
        }
    }
    if (keep_block_size == 0) {
        ROS_WARN("No parameters to keep after marginalization");
        return;
    }
    
    
    // Calculate total residual size
    int total_residual_size = 0;
    for (const auto& rbi : residual_block_infos) {
        if (!rbi) continue;
        total_residual_size += rbi->num_residuals;
    }
    
    if (total_residual_size == 0) {
        ROS_WARN("No residuals in marginalization");
        return;
    }
    
    // Construct the linearized system: Jacobian and residuals
    Eigen::MatrixXd linearized_jacobians(total_residual_size, total_block_size);
    linearized_jacobians.setZero();
    Eigen::VectorXd linearized_residuals(total_residual_size);
    linearized_residuals.setZero();
    
    // Fill the jacobian and residual
    int residual_idx = 0;
    for (const auto& rbi : residual_block_infos) {
        if (!rbi) continue;
        
        // Copy residuals
        linearized_residuals.segment(residual_idx, rbi->num_residuals) = rbi->residuals;
        
        // Copy jacobians for each parameter block
        for (int i = 0; i < static_cast<int>(rbi->parameter_blocks.size()); i++) {
            double* addr = rbi->parameter_blocks[i];
            // Skip null parameter blocks
            if (!addr) continue;
            
            if (parameter_block_idx.find(addr) == parameter_block_idx.end()) {
                ROS_ERROR("Parameter block %p index not found during linearization", addr);
                continue;
            }
            
            int idx = parameter_block_idx[addr];
            int size = parameter_block_size[addr];
            // if (size == 7) {size = 6;}
            
            // Safety check for bounds
            if (residual_idx + rbi->num_residuals > linearized_jacobians.rows() ||
                idx + size > linearized_jacobians.cols()) {
                ROS_ERROR("Jacobian index out of bounds: residual_idx=%d, num_residuals=%d, idx=%d, size=%d",
                            residual_idx, rbi->num_residuals, idx, size);
                continue;
            }
            
            // Copy jacobian block
            linearized_jacobians.block(residual_idx, idx, rbi->num_residuals, size) = rbi->jacobians[i];
            // ROS_INFO_STREAM("Linearized Jacobian for block " << i << " at index " << idx 
            //     << " with size " << size << " and residuals size " << rbi->num_residuals);
            // ROS_INFO_STREAM("Jacobian block:\n" << rbi->jacobians[i]);
        }
        
        residual_idx += rbi->num_residuals;
    }

    // Reorder the Jacobian to have [kept_params | marg_params] in pose7-vel3-bias6 order
    Eigen::MatrixXd reordered_jacobians = Eigen::MatrixXd::Zero(total_residual_size, total_block_size_local);
    
    // 定义参数块顺序：pose(7) -> vel(3) -> bias(6)
    std::vector<int> param_order = {7, 3, 6}; // pose, vel, bias
    
    // 按照固定顺序重新排列kept参数
    int col_idx = 0;
    int reorderd_block_size = 0;
    std::vector<int> reordered_block_sizes;
    std::vector<double*> reordered_keep_addr;
    std::vector<double*> reordered_keep_data;
    std::vector<int> reordered_keep_idx;
    
    // 先按照参数类型顺序排列kept参数
    for (int param_size : param_order) {
        for (int i = 0; i < static_cast<int>(keep_block_addr.size()); i++) {
            double* addr = keep_block_addr[i];
            if (!addr) continue;
            
            int size = parameter_block_size[addr];
            if (size != param_size) continue; // 只处理当前大小的参数
            reordered_block_sizes.push_back(size);
            size = localSize(size); // 调整大小
            reorderd_block_size += size;
            
            int idx = keep_block_idx[i];
            
            // Safety check for bounds
            if (idx + size > linearized_jacobians.cols() || 
                col_idx + size > reordered_jacobians.cols()) {
                ROS_ERROR("Reordering jacobian index out of bounds");
                continue;
            }
            
            reordered_jacobians.block(0, col_idx, total_residual_size, size) = 
                linearized_jacobians.block(0, idx, total_residual_size, size);
            
            // 保存重新排序后的keep参数信息
            reordered_keep_addr.push_back(addr);
            reordered_keep_data.push_back(keep_block_data[i]);
            reordered_keep_idx.push_back(col_idx); // 新的索引位置
            
            col_idx += size;
        }
    }
    
    // 按照固定顺序排列marginalized参数
    for (int param_size : param_order) {
        for (const auto& it : parameter_block_idx) {
            double* addr = it.first;
            if (!addr) continue;
            
            int size = parameter_block_size[addr];
            if (size != param_size) continue; // 只处理当前大小的参数

            size = localSize(size); // 调整大小
            
            // Skip if this parameter is kept
            if (std::find(keep_block_addr.begin(), keep_block_addr.end(), addr) != keep_block_addr.end()) {
                continue;
            }
            
            int idx = it.second;
            
            // Safety check for bounds
            if (idx + size > linearized_jacobians.cols() || 
                col_idx + size > reordered_jacobians.cols()) {
                ROS_ERROR("Reordering marg jacobian index out of bounds");
                continue;
            }
            
            reordered_jacobians.block(0, col_idx, total_residual_size, size) = 
                linearized_jacobians.block(0, idx, total_residual_size, size);
            
            col_idx += size;
        }
    }
    
    // 更新keep参数的信息为重新排序后的
    keep_block_sizes = reordered_block_sizes;
    keep_block_size = reorderd_block_size;
    keep_block_addr = reordered_keep_addr;
    keep_block_data = reordered_keep_data;
    keep_block_idx = reordered_keep_idx;

    // Calculate marginalized block size
    int marg_block_size = total_block_size_local - keep_block_size;
    if (marg_block_size <= 0) {
        ROS_WARN("No parameters to marginalize");
        return;
    }
    
    // Split into kept and marginalized parts
    Eigen::MatrixXd jacobian_keep = reordered_jacobians.leftCols(keep_block_size);
    Eigen::MatrixXd jacobian_marg = reordered_jacobians.rightCols(marg_block_size);
    
    // Form the normal equations: J^T * J * delta_x = -J^T * r
    Eigen::MatrixXd H_marg = jacobian_marg.transpose() * jacobian_marg;
    Eigen::MatrixXd H_keep_marg = jacobian_keep.transpose() * jacobian_marg;
    Eigen::VectorXd b = -reordered_jacobians.transpose() * linearized_residuals;
    
    Eigen::VectorXd b_keep = b.head(keep_block_size);
    Eigen::VectorXd b_marg = b.tail(marg_block_size);
    
    // Add regularization to H_marg for numerical stability
    double lambda = 1e-4;
    for (int i = 0; i < H_marg.rows(); i++) {
        H_marg(i, i) += lambda;
    }
    
    // Compute Schur complement with regularization for numerical stability
    // First, compute eigendecomposition of H_marg
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H_marg);
    Eigen::VectorXd S = saes.eigenvalues();
    Eigen::MatrixXd V = saes.eigenvectors();
    
    // Apply regularization to eigenvalues
    Eigen::VectorXd S_inv = Eigen::VectorXd::Zero(S.size());
    double lambda_threshold = 1e-8;
    for (int i = 0; i < S.size(); i++) {
        if (S(i) > lambda_threshold) {
            S_inv(i) = 1.0 / S(i);
        } else {
            S_inv(i) = 0.0;
        }
    }
    // Eigen::MatrixXd S_inv = S.inverse();
    
    // Compute inverse of H_marg using eigendecomposition
    Eigen::MatrixXd H_marg_inv = V * S_inv.asDiagonal() * V.transpose();
    
    // Compute Schur complement for prior
    Eigen::MatrixXd schur_complement = H_keep_marg * H_marg_inv * H_keep_marg.transpose();
    
    // Final linearized system for prior
    Eigen::MatrixXd H_prior = jacobian_keep.transpose() * jacobian_keep - schur_complement;
    Eigen::VectorXd b_prior = b_keep - H_keep_marg * H_marg_inv * b_marg;

    Eigen::MatrixXd H_ = H_prior;

    // Firstly, decompose the hessian first
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_);
    // check if the decomposation is successfull or not
    if(saes2.info()!=Eigen::Success){
        ROS_ERROR("Eigen Solver Fail!!!");
    }

    // compute sqrt_D
    Eigen::MatrixXd P = saes2.eigenvectors();
    Eigen::VectorXd evs = saes2.eigenvalues();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> sqrt_D(evs.size());
    Eigen::VectorXd sqrt_evs = evs.array().sqrt().matrix();
    sqrt_D.diagonal() = sqrt_evs;

    // print out the size of the P, D, and b_
    // ROS_INFO("P size: %d, D size: %d, b_ size: %d", P.rows(), sqrt_D.diagonal().size(), linearized_residuals.size());

    // compute the whole jocobian, and residual
    linearized_jacobians_ = sqrt_D * P.transpose();
    linearized_residuals_ = - (sqrt_D.inverse() * P.transpose() * b_prior);
    // ROS_INFO_STREAM("Linearized Jacobians after marginalization:\n" << linearized_jacobians_);
}

// --- Accessor Implementations ---

const Eigen::MatrixXd& MarginalizationInfo::getLinearizedJacobians() const {
    return linearized_jacobians_;
}

const Eigen::VectorXd& MarginalizationInfo::getLinearizedResiduals() const {
    return linearized_residuals_;
}

const std::vector<double*>& MarginalizationInfo::getKeepBlockData() const {
    return keep_block_data;
}

const std::vector<int>& MarginalizationInfo::getKeepBlockSizes() const {
    return keep_block_sizes;
}

int MarginalizationInfo::getKeepBlockSize() const {
    return keep_block_size;
}

const std::vector<int>& MarginalizationInfo::getKeepBlockIdx() const {
    return keep_block_idx;
}
