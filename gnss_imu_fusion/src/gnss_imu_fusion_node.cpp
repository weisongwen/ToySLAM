#include <ros/ros.h>
#include <memory>

#include "parameter_server.h"
#include "data_processor.h"
#include "backend_optimizer.h"
#include "visualizer.h"
#include "ceres_logger.h"

// Main GPS-IMU fusion class
class GnssImuFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    GnssImuFusion() {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        params.loadParameters(private_nh);
        params.initLogging(logger_);
        backend_optimizer_ = std::make_unique<BackendOptimizer>(params);
        visualizer_ = std::make_unique<Visualizer>(nh, params);
        data_processor_ = std::make_unique<DataProcessor>(nh, params, 
                                                          backend_optimizer_.get(), 
                                                          visualizer_.get());
        
        optimization_timer_ = nh.createTimer(
            ros::Duration(1.0 / params.optimization_frequency),
            &DataProcessor::optimizationTimerCallback, 
            data_processor_.get()
        );
        
    }

    ~GnssImuFusion() {}

private:
    ParameterServer params;
    ros::Timer optimization_timer_;


    std::unique_ptr<Visualizer> visualizer_; 
    std::unique_ptr<BackendOptimizer> backend_optimizer_;
    std::unique_ptr<DataProcessor> data_processor_; 
    
    CeresLogger logger_;


};

int main(int argc, char **argv) {
    try {
        ros::init(argc, argv, "uwb_imu_batch_node");
        
        {
            GnssImuFusion fusion; 
            ros::spin();
        }
        
        return 0;
    } catch (const std::exception& e) {
        ROS_ERROR("Fatal exception in main: %s", e.what());
        return 1;
    } catch (...) {
        ROS_ERROR("Unknown fatal exception in main");
        return 1;
    }
}