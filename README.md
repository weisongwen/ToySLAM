# ToySLAM: A ROS Package for 3D Point Cloud Registration using NDT for PolyU course AAE4011

## Overview
**ToySLAM** is a lightweight ROS package for 3D point cloud registration using the Normal Distributions Transform (NDT) algorithm. Designed for educational purposes and experimentation, this package provides a modular pipeline to process, align, and visualize point clouds in real-time. It serves as a foundational framework for SLAM (Simultaneous Localization and Mapping) applications and can be extended for robotics or autonomous systems.

[SLAM Dataset](https://www.dropbox.com/scl/fi/c9a4spcbqupvvcsacwbtf/2025-02-06-17-20-03.bag?rlkey=jkk60x2sn3awbcd5w1tx0mxgf&dl=0)

## Features
- **Modular Architecture**: Four standalone nodes for processing, alignment, mapping, and visualization.
- **NDT-Based Registration**: Efficient alignment of 3D point clouds using the NDT implementation.
- **Real-Time Visualization**: Live preview of registered point clouds and **trajectory** in RViz.
- **Configurable Parameters**: Tune algorithm behavior via ROS parameters for optimal performance.
- **Configurable Parameters**: Tune algorithm behavior via ROS parameters for optimal performance.

### Nodes
1. **lidar_subscriber_node.cpp**: subscribe the 3D LiDAR point clouds from the rosbag file and save the point clouds to the PCD files.
2. **ndt_omp_mapping_node.cpp**: read the PCD file continuously and perform the ICP for the continuous PCD files. Output the 4x4 transformation matrix.
3. **ndt_omp_node.cpp**: similar to the File 2, but also plot the trajectory in the Rviz of ROS.
4. **ndt_rosbag_mapping_node.cpp**: Directly read the 3D point clouds from the rosbag file and perform the ICP continuously.

### Folder tree

```
.
├── CMakeLists.txt
├── package.xml
└── src
    ├── lidar_subscriber_node.cpp
    ├── ndt_omp_mapping_node.cpp
    ├── ndt_omp_node.cpp
    └── ndt_rosbag_mapping_node.cpp
```

## Dependencies
- **ROS Noetic** (or other [ROS](http://www.ros.org/) distributions)
- **PCL 1.10+** (`libpcl-dev`, `ros-<distro>-pcl-ros`)
- **Eigen3**
- **RViz** for visualization

## Installation
1. Clone this repository into your ROS workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/weisongwen/ToySLAM
   catkin_make
   rosrun ToySLAM  ndt_rosbag_mapping_node /home/wws/Download/UrbanNav-HK_Whampoa-20210521_sensors.bag
   ```


## NEW updates for further extension (Optional)
1. ```uwb_node.cpp```
    - simulate the UWB ranging measurements and do the positioning
    - ```roslaunch ToySLAM fusion.launch ```

2. ```uwb_imu_node.cpp```
    - UWB/IMU fusion via sliding window optimization
    - ```rosrun ToySLAM uwb_imu_node ```

3. ```uwb_imu_sim_node.cpp```
    - UWB/IMU data simulation. Perform the least square estimation for UWB ranging measurements with visualization
    - ```roslaunch ToySLAM uwb_imu_fusion_sim.launch ```
4. ```uwb_imu_EKF_node.cpp```
    - UWB/IMU fusion with EKF. The simulated IMU data is not correct, please use the dataset ```2025-02-06-16-30-08.bag```
    - ```rosrun ToySLAM uwb_imu_EKF_node ```