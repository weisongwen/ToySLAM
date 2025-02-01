# ToySLAM: A ROS Package for 3D Point Cloud Registration using NDT PolyU course AAE4011

## Overview
**ToySLAM** is a lightweight ROS package for 3D point cloud registration using the Normal Distributions Transform (NDT) algorithm. Designed for educational purposes and experimentation, this package provides a modular pipeline to process, align, and visualize point clouds in real-time. It serves as a foundational framework for SLAM (Simultaneous Localization and Mapping) applications and can be extended for robotics or autonomous systems.

## Features
- **Modular Architecture**: Four standalone nodes for processing, alignment, mapping, and visualization.
- **NDT-Based Registration**: Efficient alignment of 3D point clouds using the NDT implementation.
- **Real-Time Visualization**: Live preview of registered point clouds and **trajectory** in RViz.
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