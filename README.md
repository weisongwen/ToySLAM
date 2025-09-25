# ToySLAM

This project is a factor graph-based, GNSS/IMU fusion system designed to provide localization for autonomous driving and robotics applications. It is developed within the Robot Operating System (ROS) framework.

## Project Structure

This workspace (`toyslam_ws/src`) contains the following core ROS packages:

- `gnss_imu_fusion`: The core package of the project, implementing the full pipeline for sensor data processing, factor graph construction, and state estimation.
- `gnss_comm`: Provides common GNSS data structures (message definitions) and processing utilities.
- `nmea_parser`: A parser for NMEA-formatted GNSS data.
- `novatel_span_driver`: A ROS driver for NovAtel SPAN series of integrated navigation systems.
- `nlosexclusion`: A utility package for processing and excluding Non-Line-of-Sight (NLOS) satellite signals.

## Dependencies

- **ROS**: Melodic or Noetic
- **Ceres Solver**: For non-linear optimization
- **Eigen**: For matrix operations
- **Glog**: Google Logging Library

## How to Build

1.  Clone this repository into the `src` directory of your Catkin workspace.
2.  Build the project by running `catkin_make` or `catkin build` from the root of your workspace.

```bash
cd ~/your_catkin_ws
catkin_make
```

## How to Run

1.  First, modify the configuration file `src/gnss_imu_fusion/config/params.yaml` to match your sensor topic names, extrinsic parameters, etc.
2.  Launch the fusion node by running the launch file:

```bash
source devel/setup.bash
roslaunch gnss_imu_fusion gnss_imu_fusion.launch
```
3.  Play your sensor data from a `rosbag` file.
4.  You can visualize the real-time pose and trajectory in RViz. A pre-configured RViz file is available in `src/gnss_imu_fusion/rviz/`.
