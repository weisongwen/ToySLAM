cmake_minimum_required(VERSION 3.0.2)
project(ToySLAM)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  pcl_ros
  tf2_ros
  ndt_omp

  roscpp
  nav_msgs
  pcl_ros
  tf2_ros
)

find_package(PCL 1.8 REQUIRED COMPONENTS common io filters)
find_package(OpenMP REQUIRED)
# find_package(pclomp REQUIRED)


catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
)

add_executable(lidar_subscriber_node src/lidar_subscriber_node.cpp)
target_link_libraries(lidar_subscriber_node
  ${catkin_LIBRARIES}
)

add_executable(ndt_omp_node src/ndt_omp_node.cpp)
target_link_libraries(ndt_omp_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # pclomp
  OpenMP::OpenMP_CXX
)

add_executable(ndt_omp_mapping_node src/ndt_omp_mapping_node.cpp)
target_link_libraries(ndt_omp_mapping_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # pclomp
  OpenMP::OpenMP_CXX
)

add_executable(ndt_rosbag_mapping_node src/ndt_rosbag_mapping_node.cpp)
target_link_libraries(ndt_rosbag_mapping_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # pclomp
  OpenMP::OpenMP_CXX
)