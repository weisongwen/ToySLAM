cmake_minimum_required(VERSION 3.0.2)
project(ToySLAM)

# # set(CMAKE_CXX_FLAGS "-std=c++11")
# set(CMAKE_CXX_FLAGS "-std=c++14")
# # C++ 14 is required
# add_definitions(-std=c++14)
# set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

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

find_package(PCL 1.8 REQUIRED COMPONENTS common io filters kdtree)
# find_package(PCL REQUIRED)

find_package(OpenMP REQUIRED)

find_package(Ceres REQUIRED)

find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# find_package(Ceres REQUIRED)


find_package(Gflags)
find_package(Glog)
include_directories(${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})
include_directories(include)

# find_package(pclomp REQUIRED)


catkin_package()



include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  # ${CERES_INCLUDE_DIRS}
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

add_executable(GPSRAIM_node src/GPSRAIM.cpp)
target_link_libraries(GPSRAIM_node ${catkin_LIBRARIES} ${CERES_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  # pclomp
  OpenMP::OpenMP_CXX
)

# add_executable(loam_mapping_node src/loam_mapping_node.cpp)
# target_link_libraries(loam_mapping_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${CERES_LIBRARIES}
#   # pclomp
#   OpenMP::OpenMP_CXX
# )