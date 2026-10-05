cmake_minimum_required(VERSION 3.16)
project(spot_orbslam3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Keep the directories of the linked ORB-SLAM3 / Pangolin libraries in the installed binary's RUNPATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Root of an ORB-SLAM3 checkout that has been built (contains lib/libORB_SLAM3.so).
# Pass -DORB_SLAM3_ROOT=... or export ORB_SLAM3_ROOT before running colcon build.
if(NOT ORB_SLAM3_ROOT AND DEFINED ENV{ORB_SLAM3_ROOT})
  set(ORB_SLAM3_ROOT $ENV{ORB_SLAM3_ROOT})
endif()
if(NOT ORB_SLAM3_ROOT)
  message(FATAL_ERROR "Set ORB_SLAM3_ROOT (env var or -DORB_SLAM3_ROOT=...) to the built ORB-SLAM3 directory")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

find_library(ORB_SLAM3_LIB ORB_SLAM3 PATHS ${ORB_SLAM3_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
find_library(DBOW2_LIB DBoW2 PATHS ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib NO_DEFAULT_PATH REQUIRED)
find_library(G2O_LIB g2o PATHS ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib NO_DEFAULT_PATH REQUIRED)

add_executable(orbslam3_rgbd_node src/orbslam3_rgbd_node.cpp)
target_include_directories(orbslam3_rgbd_node PRIVATE
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_rgbd_node
  ${ORB_SLAM3_LIB} ${DBOW2_LIB} ${G2O_LIB} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
# Classic RPATH (not RUNPATH) so the directories also apply to transitive deps such as libpango_windowing.
target_link_options(orbslam3_rgbd_node PRIVATE -Wl,--disable-new-dtags)
ament_target_dependencies(orbslam3_rgbd_node rclcpp sensor_msgs geometry_msgs std_msgs message_filters)

install(TARGETS orbslam3_rgbd_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
