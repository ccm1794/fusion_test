cmake_minimum_required(VERSION 3.5)
project(fusion_test)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(interfaces REQUIRED)

set(dependencies
  rclcpp
  std_msgs
  PCL
  pcl_conversions
  image_transport
  cv_bridge
  geometry_msgs
  #kbub_interfaces
  sensor_msgs
  OpenCV
  vision_msgs
  visualization_msgs
  #visualization_msgs
  message_filters
  interfaces
)

link_libraries(
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${pcl_ros_LIBRARIES}
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(plz src/plz.cpp)
ament_target_dependencies(plz ${dependencies})

add_executable(base src/base.cpp)
ament_target_dependencies(base ${dependencies})

add_executable(coord_test_2 src/coord_test_2.cpp)
ament_target_dependencies(coord_test_2 ${dependencies})

# add_executable(lidar-image_fusion_sync src/lidar-image_fusion_sync.cpp)
# ament_target_dependencies(lidar-image_fusion_sync ${dependencies})

add_executable(projection src/projection.cpp)
ament_target_dependencies(projection ${dependencies})

install(TARGETS
plz
base
coord_test_2
#lidar-image_fusion_sync
projection
DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  param
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
