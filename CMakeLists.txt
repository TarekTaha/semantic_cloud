cmake_minimum_required(VERSION 2.8)
project(semantic_cloud)

set(PACKAGE_DEPENDENCIES
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_message_files(
  FILES
  SemanticColoredLabel.msg
  SemanticColoredLabels.msg
)

add_service_files(
  FILES
  GetSemanticColoredLabels.srv
)

# Make modules in include/ visible to work place
catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS
 ${PACKAGE_DEPENDENCIES}
  message_runtime
)

