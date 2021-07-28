cmake_minimum_required(VERSION 3.8.3)
project(ROS_humanSensing)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  rospy
  roscpp
)

#######################################
## Declare ROS messages and services ##
#######################################

add_message_files(
  FILES
  JointVelocity.msg
  JointAngles.msg

)

add_service_files(
  FILES
  SaveHands.srv

)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS  message_runtime
)


