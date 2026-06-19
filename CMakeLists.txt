cmake_minimum_required(VERSION 3.16)
project(open_diffloco_deploy LANGUAGES CXX)

option(OPEN_DIFFLOCO_ENABLE_ROS2 "Build optional ROS2 deployment integrations" OFF)

add_subdirectory(src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp)
add_subdirectory(src/envs/go2/variants/blind_linvel_nokinref/deploy_cpp)
add_subdirectory(src/envs/go2/variants/blind_linvel_kinref/deploy_cpp)
add_subdirectory(src/envs/go2/variants/highspeed_nokinref/deploy_cpp)
