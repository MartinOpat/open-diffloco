cmake_minimum_required(VERSION 3.16)
project(deploy_blind_nolinvel_nokinref LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# zlib (for reading compressed .npz files)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# unitree_sdk2
find_package(unitree_sdk2 REQUIRED)

if(OPEN_DIFFLOCO_ENABLE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(geometry_msgs REQUIRED)
endif()

# Core library (policy inference)
add_library(deploy_blind_nolinvel_nokinref_core
    src/policy.cpp
)
target_include_directories(deploy_blind_nolinvel_nokinref_core PUBLIC include)
target_link_libraries(deploy_blind_nolinvel_nokinref_core PUBLIC Eigen3::Eigen ZLIB::ZLIB)

# Controller library (state machine + DDS)
add_library(deploy_blind_nolinvel_nokinref_controller
    src/controller.cpp
)
target_link_libraries(deploy_blind_nolinvel_nokinref_controller PUBLIC deploy_blind_nolinvel_nokinref_core unitree_sdk2)
if(OPEN_DIFFLOCO_ENABLE_ROS2)
  target_compile_definitions(deploy_blind_nolinvel_nokinref_controller PUBLIC OPEN_DIFFLOCO_ENABLE_ROS2)
  ament_target_dependencies(deploy_blind_nolinvel_nokinref_controller PUBLIC rclcpp geometry_msgs)
endif()

# Deployment executable
add_executable(deploy_blind_nolinvel_nokinref src/main.cpp)
target_compile_definitions(deploy_blind_nolinvel_nokinref PRIVATE
    JAVE_DEPLOY_REPO_ROOT="${CMAKE_SOURCE_DIR}/")
target_link_libraries(deploy_blind_nolinvel_nokinref PRIVATE deploy_blind_nolinvel_nokinref_controller)

# Validation tool
add_executable(test_policy_blind_nolinvel_nokinref src/test_policy.cpp)
target_link_libraries(test_policy_blind_nolinvel_nokinref PRIVATE deploy_blind_nolinvel_nokinref_core)
