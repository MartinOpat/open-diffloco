#pragma once
/// @file controller.hpp
/// Go2 deployment controller - state machine, DDS communication, motor
/// commands.
///
/// Requires unitree_sdk2 C++ SDK for compilation.
/// See: https://github.com/unitreerobotics/unitree_sdk2

#include "policy.hpp"

#include <Eigen/Core>
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace jave {

// Constants

inline constexpr int NUM_MOTORS = 12;
inline constexpr int NUM_MOTOR_SLOTS = 20;

/// Joint reordering:  simulation <--> hardware.
/// Both directions happen to be the same permutation for Go2.
inline constexpr int SIM_TO_HW[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
inline constexpr int HW_TO_SIM[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};

/// Crouched pose in hardware ordering (for sit-down sequence).
inline const Eigen::Matrix<double, 12, 1> &crouch_pos_hw() {
  static const Eigen::Matrix<double, 12, 1> v =
      (Eigen::Matrix<double, 12, 1>() << -0.35, 1.36, -2.65, // FR
       0.35, 1.36, -2.65,                                    // FL
       -0.50, 1.36, -2.65,                                   // RR
       0.50, 1.36, -2.65                                     // RL
       )
          .finished();
  return v;
}

// Reorder helpers

/// Reorder a 12-vector from simulation to hardware ordering.
inline Eigen::Matrix<double, 12, 1>
sim_to_hw(const Eigen::Matrix<double, 12, 1> &v) {
  Eigen::Matrix<double, 12, 1> out;
  for (int i = 0; i < 12; ++i)
    out(SIM_TO_HW[i]) = v(i);
  return out;
}

/// Overload for dynamic VectorXd (e.g. default_joints).
inline Eigen::Matrix<double, 12, 1> sim_to_hw(const Eigen::VectorXd &v) {
  Eigen::Matrix<double, 12, 1> out;
  for (int i = 0; i < 12; ++i)
    out(SIM_TO_HW[i]) = v(i);
  return out;
}

/// Reorder a 12-vector from hardware to simulation ordering.
inline Eigen::Matrix<double, 12, 1>
hw_to_sim(const Eigen::Matrix<double, 12, 1> &v) {
  Eigen::Matrix<double, 12, 1> out;
  for (int i = 0; i < 12; ++i)
    out(HW_TO_SIM[i]) = v(i);
  return out;
}

// State machine

enum class State {
  IDLE,
  STANDUP,
  READY,
  WALKING,
  SITDOWN,
  ESTOP,
};

const char *state_name(State s);

enum class CommandSource {
  TERMINAL,
  WIRELESS,
  ROS2,
};

// Controller

class Go2Deploy {
public:
  Go2Deploy(std::shared_ptr<NumpyPolicy> policy,
            const std::string &interface = "lo", int domain_id = 0,
            CommandSource command_source = CommandSource::TERMINAL,
            const std::string &cmd_topic = "/velocity_command");
  ~Go2Deploy();

  /// Main blocking loop (keyboard input on calling thread).
  void run();

  // PD gains - set before run(), or auto-loaded from policy.
  double kp = 35.0;
  double kd = 0.5;

  // Smooth stand/sit gains. The high-gain linear Unitree example vibrates in
  // sim.
  static constexpr double STANDUP_KP = 50.0;
  static constexpr double STANDUP_KD = 3.5;
  static constexpr double STANDUP_KP_START = 20.0;
  static constexpr double STANDUP_TANH_SCALE = 1.2;
  static constexpr double SAFETY_TILT_MAX = 1.05;

private:
  // SDK initialisation
  void init_sdk(const std::string &interface, int domain_id);
  void release_sport_mode();

  // 500 Hz command loop (called by CreateRecurrentThreadEx)
  void LowCmdWrite();

  // SDK callback
  void LowStateHandler(const void *message);
  void update_wireless_command(const uint8_t *data, std::size_t size);

  // Motor / publish helpers
  void set_motor(int i, float q, float kp_val, float dq, float kd_val,
                 float tau);
  void publish_cmd();

  // State handlers (called from LowCmdWrite)
  void handle_idle();
  void handle_standup();
  void handle_ready();
  void handle_walking();
  void handle_sitdown();
  void handle_estop();

  // Observation + safety
  Eigen::VectorXd build_obs();
  bool check_safety();

  // State transitions
  void transition(State to);

  // Keyboard processing
  void process_key(const std::string &key);
  void set_cmd(double vx, double vy, double yaw_rate);
  Eigen::Vector3d get_cmd() const;

  // Members
  std::shared_ptr<NumpyPolicy> policy_;
  std::string interface_;
  CommandSource command_source_;

  // Control timing
  double dt_cmd_ = 0.002;      // 500 Hz command rate
  int policy_decimation_ = 10; // 50 Hz policy

  // State machine (written from keyboard thread, read from cmd thread)
  std::atomic<State> state_{State::IDLE};
  mutable std::mutex cmd_mutex_;
  Eigen::Vector3d cmd_ = Eigen::Vector3d::Zero(); // [vx, vy, yaw_rate]

  // Walking state
  Eigen::Matrix<double, 12, 1> last_action_ =
      Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::Matrix<double, 12, 1> walking_target_hw_ =
      Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::VectorXd actor_obs_history_;
  int step_count_ = 0;

  // Standup state
  double standup_time_ = 0.0;
  Eigen::Matrix<double, 12, 1> standup_start_pos_ =
      Eigen::Matrix<double, 12, 1>::Zero();
  bool standup_first_run_ = true;

  // Sitdown state
  double sitdown_time_ = 0.0;
  Eigen::Matrix<double, 12, 1> sitdown_start_pos_ =
      Eigen::Matrix<double, 12, 1>::Zero();
  std::atomic<bool> sitdown_done_{false};

  // ESTOP state
  Eigen::Matrix<double, 12, 1> estop_hold_pos_ =
      Eigen::Matrix<double, 12, 1>::Zero();

  // Motion counter (incremented every LowCmdWrite = 500Hz)
  int motiontime_ = 0;

  // Sensor data - written by subscriber callback, read by LowCmdWrite.
  // Protected by mutex since SDK callback may be on a different thread.
  mutable std::mutex sensor_mutex_;
  std::atomic<bool> state_received_{false};

  // Cached sensor readings (under sensor_mutex_)
  Eigen::Matrix<double, 12, 1> hw_pos_ = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::Matrix<double, 12, 1> hw_vel_ = Eigen::Matrix<double, 12, 1>::Zero();
  Eigen::Vector4d imu_quat_ = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector3d imu_gyro_ = Eigen::Vector3d::Zero();

  // SDK handles (opaque - only used in controller.cpp)
  // Forward-declared in the .cpp to avoid pulling SDK headers here.
  struct SdkHandles;
  std::unique_ptr<SdkHandles> sdk_;
};

} // namespace jave
