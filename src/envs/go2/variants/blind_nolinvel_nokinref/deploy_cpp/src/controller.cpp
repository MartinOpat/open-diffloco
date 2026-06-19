/// @file controller.cpp
/// Go2Deploy - full deployment controller with unitree_sdk2.

#include "controller.hpp"
#include "math_utils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <csignal>

#ifdef OPEN_DIFFLOCO_ENABLE_ROS2
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#endif

//  unitree_sdk2
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

namespace jave {

//  CRC32 (from unitree_sdk2 examples)

static uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;

  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit)
        CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

//  State name

const char *state_name(State s) {
  switch (s) {
  case State::IDLE:
    return "IDLE";
  case State::STANDUP:
    return "STANDUP";
  case State::READY:
    return "READY";
  case State::WALKING:
    return "WALKING";
  case State::SITDOWN:
    return "SITDOWN";
  case State::ESTOP:
    return "ESTOP";
  }
  return "UNKNOWN";
}

//  SDK handle storage

struct Go2Deploy::SdkHandles {
  unitree_go::msg::dds_::LowCmd_ low_cmd{};
  unitree_go::msg::dds_::LowState_ low_state{};

  ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> pub;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> sub;
  ThreadPtr cmd_thread;

#ifdef OPEN_DIFFLOCO_ENABLE_ROS2
  rclcpp::Node::SharedPtr ros_node;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr cmd_sub;
  rclcpp::executors::SingleThreadedExecutor ros_exec;
  std::thread ros_thread;
#endif
};

//  Construction / destruction

Go2Deploy::~Go2Deploy() {
#ifdef OPEN_DIFFLOCO_ENABLE_ROS2
  if (sdk_) {
    sdk_->ros_exec.cancel();
    if (sdk_->ros_thread.joinable())
      sdk_->ros_thread.join();
  }
  if (rclcpp::ok())
    rclcpp::shutdown();
#endif
}

Go2Deploy::Go2Deploy(std::shared_ptr<NumpyPolicy> policy,
                     const std::string &interface, int domain_id,
                     CommandSource command_source, const std::string &cmd_topic)
    : policy_(std::move(policy)), interface_(interface),
      command_source_(command_source) {
#ifndef OPEN_DIFFLOCO_ENABLE_ROS2
  if (command_source_ == CommandSource::ROS2) {
    throw std::runtime_error("command-source=ros2 requires building with "
                             "OPEN_DIFFLOCO_ENABLE_ROS2=ON");
  }
#endif

  sdk_ = std::make_unique<SdkHandles>();

  policy_decimation_ = static_cast<int>(std::round(policy_->dt / dt_cmd_));
  if (policy_decimation_ < 1)
    policy_decimation_ = 1;

  if (!std::isnan(policy_->training_kp)) {
    kp = policy_->training_kp;
    kd = policy_->training_kd;
    std::cout << "  Using training gains: kp=" << kp << ", kd=" << kd << "\n";
  } else {
    std::cout << "  Using default gains: kp=" << kp << ", kd=" << kd
              << " (override with --kp/--kd)\n";
  }

  init_sdk(interface, domain_id);

#ifdef OPEN_DIFFLOCO_ENABLE_ROS2
  if (command_source_ == CommandSource::ROS2) {
    if (!rclcpp::ok()) {
      int argc = 0;
      char **argv = nullptr;
      rclcpp::init(argc, argv);
    }
    sdk_->ros_node =
        std::make_shared<rclcpp::Node>("open_diffloco_cmd_subscriber");
    sdk_->cmd_sub =
        sdk_->ros_node->create_subscription<geometry_msgs::msg::PointStamped>(
            cmd_topic, rclcpp::QoS(10),
            [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
              set_cmd(msg->point.x, msg->point.y, msg->point.z);
            });
    sdk_->ros_exec.add_node(sdk_->ros_node);
    sdk_->ros_thread = std::thread([this]() { sdk_->ros_exec.spin(); });
  }
#endif

  const char *source_name = "terminal";
  if (command_source_ == CommandSource::WIRELESS)
    source_name = "wireless";
  else if (command_source_ == CommandSource::ROS2)
    source_name = "ros2";
  std::cout << "  Command source: " << source_name << "\n";
}

//  SDK initialisation (matches go2_low_level.cpp)

void Go2Deploy::init_sdk(const std::string &interface, int domain_id) {
  std::cout << "  DDS init: interface=" << interface << ", domain_id="
            << domain_id << "\n";

  ChannelFactory::Instance()->Init(domain_id, interface);

  // InitLowCmd
  auto &cmd = sdk_->low_cmd;
  cmd.head()[0] = 0xFE;
  cmd.head()[1] = 0xEF;
  cmd.level_flag() = 0xFF;
  cmd.gpio() = 0;
  for (int i = 0; i < NUM_MOTOR_SLOTS; ++i) {
    cmd.motor_cmd()[i].mode() = 0x01;
    cmd.motor_cmd()[i].q() = 0.0f;
    cmd.motor_cmd()[i].kp() = 0.0f;
    cmd.motor_cmd()[i].dq() = 0.0f;
    cmd.motor_cmd()[i].kd() = 0.0f;
    cmd.motor_cmd()[i].tau() = 0.0f;
  }

  // Publisher
  sdk_->pub.reset(
      new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>("rt/lowcmd"));
  sdk_->pub->InitChannel();

  // Subscriber
  sdk_->sub.reset(
      new ChannelSubscriber<unitree_go::msg::dds_::LowState_>("rt/lowstate"));
  sdk_->sub->InitChannel(
      std::bind(&Go2Deploy::LowStateHandler, this, std::placeholders::_1), 1);

  release_sport_mode();

  // 500 Hz command thread (interval in microseconds)
  sdk_->cmd_thread = CreateRecurrentThreadEx(
      "writebasiccmd", UT_CPU_ID_NONE, 2000, &Go2Deploy::LowCmdWrite, this);

  std::cout << "  SDK initialized\n";
}

void Go2Deploy::release_sport_mode() {
  if (interface_.empty() || interface_.rfind("lo", 0) == 0) {
    std::cout << "  MotionSwitcherClient: skipped for simulator/loopback\n";
    return;
  }

  MotionSwitcherClient msc;
  msc.SetTimeout(10.0f);
  msc.Init();

  std::string robot_form;
  std::string motion_name;
  int32_t ret = msc.CheckMode(robot_form, motion_name);
  if (ret != 0) {
    throw std::runtime_error(
        "CheckMode failed while verifying sport mode is off");
  }

  int attempts = 0;
  while (!motion_name.empty()) {
    std::cout << "  Active motion mode: " << motion_name << ", releasing...\n";
    ret = msc.ReleaseMode();
    if (ret == 0)
      std::cout << "  ReleaseMode succeeded\n";
    else
      std::cout << "  ReleaseMode failed: ret=" << ret << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(5));
    ret = msc.CheckMode(robot_form, motion_name);
    if (ret != 0) {
      throw std::runtime_error("CheckMode failed after ReleaseMode while "
                               "verifying sport mode is off");
    }

    ++attempts;
    if (attempts >= 20 && !motion_name.empty()) {
      throw std::runtime_error(
          "motion mode still active after repeated ReleaseMode calls");
    }
  }

  std::cout << "  motion-control service is deactivated\n";
}

//  Low-state callback (runs on SDK subscriber thread)

void Go2Deploy::LowStateHandler(const void *message) {
  const auto &msg =
      *static_cast<const unitree_go::msg::dds_::LowState_ *>(message);

  std::lock_guard<std::mutex> lock(sensor_mutex_);
  sdk_->low_state = msg;

  for (int i = 0; i < NUM_MOTORS; ++i) {
    hw_pos_(i) = msg.motor_state()[i].q();
    hw_vel_(i) = msg.motor_state()[i].dq();
  }

  const auto &imu = msg.imu_state();
  imu_quat_ << imu.quaternion()[0], imu.quaternion()[1], imu.quaternion()[2],
      imu.quaternion()[3];
  imu_gyro_ << imu.gyroscope()[0], imu.gyroscope()[1], imu.gyroscope()[2];

  if (command_source_ == CommandSource::WIRELESS) {
    const auto &remote = msg.wireless_remote();
    update_wireless_command(remote.data(), remote.size());
  }

  state_received_.store(true, std::memory_order_release);
}

void Go2Deploy::update_wireless_command(const uint8_t *data, std::size_t size) {
  if (size < 24)
    return;

  auto read_float = [data](std::size_t offset) {
    float value = 0.0f;
    std::memcpy(&value, data + offset, sizeof(float));
    return static_cast<double>(value);
  };

  auto deadzone = [](double value) {
    return std::abs(value) < 0.1 ? 0.0 : value;
  };

  const double lx = deadzone(read_float(4));
  const double rx = deadzone(read_float(8));
  const double ly = deadzone(read_float(20));

  const double vx = ly * policy_->cmd_vel_x_range(1);
  const double vy = -lx * policy_->cmd_vel_y_range(1);
  const double wz = -rx * policy_->cmd_yaw_rate_range(1);
  set_cmd(vx, vy, wz);
}

//  Motor helper

void Go2Deploy::set_motor(int i, float q, float kp_val, float dq, float kd_val,
                          float tau) {
  sdk_->low_cmd.motor_cmd()[i].q() = q;
  sdk_->low_cmd.motor_cmd()[i].kp() = kp_val;
  sdk_->low_cmd.motor_cmd()[i].dq() = dq;
  sdk_->low_cmd.motor_cmd()[i].kd() = kd_val;
  sdk_->low_cmd.motor_cmd()[i].tau() = tau;
}

void Go2Deploy::publish_cmd() {
  sdk_->low_cmd.crc() =
      crc32_core(reinterpret_cast<uint32_t *>(&sdk_->low_cmd),
                 (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
  sdk_->pub->Write(sdk_->low_cmd);
}

//  500 Hz command loop

void Go2Deploy::LowCmdWrite() {
  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    if (!state_received_.load(std::memory_order_acquire)) {
      publish_cmd();
      return;
    }
  }

  ++motiontime_;

  switch (state_.load()) {
  case State::IDLE:
    handle_idle();
    break;
  case State::STANDUP:
    handle_standup();
    break;
  case State::READY:
    handle_ready();
    break;
  case State::WALKING:
    handle_walking();
    break;
  case State::SITDOWN:
    handle_sitdown();
    break;
  case State::ESTOP:
    handle_estop();
    break;
  }

  publish_cmd();
}

//  State handlers

void Go2Deploy::handle_idle() {
  for (int i = 0; i < NUM_MOTORS; ++i)
    set_motor(i, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

void Go2Deploy::handle_standup() {
  if (standup_first_run_) {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    standup_start_pos_ = hw_pos_;
    standup_first_run_ = false;
  }

  standup_time_ += dt_cmd_;
  const double phase = std::tanh(standup_time_ / STANDUP_TANH_SCALE);
  const double kp_i = phase * STANDUP_KP + (1.0 - phase) * STANDUP_KP_START;
  Eigen::Matrix<double, 12, 1> stand_target =
      sim_to_hw(policy_->default_joints);
  Eigen::Matrix<double, 12, 1> target =
      (1.0 - phase) * standup_start_pos_ + phase * stand_target;

  for (int i = 0; i < NUM_MOTORS; ++i) {
    set_motor(i, static_cast<float>(target(i)), static_cast<float>(kp_i), 0.0f,
              static_cast<float>(STANDUP_KD), 0.0f);
  }

  if (standup_time_ > 3.0 * STANDUP_TANH_SCALE) {
    std::cout << "\n  Standup complete. Press Enter for WALKING.\n";
    state_.store(State::READY);
  }
}

void Go2Deploy::handle_ready() {
  auto default_hw = sim_to_hw(policy_->default_joints);
  for (int i = 0; i < NUM_MOTORS; ++i)
    set_motor(i, static_cast<float>(default_hw(i)),
              static_cast<float>(STANDUP_KP), 0.0f,
              static_cast<float>(STANDUP_KD), 0.0f);
}

void Go2Deploy::handle_walking() {
  if (motiontime_ % policy_decimation_ == 0) {
    if (!check_safety())
      return;

    Eigen::VectorXd obs = build_obs();
    Eigen::VectorXd action = (*policy_)(obs);
    last_action_ = action;

    auto target_sim = policy_->get_target_joints(action);
    walking_target_hw_ = sim_to_hw(target_sim);
    ++step_count_;

    if (step_count_ <= 3) {
      const int frame_start =
          (policy_->actor_history_len - 1) * policy_->actor_frame_obs_dim;
      const auto frame = obs.segment(frame_start, policy_->actor_frame_obs_dim);
      std::cout << "\n  [step " << step_count_ << "] DIAGNOSTIC:\n";
      std::cout << "    obs angvel  = " << frame.head<3>().transpose() << "\n";
      std::cout << "    obs gravity = " << frame.segment<3>(3).transpose()
                << "\n";
      std::cout << "    obs cmd     = " << frame.segment<3>(6).transpose()
                << "\n";
      std::cout << "    action[:4]  = " << action.head(4).transpose() << "\n";
      std::cout << "    target[:4]  = " << target_sim.head(4).transpose()
                << "\n";
    } else if (step_count_ % 250 == 0) {
      Eigen::Vector3d cmd = get_cmd();
      std::cout << "  [step " << step_count_ << "]  cmd=(" << cmd(0) << ", "
                << cmd(1) << ", " << cmd(2) << ")\n";
    }
  }

  if (state_.load() == State::WALKING) {
    for (int i = 0; i < NUM_MOTORS; ++i)
      set_motor(i, static_cast<float>(walking_target_hw_(i)),
                static_cast<float>(kp), 0.0f, static_cast<float>(kd), 0.0f);
  }
}

void Go2Deploy::handle_sitdown() {
  if (sitdown_done_.load())
    return;

  sitdown_time_ += dt_cmd_;
  const double phase = std::tanh(sitdown_time_ / STANDUP_TANH_SCALE);
  const auto &crouch = crouch_pos_hw();

  for (int i = 0; i < NUM_MOTORS; ++i) {
    float q = static_cast<float>((1.0 - phase) * sitdown_start_pos_(i) +
                                 phase * crouch(i));
    set_motor(i, q, static_cast<float>(STANDUP_KP), 0.0f,
              static_cast<float>(STANDUP_KD), 0.0f);
  }

  if (sitdown_time_ > 3.0 * STANDUP_TANH_SCALE) {
    sitdown_done_.store(true);
    state_.store(State::IDLE);
  }
}

void Go2Deploy::handle_estop() {
  for (int i = 0; i < NUM_MOTORS; ++i)
    set_motor(i, static_cast<float>(estop_hold_pos_(i)), 20.0f, 0.0f, 3.5f,
              0.0f);
}

//  Observation builder

Eigen::VectorXd Go2Deploy::build_obs() {
  Eigen::Matrix<double, 12, 1> sim_pos, sim_vel;
  Eigen::Vector4d quat;
  Eigen::Vector3d gyro;

  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    sim_pos = hw_to_sim(hw_pos_);
    sim_vel = hw_to_sim(hw_vel_);
    quat = imu_quat_;
    gyro = imu_gyro_;
  }

  Eigen::Vector4d inv_q = quat_inv(quat);
  Eigen::Vector3d gravity = quat_rotate(Eigen::Vector3d(0, 0, -1), inv_q);

  Eigen::VectorXd joint_pos_err =
      sim_pos.cast<double>() - policy_->default_joints;

  // Actor frame: angvel(3), gravity(3), cmd(3), qpos_err(12),
  // qvel(12), last_action(12).
  const int obs_dim = policy_->actor_frame_obs_dim;
  Eigen::VectorXd frame = Eigen::VectorXd::Zero(obs_dim);

  int idx = 0;
  auto put3 = [&](const Eigen::Vector3d &v) {
    if (idx + 3 <= obs_dim) {
      frame.segment<3>(idx) = v;
      idx += 3;
    }
  };
  auto put12 = [&](const Eigen::Matrix<double, 12, 1> &v) {
    if (idx + 12 <= obs_dim) {
      frame.segment<12>(idx) = v;
      idx += 12;
    }
  };
  auto putN = [&](const Eigen::VectorXd &v) {
    int n = static_cast<int>(v.size());
    if (idx + n <= obs_dim) {
      frame.segment(idx, n) = v;
      idx += n;
    }
  };

  put3(gyro);
  put3(gravity);
  put3(get_cmd());
  putN(joint_pos_err);
  put12(sim_vel);
  put12(last_action_);
  if (idx != obs_dim)
    throw std::runtime_error(
        "Actor frame dimension does not match policy metadata");

  if (actor_obs_history_.size() == 0) {
    actor_obs_history_.resize(policy_->actor_history_len * obs_dim);
    for (int i = 0; i < policy_->actor_history_len; ++i)
      actor_obs_history_.segment(i * obs_dim, obs_dim) = frame;
  } else {
    actor_obs_history_.head((policy_->actor_history_len - 1) * obs_dim) =
        actor_obs_history_.tail((policy_->actor_history_len - 1) * obs_dim)
            .eval();
    actor_obs_history_.tail(obs_dim) = frame;
  }
  return actor_obs_history_;
}

//  Safety

bool Go2Deploy::check_safety() {
  Eigen::Vector4d quat;
  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    quat = imu_quat_;
  }
  Eigen::Vector3d gravity =
      quat_rotate(Eigen::Vector3d(0, 0, -1), quat_inv(quat));
  double tilt = std::sqrt(gravity(0) * gravity(0) + gravity(1) * gravity(1));

  if (tilt > SAFETY_TILT_MAX) {
    double deg = std::asin(std::min(tilt, 1.0)) * 180.0 / M_PI;
    std::cout << "\n  SAFETY: tilt " << deg << " deg\n";
    transition(State::ESTOP);
    return false;
  }
  return true;
}

//  State transitions

void Go2Deploy::transition(State to) {
  std::cout << "  State: " << state_name(state_.load()) << " -> "
            << state_name(to) << "\n";

  switch (to) {
  case State::STANDUP:
    standup_time_ = 0.0;
    standup_first_run_ = true;
    motiontime_ = 0;
    break;

  case State::WALKING:
    last_action_.setZero();
    actor_obs_history_.resize(0);
    set_cmd(0.0, 0.0, 0.0);
    walking_target_hw_ = sim_to_hw(policy_->default_joints);
    step_count_ = 0;
    motiontime_ = 0;
    break;

  case State::SITDOWN: {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    sitdown_start_pos_ = state_received_.load(std::memory_order_acquire)
                             ? hw_pos_
                             : sim_to_hw(policy_->default_joints);
    sitdown_time_ = 0.0;
    sitdown_done_.store(false);
    break;
  }

  case State::ESTOP: {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    estop_hold_pos_ = state_received_.load(std::memory_order_acquire)
                          ? hw_pos_
                          : sim_to_hw(policy_->default_joints);
    break;
  }

  default:
    break;
  }

  state_.store(to);
}

//  Keyboard input

void Go2Deploy::set_cmd(double vx, double vy, double yaw_rate) {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  cmd_(0) =
      std::clamp(vx, policy_->cmd_vel_x_range(0), policy_->cmd_vel_x_range(1));
  cmd_(1) =
      std::clamp(vy, policy_->cmd_vel_y_range(0), policy_->cmd_vel_y_range(1));
  cmd_(2) = std::clamp(yaw_rate, policy_->cmd_yaw_rate_range(0),
                       policy_->cmd_yaw_rate_range(1));
}

Eigen::Vector3d Go2Deploy::get_cmd() const {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  return cmd_;
}

void Go2Deploy::process_key(const std::string &key) {
  constexpr double S = 0.1;
  Eigen::Vector3d cmd = get_cmd();

  if (key.empty()) {
    State s = state_.load();
    if (s == State::IDLE)
      transition(State::STANDUP);
    else if (s == State::READY)
      transition(State::WALKING);
    else if (s == State::ESTOP)
      std::cout << "  In ESTOP. Restart to continue.\n";
  } else if (key == "x") {
    transition(State::ESTOP);
  } else if (key == "w") {
    set_cmd(cmd(0) + S, cmd(1), cmd(2));
    std::cout << "  vx=" << get_cmd()(0) << "\n";
  } else if (key == "s") {
    set_cmd(cmd(0) - S, cmd(1), cmd(2));
    std::cout << "  vx=" << get_cmd()(0) << "\n";
  } else if (key == "a") {
    set_cmd(cmd(0), cmd(1) + S, cmd(2));
    std::cout << "  vy=" << get_cmd()(1) << "\n";
  } else if (key == "d") {
    set_cmd(cmd(0), cmd(1) - S, cmd(2));
    std::cout << "  vy=" << get_cmd()(1) << "\n";
  } else if (key == "q") {
    set_cmd(cmd(0), cmd(1), cmd(2) + S);
    std::cout << "  yaw=" << get_cmd()(2) << "\n";
  } else if (key == "e") {
    set_cmd(cmd(0), cmd(1), cmd(2) - S);
    std::cout << "  yaw=" << get_cmd()(2) << "\n";
  } else if (key == "0") {
    set_cmd(0.0, 0.0, 0.0);
    std::cout << "  cmd: zeroed\n";
  }
}

//  Signal handling

static volatile std::sig_atomic_t g_shutdown_requested = 0;

static void sigint_handler(int /*sig*/) { g_shutdown_requested = 1; }

//  Main run loop

void Go2Deploy::run() {
  std::cout << "\n"
            << std::string(60, '=') << "\n"
            << "  JAVE/SHAC Go2 Deployment Controller (C++)\n"
            << std::string(60, '=') << "\n"
            << "  Interface:    " << interface_ << "\n"
            << "  Command rate: " << 1.0 / dt_cmd_ << " Hz\n"
            << "  Policy rate:  " << 1.0 / (dt_cmd_ * policy_decimation_)
            << " Hz (decimation=" << policy_decimation_ << ")\n"
            << "  Walk gains:   kp=" << kp << ", kd=" << kd << "\n"
            << "  Standup gains: kp=" << STANDUP_KP << ", kd=" << STANDUP_KD
            << "\n"
            << "\n  Waiting for robot state...\n";

  // Install SIGINT handler so Ctrl+C triggers graceful sit-down
  std::signal(SIGINT, sigint_handler);

  {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!state_received_.load(std::memory_order_acquire) &&
           !g_shutdown_requested &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  if (g_shutdown_requested) {
    std::cout << "\n  Interrupted during startup.\n";
    return;
  }

  if (!state_received_.load(std::memory_order_acquire)) {
    std::cout << "  ERROR: No state after 10s.\n"
              << "  Check: robot on? interface correct? domain_id matching?\n";
    return;
  }
  std::cout
      << "  Robot state received!\n\n"
      << "  State: IDLE (zero torque, joints free)\n"
      << "  Controls: Enter=advance  x=estop  w/s a/d q/e=vel  0=zero\n\n";

  // Command thread already started by CreateRecurrentThreadEx

  // Keyboard input - breaks on EOF, failed read, or SIGINT
  std::string line;
  while (!g_shutdown_requested && std::getline(std::cin, line)) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
      line.pop_back();
    for (auto &c : line)
      c = static_cast<char>(std::tolower(c));
    process_key(line);
  }

  //  Graceful shutdown
  std::cout << "\n  Shutting down gracefully...\n";
  transition(State::SITDOWN);

  // Wait for sit-down to complete (transitions to IDLE automatically)
  auto t0 = std::chrono::steady_clock::now();
  while (!sitdown_done_.load() &&
         std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  if (sitdown_done_.load())
    std::cout << "  Sit-down complete.\n";
  else {
    std::cout << "  Sit-down timeout, forcing idle.\n";
    state_.store(State::IDLE);
  }

  // Let IDLE send a few zero-torque commands before exiting
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::cout << "  Done.\n";
}

} // namespace jave
