/// @file policy.cpp
#include "policy.hpp"
#include "math_utils.hpp"
#include "npz_reader.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace jave {

//  helpers

namespace {

Eigen::VectorXd load_vec(const npz::NpzFile &npz, const std::string &key) {
  auto it = npz.find(key);
  if (it == npz.end())
    throw std::runtime_error("Missing key in .npz: " + key);
  return it->second.to_vector();
}

Eigen::MatrixXd load_mat(const npz::NpzFile &npz, const std::string &key) {
  auto it = npz.find(key);
  if (it == npz.end())
    throw std::runtime_error("Missing key in .npz: " + key);
  return it->second.to_matrix();
}

double load_scalar(const npz::NpzFile &npz, const std::string &key) {
  auto it = npz.find(key);
  if (it == npz.end())
    throw std::runtime_error("Missing key in .npz: " + key);
  return it->second.to_scalar();
}

Eigen::MatrixXd make_kinematic_ref(int step_k, double scale) {
  Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(step_k * 2, 12);
  for (int t = 0; t < ref.rows(); ++t) {
    const double wave = -std::cos((2.0 * M_PI / step_k) * t) * (scale / 2.0) +
                        (scale / 2.0);
    ref(t, 2) = wave;
    ref(t, 5) = scale - wave;
    ref(t, 8) = scale - wave;
    ref(t, 11) = wave;
  }
  return ref;
}

} // namespace

//  Construction

NumpyPolicy::NumpyPolicy(const std::string &npz_path) {
  std::cout << "Loading policy: " << npz_path << "\n";
  auto npz = npz::load_npz(npz_path);

  // Normalizer
  norm_mean = load_vec(npz, "norm_mean");
  norm_var = load_vec(npz, "norm_var");

  // Network layers
  const int n_hidden = static_cast<int>(load_scalar(npz, "n_hidden"));
  layers_.resize(n_hidden);
  for (int i = 0; i < n_hidden; ++i) {
    auto &L = layers_[i];
    L.kernel = load_mat(npz, "dense_" + std::to_string(i) + "_kernel");
    L.bias = load_vec(npz, "dense_" + std::to_string(i) + "_bias");
    L.ln_scale = load_vec(npz, "ln_" + std::to_string(i) + "_scale");
    L.ln_bias = load_vec(npz, "ln_" + std::to_string(i) + "_bias");
  }
  out_kernel_ = load_mat(npz, "dense_" + std::to_string(n_hidden) + "_kernel");
  out_bias_ = load_vec(npz, "dense_" + std::to_string(n_hidden) + "_bias");

  // Print architecture
  const int in_dim = static_cast<int>(layers_[0].kernel.rows());
  const int out_dim = static_cast<int>(out_kernel_.cols());
  std::cout << "  Actor: " << in_dim;
  for (int i = 0; i < n_hidden; ++i)
    std::cout << " -> " << layers_[i].kernel.cols();
  std::cout << " -> " << out_dim << "\n";
  std::cout << "  Hidden layers: " << n_hidden
            << " (each Dense + LayerNorm + ELU)\n";

  // Environment config
  default_joints = load_vec(npz, "default_joints");
  action_scale = load_vec(npz, "action_scale");
  dt = load_scalar(npz, "dt");
  if (!npz::has_key(npz, "actor_history_len") ||
      !npz::has_key(npz, "actor_frame_obs_dim"))
    throw std::runtime_error(
        "Policy predates the reduced actor observation history format");
  actor_history_len = static_cast<int>(load_scalar(npz, "actor_history_len"));
  actor_frame_obs_dim =
      static_cast<int>(load_scalar(npz, "actor_frame_obs_dim"));
  if (in_dim != actor_history_len * actor_frame_obs_dim)
    throw std::runtime_error(
        "Actor input dimension does not match observation history metadata");
  if (norm_mean.size() != actor_frame_obs_dim)
    throw std::runtime_error(
        "Normalizer dimension does not match actor frame size");

  std::cout << "  Deployment obs: " << actor_history_len << "x"
            << actor_frame_obs_dim << "D\n";

  std::cout << "  dt=" << dt << "s\n";

  if (npz::has_key(npz, "gait_ref")) {
    gait_ref = load_mat(npz, "gait_ref");
    cycle_len = static_cast<int>(gait_ref.rows());
  } else {
    gait_ref = make_kinematic_ref(20, 0.3);
    cycle_len = static_cast<int>(gait_ref.rows());
  }

  // Actuator gains
  if (npz::has_key(npz, "actuator_kp")) {
    training_kp = load_vec(npz, "actuator_kp")(0);
    double kd_act = npz::has_key(npz, "actuator_kd")
                        ? load_vec(npz, "actuator_kd")(0)
                        : 0.0;
    double kd_joint = npz::has_key(npz, "joint_damping")
                          ? load_vec(npz, "joint_damping")(0)
                          : 0.0;
    training_kd = kd_act + kd_joint;
    std::cout << "  Training gains: kp=" << training_kp
              << ", kd=" << training_kd << " (actuator=" << kd_act
              << " + joint_damping=" << kd_joint << ")\n";
  } else {
    std::cout << "  WARNING: No actuator gains in .npz!\n";
  }
}

//  Forward pass

Eigen::VectorXd NumpyPolicy::operator()(const Eigen::VectorXd &obs) const {
  if (obs.size() != actor_history_len * actor_frame_obs_dim)
    throw std::runtime_error("Unexpected actor observation dimension");
  Eigen::VectorXd x(obs.size());
  for (int i = 0; i < actor_history_len; ++i) {
    x.segment(i * actor_frame_obs_dim, actor_frame_obs_dim) =
        (obs.segment(i * actor_frame_obs_dim, actor_frame_obs_dim) - norm_mean)
            .array() /
        (norm_var.array() + NORM_EPS).sqrt();
  }
  return forward_raw(x);
}

Eigen::VectorXd NumpyPolicy::forward_raw(const Eigen::VectorXd &x_in) const {
  Eigen::VectorXd x = x_in;

  // Hidden layers: Dense --> LayerNorm --> ELU
  for (const auto &L : layers_) {
    x = (x.transpose() * L.kernel).transpose() + L.bias; // Dense
    x = layer_norm(x, L.ln_scale, L.ln_bias);            // LN
    x = elu(x);                                          // ELU
  }

  // Output layer: Dense --> tanh
  x = (x.transpose() * out_kernel_).transpose() + out_bias_;
  return tanh_vec(x);
}

//  Target joints

Eigen::VectorXd
NumpyPolicy::get_target_joints(const Eigen::VectorXd &action) const {
  Eigen::VectorXd a = action.cwiseMax(-1.0).cwiseMin(1.0);
  // action_scale may be scalar (1,) or per-joint (12,)
  if (action_scale.size() == 1) {
    return default_joints + a * action_scale(0);
  }
  return default_joints + a.cwiseProduct(action_scale);
}

Eigen::VectorXd NumpyPolicy::get_target_joints(const Eigen::VectorXd &action,
                                               int phase_idx) const {
  Eigen::VectorXd target = get_target_joints(action);
  const int idx = ((phase_idx % cycle_len) + cycle_len) % cycle_len;
  return target + gait_ref.row(idx).transpose();
}

} // namespace jave
