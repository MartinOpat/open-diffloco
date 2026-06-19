#pragma once
/// @file policy.hpp
/// Loads exported .npz weights and runs actor MLP inference.
/// Network: [Dense + LayerNorm + ELU] x N  -->  Dense + tanh

#include <Eigen/Core>
#include <limits>
#include <string>
#include <vector>

namespace jave {

/// One hidden layer: Dense + LayerNorm + ELU.
struct HiddenLayer {
  Eigen::MatrixXd kernel;   // (in, out)
  Eigen::VectorXd bias;     // (out,)
  Eigen::VectorXd ln_scale; // (out,)
  Eigen::VectorXd ln_bias;  // (out,)
};

class NumpyPolicy {
public:
  explicit NumpyPolicy(const std::string &npz_path);

  /// Run actor forward pass. obs -> action in [-1, 1]^12.
  Eigen::VectorXd operator()(const Eigen::VectorXd &obs) const;

  /// Run MLP forward pass on pre-normalized input (skip normalization).
  Eigen::VectorXd forward_raw(const Eigen::VectorXd &x_norm) const;

  /// Compute PD target joint positions from action.
  Eigen::VectorXd get_target_joints(const Eigen::VectorXd &action) const;

  //  Public config (read after construction)

  // Normalizer
  Eigen::VectorXd norm_mean;
  Eigen::VectorXd norm_var;
  int actor_history_len = 1;
  int actor_frame_obs_dim = 0;
  static constexpr double NORM_EPS = 1e-4;

  // Environment config
  Eigen::VectorXd default_joints; // (12,)
  Eigen::VectorXd action_scale;   // scalar (1,) or per-joint (12,)
  Eigen::Vector2d cmd_vel_x_range = Eigen::Vector2d(-1.5, 1.5);
  Eigen::Vector2d cmd_vel_y_range = Eigen::Vector2d(-1.0, 1.0);
  Eigen::Vector2d cmd_yaw_rate_range = Eigen::Vector2d(-1.5, 1.5);
  double dt = 0.0;
  // Actuator gains from training (NaN if not available)
  double training_kp = std::numeric_limits<double>::quiet_NaN();
  double training_kd = std::numeric_limits<double>::quiet_NaN();

private:
  std::vector<HiddenLayer> layers_;
  Eigen::MatrixXd out_kernel_; // (hidden, 12)
  Eigen::VectorXd out_bias_;   // (12,)
};

} // namespace jave
