#pragma once
/// @file math_utils.hpp
/// Quaternion and rotation helpers for Go2 deployment.
/// All quaternions are (w, x, y, z) convention (MuJoCo/Eigen default).

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

namespace jave {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;

// Quaternion helpers

/// Quaternion conjugate (= inverse for unit quaternions).
inline Vec4 quat_inv(const Vec4 &q) { return {q(0), -q(1), -q(2), -q(3)}; }

/// Rotate vector v by quaternion q  (Hamilton product shortcut).
inline Vec3 quat_rotate(const Vec3 &v, const Vec4 &q) {
  const double w = q(0);
  const Vec3 u = q.tail<3>();
  return 2.0 * u.dot(v) * u + (w * w - u.dot(u)) * v + 2.0 * w * u.cross(v);
}

/// Quaternion (w,x,y,z) --> 3x3 rotation matrix.
inline Mat3 quat_to_rotmat(const Vec4 &q) {
  const double w = q(0), x = q(1), y = q(2), z = q(3);
  Mat3 R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return R;
}

// Activation functions

/// Element-wise ELU (alpha=1).
template <int N>
inline Eigen::Matrix<double, N, 1> elu(const Eigen::Matrix<double, N, 1> &x) {
  return x.array().max(0.0) + (x.array().min(0.0).exp() - 1.0).min(0.0);
}

/// Dynamic-size ELU.
inline Eigen::VectorXd elu(const Eigen::VectorXd &x) {
  return x.array().max(0.0) + (x.array().min(0.0).exp() - 1.0).min(0.0);
}

/// Element-wise tanh (just wraps Eigen, here for symmetry).
inline Eigen::VectorXd tanh_vec(const Eigen::VectorXd &x) {
  return x.array().tanh();
}

// Layer norm

/// Layer normalization:  scale * (x - mean) / sqrt(var + eps) + bias
inline Eigen::VectorXd layer_norm(const Eigen::VectorXd &x,
                                  const Eigen::VectorXd &scale,
                                  const Eigen::VectorXd &bias,
                                  double eps = 1e-6) {
  const double mean = x.mean();
  const double var = (x.array() - mean).square().mean();
  return scale.array() * (x.array() - mean) / std::sqrt(var + eps) +
         bias.array();
}

} // namespace jave
