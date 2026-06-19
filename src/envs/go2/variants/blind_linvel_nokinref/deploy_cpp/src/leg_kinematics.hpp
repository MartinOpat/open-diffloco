#pragma once
/// @file leg_kinematics.hpp
/// Go2 leg forward kinematics and analytic Jacobian.
/// Leg indices: 0=FL, 1=FR, 2=RL, 3=RR  (simulation order).

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace jave {

// URDF constants

inline constexpr double THIGH_LEN = 0.213;
inline constexpr double CALF_LEN = 0.213;

/// Hip offsets from body CoM (x_fwd, y_left, z_up) in body frame.
///  Row order: FL, FR, RL, RR.
inline const Eigen::Matrix<double, 4, 3> &hip_offsets() {
  static const Eigen::Matrix<double, 4, 3> H =
      (Eigen::Matrix<double, 4, 3>() << +0.1934, +0.0465, 0.0, // FL
       +0.1934, -0.0465, 0.0,                                  // FR
       -0.1934, +0.0465, 0.0,                                  // RL
       -0.1934, -0.0465, 0.0                                   // RR
       )
          .finished();
  return H;
}

/// Lateral offset from hip joint to thigh joint (positive = outward).
inline constexpr double HIP_LENGTHS[4] = {+0.0955, -0.0955, +0.0955, -0.0955};

// Forward kinematics

/// Forward kinematics for one leg.
/// @param leg  0=FL, 1=FR, 2=RL, 3=RR
/// @param q    [hip_abduction, thigh, calf]
/// @return     Foot position in body frame (3,).
inline Eigen::Vector3d leg_fk(int leg, const Eigen::Vector3d &q) {
  assert(leg >= 0 && leg < 4);
  const double q0 = q(0), q1 = q(1), q2 = q(2);
  const double d = HIP_LENGTHS[leg];

  const double dx = -THIGH_LEN * std::sin(q1) - CALF_LEN * std::sin(q1 + q2);
  const double dz = -THIGH_LEN * std::cos(q1) - CALF_LEN * std::cos(q1 + q2);

  const double c0 = std::cos(q0), s0 = std::sin(q0);
  Eigen::Vector3d foot_rel_hip;
  foot_rel_hip << dx, d * c0 - dz * s0, d * s0 + dz * c0;

  return hip_offsets().row(leg).transpose() + foot_rel_hip;
}

// Analytic Jacobian

/// Analytic Jacobian d(foot_pos)/d(q) for one leg.
/// @return 3x3 matrix, columns = [d/dq0, d/dq1, d/dq2].
inline Eigen::Matrix3d leg_jacobian(int leg, const Eigen::Vector3d &q) {
  assert(leg >= 0 && leg < 4);
  const double q0 = q(0), q1 = q(1), q2 = q(2);
  const double d = HIP_LENGTHS[leg];
  const double c0 = std::cos(q0), s0 = std::sin(q0);

  const double dz = -THIGH_LEN * std::cos(q1) - CALF_LEN * std::cos(q1 + q2);

  const double ddx_dq1 =
      -THIGH_LEN * std::cos(q1) - CALF_LEN * std::cos(q1 + q2);
  const double ddz_dq1 =
      THIGH_LEN * std::sin(q1) + CALF_LEN * std::sin(q1 + q2);

  const double ddx_dq2 = -CALF_LEN * std::cos(q1 + q2);
  const double ddz_dq2 = CALF_LEN * std::sin(q1 + q2);

  Eigen::Matrix3d J;
  // d/dq0
  J(0, 0) = 0.0;
  J(1, 0) = -d * s0 - dz * c0;
  J(2, 0) = d * c0 - dz * s0;
  // d/dq1
  J(0, 1) = ddx_dq1;
  J(1, 1) = -ddz_dq1 * s0;
  J(2, 1) = ddz_dq1 * c0;
  // d/dq2
  J(0, 2) = ddx_dq2;
  J(1, 2) = -ddz_dq2 * s0;
  J(2, 2) = ddz_dq2 * c0;

  return J;
}

} // namespace jave
