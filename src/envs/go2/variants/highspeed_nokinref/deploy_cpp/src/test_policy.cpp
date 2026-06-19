/// @file test_policy.cpp
/// Validation tool for Phase 1+2.
///
/// Usage:
///   ./test_policy                        # run math unit tests only
///   ./test_policy <policy_deploy.npz>    # also load + test forward pass
///
/// Compare outputs with the Python equivalents to validate correctness.

#include "leg_kinematics.hpp"
#include "math_utils.hpp"
#include "policy.hpp"

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <string>

using namespace jave;

static int g_pass = 0, g_fail = 0;

static void check(const char *name, bool ok) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::cout << "  FAIL: " << name << "\n";
  }
}

static void check_near(const char *name, double a, double b,
                       double tol = 1e-10) {
  check(name, std::abs(a - b) < tol);
}

template <int N>
static void check_vec(const char *name, const Eigen::Matrix<double, N, 1> &a,
                      const Eigen::Matrix<double, N, 1> &b,
                      double tol = 1e-10) {
  check(name, (a - b).norm() < tol);
}

// Math utils tests

static void test_quat_inv() {
  Vec4 q{0.5, 0.5, 0.5, 0.5};
  Vec4 qi = quat_inv(q);
  Vec4 expected{0.5, -0.5, -0.5, -0.5};
  check_vec<4>("quat_inv", qi, expected);
}

static void test_quat_rotate_identity() {
  Vec4 q{1, 0, 0, 0}; // identity
  Vec3 v{1, 2, 3};
  Vec3 r = quat_rotate(v, q);
  check_vec<3>("quat_rotate identity", r, v);
}

static void test_quat_rotate_90z() {
  // 90 deg about z: q = (cos45, 0, 0, sin45)
  double s = std::sqrt(2.0) / 2.0;
  Vec4 q{s, 0, 0, s};
  Vec3 v{1, 0, 0};
  Vec3 r = quat_rotate(v, q);
  Vec3 expected{0, 1, 0};
  check_vec<3>("quat_rotate 90z", r, expected, 1e-9);
}

static void test_quat_to_rotmat_identity() {
  Vec4 q{1, 0, 0, 0};
  Mat3 R = quat_to_rotmat(q);
  check("rotmat identity", (R - Mat3::Identity()).norm() < 1e-12);
}

static void test_layer_norm() {
  Eigen::VectorXd x(4);
  x << 1, 2, 3, 4;
  Eigen::VectorXd s(4);
  s << 1, 1, 1, 1;
  Eigen::VectorXd b(4);
  b << 0, 0, 0, 0;
  Eigen::VectorXd r = layer_norm(x, s, b);
  // Mean-centered, unit var
  check_near("layer_norm mean", r.mean(), 0.0, 1e-10);
  double var = (r.array() - r.mean()).square().mean();
  check_near("layer_norm var", var, 1.0, 0.01);
}

static void test_elu() {
  Eigen::VectorXd x(4);
  x << -2, -1, 0, 1;
  Eigen::VectorXd r = elu(x);
  check_near("elu positive", r(3), 1.0);
  check_near("elu zero", r(2), 0.0);
  check("elu negative", r(0) < 0 && r(0) > -1.0);
}

// Leg kinematics tests

static void test_leg_fk_zero() {
  // At q=0 the leg should hang straight down
  Eigen::Vector3d q{0, 0, 0};
  Eigen::Vector3d p = leg_fk(0, q); // FL
  // x should be hip offset x, z should be -(thigh+calf)
  check_near("fk_zero z", p(2), -(THIGH_LEN + CALF_LEN), 0.01);
}

static void test_leg_jacobian_fd() {
  // Finite-difference check
  Eigen::Vector3d q{0.1, 0.5, -1.0};
  const double h = 1e-7;
  Eigen::Matrix3d J = leg_jacobian(1, q); // FR
  for (int j = 0; j < 3; ++j) {
    Eigen::Vector3d qp = q, qm = q;
    qp(j) += h;
    qm(j) -= h;
    Eigen::Vector3d fd = (leg_fk(1, qp) - leg_fk(1, qm)) / (2 * h);
    double err = (J.col(j) - fd).norm();
    std::string name = "jacobian_fd col " + std::to_string(j);
    check(name.c_str(), err < 1e-5);
  }
}

// Policy test (optional, needs .npz file)

static void test_policy(const std::string &path) {
  std::cout << "\n-- Policy load test\n";
  NumpyPolicy pol(path);

  // Sanity checks
  check("norm_mean dim", pol.norm_mean.size() > 0);
  check("default_joints dim", pol.default_joints.size() == 12);
  check("dt > 0", pol.dt > 0);

  // Forward pass with zeros --> should produce valid output
  Eigen::VectorXd obs =
      Eigen::VectorXd::Zero(pol.actor_history_len * pol.actor_frame_obs_dim);
  Eigen::VectorXd act = pol(obs);
  check("action dim", act.size() == 12);
  check("action bounded", act.cwiseAbs().maxCoeff() <= 1.0 + 1e-9);
  std::cout << "  action(zeros): " << act.head(4).transpose() << " ...\n";

  // Target joints
  auto tgt = pol.get_target_joints(act);
  check("target dim", tgt.size() == 12);
  std::cout << "  target(0):     " << tgt.head(4).transpose() << " ...\n";
}

// Main

int main(int argc, char **argv) {
  std::cout << "-- Phase 1: math_utils\n";
  test_quat_inv();
  test_quat_rotate_identity();
  test_quat_rotate_90z();
  test_quat_to_rotmat_identity();
  test_layer_norm();
  test_elu();

  std::cout << "\n-- Phase 1: leg_kinematics\n";
  test_leg_fk_zero();
  test_leg_jacobian_fd();

  if (argc > 1) {
    test_policy(argv[1]);
  } else {
    std::cout << "\n  (pass .npz path as arg to test policy loading)\n";
  }

  std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
  return g_fail > 0 ? 1 : 0;
}
