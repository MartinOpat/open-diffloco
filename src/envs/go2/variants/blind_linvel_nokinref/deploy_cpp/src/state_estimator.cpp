/// @file state_estimator.cpp
#include "state_estimator.hpp"

#include "leg_kinematics.hpp"
#include "math_utils.hpp"

#include <cmath>

namespace jave {

ContactAidedKF::ContactAidedKF(double dt, double accel_noise,
                               double contact_vel_noise,
                               double foot_force_threshold,
                               double kinematic_height_threshold,
                               double kinematic_vertical_vel_threshold)
    : dt_(dt), v_(Eigen::Vector3d::Zero()),
      P_(Eigen::Matrix3d::Identity() * 0.1),
      foot_force_threshold_(foot_force_threshold),
      kinematic_height_threshold_(kinematic_height_threshold),
      kinematic_vertical_vel_threshold_(kinematic_vertical_vel_threshold),
      last_contact_mask_{false, false, false, false} {
  double sigma_v = accel_noise * dt;
  Q_ = Eigen::Matrix3d::Identity() * (sigma_v * sigma_v);
  R_foot_ =
      Eigen::Matrix3d::Identity() * (contact_vel_noise * contact_vel_noise);
}

std::array<bool, 4>
ContactAidedKF::get_contact_mask(
    const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
    const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
    const double *foot_forces) const {
  if (foot_forces) {
    std::array<bool, 4> mask{false, false, false, false};
    bool any = false;
    for (int i = 0; i < 4; ++i) {
      if (foot_forces[i] > foot_force_threshold_) {
        mask[i] = true;
        any = true;
      }
    }
    if (any)
      return mask;
  }

  std::array<bool, 4> mask{false, false, false, false};
  Eigen::Matrix<double, 4, 1> foot_z;
  Eigen::Matrix<double, 4, 1> foot_vz;

  for (int leg = 0; leg < 4; ++leg) {
    Eigen::Vector3d q = joint_pos_sim.segment<3>(leg * 3);
    Eigen::Vector3d dq = joint_vel_sim.segment<3>(leg * 3);
    foot_z(leg) = leg_fk(leg, q)(2);
    foot_vz(leg) = (leg_jacobian(leg, q) * dq)(2);
  }

  Eigen::Index lowest_idx = 0;
  const double lowest = foot_z.minCoeff(&lowest_idx);
  bool any = false;
  for (int leg = 0; leg < 4; ++leg) {
    const bool near_ground =
        foot_z(leg) <= lowest + kinematic_height_threshold_;
    const bool low_vz =
        std::abs(foot_vz(leg)) <= kinematic_vertical_vel_threshold_;
    mask[leg] = near_ground && low_vz;
    any = any || mask[leg];
  }

  if (!any)
    mask[static_cast<int>(lowest_idx)] = true;

  return mask;
}

void ContactAidedKF::predict(const Eigen::Vector3d &accel_body,
                             const Eigen::Vector4d &quat) {
  Eigen::Matrix3d R = quat_to_rotmat(quat);
  Eigen::Vector3d a_world = R * accel_body + Eigen::Vector3d(0.0, 0.0, -9.81);

  v_ += a_world * dt_;
  P_ += Q_;
}

void ContactAidedKF::update(const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
                            const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
                            const Eigen::Vector4d &quat,
                            const Eigen::Vector3d &gyro,
                            const std::array<bool, 4> &contact_mask) {
  Eigen::Matrix3d R = quat_to_rotmat(quat);

  for (int leg = 0; leg < 4; ++leg) {
    if (!contact_mask[leg])
      continue;

    Eigen::Vector3d q = joint_pos_sim.segment<3>(leg * 3);
    Eigen::Vector3d dq = joint_vel_sim.segment<3>(leg * 3);

    Eigen::Vector3d p_foot = leg_fk(leg, q);
    Eigen::Matrix3d J = leg_jacobian(leg, q);
    Eigen::Vector3d v_foot_body = J * dq;
    Eigen::Vector3d v_body_meas = -(gyro.cross(p_foot) + v_foot_body);
    Eigen::Vector3d z = R * v_body_meas;

    Eigen::Matrix3d S = P_ + R_foot_;
    Eigen::Matrix3d K = P_ * S.inverse();
    v_ = v_ + K * (z - v_);
    P_ = (Eigen::Matrix3d::Identity() - K) * P_;
  }
}

Eigen::Vector3d
ContactAidedKF::get_body_velocity(const Eigen::Vector4d &quat) const {
  Eigen::Matrix3d R = quat_to_rotmat(quat);
  return R.transpose() * v_;
}

double
ContactAidedKF::get_height(const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
                           const std::array<bool, 4> &contact_mask) const {
  double sum = 0.0;
  int count = 0;

  for (int leg = 0; leg < 4; ++leg) {
    if (contact_mask[leg]) {
      Eigen::Vector3d q = joint_pos_sim.segment<3>(leg * 3);
      Eigen::Vector3d p = leg_fk(leg, q);
      sum += -p(2);
      ++count;
    }
  }

  if (count == 0) {
    for (int leg = 0; leg < 4; ++leg) {
      Eigen::Vector3d q = joint_pos_sim.segment<3>(leg * 3);
      Eigen::Vector3d p = leg_fk(leg, q);
      sum += -p(2);
      ++count;
    }
  }

  return sum / count;
}

std::pair<Eigen::Vector3d, double>
ContactAidedKF::step(const Eigen::Vector3d &accel_body,
                     const Eigen::Vector4d &quat, const Eigen::Vector3d &gyro,
                     const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
                     const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
                     const double *foot_forces) {
  auto mask = get_contact_mask(joint_pos_sim, joint_vel_sim, foot_forces);
  last_contact_mask_ = mask;
  predict(accel_body, quat);
  update(joint_pos_sim, joint_vel_sim, quat, gyro, mask);

  Eigen::Vector3d body_vel = get_body_velocity(quat);
  double height = get_height(joint_pos_sim, mask);

  return {body_vel, height};
}

const std::array<bool, 4> &ContactAidedKF::last_contact_mask() const {
  return last_contact_mask_;
}

void ContactAidedKF::reset() {
  v_.setZero();
  P_ = Eigen::Matrix3d::Identity() * 0.1;
  last_contact_mask_ = {false, false, false, false};
}

} // namespace jave
