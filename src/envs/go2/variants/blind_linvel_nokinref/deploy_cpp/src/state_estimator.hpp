#pragma once
/// @file state_estimator.hpp
/// Contact-aided Kalman filter for Go2 body velocity and height estimation.

#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>

namespace jave {

class ContactAidedKF {
public:
  ContactAidedKF(double dt, double accel_noise = 1.0,
                 double contact_vel_noise = 0.02,
                 double foot_force_threshold = 20.0,
                 double kinematic_height_threshold = 0.035,
                 double kinematic_vertical_vel_threshold = 0.35);

  std::array<bool, 4>
  get_contact_mask(const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
                   const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
                   const double *foot_forces = nullptr) const;

  void predict(const Eigen::Vector3d &accel_body, const Eigen::Vector4d &quat);

  void update(const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
              const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
              const Eigen::Vector4d &quat, const Eigen::Vector3d &gyro,
              const std::array<bool, 4> &contact_mask);

  Eigen::Vector3d get_body_velocity(const Eigen::Vector4d &quat) const;

  double get_height(const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
                    const std::array<bool, 4> &contact_mask) const;

  std::pair<Eigen::Vector3d, double>
  step(const Eigen::Vector3d &accel_body, const Eigen::Vector4d &quat,
       const Eigen::Vector3d &gyro,
       const Eigen::Matrix<double, 12, 1> &joint_pos_sim,
       const Eigen::Matrix<double, 12, 1> &joint_vel_sim,
       const double *foot_forces = nullptr);

  const std::array<bool, 4> &last_contact_mask() const;

  void reset();

private:
  double dt_;

  Eigen::Vector3d v_;
  Eigen::Matrix3d P_;
  Eigen::Matrix3d Q_;
  Eigen::Matrix3d R_foot_;

  double foot_force_threshold_;
  double kinematic_height_threshold_;
  double kinematic_vertical_vel_threshold_;
  std::array<bool, 4> last_contact_mask_;
};

} // namespace jave
