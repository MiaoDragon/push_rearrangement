/**
 * @file inverse_kinematics.hpp
 * @author your name (you@domain.com)
 * @brief 
 * implement the
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cmath>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include "../utilities/utilities.h"

void pseudo_inv_ik_vel(const Vector6d& twist, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq);

void damped_inv_ik_vel(const Vector6d& twist, const double damp, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq);
void pseudo_inv_ik_position_vel(const Vector3d& linear_v, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq);

void damped_inv_ik_position_vel(const Vector3d& linear_v, const double damp, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq);