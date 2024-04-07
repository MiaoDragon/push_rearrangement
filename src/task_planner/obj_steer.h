/**
 * @file obj_steer.h
 * @author your name (you@domain.com)
 * @brief 
 * implement the behavior of steering in the object level.
 * @version 0.1
 * @date 2024-04-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "mujoco/mjdata.h"
#include "mujoco/mjmodel.h"
#include "../constraint/constraint.h"
#include "../contact/contact.h"
#include "../utilities/utilities.h"
#include "../utilities/trajectory.h"
#include <memory>


/**
 * @brief 
 * steer for a single object to get from start to goal
 * assume the end-effector body is called ee_position.
 * The joint names are ee_position_x, ee_position_y, ee_position_z
 * NOTE: this does not check for collision
 * TODO: check for collision elsewhere
 * @param m 
 * @param d 
 * @param obj_body_indices 
 * @param start_T 
 * @param goal_T 
 * @param ee_contact_in_obj 
 * @param robot_ee_v 
 * @param obj_twist 
 * @param robot_ee_traj 
 * @param obj_pose_traj 
 */

bool single_obj_steer_ee_position(const mjModel* m, mjData* d, 
                                //   const std::vector<int>& obj_body_indices,
                                  const int& obj_body_id, // automatically already root id of object
                                  const Matrix4d& start_T, const Matrix4d& goal_T, 
                                  const Vector3d& ee_contact_in_obj,
                                  Vector6d& obj_twist,
                                  std::shared_ptr<PositionTrajectory>& robot_ee_traj,
                                  std::shared_ptr<PoseTrajectory>& obj_pose_traj);

