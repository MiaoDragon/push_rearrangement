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

void pseudo_inv_ik_position_vel_nullspace(const Vector3d& linear_v, const int link_idx, const VectorXd& nullspace_v,
                                            const mjModel* m, const mjData* d,
                                            const std::vector<int>& select_dofs,
                                            VectorXd& dq);

void damped_inv_ik_position_vel_nullspace(const Vector3d& linear_v, const double damp, const int link_idx, const VectorXd& nullspace_v,
                                            const mjModel* m, const mjData* d,
                                            const std::vector<int>& select_dofs,
                                            VectorXd& dq);




/**
 * @brief construct the cbf constriants and return the constraint matrix CI and vector ci
 * 
 * @param m 
 * @param d 
 * @param select_qpos 
 * @param select_dofs 
 * @param collision_pairs 
 * @param robot_geom_ids 
 * @param max_angvel 
 * @param CI 
 * @param ci 
 */
void cbf_constraints(const mjModel* m, const mjData* d, const std::vector<int>& select_qpos, const std::vector<int>& select_dofs,
                     const IntPairVector& collision_pairs, const std::vector<int>& robot_geom_ids, const double max_angvel,
                     MatrixXd& CI, VectorXd& ci0, int& ci_size);


/**
 * @brief given a goal pose, compute the joint velocities to reach the goal pose
 * by reducing the error. The error e(t) has two parts, e_p(t) and e_r(t).
 * e_p(t) = p_d(t) - p(t), e_r(t) = R_d(t) * R(t)^T
 * set dot(e_p(t)) = -K_p * e_p(t), vee(dot(e_r(t))*e_r(t)^{-1}) = -1/T(t) * vee(log(e_r(t)))
 * thus we are setting dot(p(t)) = K_p*e_p(t), and w(t) = -1/T(t) * vee(log(e_r(t)))
 * 
 * This corresponds to solving a QP with collision avoidance and joint limits constraints as CBF
    min_u ||([J_p,J_r]^Tu-[K_d*e_p(t),1/T(t)log(e_r(t))_V])||
    constraints:
    - joint angle control barrier function
        h_lb[j] = q[j]-q_lb[j], h_ub[j] = q_ub[j]-q[j]
    - collision avoidance control barrier function
        alpha = 1*L_{max}*||q_dot_max||/h_min
        dh/dp*dp/dq*u+alpha*h(q) >= 0
    - self collision avoidance control barrier function
 * @param goal 
 * @param link_idx 
 * @param m 
 * @param d 
 * @param select_dofs 
 * @param exclude_body_pairs 
 * @param robot_geom_ids 
 * @param dq 
 */
void ik_pose_cbf(const Matrix4d& goal, const int link_idx, const mjModel* m, const mjData* d,
                 const std::vector<int>& select_qpos, const std::vector<int>& select_dofs, 
                 const IntPairVector& collision_pairs,
                 const std::vector<int>& robot_geom_ids, const double max_angvel,
                 VectorXd& dq);

void ik_position_cbf(const Vector3d& goal, const int link_idx, const mjModel* m, const mjData* d,
                    const std::vector<int>& select_qpos, const std::vector<int>& select_dofs, 
                    const IntPairVector& collision_pairs,
                    const std::vector<int>& robot_geom_ids, const double max_angvel,
                    VectorXd& dq);