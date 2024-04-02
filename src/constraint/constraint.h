/**
 * @file constraint.h
 * @author your name (you@domain.com)
 * @brief 
 * Implement the constraint functions
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>

#include "../utilities/utilities.h"
#include "../contact/contact.h"


/**
 * @brief 
 * Given cs mode and ss mode, generate the matrix to represent the constraint.
 * matrix involved:
 * - equality constraints for v_c
 * - equality/inequality constraint for v_c given cs_mode
 * - equality/inequality constraint 
 * - part of force balance constraint
 *
 * NOTE:
 * this is a general formulation of the constraints.
 * variables: V1, V2 (twist in body or spatial frame),
 *            vc (linear velocity at contact, written in contact frame)
 *               body 1 relative to body 2
 *            fc (linear force at contact, written in contact frame)
 * 
 * ss_mode refers to the tangent polyhedron approx to the friction cone
 * @param contact 
 * @param cs_mode: 1 means NOT in contact. 0 means in contact.
 * @param ss_mode 
 */
void contact_constraint(const mjModel* m, const mjData* d, const Contact* contact, 
                             const int cs_mode, const std::vector<int> ss_mode,
                             MatrixXd& Ce, VectorXd& ce0, int& ce_size,
                             MatrixXd& Ci, VectorXd& ci0, int& ci_size,
                             MatrixXd& Fe1, MatrixXd& Fe2,
                             MatrixXd& Te1, MatrixXd& Te2);

/**
 * @brief 
 * copy over the constraints at contact, to create matrix for optimization.
 * 
 * @param m 
 * @param d 
 * @param robot_v_indices 
 * @param obj_body_indices 
 * @param contacts 
 * @param cs_modes 
 * @param ss_modes 
 * @param Ae 
 * @param ae0 
 * @param ae_size 
 * @param Ai 
 * @param ai0 
 * @param ai_size 
 */
void total_constraints(const mjModel* m, const mjData* d, const std::vector<int>& robot_v_indices,
                       const std::vector<int>& obj_body_indices,
                       const Contacts& contacts, 
                       const std::vector<int> cs_modes, 
                       const std::vector<std::vector<int>> ss_modes,
                       MatrixXd& Ae, VectorXd& ae0, int& ae_size,
                       MatrixXd& Ai, VectorXd& ai0, int& ai_size
                       );
/**
 * @brief 
 * objective: 0.5 x^TGx + g0^Tx
 * equivalent: x^TGx + 2*g0^Tx
 * vel objective:
 * sum_i (v(i)-v_target(i))^2 + epsilon_lambda*(w(i)-w_target(i))^2 + epsilon_lambda(q^Tq + sum_i contact_i^Tcontact_i)
 * @param m 
 * @param d 
 * @param robot_v_indices 
 * @param target_vs: the target velocities for each object in the optimization problem
 * @param active_vs: boolean indicator denoting if we are solving for the velocities for each object in the optimziation problem
 * @param ncon
 * @param K
 * @param G 
 * @param g0 
 */
void vel_objective(const mjModel* m, const mjData* d, 
                   const std::vector<int>& robot_v_indices,
                   const std::vector<Vector6d>& target_vs,
                   const std::vector<int>& active_vs,  // if the vel is active
                   const int ncon, const int K,  // K: ss mode
                   MatrixXd& G, VectorXd& g0);