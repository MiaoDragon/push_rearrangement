/**
 * @file mujoco_mppi_intvel.h
 * @author your name (you@domain.com)
 * @brief 
 * This implements the mppi for using the robot to achieve a certain task
 * given:
 * - a nominal state trajectory (joint angles),
 * - a nominal vel trajectory
 *
 * sampling:
 * - method 1: sample spline vel trajectory, then integrate into state
 *   add cost for state boundary
 * - method 2: sample spline state trajectory, then derive the vel
 * - complication: there are joint bounds and joint vel bounds, so we need
 *   to add the disturbance to the state and vel, and also make sure the 
 *   result does not go over the limit.
 *
 * 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */
 
#pragma once

#include <cmath>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>

#include "../utilities/utilities.h"
#include "../utilities/sample.h"
#include "mujoco_mppi.h"

class MujocoMPPIControllerIntvel : public MujocoMPPIController
{
  public:
    MatrixXd nominal_x;              // the nominal trajectory of control (T x nx)
    VectorXd x_ll, x_ul;

    MujocoMPPIControllerIntvel(const int& H, const int& N, const double& default_sigma,
                               const MatrixXd& nominal_x,
                               const MatrixXd& nominal_u,
                               const VectorXd& x_ll, const VectorXd& x_ul,
                               const VectorXd& u_ll, const VectorXd& u_ul);
    void step(const mjModel* m, const double* sensordata, VectorXd& control) override;  // args: represent the observed data

    // double cost(mjModel* m, mjData* d);  // each timestep cost
    // double terminal_cost(mjModel* m, mjData* d);

    void set_pos_act_indices(const std::vector<int>& pos_act_indices);
    void set_vel_ctrl_indices(const std::vector<int>& vel_ctrl_indices);

  protected:
    int knot_scale = 3; // how many timesteps do a knot cover?
    std::vector<int> pos_act_indices;
    std::vector<int> vel_ctrl_indices;
    /* sample N truncated velocities, fit spline, and then obtain the states by integration */
    void sample(std::vector<MatrixXd>& samples);
    void sample(const VectorXd& start_state, std::vector<MatrixXd>& x_samples, std::vector<MatrixXd>& u_samples);
    void get_x_bound_cost(const std::vector<MatrixXd>& x_samples,
                            std::vector<double>& x_bound_costs);  // compute the bound cost for the whole samples
};