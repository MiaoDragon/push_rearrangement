/**
 * @file mppi.h
 * @author your name (you@domain.com)
 * @brief 
 * ref: https://github.com/google-deepmind/mujoco_mpc
 * 
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cmath>
#include <vector>
#include <deque>
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


class MPPI
{
  public:
    /**
     * @brief Construct a new MPPI::MPPI object
     * 
     * @param N 
     * @param H 
     * @param dt
     * the step size in the MPPI
     * @param task 
     * a class defining the task.
     * @param policy 
     * the policy to optimize
     */
    int N=10, H=10;
    double dt=0.01;
    MPPI(const int N, const int H, const double dt, const ControlTask& task, ControlPolicy& policy)
    {
        this->N = N; this->H = H; this->dt = dt;
        this->task = task; this->policy = policy;
    }

    /* rollout using the sampled policy parameters to obtain the costs */
    void rollout(const MatrixXd& policy_params, VectorXd& costs);
    /* given the sensed data, plan from the current time */
    void step(const VectorXd& sensor_data, VectorXd& control);


  protected:
    const ControlTask& task;
    ControlPolicy& policy;




};