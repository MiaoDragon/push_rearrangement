/**
 * @brief 
 * reference:
 * https://github.com/tud-airlab/mppi-isaac
 * https://github.com/google-deepmind/mujoco_mpc
 *
 * parallel implementation of MPPI controller using Mujoco
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


class MujocoMPPIController
{
  public:
    int H = 10;  // horizon
    int N = 100; // batch size
    double default_sigma = 0.1; // default value
    double dt = 0.01; // the time for each horizon step

    std::deque<VectorXd> mu; // queue of mean, size of H
    std::deque<VectorXd> sigma; // queue of covariance (assuming independent)
                                // we store the diagonal values
    MatrixXd nominal_u;              // the nominal trajectory of control (H x nu)
                                     // by default we don't store nominal state
                                     // which is a feature of a specific impl
    VectorXd u_ll, u_ul;

    int nominal_start_idx = 0;  // current index in the nominal traj

    MujocoMPPIController(const int& H, const int& N, const double& default_sigma,
                         const MatrixXd& nominal_u, const VectorXd& u_ll, const VectorXd& u_ul);

    void set_dt(const double& dt);

    /* step one time in the controller to obtain control */
    virtual void step(const mjModel* m, const double* sensordata, VectorXd& control);  // args: represent the observed data
    virtual double get_cost(const mjModel* m, const mjData* d) = 0;  // each horizon
    virtual double get_terminal_cost(const mjModel* m, const mjData* d) = 0;
    // write sensed data to the state
    // this is to be provided by the user
    // NOTE: Eigen is column-major
    virtual void set_data_by_sensor(const double* sensordata, const mjModel* m, mjData* d) = 0;
    virtual void get_state_from_data(const mjModel* m, const mjData* d, VectorXd& state) = 0;
  protected:
    /* sample a batch N of controls, using the mu and sigma stored */
    virtual void sample(std::vector<MatrixXd>& samples) = 0;
};