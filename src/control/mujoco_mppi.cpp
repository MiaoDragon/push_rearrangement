 /**
 * @file mppi.cpp
 * @author your name (you@domain.com)
 * @brief 
 * reference:
 * https://github.com/tud-airlab/mppi-isaac
 * https://github.com/google-deepmind/mujoco_mpc
 *
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "mujoco_mppi.h"
#include <mujoco/mjdata.h>


MujocoMPPIController::MujocoMPPIController(const int& H_in, const int& N_in, const double& default_sigma_in,
                                           const MatrixXd& nominal_u_in, const VectorXd& u_ll_in, const VectorXd& u_ul_in)
{
    H = H_in;  N = N_in;  default_sigma = default_sigma_in;
    nominal_u = nominal_u_in;
    u_ll = u_ll_in;  u_ul = u_ul_in;
    // initialize mu and sigma
    for (int i=0; i<H; i++)
    {
        VectorXd mu_i(u_ll_in.size());
        VectorXd sigma_i(u_ll_in.size());
        mu_i.setZero();
        mu.push_back(mu_i);
        sigma_i.setConstant(default_sigma_in);
        sigma.push_back(sigma_i);
    }
}

void MujocoMPPIController::set_dt(const double& dt_in)
{
    dt = dt_in;
}


void MujocoMPPIController::step(const mjModel* m, const double* sensordata, VectorXd& control)
{
    // observe: obtain the state vars. for simplicity, assume it's fed in
    // set obervation to mujoco data

    // sample(N) -> eps  // N x H x nu

    // parallel for:
    //   u = nominal_u + eps[i]
    //   system.reset(x)
    //   for h = 0 .. H:
    //      cost(x,u)
    //      system.propagate(x,u[i,h]) -> x
    //   terminal_cost()
}