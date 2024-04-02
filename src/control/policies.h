/**
 * @file policies.h
 * @author your name (you@domain.com)
 * @brief 
 * implement various policies to control
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <cmath>
#include <random>
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
#include "../utilities/spline.h"
#include "../utilities/sample.h"

template<typename ThetaType>
class ControlPolicy
{
  public:
    int ctrl_dim;
    double Ht;
    ControlPolicy(const double Ht, const int ctrl_dim)
    {
        this->ctrl_dim = ctrl_dim;
        this->Ht = Ht;
    };

    /* given the state and time, return control using the current theta */
    virtual void action(const VectorXd& state, const double t, VectorXd& control) = 0;
    /* given the state, time and the parameter theta, return control */
    virtual void action(const VectorXd& state, const ThetaType& theta, const double t,
                        VectorXd& control) = 0;
    virtual void sample_gauss_param(const int N, const double sigma, std::vector<ThetaType>& theta) = 0;
    // void sample_truncated_gauss_param(s)

    /* given the sampled parameters and their costs, optimize the parameter */
    virtual void optimize(const std::vector<ThetaType>& thetas, const VectorXd& Js) = 0;
    // Q: should we use std::vector or use Eigen::VectorXd?

    /* shift the time by certain amount */
    virtual void shift_by_time(const double dt) = 0;

    void get_theta(ThetaType& theta) {theta = this->theta;}
    void set_theta(ThetaType& theta) {this->theta = theta;}

  protected:
    ThetaType theta;  // parameters for the policy
};



class KnotControlPolicy : public ControlPolicy<knot_point_deque_t>
{
  public:
    int knot_level = 0;  // 0: stepping function, 1: constant acc, 2: cubic
    int knot_num = 10;   // these are sampled with uniform dt
    int bounded = 0;  // if the control inputs are bounded
    VectorXd lower_bound, upper_bound; // bound on the control
    KnotControlPolicy(const double Ht, const int ctrl_dim, const int knot_level, const int knot_num,
                      int bounded, const VectorXd& lower_bound, const VectorXd& upper_bound) : 
                    ControlPolicy<std::deque<knot_point_t>>(Ht, ctrl_dim)
    {
        // reset the parameters
        this->knot_level = knot_level;  this->knot_num = knot_num;
        theta.resize(knot_num);
        this->dt = Ht / knot_num;
        for (int i=0; i<theta.size(); i++)
        {
            knot_point_t pt_i(VectorXd::Zero(ctrl_dim), this->dt);
            theta[i] = pt_i;
        }
        this->t0 = 0; // start time
        this->bounded = bounded;
        this->lower_bound = lower_bound;
        this->upper_bound = upper_bound;
    };
    void action(const VectorXd& state, const double t, VectorXd& control);
    void action(const VectorXd& state, const knot_point_deque_t& theta, const double t,
                VectorXd& control);
    void sample_gauss_param(const int N, const double sigma, std::vector<knot_point_deque_t>& thetas);
    void optimize(const std::vector<knot_point_deque_t>& thetas, const VectorXd& Js);
    void shift_by_time(const double dt);
    void clamp_control(VectorXd& control);  // clamp the control by lower and upper limit
    double get_t0();
  protected:
    std::default_random_engine generator;
    // parameter: a list of knots as pairs (data, dt)
    double t0 = 0;  // the start time of the first knot
    double dt = 0;
    int seed = 1;
};