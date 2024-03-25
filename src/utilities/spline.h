/**
 * @file spline.h
 * @author your name (you@domain.com)
 * @brief 
 * provide functions for interpolation using spline.
 * ref: https://github.com/google-deepmind/mujoco_mpc/blob/main/mjpc/utilities.cc
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <deque>
#include "utilities.h"

typedef std::pair<VectorXd,double> knot_point_t;  // type of each knot: position and time
typedef std::deque<std::pair<VectorXd,double>> knot_point_deque_t;  // type of each knot: position and time


int get_upper_bound_idx(const knot_point_deque_t& knots, const double t);

void zero_order_spline(const knot_point_deque_t& knots, const double t, VectorXd& res);

void linear_spline(const knot_point_deque_t& knots, const double t, VectorXd& res);

void cubic_spline(const knot_point_deque_t& knots, const double t, VectorXd& res);

void finite_diff(const knot_point_deque_t& knots, const int idx, VectorXd& grad);  // evaluate the gradient at idx