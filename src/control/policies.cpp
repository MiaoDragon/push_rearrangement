#include "policies.h"
#include <omp.h>

// ControlPolicy<ThetaType>::sample_gauss_param() depends on the ThetaType since we need to sample
// each parameter according to Gaussian.

// ControlPolicy<ThetaType>::optimize() depends on the ThetaType to add the weights.


/**
 * @brief 
 * use the current policy parameter to generate control given state and t.
 * Our policy does not consider the state, so this is ignored.
 * 
 * @param state 
 * @param t 
 * @param control 
 */
void KnotControlPolicy::action(const VectorXd& state, const double t, VectorXd& control)
{

}

/**
 * @brief 
 * given the knot parameters, obtain the control at time t. (t=0 is current time)
 * knot_level=0: zero-order step function (zero_spline)
 * knot_level=1: first-order zero-acc function (linear_spline)
 * knot_level=2: second-order cubic function (cubic_spline)
 * 
 * @param state 
 * @param t 
 * @param theta 
 * @param control 
 */
void KnotControlPolicy::action(const VectorXd& state, const double t, const knot_point_deque_t& theta,
                VectorXd& control)
{
    // use the specific interpolation/extrapolation

}

void KnotControlPolicy::sample_gauss_param(const int N, const double sigma, std::vector<knot_point_deque_t>& theta)
{}

void KnotControlPolicy::optimize(const std::vector<knot_point_deque_t>& thetas, const VectorXd& Js)
{

}

void KnotControlPolicy::shift_by_time(const double dt)
{

}