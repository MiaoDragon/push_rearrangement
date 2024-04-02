#include "policies.h"
#include <cmath>
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
    if (knot_level == 0)
    {
        zero_order_spline(theta, t0, t, control);
    }
    else if (knot_level == 1)
    {
        linear_spline(theta, t0, t, control);
    }
    else
    {
        cubic_spline(theta, t0, t, control);
    }
    if (bounded)  clamp_control(control);

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
void KnotControlPolicy::action(const VectorXd& state, const knot_point_deque_t& theta, const double t,
                VectorXd& control)
{
    // use the specific interpolation/extrapolation
    if (knot_level == 0)
    {
        zero_order_spline(theta, t0, t, control);
    }
    else if (knot_level == 1)
    {
        linear_spline(theta, t0, t, control);
    }
    else
    {
        cubic_spline(theta, t0, t, control);
    }
    if (bounded)  clamp_control(control);
}

void KnotControlPolicy::clamp_control(VectorXd &control)
{
    control = control.cwiseMin(upper_bound).cwiseMax(lower_bound);
}

void KnotControlPolicy::sample_gauss_param(const int N, const double sigma, std::vector<knot_point_deque_t>& thetas)
{
    // if bounded, sample according to truncated gauss
    thetas.resize(N);
    generator.seed(seed);

    // keep the theta as the first element
    thetas[0] = this->theta;

    if (bounded)
    {
        for (int i=1; i<N; i++)
        {
            knot_point_deque_t theta_i;
            // int seed = 0;
            truncated_gauss(theta, sigma, lower_bound, upper_bound, seed, theta_i);
            thetas[i] = theta_i;
        }
    }
    else // otherwise, sample according to gauss
    {
        for (int i=1; i<N; i++)
        {
            knot_point_deque_t theta_i;
            // int seed = 0;
            gauss(theta, sigma, generator, theta_i);
            thetas[i] = theta_i;
        }
    }
}

/**
 * @brief 
 * update the weight according to the cost J. Then update theta as the weighted average of thetas.
 * @param thetas 
 * @param Js 
 */
void KnotControlPolicy::optimize(const std::vector<knot_point_deque_t>& thetas, const VectorXd& Js)
{
    double rho = Js.minCoeff();
    double beta = 0.1; // afterewards, need to update this over time

    knot_point_deque_t theta; // updated parameter
    for (int i=0; i<this->theta.size(); i++) // on the level of horizons
    {
        double w_sum = 0;
        VectorXd theta_i_sum(this->theta[i].first.size());
        theta_i_sum = VectorXd::Zero(this->theta[i].first.size());

        for (int j=0; j<thetas.size(); j++) // on the level of samples
        {
            double wj = exp(-1/beta*(Js[j]-rho));
            /* Thoughts: since parameters are kind of independent of each other, should we weight by cost up to that time? */
            w_sum += wj;
            theta_i_sum += wj * thetas[j][i].first;
        }
        theta_i_sum = theta_i_sum / w_sum;
        // update the parameter
        this->theta[i].first = theta_i_sum;
    }
}

void KnotControlPolicy::shift_by_time(const double dt)
{
    // shift t0
    t0 = t0 - dt;
    // check if can remove knots in the head
    while (true)
    {
        if (t0 + theta[0].second < 0)
        {
            t0 += theta[0].second;
            theta.pop_front();
            knot_point_t knot(VectorXd::Zero(this->ctrl_dim),this->dt);
            theta.push_back(knot);
        }
        else
        {
            break;
        }
    }
}

double KnotControlPolicy::get_t0()
{
    return t0;
}
