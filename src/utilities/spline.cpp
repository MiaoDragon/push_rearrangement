/**
 * @file spline.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "spline.h"

/**
 * @brief Get the upper bound idx object
 * implement the std::upper_bound function to find the first idx that t is smallre than ts[i]
 * if the returned index is idx, then the interval is [idx-1,idx]
 * if idx = 0, then the interval is (-inf, 0)
 * if idx = size(knots), then the interval is [knots.size()-1,inf)
 * @param knots 
 * UPDATE: now the second fvalue of knots is the dt, rather than the cummulative time.
 *         this is for the efficiency of shifting time.
 * @param t 
 * @param t_ll: lower bound in the interval for the sum of time
 * @return int 
 */
int get_upper_bound_idx(const knot_point_deque_t &knots, const double t, double& t_ll)
{
    int res = knots.size();
    if (t < t_ll)
    {
        // (-inf, t_0)
        return 0;
    }
    for (int i=0; i<knots.size()-1; i++) // last node has dt, but we treat it as a boundary pt
    {
        if (t < t_ll+knots[i].second)  // compute the cummulative time on the RHS
        {
            res = i+1; break;
        }
        t_ll += knots[i].second;
    }
    return res;  // the interval is [res-1, res]
}

void zero_order_spline(const knot_point_deque_t &knots, const double t0, const double t, VectorXd& res)
{
    double t_ll = t0;
    int idx = get_upper_bound_idx(knots, t, t_ll);
    if (idx == 0)  // interval: (-inf, 0)
    {
        res = knots[0].first;
    }
    else if (idx == knots.size())
    {
        res = knots.back().first;
    }
    else
    {
        res = knots[idx-1].first;
    }
}

void linear_spline(const knot_point_deque_t &knots, const double t0, const double t, VectorXd& res)
{
    double t_ll = t0;
    int idx = get_upper_bound_idx(knots, t, t_ll);

    if (idx == 0)
    {
        res = knots[0].first;
    }
    else if (idx == knots.size())
    {
        res = knots.back().first;
    }
    else
    {
        // interval: [idx-1,idx)
        double t1 = t_ll; //knots[idx-1].second + t_ll;
        double t2 = t_ll + knots[idx-1].second;//knots[idx].second + t_ll;
        res = (t-t1)/(t2-t1) * knots[idx].first + (t2-t)/(t2-t1) * knots[idx-1].first;
    }
}


/**
 * @brief 
 * Here we use the simple cubic spline by finite differentiation at endpoints.
 * the interpolated value then should be clamped by the control boundary.
 * a better way might be to use more advanced splines, such as TOPPRA.
 * @param knots 
 * @param t 
 * @param res 
 */
void cubic_spline(const knot_point_deque_t &knots, const double t0, const double t, VectorXd& res)
{
    double t_ll = t0;
    int idx = get_upper_bound_idx(knots, t, t_ll);

    if (idx == 0)
    {
        res = knots[0].first;
    }
    else if (idx == knots.size())
    {
        res = knots.back().first;
    }
    else
    {
        // interval: [idx-1,idx)
        // ref: https://mathworld.wolfram.com/CubicSpline.html
        double t1 = t_ll; //knots[idx-1].second + t_ll;
        double t2 = t_ll+knots[idx-1].second;;
        VectorXd d1, d2;
        finite_diff(knots, idx-1, d1);
        finite_diff(knots, idx, d2);
        VectorXd a, b, c, d;
        a = knots[idx-1].first;
        b = d1;
        c = 3*(knots[idx].first-knots[idx-1].first) - 2*d1 - d2;
        d = 2*(knots[idx-1].first-knots[idx].first) + d1 + d2;

        double t_normalized = (t-t1)/(t2-t1);
        res = a + b*t_normalized + c*t_normalized*t_normalized + d*t_normalized*t_normalized*t_normalized;
    }
}

void finite_diff(const knot_point_deque_t &knots, const int idx, VectorXd &grad)
{
    // boundary: when idx=0 or idx=knots.size()-1
    if (idx == 0)
    {
        grad = VectorXd::Zero(knots[0].first.size());
    }
    else if (idx == knots.size()-1)
    {
        grad = VectorXd::Zero(knots[0].first.size());
    }
    else
    {
        // different options here. Could be left-side, right-side, or middle
        // here we choose middle
        // grad = (knots[idx+1].first-knots[idx].first)/(knots[idx+1].second-knots[idx].second);
        // grad = (knots[idx+1].first-knots[idx-1].first)/(knots[idx+1].second-knots[idx-1].second);
        grad = (knots[idx+1].first-knots[idx-1].first)/(knots[idx].second+knots[idx-1].second);
    }
}