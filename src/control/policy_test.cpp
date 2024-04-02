#include <ostream>
#include <vector>
#include <deque>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <fstream>

#include <algorithm>

#include "policies.h"

/**
 * @brief 
 * TODO:
 * - [x] test sampling    
 * - [] test action & clamp
 * - [] test optimization
 * - [] test shift_by_time
 * - 
 * 
 */


void test_sample()
{

    VectorXd lower_bound, upper_bound;
    int knot_level = 0;
    int knot_num = 10;
    int N = 100;

    KnotControlPolicy policy(knot_num*0.01, 2, 0, knot_num, 0, lower_bound, upper_bound);

    std::vector<knot_point_deque_t> thetas;

    policy.sample_gauss_param(N, 0.1, thetas);

    std::ofstream file;
    file.open("policy_sample.txt");
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<thetas[i].size(); j++)
        {
            file << thetas[i][j].first.transpose() << std::endl;
        }
    }
    file.close();
}

void test_sample_truncated()
{
    VectorXd lower_bound(2), upper_bound(2);
    lower_bound << 0, 0.2;
    upper_bound << 0.5, 0.4;
    int knot_level = 0;
    int knot_num = 10;
    int N = 100;

    KnotControlPolicy policy(knot_num*0.01, 2, 0, knot_num, 1, lower_bound, upper_bound);

    std::vector<knot_point_deque_t> thetas;

    policy.sample_gauss_param(N, 0.1, thetas);

    std::ofstream file;
    file.open("policy_sample_truncated.txt");
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<thetas[i].size(); j++)
        {
            file << thetas[i][j].first.transpose() << std::endl;
        }
    }
    file.close();
}

void test_action()
{
    VectorXd lower_bound, upper_bound;
    int knot_level = 0;
    int knot_num = 10;
    int N = 100;
    double dt = 0.1;

    KnotControlPolicy policy(knot_num*dt, 2, 0, knot_num, 0, lower_bound, upper_bound);

    knot_point_deque_t theta;
    theta.resize(knot_num);
    for (int i=0; i<knot_num; i++)
    {
        theta[i].first = VectorXd::Random(2);
        theta[i].second = dt;
    }
    policy.set_theta(theta);

    std::ofstream file;

    file.open("policy_ts.txt");
    // print the time for each node
    double t_sum = policy.get_t0();
    for (int i=0; i<knot_num; i++)
    {
        file << t_sum << " " << std::endl;
        t_sum += theta[i].second;
    }
    file.close();

    file.open("policy_theta.txt");
    for (int i=0; i<knot_num; i++)
    {
        file << theta[i].first.transpose() << std::endl;
    }
    file.close();

    // obtain the dense sampled points
    double sample_t0 = -0.5, sample_tN = knot_num*dt + 0.5;
    int sample_num = 1000;
    VectorXd state;
    file.open("policy_sampled_actions.txt");
    for (int i=0; i<sample_num; i++)
    {
        double t = sample_t0 + (sample_tN-sample_t0)/sample_num*i;        
        VectorXd control;
        policy.action(state, t, control);
        file << control.transpose() << std::endl;
    }
    file.close();

}

void test_clamped_action()
{
    VectorXd lower_bound(2), upper_bound(2);
    int knot_level = 0;
    int knot_num = 10;
    int N = 100;
    double dt = 0.1;

    lower_bound << 0.1, 0.02;
    upper_bound << 0.3, 0.2;

    KnotControlPolicy policy(knot_num*dt, 2, 0, knot_num, 1, lower_bound, upper_bound);

    knot_point_deque_t theta;
    theta.resize(knot_num);
    for (int i=0; i<knot_num; i++)
    {
        theta[i].first = VectorXd::Random(2);
        theta[i].second = dt;
    }
    policy.set_theta(theta);

    std::ofstream file;

    file.open("clamp_policy_ts.txt");
    // print the time for each node
    double t_sum = policy.get_t0();
    for (int i=0; i<knot_num; i++)
    {
        file << t_sum << " " << std::endl;
        t_sum += theta[i].second;
    }
    file.close();

    file.open("clamp_policy_theta.txt");
    for (int i=0; i<knot_num; i++)
    {
        file << theta[i].first.transpose() << std::endl;
    }
    file.close();

    // obtain the dense sampled points
    double sample_t0 = -0.5, sample_tN = knot_num*dt + 0.5;
    int sample_num = 1000;
    VectorXd state;
    file.open("clamp_policy_sampled_actions.txt");
    for (int i=0; i<sample_num; i++)
    {
        double t = sample_t0 + (sample_tN-sample_t0)/sample_num*i;        
        VectorXd control;
        policy.action(state, t, control);
        file << control.transpose() << std::endl;
    }
    file.close();

}

void test_shift_time()
{
    VectorXd lower_bound, upper_bound;
    int knot_level = 0;
    int knot_num = 10;
    int N = 100;
    double dt = 0.1;

    KnotControlPolicy policy(knot_num*dt, 2, 0, knot_num, 0, lower_bound, upper_bound);

    knot_point_deque_t theta;
    theta.resize(knot_num);
    for (int i=0; i<knot_num; i++)
    {
        theta[i].first = VectorXd::Random(2);
        theta[i].second = dt;
    }
    policy.set_theta(theta);
    policy.shift_by_time(0.05);
    // policy.shift_by_time(2);

    std::ofstream file;

    policy.get_theta(theta);
    file.open("shift_policy_ts.txt");
    // print the time for each node
    double t_sum = policy.get_t0();
    for (int i=0; i<knot_num; i++)
    {
        file << t_sum << " " << std::endl;
        t_sum += theta[i].second;
    }
    file.close();

    file.open("shift_policy_theta.txt");
    for (int i=0; i<knot_num; i++)
    {
        file << theta[i].first.transpose() << std::endl;
    }
    file.close();

    // obtain the dense sampled points
    double sample_t0 = -0.5, sample_tN = knot_num*dt + 0.5;
    int sample_num = 1000;
    VectorXd state;
    file.open("shift_policy_sampled_actions.txt");
    for (int i=0; i<sample_num; i++)
    {
        double t = sample_t0 + (sample_tN-sample_t0)/sample_num*i;        
        VectorXd control;
        policy.action(state, t, control);
        file << control.transpose() << std::endl;
    }
    file.close();

}

void test_optimize()
{
    VectorXd lower_bound, upper_bound;
    int knot_level = 0;
    int knot_num = 5;
    int N = 10;
    double dt = 0.1;

    KnotControlPolicy policy(knot_num*dt, 2, 0, knot_num, 0, lower_bound, upper_bound);
    knot_point_deque_t theta;
    theta.resize(knot_num);
    for (int i=0; i<knot_num; i++)
    {
        theta[i].first = VectorXd::Random(2);
        theta[i].second = dt;
    }
    policy.set_theta(theta);

    VectorXd Js(10);
    // Js << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
    Js = VectorXd::Random(10);

    std::vector<knot_point_deque_t> thetas;
    policy.sample_gauss_param(N, 0.1, thetas);

    std::ofstream file;
    file.open("optimize_thetas.txt");

    for (int i=0; i<thetas.size(); i++)
    {
        for (int j=0; j<knot_num; j++)
        {
            file << thetas[i][j].first.transpose() << std::endl;
        }
    }
    file.close();

    file.open("optimize_Js.txt");
    for (int i=0; i<thetas.size(); i++)
    {
        file << Js[i] << " ";
    }
    file << std::endl;
    file.close();

    policy.optimize(thetas, Js);

}


int main(void)
{
    test_sample();
    test_sample_truncated();
    test_action();
    test_clamped_action();
    test_shift_time();
    test_optimize();
}