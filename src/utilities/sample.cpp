/**
 * @file sample.h
 * @author your name (you@domain.com)
 * @brief 
 * provide functions for various sampling methods.
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <queue>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "utilities.h"
#include "truncated_normal.hpp"
#include "sample.h"

// // generate standard Gaussian
// void standard_gauss(const VectorXd& ll, const VectorXd& ul,  // this is for the last axis
//                      const int& seed, MatrixXd& sample)
// {
//     sample.resize(mu.size(),ll.size());
//     for (int i=0; i<mu.size(); i++)
//     {
//         for (int j=0; j<ll.size(); j++)
//         {
//             samples(i,j) = truncated_normal_ab_sample(mu[i][j], mu[i][j], ll(j), ul(j), seed);
//         }
//     }
// }

// generate one sample of size H x nu
void truncated_gauss(const std::deque<VectorXd>& mu, const std::deque<VectorXd>& sigma, 
                     const VectorXd& ll, const VectorXd& ul,  // this is for the last axis
                     int& seed, MatrixXd& sample)
{
    sample.resize(mu.size(),ll.size());
    for (int i=0; i<mu.size(); i++)
    {
        for (int j=0; j<ll.size(); j++)
        {
            sample(i,j) = truncated_normal_ab_sample(mu[i][j], sigma[i][j], ll(j), ul(j), seed);
        }
    }
}

void truncated_gauss(const std::deque<VectorXd>& mu, const std::deque<VectorXd>& sigma, 
                     const VectorXd& ll, const VectorXd& ul,  // this is for the last axis
                     int& seed, std::deque<VectorXd>& sample)
{
    sample.resize(mu.size());
    for (int i=0; i<mu.size(); i++)
    {
        sample[i].resize(ll.size());
        for (int j=0; j<ll.size(); j++)
        {
            sample[i][j] = truncated_normal_ab_sample(mu[i][j], sigma[i][j], ll(j), ul(j), seed);
        }
    }
}


void truncated_gauss(const std::deque<VectorXd>& mu, const double& sigma, 
                     const VectorXd& ll, const VectorXd& ul,  // this is for the last axis
                     int& seed, std::deque<VectorXd>& sample)
{
    sample.resize(mu.size());
    for (int i=0; i<mu.size(); i++)
    {
        sample[i].resize(ll.size());
        for (int j=0; j<ll.size(); j++)
        {
            sample[i][j] = truncated_normal_ab_sample(mu[i][j], sigma, ll(j), ul(j), seed);
        }
    }
}



void truncated_gauss(const knot_point_deque_t& mu, const double& sigma, 
                     const VectorXd& ll, const VectorXd& ul,  // this is for the last axis
                     int& seed, knot_point_deque_t& sample)
{
    sample.resize(mu.size());
    for (int i=0; i<mu.size(); i++)
    {
        sample[i].first.resize(ll.size());
        for (int j=0; j<ll.size(); j++)
        {
            sample[i].first[j] = truncated_normal_ab_sample(mu[i].first[j], sigma, ll(j), ul(j), seed);
        }
    }
}

void gauss(const std::deque<VectorXd> &mu, const double &sigma, std::default_random_engine& generator, 
           std::deque<VectorXd> &sample)
{
    std::normal_distribution<double> normal_distribution(0, sigma);
    sample.resize(mu.size());
    for (int i=0; i<sample.size(); i++)
    {
        sample[i].resize(mu[i].size());
        for (int j=0; j<mu[i].size(); j++)
        {
            sample[i][j] = normal_distribution(generator) + mu[i][j];
        }
    }
}


void gauss(const knot_point_deque_t &mu, const double &sigma, std::default_random_engine& generator,
           knot_point_deque_t &sample)
{
    std::normal_distribution<double> normal_distribution(0, sigma);
    sample.resize(mu.size());
    for (int i=0; i<sample.size(); i++)
    {
        sample[i].first.resize(mu[i].first.size());
        for (int j=0; j<mu[i].first.size(); j++)
        {
            sample[i].first[j] = normal_distribution(generator) + mu[i].first[j];
        }
    }
}