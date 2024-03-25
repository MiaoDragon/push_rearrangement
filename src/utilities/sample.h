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

#pragma once

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
                     int& seed, MatrixXd& sample);