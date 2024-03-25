#include "sample.h"

#include <ostream>
#include <vector>
#include <deque>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <algorithm>

#include "mujoco_mppi_intvel.h"

void test_sample_1d()
{

    std::deque<VectorXd> mu;
    std::deque<VectorXd> sigma;

    VectorXd mu0, sigma0;
    mu0.resize(1);  sigma0.resize(1);

    mu0 << 0.5;
    sigma0 << 1;


    VectorXd ll, ul;
    ll.resize(1); ul.resize(1);
    ll << 0;
    ul << 3.5;

    int seed = 10;
    int N = 100000;

    mu.push_back(mu0);
    sigma.push_back(sigma0);

    for (int i=0; i<N; i++)
    {
        MatrixXd sample;  // 1 x 1
        truncated_gauss(mu, sigma, ll, ul, seed, sample);

        std::cout << sample(0,0) << " ";
    }
    std::cout << std::endl;

}



int main(void)
{
    test_sample_1d();
}