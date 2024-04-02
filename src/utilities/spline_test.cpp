#include "utilities.h"
#include "spline.h"

#include <ostream>
#include <string>
#include <vector>
#include <deque>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <algorithm>
#include <fstream>

void test_spline()
{
    knot_point_deque_t data;
    int n_sample_pts = 1000;
    int K = 10;

    double ts[10] = {1, 1.2, 1.5, 2, 2.5, 2.8, 3, 3.5, 3.8, 5};

    double policy_t0 = 1.0;
    double dts[10];
    for (int i=0; i<K-1; i++)
    {
        dts[i] = ts[i+1]-ts[i];
    }
    dts[K-1] = 0;


    std::ofstream file;
    file.open("spline_data.txt");
    for (int i=0; i<K; i++)
    {
        VectorXd a;
        a.resize(3);
        a = VectorXd::Random(3);
        knot_point_t data_i(a, dts[i]);
        data.push_back(data_i);
        file << a.transpose() << std::endl;
    }
    file.close();

    file.open("spline_ts.txt");
    for (int i=0; i<K; i++)
    {
        file << ts[i] << " " << std::endl;
    }
    file.close();

    double sample_t0 = 0, sample_tN = 7;
    for (int order=0; order<3; order++)
    {
        std::ofstream file;
        file.open("spline_sample_order_" + std::to_string(order) + ".txt");
        std::vector<VectorXd> samples;
        for (int i=0; i<n_sample_pts; i++)
        {
            double t = sample_t0 + (sample_tN-sample_t0) / n_sample_pts * i;
            VectorXd sample;
            if (order == 0)
            {
                zero_order_spline(data, policy_t0, t, sample);
            }
            else if (order == 1)
            {
                linear_spline(data, policy_t0, t, sample);
            }
            else
            {
                cubic_spline(data, policy_t0, t, sample);
            }

            samples.push_back(sample);
            file << sample.transpose() << std::endl;
        }
        file.close();
    }

}

int main(void)
{
    test_spline();
}