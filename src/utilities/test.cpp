#include "utilities.h"

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

void test_adjoint()
{
    Matrix4d g;
    g << 1, 0, 0, 0,
         0, 1, 0, 2,
         0, 0, 1, 0,
         0, 0, 0, 0;
    Matrix6d res;
    adjoint(g, res);
}

int main(void)
{
    test_adjoint();

}