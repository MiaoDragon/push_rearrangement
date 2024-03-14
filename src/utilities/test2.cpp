#include "utilities.h"

#include <cmath>
#include <ostream>
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <algorithm>

void test_to_SE3_funcs()
{
    Vector3d unit_w;
    double theta;
    unit_w << 0,0,1;
    theta = 0;

    Matrix3d R;
    w_to_SO3(unit_w, theta, R);

    std::cout << "test1 SO3: " << std::endl;
    std::cout << R << std::endl;

    // test SO3 to w
    Matrix3d unit_so3;
    SO3_to_so3(R, unit_so3, theta);
    std::cout << "test 1 unit so3: " << std::endl;
    std::cout << unit_so3 << std::endl;
    std::cout << "theta: " << theta << std::endl;
    SO3_to_w(R, unit_w, theta);
    std::cout << "test 1 unit w: " << std::endl;
    std::cout << unit_w << std::endl;
    std::cout << "theta: " << theta << std::endl;

    so3_to_SO3(unit_so3, theta, R);
    std::cout << "test 1 SO3: " << std::endl;
    std::cout << R << std::endl;


    /////////////////////////////////////////////////////
    unit_w[0] = 0; unit_w[1] = 0; unit_w[2] = 1;
    theta = 30.0/180*M_PI;

    w_to_SO3(unit_w, theta, R);

    std::cout << "test2 SO3: " << std::endl;
    std::cout << R << std::endl;

    // test SO3 to w
    SO3_to_so3(R, unit_so3, theta);
    std::cout << "test 2 unit so3: " << std::endl;
    std::cout << unit_so3 << std::endl;
    std::cout << "theta: " << theta << std::endl;
    SO3_to_w(R, unit_w, theta);
    std::cout << "test 2 unit w: " << std::endl;
    std::cout << unit_w << std::endl;
    std::cout << "theta: " << theta << std::endl;

    so3_to_SO3(unit_so3, theta, R);
    std::cout << "test 2 SO3: " << std::endl;
    std::cout << R << std::endl;    

    ////////////////////////////////////////////////////////
    Matrix4d T;
    Vector6d unit_twist;  // [v,w]
    unit_twist.setZero();
    unit_twist.tail(3) = unit_w;
    twist_to_SE3(unit_twist, theta, T);
    std::cout << "test 3 SE3: " << std::endl;
    std::cout << T << std::endl;

    Vector3d v;
    v << 1,0,0;
    unit_twist.head(3) = v;
    twist_to_SE3(unit_twist, theta, T);
    std::cout << "test 3 SE3: " << std::endl;
    std::cout << T << std::endl;

    Matrix4d unit_se3;
    SE3_to_se3(T, unit_se3, theta);
    std::cout << "test 3 unit_se3: " << std::endl;
    std::cout << unit_se3 << std::endl;
    std::cout << "theta: " << theta << std::endl;

    SE3_to_twist(T, unit_twist, theta);
    std::cout << "test 3 unit_twist: " << std::endl;
    std::cout << unit_twist << std::endl;
    std::cout << "theta: " << theta << std::endl;

}

void test_twist_apply()
// compare: apply twist*theta to pose. Versus iteratively apply twist to pose
{
    Vector6d unit_twist = VectorXd::Random(6); // [v,w]
    unit_twist = unit_twist / unit_twist.tail(3).norm();
    double theta = 30.0 / 180 * M_PI;

    // option 1: directly apply theta
    Vector3d axis = Vector3d::Random();
    axis = axis / axis.norm();
    double angle = 45.0 / 180 * M_PI;
    AngleAxisd axis_ang(angle, axis);
    Matrix3d R = axis_ang.toRotationMatrix();
    Vector3d p = Vector3d::Random();
    Matrix4d T;
    T.setZero();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = p;
    T(3,3) = 1;

    std::cout << "T: " << std::endl;
    std::cout << T << std::endl;

    Matrix4d T1;
    twist_to_SE3(unit_twist, theta, T1);
    Matrix4d T_after1 = T1 * T;
    std::cout << "option 1, T_after1: " << std::endl;
    std::cout << T_after1 << std::endl;

    // option 2: iteratively apply theta
    int n = 10;
    double d_theta = theta / n;
    Matrix4d T_after2 = T;
    Matrix4d T2;
    twist_to_SE3(unit_twist, d_theta, T2);
    for (int i=0; i<n; i++)
    {
        T_after2 = T2 * T_after2;
    }

    std::cout << "option 2, T_after2: " << std::endl;
    std::cout << T_after2 << std::endl;
}


int main(void)
{
    test_to_SE3_funcs();

    test_twist_apply();

}