/**
 * @file traj_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * testing the implementation of trajectory
 * @version 0.1
 * @date 2024-04-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "trajectory.h"
#include "utilities.h"
#include <cmath>


void test_pose_traj_ori() {
    Vector3d start_unit_w(0,0,1);
    double start_angle = 30.0/180*M_PI;
    Matrix3d R1;
    w_to_SO3(start_unit_w, start_angle, R1);
    Matrix4d T1;
    T1.setZero();
    T1.block<3,3>(0,0) = R1;
    T1(3,3) = 1;


    Vector3d goal_unit_w(0,0,1);
    double goal_angle = 180.0/180*M_PI;
    Matrix3d R2;
    w_to_SO3(goal_unit_w, goal_angle, R2);
    Matrix4d T2;
    T2.setZero();
    T2.block<3,3>(0,0) = R2;
    T2(3,3) = 1;

    PoseTrajectory traj(T1, T2);
    Matrix4d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated pose: " << std::endl;
    std::cout << res << std::endl;
}



void test_pose_traj_position() {
    Matrix4d T1;
    T1.setZero();
    T1.block<3,3>(0,0) = Matrix3d::Identity();
    T1.block<3,1>(0,3) = Vector3d::Random();
    T1(3,3) = 1;
    std::cout << "T1: " << std::endl;
    std::cout << T1 << std::endl;

    Matrix4d T2;
    T2.setZero();
    T2.block<3,3>(0,0) = Matrix3d::Identity();
    T2.block<3,1>(0,3) = Vector3d::Random();
    T2(3,3) = 1;
    std::cout << "T2: " << std::endl;
    std::cout << T2 << std::endl;

    PoseTrajectory traj(T1, T2);
    Matrix4d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated pose: " << std::endl;
    std::cout << res << std::endl;
}


void test_position_traj() {
    Vector3d pos1 = Vector3d::Random();
    std::cout << "pos1: " << std::endl;
    std::cout << pos1 << std::endl;

    Vector3d pos2 = Vector3d::Random();
    std::cout << "pos2: " << std::endl;
    std::cout << pos2 << std::endl;


    PositionTrajectory traj(pos1, pos2);
    Vector3d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated position: " << std::endl;
    std::cout << res << std::endl;
}



int main(void)
{
    test_pose_traj_ori();
    test_pose_traj_position();
    test_position_traj();
}