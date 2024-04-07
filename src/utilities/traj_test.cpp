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
#include <fstream>

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

    ScrewPoseTrajectory traj(T1, T2);
    Matrix4d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated pose: " << std::endl;
    std::cout << res << std::endl;

    std::cout << "twist: " << std::endl;
    Vector6d twist;
    traj.twist(0.3, twist);
    std::cout << twist.transpose() << std::endl;
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

    ScrewPoseTrajectory traj(T1, T2);
    Matrix4d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated pose: " << std::endl;
    std::cout << res << std::endl;
    Vector6d twist;
    traj.twist(0.3, twist);
    std::cout << "twist: " << std::endl;
    std::cout << twist.transpose() << std::endl;
}


void test_position_traj() {
    std::cout << "#########test_position_traj:##########" << std::endl;
    Vector3d pos1 = Vector3d::Random();
    std::cout << "pos1: " << std::endl;
    std::cout << pos1 << std::endl;

    Vector3d pos2 = Vector3d::Random();
    std::cout << "pos2: " << std::endl;
    std::cout << pos2 << std::endl;


    LinearPositionTrajectory traj(pos1, pos2);
    Vector3d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated position: " << std::endl;
    std::cout << res << std::endl;
    std::cout << "velocity: " << std::endl;
    Vector3d vel;
    traj.velocity(0.3, vel);
    std::cout << vel << std::endl;

}

void test_sinusoid_traj()
{
    Vector3d pos1 = Vector3d::Random();
    Vector3d pos2 = Vector3d::Random();
    Vector3d unit_vec = pos2 - pos1;
    unit_vec = unit_vec / unit_vec.norm();
    double period = 2*M_PI;
    double y_scale = 0.1;
    double t_scale = 0.2;
    SimpleSinusoidTrajectory traj(pos1, unit_vec, period, t_scale, y_scale);
    std::ofstream file;
    file.open("sinusoid_traj_test_data_start.txt");
    file << pos1.transpose() << std::endl;
    file.close();
    Vector3d x_vec, y_vec, z_vec;
    file.open("sinusoid_traj_test_data_x.txt");
    traj.get_x_vec(x_vec);
    file << x_vec.transpose() << std::endl;
    file.close();
    file.open("sinusoid_traj_test_data_y.txt");
    traj.get_y_vec(y_vec);
    file << y_vec.transpose() << std::endl;
    file.close();
    file.open("sinusoid_traj_test_data_z.txt");
    traj.get_z_vec(z_vec);
    file << z_vec.transpose() << std::endl;
    file.close();

    file.open("sinusoid_traj_test_data.txt");
    int sample_n = 100;
    double duration = period * 5;
    double dt = duration / sample_n;
    for (int i=0; i<sample_n; i++)
    {
        double t = dt*i;
        Vector3d res;
        traj.interpolate(t, res);
        file << res.transpose() << std::endl;
    }
    file.close();
}


void test_screw_position_traj() {
    std::cout << "#########test_screw_position_traj:##########" << std::endl;
    Vector3d pos1 = Vector3d::Random();
    std::cout << "pos1: " << std::endl;
    std::cout << pos1 << std::endl;

    Vector3d pos2 = Vector3d::Random();
    std::cout << "pos2: " << std::endl;
    std::cout << pos2 << std::endl;

    Vector3d delta_pos = pos2-pos1;
    // double distance = delta_pos.norm();

    Vector6d screw = VectorXd::Zero(6);  // rotation is zero
    screw.head<3>() = delta_pos;

    ScrewPositionTrajectory traj(pos1, screw);
    Vector3d res;
    traj.interpolate(0.3, res);
    std::cout << "interpolated position: " << std::endl;
    std::cout << res << std::endl;
    std::cout << "velocity: " << std::endl;
    Vector3d vel;
    traj.velocity(0.3, vel);
    std::cout << vel << std::endl;

    std::cout << "testing rotation..." << std::endl;
    screw[5] = 1.0;  // rotating relative to z axis while translating

    ScrewPositionTrajectory traj2(pos1, screw);
    traj2.interpolate(0.3, res);
    std::cout << "interpolated position: " << std::endl;
    std::cout << res << std::endl;
    std::cout << "velocity: " << std::endl;
    traj2.velocity(0.3, vel);
    std::cout << vel << std::endl;

    std::cout << "transformation matrix at time " << 0.3 << std::endl;
    double t = 0.3;
    
    Vector6d unit_screw;
    double theta;
    Matrix4d T;
    twist_to_unit_twist(screw*t, unit_screw, theta);
    twist_to_SE3(unit_screw, theta, T);
    // transformed points
    std::cout << T << std::endl;
    std::cout << "transformed point: " << std::endl;
    Vector3d pos = T.block<3,3>(0,0)*pos1 + T.block<3,1>(0,3);
    std::cout << pos << std::endl;
    std::cout << "twist at time t by w x r + v: " << std::endl;
    std::cout << screw.tail<3>().cross(pos) + screw.head<3>(0) << std::endl;
}


int main(void)
{
    test_pose_traj_ori();
    test_pose_traj_position();
    test_position_traj();
    test_sinusoid_traj();
    test_screw_position_traj();
}