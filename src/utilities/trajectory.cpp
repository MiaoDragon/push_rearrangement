/**
 * @file trajectory.cpp
 * @author your name (you@domain.com)
 * @brief 
 * implementation of trajectories
 * @version 0.1
 * @date 2024-04-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "trajectory.h"
#include "utilities.h"
#include <cmath>
#include <math.h>

ScrewPoseTrajectory::ScrewPoseTrajectory(const Matrix4d& start, const Matrix4d& goal)
{
    Matrix4d dT;
    this->start = start;
    pose_to_rel_transform(start, goal, dT);
    SE3_to_twist(dT, unit_screw, theta);
}

ScrewPoseTrajectory::ScrewPoseTrajectory(const Matrix4d& start, const Vector6d& unit_screw, const double& theta)
{
    this->start = start; this->unit_screw = unit_screw; this->theta = theta;
}

void ScrewPoseTrajectory::interpolate(const double t, Matrix4d &res) const
{
    // since we can still extrapolate outside of [0,1], we just do it here
    Matrix4d dT;
    twist_to_SE3(unit_screw, t*theta, dT);
    res = dT*start;
}

void ScrewPoseTrajectory::twist(const double t, Vector6d &res) const
{
    res = unit_screw * theta;
}


LinearPositionTrajectory::LinearPositionTrajectory(const Vector3d& start, const Vector3d& goal)
{
    Vector3d dx = goal - start;
    this->start = start;
    distance = dx.norm();
    if (distance == 0.0)
    {
        unit_vec = Vector3d::Zero();
    }
    else
    {
        unit_vec = dx / distance;
    }
}

LinearPositionTrajectory::LinearPositionTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double& distance)
{
    this->start = start;  this->unit_vec = unit_vec;  this->distance = distance;
}

void LinearPositionTrajectory::interpolate(double t, Vector3d &res) const
{
    res = start + t*distance*unit_vec;
}

void LinearPositionTrajectory::velocity(const double t, Vector3d &res) const
{
    res = unit_vec*distance;
}


ScrewPositionTrajectory::ScrewPositionTrajectory(const Vector3d& start, const Vector6d& screw)
{
    twist_to_unit_twist(screw, unit_screw, theta);
    this->start = start;
}

ScrewPositionTrajectory::ScrewPositionTrajectory(const Vector3d& start, const Vector6d& unit_screw, const double& theta)
{
    this->start = start; this->unit_screw = unit_screw; this->theta = theta;
}

void ScrewPositionTrajectory::interpolate(double t, Vector3d &res) const
{
    Matrix4d T;
    twist_to_SE3(unit_screw, t*theta, T);
    res = T.block<3,3>(0,0)*start + T.block<3,1>(0,3);
}

void ScrewPositionTrajectory::velocity(const double t, Vector3d &res) const
{
    // velocity: w x r + v
    // obtain position at time t
    Vector3d pos;
    interpolate(t, pos);
    Vector6d screw = unit_screw * theta;
    res = screw.tail<3>().cross(pos) + screw.head<3>();
}



SimpleSinusoidTrajectory::SimpleSinusoidTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double period, const double t_scale, const double y_scale)
{
    this->start = start; this->unit_x_vec = unit_vec;
    unit_z_vec = Vector3d::Zero();
    unit_z_vec[2] = 1;
    unit_z_vec = unit_z_vec.cross(unit_vec);
    unit_z_vec = unit_z_vec / unit_z_vec.norm();
    unit_y_vec = unit_z_vec.cross(unit_x_vec);
    this->period = period;
    this->t_scale = t_scale;
    this->y_scale = y_scale;
}

void SimpleSinusoidTrajectory::interpolate(double t, Vector3d &res) const
{
    // y = sin(2pi/period * x)
    double scaled_t = t*t_scale;
    double y = y_scale*std::sin(2*M_PI/period * scaled_t);
    res = start + unit_x_vec * scaled_t + unit_y_vec * y;
}

void SimpleSinusoidTrajectory::velocity(const double t, Vector3d &res) const
{
    res = unit_x_vec * t_scale + unit_y_vec * y_scale*std::cos(2*M_PI/period*t*t_scale)*2*M_PI/period*t_scale;
}

void SimpleSinusoidTrajectory::get_x_vec(Vector3d &x_vec) const
{
    x_vec = unit_x_vec;
}

void SimpleSinusoidTrajectory::get_y_vec(Vector3d &y_vec) const
{
    y_vec = unit_y_vec;
}

void SimpleSinusoidTrajectory::get_z_vec(Vector3d &z_vec) const
{
    z_vec = unit_z_vec;
}


LinearVectorTrajectory::LinearVectorTrajectory(const VectorXd& start, const VectorXd& goal)
{
    VectorXd dx = goal - start;
    this->start = start;
    distance = dx.norm();
    if (distance == 0.0)
    {
        unit_vec = VectorXd::Zero(start.size());
    }
    else
    {
        unit_vec = dx / distance;
    }
}

LinearVectorTrajectory::LinearVectorTrajectory(const VectorXd& start, const VectorXd& unit_vec, const double& distance)
{
    this->start = start;  this->unit_vec = unit_vec;  this->distance = distance;
}

void LinearVectorTrajectory::interpolate(double t, VectorXd &res) const
{
    res = start + t*distance*unit_vec;
}