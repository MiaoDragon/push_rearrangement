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

PoseTrajectory::PoseTrajectory(const Matrix4d& start, const Matrix4d& goal)
{
    Matrix4d dT;
    this->start = start;
    pose_to_rel_transform(start, goal, dT);
    SE3_to_twist(dT, unit_twist, theta);
}

PoseTrajectory::PoseTrajectory(const Matrix4d& start, const Vector6d& unit_twist, const double& theta)
{
    this->start = start; this->unit_twist = unit_twist; this->theta = theta;
}

void PoseTrajectory::interpolate(double t, Matrix4d &res) const
{
    // since we can still extrapolate outside of [0,1], we just do it here
    Matrix4d dT;
    twist_to_SE3(unit_twist, t*theta, dT);
    res = dT*start;
}

PositionTrajectory::PositionTrajectory(const Vector3d& start, const Vector3d& goal)
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
PositionTrajectory::PositionTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double& distance)
{
    this->start = start;  this->unit_vec = unit_vec;  this->distance = distance;
}
void PositionTrajectory::interpolate(double t, Vector3d &res) const
{
    res = start + t*unit_vec;
}