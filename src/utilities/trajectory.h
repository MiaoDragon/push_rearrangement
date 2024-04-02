/**
 * @file trajectory.h
 * @author your name (you@domain.com)
 * @brief 
 * define class of trajectories to allow interpolation.
 * @version 0.1
 * @date 2024-04-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief Abstract class defining general trajectory along time s \in [0,1]
 * 
 */

#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <random>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "utilities.h"

template<typename T>
class Trajectory
{
  public:
    Trajectory() = default;
    virtual ~Trajectory() = default;  // so that we can pass this on to class that inherits this
    virtual void interpolate(double t, T& res) const = 0; // need to make sure t is within [0,1]
};

class PoseTrajectory : public Trajectory<Matrix4d>
{
  public:
    PoseTrajectory(const Matrix4d& start, const Matrix4d& goal);
    PoseTrajectory(const Matrix4d& start, const Vector6d& unit_twist, const double& theta);
    void interpolate(double t, Matrix4d& res) const;
  protected:
    Matrix4d start;
    Vector6d unit_twist;
    double theta=0.0;
};

class PositionTrajectory : public Trajectory<Vector3d>
{
  public:
    PositionTrajectory(const Vector3d& start, const Vector3d& goal);
    PositionTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double& distance);
    void interpolate(double t, Vector3d& res) const;
  protected:
    Vector3d start;
    Vector3d unit_vec;
    double distance=0.0;
};