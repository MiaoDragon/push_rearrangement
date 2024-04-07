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
    virtual void interpolate(const double t, T& res) const = 0; // need to make sure t is within [0,1]
};

class PoseTrajectory : public Trajectory<Matrix4d>
{
  public:
    virtual void twist(const double t, Vector6d& res) const = 0;
};


class ScrewPoseTrajectory : public PoseTrajectory
{
  public:
    ScrewPoseTrajectory(const Matrix4d& start, const Matrix4d& goal);
    ScrewPoseTrajectory(const Matrix4d& start, const Vector6d& unit_screw, const double& theta);
    void interpolate(const double t, Matrix4d& res) const;
    void twist(const double t, Vector6d& res) const;
  protected:
    Matrix4d start;
    Vector6d unit_screw;
    double theta=0.0;
};


class PositionTrajectory : public Trajectory<Vector3d>
{
  public:
    virtual void velocity(const double t, Vector3d& res) const = 0;
};


class LinearPositionTrajectory : public PositionTrajectory
{
  public:
    LinearPositionTrajectory(const Vector3d& start, const Vector3d& goal);
    LinearPositionTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double& distance);
    void interpolate(const double t, Vector3d& res) const;
    void velocity(const double t, Vector3d& res) const;
  protected:
    Vector3d start;
    Vector3d unit_vec;
    double distance=0.0;
};

/**
 * @brief this represents the position trajectory undergoing screw motion.
 * here we assume the screw is in the world frame, and the interpolated object is also in the world frame.
 * t = [0,1], where t=0 at start position, and t=1 at T(screw)start
 */
class ScrewPositionTrajectory : public PositionTrajectory
{
  public:
    ScrewPositionTrajectory(const Vector3d& start, const Vector6d& screw);
    ScrewPositionTrajectory(const Vector3d& start, const Vector6d& unit_screw, const double& theta);
    void interpolate(const double t, Vector3d& res) const;
    void velocity(const double t, Vector3d& res) const;
  protected:
    Vector3d start;
    Vector6d unit_screw;
    double theta=0;
};

class SimpleSinusoidTrajectory : public PositionTrajectory
{
  public:
    SimpleSinusoidTrajectory(const Vector3d& start, const Vector3d& unit_vec, const double period, const double t_scale, const double y_scale);  // this is for general t
    void interpolate(const double t, Vector3d& res) const;
    void velocity(const double t, Vector3d& res) const;
    void get_x_vec(Vector3d& x_vec) const;
    void get_y_vec(Vector3d& y_vec) const;
    void get_z_vec(Vector3d& z_vec) const;

  protected:
    Vector3d start;
    Vector3d unit_x_vec;
    Vector3d unit_y_vec;
    Vector3d unit_z_vec;
    double period;
    double t_scale = 1.0;
    double y_scale = 1.0;
};

class VectorTrajectory : public Trajectory<VectorXd>
{};

class LinearVectorTrajectory : public VectorTrajectory
{
  public:
    LinearVectorTrajectory(const VectorXd& start, const VectorXd& goal);
    LinearVectorTrajectory(const VectorXd& start, const VectorXd& unit_vec, const double& distance);
    void interpolate(const double t, VectorXd& res) const;
  protected:
    VectorXd start;
    VectorXd unit_vec;
    double distance=0.0;
};