/**
 * @file disturb_policy.h
 * @author your name (you@domain.com)
 * @brief 
 * implement the disturbance-based policy.
 * given the nominal control trajectory, sample disturbance and add to it as control.
 * @version 0.1
 * @date 2024-04-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#pragma once

#include <memory>
#include <vector>

#include <absl/random/distributions.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/planners/policy.h"

#include "mjpc/utilities.h"
#include "mjpc/trajectory.h"

namespace mjpc {
/**
 * @brief 
 * TODO: add general interpolation method that is a function.
 * Theroetically correct way is to use screw interpolation sometimes.
 * 
 */
class NominalControlTrajectory
{
  public:
    NominalControlTrajectory(const int nu, const int nt,
                             const PolicyRepresentation& representation,
                             const std::vector<double>& parameters,
                             const std::vector<double>& times)
    {
        this->nu = nu; this->nt = nt;
        this->representation = representation;
        this->parameters = parameters;
        this->times = times;
    }

    /**
     * @brief 
     * obtain the control at time t, and put to action
     * @param t 
     */
    void set_start_time(const double& start_t)
    {
        // update each time so that start time is start_t
        double diff = start_t - times[0];
        for (int i=0; i<nt; i++)
        {
            times[i] = times[i] + diff;
        }
    }

    void at(const double& t, double* action)
    {
        int bounds[2];
        FindInterval(bounds, times, t, nt);

        if (bounds[0] == bounds[1] ||
            representation == PolicyRepresentation::kZeroSpline)
        {
            ZeroInterpolation(action, t, times, parameters.data(), nu, nt);
        }
        else if (representation == PolicyRepresentation::kLinearSpline)
        {
            LinearInterpolation(action, t, times, parameters.data(), nu, nt);
        }
        else if (representation == PolicyRepresentation::kCubicSpline)
        {
            CubicInterpolation(action, t, times, parameters.data(), nu, nt);
        }
    }

    int nu=0, nt=0;
    PolicyRepresentation representation;
    std::vector<double> parameters;
    std::vector<double> times;
};


// policy for sampling planner
class SamplingDisturbPolicy : public Policy {
 public:
  // constructor
  SamplingDisturbPolicy() = default;

  // destructor
  ~SamplingDisturbPolicy() override = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(const mjModel* model, const Task& task, int horizon) override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set action from policy
  void Action(double* action, const double* state, double time) const override;

  void Parameter(double* parameter, const double* state, double time) const;


  // copy policy
  void CopyFrom(const SamplingDisturbPolicy& policy, int horizon);

  // copy parameters
  void CopyParametersFrom(const std::vector<double>& src_parameters,
                          const std::vector<double>& src_times);

  // ----- disturbance-related methods ----- //
  void SetNominalControlTrajectory(const std::shared_ptr<NominalControlTrajectory> nominal_control_traj);

  // ----- members ----- //
  const mjModel* model;
  std::vector<double> parameters;
  std::vector<double> times;
  int num_parameters;
  int num_spline_points;
  PolicyRepresentation representation;
  // disturbance
  std::shared_ptr<NominalControlTrajectory> nominal_control_traj = nullptr;

//   std::shared_ptr<VectorTrajectory> nominal_control_traj = nullptr;
//   double nominal_start_time = 0;
//   double nominal_duration = 0;
  // disturbance should have its own range
  double disturbance_sampling_scale = 0.02;
  std::vector<double> disturbance_range; // nu x 2


};

}  // namespace mjpc

