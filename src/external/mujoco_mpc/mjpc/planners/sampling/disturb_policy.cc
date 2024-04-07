// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "disturb_policy.h"

#include <algorithm>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/policy.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void SamplingDisturbPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // parameters
  parameters.resize(model->nu * kMaxTrajectoryHorizon);

  // times
  times.resize(kMaxTrajectoryHorizon);

  // dimensions
  num_parameters = model->nu * kMaxTrajectoryHorizon;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");

  // representation
  representation = GetNumberOrDefault(PolicyRepresentation::kCubicSpline, model,
                                      "sampling_representation");

  // for disturbance
  disturbance_sampling_scale = GetNumberOrDefault(0.02, model,
                                      "disturbance_sampling_scale");

  // set the disturbance range
  for (int i=0; i<model->nu; i++)
  {
    disturbance_range.push_back(-disturbance_sampling_scale);
    disturbance_range.push_back(disturbance_sampling_scale);
  }


}

// reset memory to zeros
void SamplingDisturbPolicy::Reset(int horizon, const double* initial_repeated_action) {
  // parameters
  if (initial_repeated_action != nullptr) {
    for (int i = 0; i < num_spline_points; ++i) {
      mju_copy(parameters.data() + i * model->nu, initial_repeated_action,
               model->nu);
    }
  } else {
    std::fill(parameters.begin(),
              parameters.begin() + model->nu * num_spline_points, 0.0);
  }
  // policy parameter times
  std::fill(times.begin(), times.begin() + horizon, 0.0);
}

// set action from policy
void SamplingDisturbPolicy::Action(double* action, const double* state,
                            double time) const 
{
  // find times bounds
  int bounds[2];
  FindInterval(bounds, times, time, num_spline_points);

//   std::cout << "finding interval..." << std::endl;
//   std::cout << "times: " << std::endl;
//   for (int i=0; i<num_spline_points; i++)
//   {
//     std::cout << times[i] << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "time: " << time << std::endl;

  // ----- get action ----- //

//   std::cout << "policy parameters: " << std::endl;
//   for (int i=0; i<num_parameters; i++)
//   {
//     std::cout << parameters[i] << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "bounds[0]: " << bounds[0] << std::endl;
//   std::cout << "bounds[1]: " << bounds[1] << std::endl;
//   std::cout << "representation: " << representation << std::endl;

  if (bounds[0] == bounds[1] ||
      representation == PolicyRepresentation::kZeroSpline) {
    // std::cout << "zero" << std::endl;
    ZeroInterpolation(action, time, times, parameters.data(), model->nu,
                      num_spline_points);
  } else if (representation == PolicyRepresentation::kLinearSpline) {
    // std::cout << "linear" << std::endl;
    LinearInterpolation(action, time, times, parameters.data(), model->nu,
                        num_spline_points);
  } else if (representation == PolicyRepresentation::kCubicSpline) {
    // std::cout << "cubic" << std::endl;
    CubicInterpolation(action, time, times, parameters.data(), model->nu,
                       num_spline_points);
  }

  // * for disturbance *
  // add disturbance to the norminal control
  double nominal_action[model->nu];
  nominal_control_traj->at(time, nominal_action);

//   std::cout << "time: " << time << std::endl;
//   std::cout << "action before adding" << std::endl;
//   for (int i=0; i<model->nu; i++)
//   {
//     std::cout << action[i] << " ";
//   }
//   std::cout << std::endl;


  mju_addTo(action, nominal_action, model->nu);

//   std::cout << "nominal" << std::endl;
//   for (int i=0; i<model->nu; i++)
//   {
//     std::cout << nominal_action[i] << " ";
//   }
//   std::cout << std::endl;


//   std::cout << "action after adding nominal" << std::endl;
//   for (int i=0; i<model->nu; i++)
//   {
//     std::cout << action[i] << " ";
//   }
//   std::cout << std::endl;


  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);

//   std::cout << "action after clamping" << std::endl;
//   for (int i=0; i<model->nu; i++)
//   {
//     std::cout << action[i] << " ";
//   }
//   std::cout << std::endl;


}

/**
 * @brief 
 * obtain parameter, which is the distur bance knots
 * TODO: need to have a clamping for the disturbance
 * 
 * @param parameter 
 * @param state 
 * @param time 
 */
void SamplingDisturbPolicy::Parameter(double* parameter, const double* state, double time) const
{
  // * for disturbance *
  // add disturbance to the norminal control
  // find times bounds
  int bounds[2];
  FindInterval(bounds, times, time, num_spline_points);

//   std::cout << "finding interval..." << std::endl;
//   std::cout << "times: " << std::endl;
//   for (int i=0; i<num_spline_points; i++)
//   {
//     std::cout << times[i] << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "time: " << time << std::endl;

  // ----- get action ----- //

//   std::cout << "policy parameters: " << std::endl;
//   for (int i=0; i<num_parameters; i++)
//   {
//     std::cout << parameters[i] << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "bounds[0]: " << bounds[0] << std::endl;
//   std::cout << "bounds[1]: " << bounds[1] << std::endl;
//   std::cout << "representation: " << representation << std::endl;

  if (bounds[0] == bounds[1] ||
      representation == PolicyRepresentation::kZeroSpline) {
    // std::cout << "zero" << std::endl;
    ZeroInterpolation(parameter, time, times, parameters.data(), model->nu,
                      num_spline_points);
  } else if (representation == PolicyRepresentation::kLinearSpline) {
    // std::cout << "linear" << std::endl;
    LinearInterpolation(parameter, time, times, parameters.data(), model->nu,
                        num_spline_points);
  } else if (representation == PolicyRepresentation::kCubicSpline) {
    // std::cout << "cubic" << std::endl;
    CubicInterpolation(parameter, time, times, parameters.data(), model->nu,
                       num_spline_points);
  }

  // * for disturbance *
  // add disturbance to the norminal control

//   std::cout << "time: " << time << std::endl;
//   std::cout << "parameter" << std::endl;
//   for (int i=0; i<model->nu; i++)
//   {
//     std::cout << parameter[i] << " ";
//   }
//   std::cout << std::endl;


//   // Clamp controls
//   Clamp(action, model->actuator_ctrlrange, model->nu);
  Clamp(parameter, disturbance_range.data(), model->nu);

}


// copy policy
void SamplingDisturbPolicy::CopyFrom(const SamplingDisturbPolicy& policy, int horizon) {
  mju_copy(parameters.data(), policy.parameters.data(), policy.num_parameters);
  mju_copy(times.data(), policy.times.data(), policy.num_spline_points);
  num_spline_points = policy.num_spline_points;
  num_parameters = policy.num_parameters;
  nominal_control_traj = policy.nominal_control_traj;
}

// copy parameters
void SamplingDisturbPolicy::CopyParametersFrom(
    const std::vector<double>& src_parameters,
    const std::vector<double>& src_times) {
  mju_copy(parameters.data(), src_parameters.data(),
           num_spline_points * model->nu);
  mju_copy(times.data(), src_times.data(), num_spline_points);
}

void SamplingDisturbPolicy::SetNominalControlTrajectory(const std::shared_ptr<NominalControlTrajectory> nominal_control_traj)
{
    this->nominal_control_traj = nominal_control_traj;
}


}  // namespace mjpc
