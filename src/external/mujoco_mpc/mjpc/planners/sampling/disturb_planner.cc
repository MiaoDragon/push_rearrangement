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

// #include "mjpc/planners/sampling/planner.h"
#include "disturb_planner.h"
#include "disturb_policy.h"

#include <algorithm>
#include <chrono>
#include <shared_mutex>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
// #include "mjpc/planners/sampling/policy.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include "mujoco/mjmodel.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void SamplingDisturbPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();
  // allocate one mjData for nominal.
  ResizeMjData(model, 1);
  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise
  noise_exploration = GetNumberOrDefault(0.1, model, "sampling_exploration");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  // for disturbance
  disturbance_sampling_scale = GetNumberOrDefault(0.02, model,
                                      "disturbance_sampling_scale");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  winner = 0;
}

// allocate memory
void SamplingDisturbPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // trajectory and parameters
  winner = -1;
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
}

// reset memory to zeros
void SamplingDisturbPlanner::Reset(int horizon,
                            const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon, initial_repeated_action);
  }

  for (const auto& d : data_) {
    if (initial_repeated_action) {
      mju_copy(d->ctrl, initial_repeated_action, model->nu);
    } else {
      mju_zero(d->ctrl, model->nu);
    }
  }

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;
}

// set state
void SamplingDisturbPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int SamplingDisturbPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                              ThreadPool& pool) {
  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;


  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());


  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  this->Rollouts(num_trajectory, horizon, pool);

//   {
//     std::cout << "after rollout..." << std::endl;
//     std::cout << "candidate policy parametesr:" << std::endl;
//     for (int i=0; i<num_trajectory; i++)
//     {
//         std::cout << i << "-th num_spline_points: " << candidate_policy[i].num_spline_points << std::endl;
//         for (int j=0; j<candidate_policy[i].num_spline_points; j++)
//         {
//             for (int k=0; k<model->nu; k++)
//             {
//                 std::cout << candidate_policy[i].parameters[j*model->nu+k] << " ";

//             }
//             std::cout << std::endl;
//         }
//         std::cout << i << "-th total return: " << trajectory[i].total_return << std::endl;
//     }

//   }


  // sort candidate policies and trajectories by score
  trajectory_order.clear();
  trajectory_order.reserve(num_trajectory);
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order.push_back(i);
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + ncandidates,
      trajectory_order.end(), [trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  return ncandidates;
}

// optimize nominal policy using random sampling
void SamplingDisturbPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // resample nominal policy to current time
  this->UpdateNominalPolicy(horizon);

  OptimizePolicyCandidates(1, horizon, pool);


//   {
//     std::cout << "optimal trajectory: " << std::endl;
//     std::cout << "action: " << std::endl;
//     for (int ti=0; ti<trajectory[trajectory_order[0]].horizon-1; ti++)
//     {
//         for (int ai=0; ai<trajectory[trajectory_order[0]].dim_action; ai++)
//         {
//             std::cout << trajectory[trajectory_order[0]].actions[ti*trajectory[trajectory_order[0]].dim_action+ai];
//             std::cout << " ";        
//         }
//         std::cout << std::endl;
//     }
//   }


  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  CopyCandidateToPolicy(0);

  // improvement: compare nominal to winner
  double best_return = trajectory[0].total_return;
  improvement = mju_max(best_return - trajectory[winner].total_return, 0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void SamplingDisturbPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [&cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
}

// set action from policy
void SamplingDisturbPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy via resampling
void SamplingDisturbPlanner::UpdateNominalPolicy(int horizon) {
  // dimensions
  int num_spline_points = candidate_policy[winner].num_spline_points;

  // set time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    candidate_policy[winner].Parameter(DataAt(parameters_scratch, t * model->nu),
                               nullptr, nominal_time);
    // candidate_policy[winner].Action(DataAt(parameters_scratch, t * model->nu),
    //                            nullptr, nominal_time);
    nominal_time += time_shift;
  }

//   {
//     std::cout << "updating nominal policy..." << std::endl;
//     std::cout << "updated time_scratch: " << std::endl;
//     for (int i=0; i<times_scratch.size(); i++)
//     {
//         std::cout << times_scratch[i] << " ";
//     }
//     std::cout << std::endl;
//   }
// here we update the this->policy, which may be a new policy

  // update
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    // parameters
    policy.CopyParametersFrom(parameters_scratch, times_scratch);

    LinearRange(policy.times.data(), time_shift, policy.times[0],
                num_spline_points);
  }
}

// add random noise to nominal policy
void SamplingDisturbPlanner::AddNoiseToPolicy(int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;
  int num_parameters = candidate_policy[i].num_parameters;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample noise
//   std::cout << "addnoiseToPolicy: " << i << std::endl;
  for (int t = 0; t < num_spline_points; t++) {
    //   std::cout << "sampled noise for spline " << t << std::endl;
    for (int k = 0; k < model->nu; k++) {
    double scale = disturbance_sampling_scale;
    //   double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
    //                         model->actuator_ctrlrange[2 * k]);
      noise[shift + t * model->nu + k] =
          absl::Gaussian<double>(gen_, 0.0, scale * noise_exploration);

    //   std::cout << noise[shift + t * model->nu + k];
    //   std::cout << " ";
    }
    // std::cout << std::endl;
  }

  // add noise
//   std::cout << "before adding noise, parametsr:" << std::endl;
//   for (int j=0; j<num_parameters; j++)
//   {
//     std::cout << candidate_policy[i].parameters[j] << " ";
//   }
//   std::cout << std::endl;
  mju_addTo(candidate_policy[i].parameters.data(), DataAt(noise, shift),
            num_parameters);
//   std::cout << "after adding noise, parametsr:" << std::endl;
//   for (int j=0; j<num_parameters; j++)
//   {
//     std::cout << candidate_policy[i].parameters[j] << " ";
//   }
//   std::cout << std::endl;


  // clamp parameters
  for (int t = 0; t < num_spline_points; t++) {
    // TODO: clamp the parameters by the noise exploration bound
    Clamp(DataAt(candidate_policy[i].parameters, t * model->nu),
          candidate_policy[i].disturbance_range.data(), model->nu);

    // Clamp(DataAt(candidate_policy[i].parameters, t * model->nu),
    //       model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void SamplingDisturbPlanner::Rollouts(int num_trajectory, int horizon,
                               ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;
  policy.num_parameters = model->nu * policy.num_spline_points;

  // print out policy info
//   std::cout << "before rollouts, policy representation: " << policy.representation << std::endl;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      // copy nominal policy
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);

        s.candidate_policy[i].CopyFrom(s.policy, s.policy.num_spline_points);
        s.candidate_policy[i].representation = s.policy.representation;
      }

      // sample noise policy
      if (i != 0) s.AddNoiseToPolicy(i);

      // ----- rollout sample policy ----- //

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
    //   std::cout << "rolling out trajectory " << i << std::endl;
      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }

  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();


  // print out the rollout after trajectroy
//   std::cout << "after rollout, cost: " << std::endl;
//   for (int i=0; i<num_trajectory; i++)
//   {
//     std::cout << "trajectory[" << i << "]: " << std::endl;
//     std::cout << "action: " << std::endl;
//     for (int ti=0; ti<trajectory[i].horizon-1; ti++)
//     {
//         for (int ai=0; ai<trajectory[i].dim_action; ai++)
//         {
//             std::cout << trajectory[i].actions[ti*trajectory[i].dim_action+ai];
//             std::cout << " ";        
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "state: " << std::endl;

//     int jnt_qpos_idx = model->jnt_qposadr[mj_name2id(model, mjOBJ_JOINT, "ee_position_x")];

//     std::cout << "ee_position qpos: " << std::endl;
//     for (int ti=0; ti<trajectory[i].horizon; ti++)
//     {
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+jnt_qpos_idx+0];
//         std::cout << " ";
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+jnt_qpos_idx+1];
//         std::cout << " ";
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+jnt_qpos_idx+2];
//         std::cout << std::endl;
//     }

//     std::cout << "ee_position qvel: " << std::endl;
//     int jnt_qvel_idx = model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, "ee_position_x")];
//     for (int ti=0; ti<trajectory[i].horizon; ti++)
//     {
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+model->nq+jnt_qvel_idx+0];
//         std::cout << " ";
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+model->nq+jnt_qvel_idx+1];
//         std::cout << " ";
//         std::cout << trajectory[i].states[ti*trajectory[i].dim_state+model->nq+jnt_qvel_idx+2];
//         std::cout << std::endl;
//     }


//     std::cout << "residual: " << std::endl;
//     for (int ti=0; ti<trajectory[i].horizon-1; ti++)
//     {
//         for (int j=0; j<task->num_residual; j++)
//         {
//             std::cout << trajectory[i].residual[ti*task->num_residual+j];
//             std::cout << " ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout << "total return: " << trajectory[i].total_return << std::endl;
//   }


}

// return trajectory with best total return
const Trajectory* SamplingDisturbPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void SamplingDisturbPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  for (int k = 0; k < num_trajectory_; k++) {
    // skip winner
    if (k == winner) continue;

    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // make geometry
        mjv_makeConnector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[k].trace[3 * task->num_trace * i + 3 * j],
            trajectory[k].trace[3 * task->num_trace * i + 1 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * i + 2 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
            trajectory[k].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void SamplingDisturbPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &policy.representation,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, &noise_exploration, "0 1"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", DisturbMinSamplingSplinePoints,
                   DisturbMaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", DisturbMinNoiseStdDev,
                   DisturbMaxNoiseStdDev);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void SamplingDisturbPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                            int planner_shift, int timer_shift, int planning,
                            int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * noise_compute_time * planning, 100,
                 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

double SamplingDisturbPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void SamplingDisturbPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void SamplingDisturbPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}

void SamplingDisturbPlanner::SetPolicyNominalControlTrajectory(const std::shared_ptr<NominalControlTrajectory> nominal_control_traj)
{
    this->nominal_control_traj = nominal_control_traj;
    policy.SetNominalControlTrajectory(nominal_control_traj);
}


}  // namespace mjpc
