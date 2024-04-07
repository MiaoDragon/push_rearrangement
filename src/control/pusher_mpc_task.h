/**
 * @file pusher_task.h
 * @author your name (you@domain.com)
 * @brief 
 * define control task for tracking pushing trajectory
 * @version 0.1
 * @date 2024-04-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <mjpc/interface.h>
#include <mjpc/task.h>
#include <mjpc/planners/sampling/planner.h>
#include <mjpc/states/state.h>
#include <mjpc/threadpool.h>
#include "mjpc/norm.h"

#include "../utilities/utilities.h"
#include "../utilities/trajectory.h"
#include "mujoco/mjdata.h"
#include "mujoco/mjmodel.h"
#include "mujoco/mujoco.h"
#include <memory>

class EEPositionPusherTask : public mjpc::Task
{
  public:
    std::string Name() const override
    {
        return "EEPositionPusherTask";
    }
    std::string XmlPath() const override
    {
        return std::string(this->xml_path);
    }

    class ResidualFn : public mjpc::BaseResidualFn
    {
      public:
        explicit ResidualFn(const EEPositionPusherTask* task) : mjpc::BaseResidualFn(task)
        {
            this->obj_pose_traj = task->obj_pose_traj;
            this->robot_ee_traj = task->robot_ee_traj;
            this->duration = task->duration;
            this->start_time = task->start_time;
            this->robot_qpos_ids = task->robot_qpos_ids;
            this->robot_qvel_ids = task->robot_qvel_ids;
            this->obj_bids = task->obj_bids;
        }

        void set_target(const std::shared_ptr<PoseTrajectory> obj_pose_traj, const std::shared_ptr<PositionTrajectory> robot_ee_traj)
        {
            this->obj_pose_traj = obj_pose_traj;
            this->robot_ee_traj = robot_ee_traj;
        }
        void set_duration(const double duration)
        {
            this->duration = duration;
        }
        void set_start_time(const double start_time)
        {
            this->start_time = start_time;
        }

        void set_robot_qpos_ids(const std::vector<int>& robot_qpos_ids)
        {
            this->robot_qpos_ids = robot_qpos_ids;
        }

        void set_robot_qvel_ids(const std::vector<int>& robot_qvel_ids)
        {
            this->robot_qvel_ids = robot_qvel_ids;
        }


        void set_obj_bids(const std::vector<int>& obj_bids)
        {
            this->obj_bids = obj_bids;
        }


        /**
         * @brief 
         * residual:
         * - robot position
         * - objects
         *   - position
         *   - orientation
         * @param model 
         * @param data 
         * @param residual 
         */
        void Residual(const mjModel* model, const mjData* data, double* residual) const override
        {
            // evaluate the trajectories at current time
            double dT = data->time - this->start_time;
            Matrix4d nominal_obj_pose;
            double interp_t = std::min(1.0, dT/duration);
            obj_pose_traj->interpolate(interp_t, nominal_obj_pose);
            Vector3d nominal_ee_position;
            robot_ee_traj->interpolate(interp_t, nominal_ee_position);
            Vector3d nominal_ee_v;
            robot_ee_traj->velocity(interp_t, nominal_ee_v);
            if (interp_t == 1.0) // resting
            {
                nominal_ee_v = Vector3d::Zero();
            }
            int counter = 0;

            /* residual: robot_ee_position - nominal_robot_ee_position */
            for (int i=0; i<3; i++)  // in total the size should be 3 for position
            {
                residual[counter] = data->qpos[robot_qpos_ids[i]] - nominal_ee_position[i];
                counter += 1;
            }

            /* residual: robot_ee_v - nominal_robot_ee_v */
            for (int i=0; i<3; i++)  // in total the size should be 3 for position
            {
                residual[counter] = data->qvel[robot_qvel_ids[i]] - nominal_ee_v[i];
                counter += 1;
            }

            // term: one term (normed distance) or multiple terms

            /* residual for each object in the list */
            for (int i=0; i<obj_bids.size(); i++)
            {
                int obj_bid = obj_bids[i];
                // residual: obj_pose_position - nominal_obj_pose_position
                residual[counter] = data->xpos[obj_bid*3] - nominal_obj_pose(0,3);
                residual[counter+1] = data->xpos[obj_bid*3+1] - nominal_obj_pose(1,3);
                residual[counter+2] = data->xpos[obj_bid*3+2] - nominal_obj_pose(2,3);
                counter += 3;
                // term

                // residual: obj_pose_orientation - nominal_obj_pose_orientation (size of 3)
                Quaterniond quat(nominal_obj_pose.block<3,3>(0,0));
                double ori[4] = {quat.w(), quat.x(), quat.y(), quat.z()};
                mju_subQuat(residual + counter, data->xquat+obj_bid*4, ori);
                counter += 3;
                // (mujoco impl) Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
                // (mujoco impl) mju_subQuat(residual + counter, goal_orientation, orientation);
                // term
                // mju_copy(residual+counter, data->ctrl, 6);//model->nu);  // ctrl is velocity
            }
        }
      protected:
        std::shared_ptr<PoseTrajectory> obj_pose_traj = nullptr;
        std::shared_ptr<PositionTrajectory> robot_ee_traj = nullptr;
        double duration = 0;
        double start_time = 0;
        std::vector<int> robot_qpos_ids = {};
        std::vector<int> robot_qvel_ids = {};
        std::vector<int> obj_bids = {};
    };

    EEPositionPusherTask(const int num_term, const std::vector<int>& dim_norm_residual, const std::vector<mjpc::NormType>& norm,
                         const std::vector<int>& num_norm_parameter, const std::vector<double>& norm_parameter, const std::vector<double>& weight,
                         const std::vector<double>& parameters, const double risk,
                         const std::shared_ptr<PoseTrajectory> obj_pose_traj,
                         const std::shared_ptr<PositionTrajectory> robot_ee_traj,
                         const double duration,
                         const std::vector<int>& robot_qpos_ids, 
                         const std::vector<int>& robot_qvel_ids,
                         const std::vector<int>& obj_bids) : mjpc::Task()
    {
        /* set parameters for the cartpole task */
        this->num_residual = 6+3+3;  // robot_ee position, robot_ee velocity obj position, obj orientation (3)
        /* TODO: also track velocities */
        this->num_term = num_term; // each one is a term, or merging multiple
        this->num_trace = 0;
        this->dim_norm_residual = dim_norm_residual;
        this->norm = norm;
        // this->norm = {mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic};
        this->num_norm_parameter = num_norm_parameter;
        // provide the params
        this->weight = weight;
        this->norm_parameter = norm_parameter;
        this->risk = risk; //0.0;
        this->parameters = parameters;
        this->obj_pose_traj = obj_pose_traj;
        this->robot_ee_traj = robot_ee_traj;
        this->duration = duration;

        this->robot_qpos_ids = robot_qpos_ids;
        this->robot_qvel_ids = robot_qvel_ids;
        this->obj_bids = obj_bids;

        residual_ = new ResidualFn(this);
    }

    ~EEPositionPusherTask()
    {
        delete residual_;
        this->residual_ = nullptr;
        this->obj_pose_traj = nullptr;
        this->robot_ee_traj = nullptr;
    }

    void set_duration(const double duration)
    {
        this->duration =  duration;
        residual_->set_duration(duration);
    }

    void set_start_time(const double start_time)
    {
        this->start_time = start_time;
        residual_->set_start_time(start_time);
    }

    void set_robot_qpos_ids(const std::vector<int>& robot_qpos_ids)
    {
        this->robot_qpos_ids = robot_qpos_ids;
        residual_->set_robot_qpos_ids(robot_qpos_ids);
    }

    void set_robot_qvel_ids(const std::vector<int>& robot_qvel_ids)
    {
        this->robot_qvel_ids = robot_qvel_ids;
        residual_->set_robot_qvel_ids(robot_qvel_ids);
    }

    void set_obj_bids(const std::vector<int>& obj_bids)
    {
        this->obj_bids = obj_bids;
        residual_->set_obj_bids(obj_bids);
    }

    void set_target(std::shared_ptr<PoseTrajectory> obj_pose_traj, std::shared_ptr<PositionTrajectory> robot_ee_traj)
    {
        this->obj_pose_traj = obj_pose_traj;
        this->robot_ee_traj = robot_ee_traj;
        residual_->set_target(obj_pose_traj, robot_ee_traj);
    }

  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
        return std::make_unique<ResidualFn>(this);
    }
    ResidualFn* InternalResidual() override { return residual_; }

    std::shared_ptr<PoseTrajectory> obj_pose_traj = nullptr;
    std::shared_ptr<PositionTrajectory> robot_ee_traj = nullptr;
    double duration = 0.0;
    double start_time = 0.0;
    const char* xml_path = "";

    std::vector<int> robot_qpos_ids = {};
    std::vector<int> robot_qvel_ids = {};
    std::vector<int> obj_bids = {};


  private:
    ResidualFn* residual_ = nullptr;

};