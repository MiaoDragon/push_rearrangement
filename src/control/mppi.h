/**
 * @file mppi.h
 * @author your name (you@domain.com)
 * @brief 
 * ref: https://github.com/google-deepmind/mujoco_mpc
 * 
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>

#include "../utilities/utilities.h"
#include "policies.h"
#include "task.h"

template<typename ModelType, typename DataType, typename SensorType>
class MPPI
{
  public:
    /**
     * @brief Construct a new MPPI::MPPI object
     * 
     * @param N 
     * @param H 
     * @param dt
     * the step size in the MPPI
     * @param task 
     * a class defining the task.
     * @param policy 
     * the policy to optimize
     */
    int N=10, H=10;
    double dt=0.01;
    double sigma=0.1;

    MPPI(const int N, const int H, const double dt, const double sigma, ControlTask<SensorType>* task, KnotControlPolicy* policy,
         const ModelType* m)
    {
        this->N = N; this->H = H; this->dt = dt; this->sigma = sigma;
        this->task = task; this->policy = policy;
        set_model(m);
        file.open("policy_cost.txt");
    }

    ~MPPI()
    {
        delete m;
        m = nullptr;
        delete start_d;
        start_d = nullptr;
        file.close();
    }

    virtual void set_model(const ModelType* m)
    {
        if (this->m != nullptr)
        {
            delete this->m;
            this->m = nullptr;
        }
        this->m = new ModelType(*m);
    }
    virtual void set_start_data(const DataType* d)
    {
        if (this->start_d != nullptr)
        {
            delete this->start_d;
            this->start_d = nullptr;
        }
        this->start_d = new DataType(*d);
    }

    /* set the data from the sensor */
    // create a new data to hold the info
    virtual void set_data_from_sensor(const SensorType& sensor_data, DataType*& d) = 0;
    virtual void get_state_from_data(const DataType* d, VectorXd& state) = 0;
    /* rollout using the sampled policy parameters to obtain the costs */
    // start from the start data
    virtual void rollout(const std::vector<knot_point_deque_t>& thetas, VectorXd& costs) = 0;
    /* given the sensed data, plan from the current time */
    void step(const SensorType& sensor_data, VectorXd& control)
    {
        std::cout << "stepping..." << std::endl;
        std::cout << "sensor_data: " << sensor_data.transpose() << std::endl;
        DataType* d;
        std::cout << "before setting data from sensor" << std::endl;
        set_data_from_sensor(sensor_data, d);
        VectorXd state;
        std::cout << "before setting state from data" << std::endl;
        get_state_from_data(d, state);
        /* sample noisy policy params */
        std::vector<knot_point_deque_t> thetas;
        std::cout << "before sample gauss param" << std::endl;

        policy->sample_gauss_param(this->N, this->sigma, thetas);
        std::cout << "before rollout" << std::endl;

        /* rollout */
        set_start_data(d);
        VectorXd costs;
        rollout(thetas, costs);

        file << costs.transpose() << std::endl;

        /* optimize the pollicy */
        policy->optimize(thetas, costs);
        policy->action(state, 0, control);
        delete d;
    }

  protected:
    ControlTask<SensorType>* task = nullptr;
    KnotControlPolicy* policy = nullptr;
    ModelType* m = nullptr;
    DataType* start_d = nullptr;
    std::ofstream file;
};


template<typename SensorType>
class MujocoMPPI : public MPPI<mjModel, mjData, SensorType>
{
  public:
    MujocoMPPI(const int N, const int H, const double dt, const double sigma,
               ControlTask<VectorXd>* task, KnotControlPolicy* policy, const mjModel* m) :
               MPPI<mjModel, mjData, VectorXd>(N, H, dt, sigma, task, policy, m)
               {
                    this->m->opt.timestep = dt; // set dt
               }

    void set_start_data(const mjData* d) override
    {
        if (this->start_d != nullptr)
        {
            delete this->start_d;
            this->start_d = nullptr;
        }
        this->start_d = mj_makeData(this->m);
        // faster copying the data
        // ref: https://mujoco.readthedocs.io/en/stable/programming/simulation.html   
        copy_data(d, this->start_d);
        mj_forward(this->m, this->start_d);
    }

    // rollout from the start data
    void rollout(const std::vector<knot_point_deque_t>& thetas, VectorXd& costs)
    {
        std::cout << "rollout..." << std::endl;
        std::cout << "thetas: " << std::endl;
        for (int i=0; i<thetas.size(); i++)
        {
            std::cout << "sample " << i << ": " << std::endl;
            for (int j=0; j<thetas[i].size(); j++)
            {
                std::cout << thetas[i][j].first.transpose() << std::endl;
            }
        }


        VectorXd state0, sensor0;
        this->get_state_from_data(this->start_d, state0);
        set_sensor_from_data(this->start_d, sensor0);


        costs.resize(this->N);
        costs = VectorXd::Zero(this->N);

        std::vector<mjData*> ds(this->N);
        for (int i=0; i<this->N; i++)
        {
            ds[i] = mj_makeData(this->m);
            copy_data(this->start_d, ds[i]);
        }

        for (int i=0; i<this->N; i++)
        {
            std::cout << "sample " << i << std::endl;
            for (int j=0; j<this->H; j++)
            {
                VectorXd state, control, sensor;
                this->get_state_from_data(ds[i], state);
                set_sensor_from_data(ds[i], sensor);

                costs[i] += this->task->cost(state, sensor, control);
                std::cout << "state: " << state.transpose() << std::endl;
                std::cout << "sensor: " << sensor.transpose() << std::endl; 

                this->policy->action(state, thetas[i], j*this->dt, control);
                // apply the control to the pendulum
                mju_copy(ds[i]->ctrl, control.data(), control.size());
                mj_step(this->m, ds[i]);
            }
            VectorXd state, control, sensor;
            this->get_state_from_data(ds[i], state);
            set_sensor_from_data(ds[i], sensor);
            costs[i] += this->task->terminal_cost(state, sensor, control);

            std::cout << "cost[i]:" << std::endl;
            std::cout << costs[i] << std::endl;

        }

        for (int i=0; i<this->N; i++)
        {
            delete ds[i];
        }

    }

  protected:
    void copy_data(const mjData* src, mjData* dst)
    {
        // copy simulation state
        dst->time = src->time;
        mju_copy(dst->qpos, src->qpos, this->m->nq);
        mju_copy(dst->qvel, src->qvel, this->m->nv);
        mju_copy(dst->act,  src->act,  this->m->na);

        // copy mocap body pose and userdata
        mju_copy(dst->mocap_pos,  src->mocap_pos,  3*this->m->nmocap);
        mju_copy(dst->mocap_quat, src->mocap_quat, 4*this->m->nmocap);
        mju_copy(dst->userdata, src->userdata, this->m->nuserdata);

        // copy sensordata
        mju_copy(dst->sensordata, src->sensordata, this->m->nsensordata);

        // copy positions
        mju_copy(dst->xpos, src->xpos, this->m->nbody*3);
        mju_copy(dst->xquat, src->xquat, this->m->nbody*4);
        mju_copy(dst->xmat, src->xmat, this->m->nbody*9);
        mju_copy(dst->xipos, src->xipos, this->m->nbody*3);
        mju_copy(dst->ximat, src->ximat, this->m->nbody*9);
        mju_copy(dst->geom_xpos, src->geom_xpos, this->m->ngeom*3);
        mju_copy(dst->geom_xmat, src->geom_xmat, this->m->ngeom*9);
        mju_copy(dst->site_xpos, src->site_xpos, this->m->nsite*3);
        mju_copy(dst->site_xmat, src->site_xmat, this->m->nsite*9);


        // copy warm-start acceleration
        mju_copy(dst->qacc_warmstart, src->qacc_warmstart, this->m->nv);
    }

    virtual void set_sensor_from_data(const mjData* d, SensorType& sensor) = 0;  // used during rollout
};