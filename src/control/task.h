/**
 * @file task.h
 * @author your name (you@domain.com)
 * @brief 
 * define the control task. The task implements the cost function, and obtain the state.
 * @version 0.1
 * @date 2024-03-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <limits>
#include "../utilities/utilities.h"

template<typename SensorType>
class ControlTask
{
  public:
    ControlTask(const VectorXd& state_lb, const VectorXd& state_ub, const VectorXd& control_lb, const VectorXd& control_ub)
    {
        this->state_lb = state_lb;
        this->state_ub = state_ub;
        this->control_lb = control_lb;
        this->control_ub = control_ub;
    }

    VectorXd state_lb, state_ub;
    VectorXd control_lb, control_ub;

    virtual void sensor_to_state(const SensorType& sensor, VectorXd& state) = 0;
    /* obtain the cost given the planning state, the sensed info and the control */
    virtual double cost(const VectorXd& state, const SensorType& sensor, const VectorXd& control) = 0;
    virtual double terminal_cost(const VectorXd& state, const SensorType& sensor, const VectorXd& control) = 0;
    
};