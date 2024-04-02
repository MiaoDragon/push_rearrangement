/**
 * @file pendulum_example.cpp
 * @author your name (you@domain.com)
 * @brief 
 * test the mpc controller in the pendulum example
 * @version 0.1
 * @date 2024-03-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cmath>
#include <limits>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <GLFW/glfw3.h>

#include "../utilities/utilities.h"
#include "../utilities/sample.h"

#include "mujoco_mppi_intvel.h"
#include "mujoco_mppi.h"
#include "task.h"
#include "policies.h"
#include "mppi.h"


class CartpoleMPPI : public MujocoMPPI<VectorXd>
{
  public:
    using MujocoMPPI::MujocoMPPI;

    void set_data_from_sensor(const VectorXd& sensordata, mjData*& d)
    {
        d = mj_makeData(m);
        // sensordata: orientation of the ball joint. qw, qx, qy, qz
        //             position of the end-effector: x, y, z
        // obtain the current activation from the orientation
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, "slider");
        // std::cout << "joint number: " << m->jnt_
        int jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        d->qpos[jnt_qpos_idx+0] = sensordata[0];
        joint_idx = mj_name2id(m, mjOBJ_JOINT, "hinge_1");
        jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        d->qpos[jnt_qpos_idx+0] = sensordata[1];
    }
    void get_state_from_data(const mjData* d, VectorXd& state)
    {
        // state: slider position, hinge position
        state.resize(2);
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, "slider");
        int jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        state[0] = d->qpos[jnt_qpos_idx+0];
        joint_idx = mj_name2id(m, mjOBJ_JOINT, "hinge_1");
        jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        state[1] = d->qpos[jnt_qpos_idx+0];
    }
  protected:
    void set_sensor_from_data(const mjData* d, VectorXd& sensor)
    {
        sensor.resize(2);
        int sensor_idx = mj_name2id(m, mjOBJ_SENSOR, "slider_pos");
        int sensor_adr = m->sensor_adr[sensor_idx];
        sensor[0] = d->sensordata[sensor_adr];

        sensor_idx = mj_name2id(m, mjOBJ_SENSOR, "hinge_pos");
        sensor_adr = m->sensor_adr[sensor_idx];
        sensor[1] = d->sensordata[sensor_adr];
    }
};

class CartpoleTask : public ControlTask<VectorXd>
{
  public:
    using ControlTask::ControlTask;

    double goal_hinge;
    double goal_pos;

    void sensor_to_state(const VectorXd& sensor, VectorXd& state)
    {
        state.resize(2);
        // state: slider position, hinge
        state[0] = sensor[0];
        state[1] = sensor[1];
    }

    double cost(const VectorXd& state, const VectorXd& sensor, const VectorXd& control)
    {
        // check state bound
        bool in_bound = true;
        in_bound &= compare_vector_smaller_eq(state_lb, state);
        in_bound &= compare_vector_smaller_eq(state, state_ub);
        in_bound &= compare_vector_smaller_eq(control_lb, control);
        in_bound &= compare_vector_smaller_eq(control, control_ub);

        double res = 0;
        double bound_cost = 10.0;
        if (!in_bound)
        {
            res += bound_cost;
        }

        // task: track the hinge and position
        res += (sensor[0] - goal_pos)*(sensor[0] - goal_pos);
        res += (sensor[1] - goal_hinge)*(sensor[1] - goal_hinge);
        return res;        
    }

    double terminal_cost(const VectorXd& state, const VectorXd& sensor, const VectorXd& control)
    {
        return 0;
    }

    void set_goal(const double goal_hinge, const double goal_pos)
    {
        this->goal_hinge = goal_hinge; this->goal_pos = goal_pos;
    }

};

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);

    // also reset the trajectory

  }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}



int main(void)
{

    // read mujoco scene
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/cartpole_task.xml";
    const char* c = filename.c_str();
    char loadError[1024] = "";

    m = mj_loadXML(c, 0, loadError, 1024);
    if (!m)
    {
        mju_error("Could not init model");
    }
    d = mj_makeData(m);

    if (!glfwInit())
    {
        mju_error("Could not init glfw");
    }

    // init GLFW, create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);


    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    mj_forward(m, d);

    std::vector<const char*> act_names{"slide"};
    VectorXd x_ll(2), x_ul(2), u_ll(1), u_ul(1);

    std::vector<int> ctrl_indices;
    
    /* obtain range for x and u */
    for (int i=0; i<act_names.size(); i++)
    {
        int act_id = mj_name2id(m, mjOBJ_ACTUATOR, act_names[i]);
        ctrl_indices.push_back(act_id);

        u_ll[i] = m->actuator_ctrlrange[i*2];
        u_ul[i] = m->actuator_ctrlrange[i*2+1];
    }
    x_ll[0] = -1.8; x_ul[0] = 1.8;
    x_ll[1] = -std::numeric_limits<double>::infinity(); x_ul[1] = std::numeric_limits<double>::infinity();


    int H = 100, N = 10;
    double default_sigma = 1.0;
    double dt = 0.01;

    // PendulumMPPI controller(H, N, default_sigma, nominal_x, nominal_u, x_ll, x_ul, u_ll, u_ul);
    // controller.set_pos_act_indices(pos_act_indices);
    // controller.set_vel_ctrl_indices(vel_ctrl_indices);
    // controller.set_dt(0.01);//m->opt.timestep);


    CartpoleTask task(x_ll, x_ul, u_ll, u_ul);
    int goal_bid = mj_name2id(m, mjOBJ_BODY, "cart_goal");
    // compute the position, and track it
    task.set_goal(0, d->xpos[goal_bid*3+0]);

    KnotControlPolicy policy(H*dt, 1, 0, 30, 1, u_ll, u_ul);
    CartpoleMPPI controller(N, H, dt, default_sigma, &task, &policy, m);
    mjtNum total_simstart = d->time;


    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize

    total_simstart = d->time;
    // traj_idx = 0;
    int step_idx = 0;
    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        // if (d->time - total_simstart < 1)
        {
            step_idx++;
            int sensor_idx = mj_name2id(m, mjOBJ_SENSOR, "slider_pos");
            int sensor_adr = m->sensor_adr[sensor_idx];
            double* sensordata = d->sensordata;
            VectorXd sensor(2);
            sensor[0] = d->sensordata[sensor_adr];
            sensor_idx = mj_name2id(m, mjOBJ_SENSOR, "hinge_pos");
            sensor_adr = m->sensor_adr[sensor_idx];
            sensor[1] = d->sensordata[sensor_adr];
            
            // sensordata[0] = 30.0/180 * M_PI;
            VectorXd control;
            controller.step(sensor, control);
            policy.shift_by_time(m->opt.timestep);

            for (int i=0; i<ctrl_indices.size(); i++)
            {
                d->ctrl[ctrl_indices[i]] = control[i];
            }

            mj_step(m, d);

        }
        // get framebuffer viewport
        mjrRect viewport = {0,0,0,0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffer
        glfwSwapBuffers(window);

        glfwPollEvents();
    }
    // free vis storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free mujoco model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
}