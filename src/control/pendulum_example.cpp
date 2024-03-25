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


class PendulumMPPI : public MujocoMPPIControllerIntvel
{
  public:
    using MujocoMPPIControllerIntvel::MujocoMPPIControllerIntvel;

    /* given sensordata, set the corresponding components of Mujoco data */
    void set_data_by_sensor(const double* sensordata, const mjModel* m, mjData* d)
    {
        // sensordata: orientation of the ball joint. qw, qx, qy, qz
        // obtain the current activation from the orientation
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, "ball_joint");
        // std::cout << "joint number: " << m->jnt_
        int jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        std::cout << "q->pos: " << d->qpos[jnt_qpos_idx+0] << ", " << d->qpos[jnt_qpos_idx+1] << ", " << d->qpos[jnt_qpos_idx+2] << ", " << d->qpos[jnt_qpos_idx+3] << std::endl; 
        std::cout << "sensor: " << sensordata[0] << ", " << sensordata[1] << ", " << sensordata[2] << ", " << sensordata[3] << std::endl;
        d->qpos[jnt_qpos_idx+0] = sensordata[0];
        d->qpos[jnt_qpos_idx+1] = sensordata[1];
        d->qpos[jnt_qpos_idx+2] = sensordata[2];
        d->qpos[jnt_qpos_idx+3] = sensordata[3];
    }

    void get_state_from_data(const mjModel* m, const mjData* d, VectorXd& state)
    {
        Vector3d axis1, axis2, axis3;
        axis1 << 1, 0, 0;
        axis2 << 0, 1, 0;
        axis3 << 0, 0, 1;

        int joint_idx = mj_name2id(m, mjOBJ_JOINT, "ball_joint");
        int jnt_qpos_idx = m->jnt_qposadr[joint_idx];
        
        Quaterniond quat(d->qpos[jnt_qpos_idx+0],
                         d->qpos[jnt_qpos_idx+1],
                         d->qpos[jnt_qpos_idx+2],
                         d->qpos[jnt_qpos_idx+3]);
        AngleAxisd ang_axis(quat);
        double ang = ang_axis.angle();
        Vector3d axis = ang_axis.axis();

        state.resize(3);
        state[0] = ang * axis.transpose() * axis1;
        state[1] = ang * axis.transpose() * axis2;
        state[2] = ang * axis.transpose() * axis3;

        std::cout << "state from data: " << std::endl;
        std::cout << state << std::endl;

    }

    double get_cost(const mjModel* m, const mjData* d)
    {
        // task: track the tip position
        int tip_bid = mj_name2id(m, mjOBJ_BODY, "tip");
        int goal_bid = mj_name2id(m, mjOBJ_BODY, "goal_tip");
        // compute the position, and track it
        Vector3d tip_pos, goal_pos;
        tip_pos[0] = d->xpos[tip_bid*3+0];
        tip_pos[1] = d->xpos[tip_bid*3+1];
        tip_pos[2] = d->xpos[tip_bid*3+2];
        goal_pos[0] = d->xpos[goal_bid*3+0];
        goal_pos[1] = d->xpos[goal_bid*3+1];
        goal_pos[2] = d->xpos[goal_bid*3+2];
        // compute the difference
        Vector3d diff = tip_pos - goal_pos;
        return diff.norm();
    }

    double get_terminal_cost(const mjModel* m, const mjData* d)
    {
        return 0;
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
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/3d_pendulum.xml";
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


    std::vector<const char*> act_names{"axis_x", "axis_y", "axis_z"};
    VectorXd x_ll(3), x_ul(3), u_ll(3), u_ul(3);

    std::vector<int> pos_act_indices, vel_ctrl_indices;
    
    /* obtain range for x and u */
    for (int i=0; i<act_names.size(); i++)
    {
        int act_id = mj_name2id(m, mjOBJ_ACTUATOR, act_names[i]);
        pos_act_indices.push_back(m->actuator_actadr[act_id]);  // activation
        vel_ctrl_indices.push_back(act_id);

        x_ll[i] = m->actuator_actrange[i*2];
        x_ul[i] = m->actuator_actrange[i*2+1];
        u_ll[i] = m->actuator_ctrlrange[i*2];
        u_ul[i] = m->actuator_ctrlrange[i*2+1];
    }
    MatrixXd nominal_x(1,3), nominal_u(1,3);
    nominal_x.setZero();  nominal_u.setZero();

    int H = 100, N = 10;
    double default_sigma = 0.3;

    PendulumMPPI controller(H, N, default_sigma, nominal_x, nominal_u, x_ll, x_ul, u_ll, u_ul);
    controller.set_pos_act_indices(pos_act_indices);
    controller.set_vel_ctrl_indices(vel_ctrl_indices);
    controller.set_dt(0.01);//m->opt.timestep);

    mjtNum total_simstart = d->time;


    mj_forward(m, d);
    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize

    total_simstart = d->time;
    // traj_idx = 0;
    int step_idx = 0;
    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        // if (d->time - total_simstart < 1)
        {
            std::cout << "stepping... " << step_idx << std::endl;
            step_idx ++;

            double* sensordata = d->sensordata;
            // try to set the sensor data to a different value
            
            // sensordata[0] = 30.0/180 * M_PI;
            VectorXd control;
            controller.step(m, sensordata, control);

            for (int i=0; i<pos_act_indices.size(); i++)
            {
                // d->act[pos_act_indices[i]] = control[i];
            }

            std::cout << "control: " << std::endl;
            std::cout << control << std::endl;

            for (int i=0; i<vel_ctrl_indices.size(); i++)
            {
                d->ctrl[vel_ctrl_indices[i]] = control[pos_act_indices.size()+i];
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