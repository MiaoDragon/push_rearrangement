/**
 * @file motoman_ik_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * this tests the inverse kinematics solver.
 * @version 0.1
 * @date 2024-06-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @file obj_steer_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * TODO:
 * [] testing single object steering from start to goal, with end-effector position as contact
 * [] multi-thread testing
 *
 * @version 0.1
 * @date 2024-04-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cmath>
#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjvisualize.h>
#include <GLFW/glfw3.h>

#include <ctime>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <cstring>

#include <mjpc/states/state.h>
#include <mjpc/threadpool.h>
#include <mjpc/norm.h>

#include "../utilities/utilities.h"
#include "../utilities/trajectory.h"

#include "inverse_kinematics.h"

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
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/robot_ik.xml";
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


    // d->qpos[m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")]] = 0.1; // for stable start


    // get the robot ee link id, joint indices
    std::vector<const char*> joint_names{"torso_joint_b1",
                                        "arm_left_joint_1_s",
                                        "arm_left_joint_2_l",
                                        "arm_left_joint_3_e",
                                        "arm_left_joint_4_u",
                                        "arm_left_joint_5_r",
                                        "arm_left_joint_6_b",
                                        "arm_left_joint_7_t"};

    // double q[8] = {0, 1.75, 0.8, 0, -0.66, 0, 0, 0};
    double q[8] = {0, 1.75, 0.8, 0, -0.66, 0, 0, 0};

    double lb[8] = { -1.58, -3.13, -1.9, -2.95, -2.36, -3.13, -1.9, -3.13 }; /* lower bounds */
    double ub[8] = { 1.58, 3.13, 1.9, 2.95, 2.36, 3.13, 1.9, 3.13 }; /* upper bounds */

    std::vector<const char*> actuator_names{"torso_intv_b1",
                                            "arm_left_intv_1_s",
                                            "arm_left_intv_2_l",
                                            "arm_left_intv_3_e",
                                            "arm_left_intv_4_u",
                                            "arm_left_intv_5_r",
                                            "arm_left_intv_6_b",
                                            "arm_left_intv_7_t"};


    const char* link_name = "left_ee_tip";
    double pose[7];

    int ee_idx = mj_name2id(m, mjOBJ_BODY, link_name);
    std::vector<int> select_dofs;
    std::vector<int> select_qpos;
    std::vector<int> select_actid;
    for (int i=0; i<joint_names.size(); i++)
    {
        int jnt_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        int jnt_qposadr = m->jnt_qposadr[jnt_idx];
        int jnt_dofadr = m->jnt_dofadr[jnt_idx];
        select_qpos.push_back(jnt_qposadr);
        select_dofs.push_back(jnt_dofadr);
        int actid = mj_name2id(m, mjOBJ_ACTUATOR, actuator_names[i]);
        select_actid.push_back(actid);
    }

    for (int i=0; i<select_qpos.size(); i++)
    {
        d->qpos[select_qpos[i]] = q[i];
    }

    mj_forward(m, d);

    // for (int i=0; i<20; i++) mj_step(m, d);  // try to stablize
    std::cout << "after step" << std::endl;

    int ee_goal_bid = mj_name2id(m, mjOBJ_BODY, "ee_goal");
    Matrix4d ee_goal_T;
    mj_to_transform(m, d, ee_goal_bid, ee_goal_T);




    double duration = 2; // seconds

    /* start execution */
    // int n_samples = ceil(duration / m->opt.timestep);
    int traj_idx = 0;
    int step_idx = 0;
    m->opt.integrator = mjtIntegrator::mjINT_RK4;
    // double duration = 5.0;

    double start_time = d->time;

    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        {
            // obtain the current ee pose, and get the pose difference to the target.
            // compute the joint change to move toward the ee pose
            Matrix4d ee_T;
            mj_to_transform(m, d, ee_idx, ee_T);
            Vector6d unit_twist;
            double theta;
            VectorXd dq(8);
            pose_to_twist(ee_T, ee_goal_T, unit_twist, theta);
            std::cout << "twist: " << std::endl;
            std::cout << unit_twist*theta << std::endl;
            unit_twist.head(3) = (ee_goal_T.block<3,1>(0,3) - ee_T.block<3,1>(0,3)) / theta;
            // it seems that this step is necessary to make it robust.
            // This is different from straight-line interpolation, but follows a different path.
            pseudo_inv_ik_vel(unit_twist*theta/duration, ee_idx, m, d, select_dofs, dq);
            // damped_inv_ik_vel(unit_twist*theta/duration, 1e-4, ee_idx, m, d, select_dofs, dq);
            // change the size of dq
            double max_vel = 0.5;
            // double norm = dq.norm();
            double norm = dq.lpNorm<Eigen::Infinity>();
            if (norm < max_vel) max_vel = norm;
            dq = dq/dq.norm()*max_vel;
            double qpos_i[m->nq], qvel_i[m->nv];
            std::cout << "before integrating, qpos: " << std::endl;
            for (int i=0; i<m->nq; i++)
            {
                qpos_i[i] = d->qpos[i];
            }
            std::cout << std::endl;
            for (int i=0; i<select_qpos.size(); i++)
            {
                std::cout << qpos_i[select_qpos[i]] << " ";
            }
            std::cout << std::endl;
            for (int i=0; i<m->nv; i++)
            {
                qvel_i[i] = 0;
            }
            for (int i=0; i<select_dofs.size(); i++)
            {
                qvel_i[select_dofs[i]] = dq[i];
            }
            // for (int i=0; i<7; i++)
            // {
            //     qpos_i[i] = d->qpos[select_qpos[i]];
            //     qvel_i[i] = dq[i];
            // }
            std::cout << "dq: " << std::endl;
            std::cout << dq << std::endl;

            mj_integratePos(m, qpos_i, qvel_i, m->opt.timestep);
            // clipping the updated position
            for (int i=0; i<select_qpos.size(); i++)
            {
                qpos_i[select_qpos[i]] = mju_clip(qpos_i[select_qpos[i]], lb[i], ub[i]);
            }
            // for (int i=0; i<m->nq; i++)
            // {
            //     d->qpos[i] = qpos_i[i];
            // }
            // set the control
            for (int i=0; i<actuator_names.size(); i++)
            {
                d->act[m->actuator_actadr[select_actid[i]]] = qpos_i[select_qpos[i]];
                d->ctrl[select_actid[i]] = dq[i];
            }


        }
        // mj_forward(m, d);
        mj_step(m, d);
        std::cout << "stepping" << std::endl;
        // d->time = d->time + m->opt.timestep;

        step_idx += 1;
        // if (step_idx == 11500) break;

        if (step_idx % 10 != 0) continue;

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