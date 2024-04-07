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

#include "obj_steer.h"
#include "../utilities/utilities.h"
#include "../utilities/trajectory.h"

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
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/point_task1.xml";
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
    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize

    int obj_bid = mj_name2id(m, mjOBJ_BODY, "object_0");
    int obj_goal_bid = mj_name2id(m, mjOBJ_BODY, "goal");
    Matrix4d obj_start_T, obj_goal_T;
    mj_to_transform(m, d, obj_bid, obj_start_T);
    mj_to_transform(m, d, obj_goal_bid, obj_goal_T);
    
    Vector3d ee_contact_in_obj;
    ee_contact_in_obj[0] = -0.04;//obj_start_pos[0] - 0.04; //0.7000660902822591; 
    ee_contact_in_obj[1] = 0;//obj_start_pos[1];
    ee_contact_in_obj[2] = 0;//obj_start_pos[2];

    Vector6d obj_twist;
    std::shared_ptr<PositionTrajectory> robot_ee_traj = nullptr;
    std::shared_ptr<PoseTrajectory> obj_pose_traj = nullptr;

    bool status = single_obj_steer_ee_position(m, d, obj_bid, obj_start_T, obj_goal_T, ee_contact_in_obj, 
                                               obj_twist, robot_ee_traj, obj_pose_traj);
    if (status)
    {
        std::cout << "planning success!" << std::endl;
    }
    else
    {
        std::cout << "planning failed" << std::endl;
    }

    double duration = 5; // seconds
    int n_samples = ceil(duration / m->opt.timestep);
    int traj_idx = 0;

    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        {
            // set the object trajectory
            Matrix4d obj_pose;
            obj_pose_traj->interpolate(1.0/n_samples*traj_idx, obj_pose);
            Vector3d robot_pos;
            std::cout << "interpolation t: " << 1.0/n_samples*traj_idx << std::endl;
            robot_ee_traj->interpolate(1.0/n_samples*traj_idx, robot_pos);
            std::cout << "robot_pos: " << std::endl;
            std::cout << robot_pos << std::endl;

            // std::cout << "robot_pos: " << std::endl;
            // std::cout << robot_pos << std::endl;
            // int obj_jntnum = m->body_jntnum[obj_bid];
            int obj_jntadr = m->body_jntadr[obj_bid];
            int obj_qposadr = m->jnt_qposadr[obj_jntadr];  // x,y,z,qw,qx,qy,qz
            Quaterniond obj_q(obj_pose.block<3,3>(0,0));
            d->qpos[obj_qposadr+0] = obj_pose(0,3);
            d->qpos[obj_qposadr+1] = obj_pose(1,3);
            d->qpos[obj_qposadr+2] = obj_pose(2,3);
            d->qpos[obj_qposadr+3+0] = obj_q.w();
            d->qpos[obj_qposadr+3+1] = obj_q.x();
            d->qpos[obj_qposadr+3+2] = obj_q.y();
            d->qpos[obj_qposadr+3+3] = obj_q.z();

            // robot pose
            int ee_bid = mj_name2id(m, mjOBJ_BODY, "ee_position");
            // int robot_jntadr = m->body_jntadr[robot_bid];
            // int robot_qposadr = m->jnt_qposadr[robot_jntadr];  // x,y,z

            int qadr1 = m->jnt_qposadr[m->body_jntadr[ee_bid]];
            int qadr2 = m->jnt_qposadr[m->body_jntadr[ee_bid]+1];
            int qadr3 = m->jnt_qposadr[m->body_jntadr[ee_bid]+2];

            d->qpos[qadr1] = robot_pos[0];
            d->qpos[qadr2] = robot_pos[1];
            d->qpos[qadr3] = robot_pos[2];

            // mj_step(m, d);
            mj_forward(m, d);

            traj_idx = (traj_idx + 1) % n_samples;

        }
        if (traj_idx % 10 != 0) continue;
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