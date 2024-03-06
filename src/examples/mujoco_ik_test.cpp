/**
  Testing mujoco inverse kinematics with Motoman.
  Specify the target position/orientation in XML
  select which body to go there. And do optimziation through NLOPT
 */

 /**
use robot to push an object with multiple steps.
First step: 
  sample body and configurations to contact with the object.
Second step:
  IK to filter out invalid samples
Third step:
  IK to find collision-free samples
Fourth step:
  MPC for pushing
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
#include <vector>
#include <string>
#include <cstring>

#include "mujoco_ik_nlopt.cpp"

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

// global vars


// void simulate(std::vector<const char*>& joint_names, double* q, const char* link_name, double pose[7])
// // pose: x y z qw qx qy qz
// {
//     // get the joint ids of the joint names
//     for (int i=0; i<joint_names.size(); i++)
//     {
//         int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
//         int jnt_qpos = m->jnt_qposadr[joint_idx];
//         d->qpos[jnt_qpos] = q[i];
//     }
//     mj_forward(m, d);

//     // get the link id
//     int body_id = mj_name2id(m, mjOBJ_BODY, link_name);

//     // get the pose    
//     pose[0] = d->xpos[body_id*3+0];
//     pose[1] = d->xpos[body_id*3+1];
//     pose[2] = d->xpos[body_id*3+2];
//     pose[3] = d->xquat[body_id*4+0];
//     pose[4] = d->xquat[body_id*4+1];
//     pose[5] = d->xquat[body_id*4+2];
//     pose[6] = d->xquat[body_id*4+3];

// }



/* optimization functions */

// given the joint names, init joint angles, link name and target pose, find the joint angle to achieve the
// void inverse_kinematics(std::vector<const char*>& joint_names, double* q, const char* link_name, double pose[7])
// {
//     // set the robot position at q, and then extract the robot link pose
//     simulate(joint_names, q, link_name, pose);
//     std::cout << "pose: " << pose[0] << ", " << pose[1] << ", " 
//             << pose[2] << ", " << pose[3] << ", " << pose[4] << ", " << pose[5] << ", " << pose[6] << std::endl;
// }



/* Mujoco functions */

int t=0;  // 0: rotate around x-axis. 1, 2 similar
const double axis_list[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
  // w,s,a,d: control the position of target
  // q,e: rotate the target around given axis (changed by t)
  int target_bid = mj_name2id(m, mjOBJ_BODY, "target");
  int jnt_idx = m->body_jntadr[target_bid];
  int qadr = m->jnt_qposadr[jnt_idx];
  double dx = 0.01, dy = 0.01, dz = 0.01;
  if (act==GLFW_PRESS && key==GLFW_KEY_W)
  {
    d->qpos[qadr] += dx;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_S)
  {
    d->qpos[qadr] -= dx;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_A)
  {
    d->qpos[qadr+1] += dy;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_D)
  {
    d->qpos[qadr+1] -= dy;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_Z)
  {
    d->qpos[qadr+2] += dz;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_C)
  {
    d->qpos[qadr+2] -= dz;
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_R)
  {
    t = (t+1) % 3;  // change rotation mode
  }
  double theta = 5*M_PI/180;
  if (act==GLFW_PRESS && key==GLFW_KEY_Q)
  {
    // rotating around x-axis
    double quat[4] = {0};
    mju_axisAngle2Quat(quat, axis_list[t], -theta);
    double res[4] = {0};
    mju_mulQuat(res, d->qpos+qadr+3, quat);
    d->qpos[qadr+3] = res[0];
    d->qpos[qadr+4] = res[1];
    d->qpos[qadr+5] = res[2];
    d->qpos[qadr+6] = res[3];
  }
  if (act==GLFW_PRESS && key==GLFW_KEY_E)
  {
    // rotating around x-axis
    double quat[4] = {0};
    mju_axisAngle2Quat(quat, axis_list[t], theta);
    double res[4] = {0};
    mju_mulQuat(res, d->qpos+qadr+3, quat);
    d->qpos[qadr+3] = res[0];
    d->qpos[qadr+4] = res[1];
    d->qpos[qadr+5] = res[2];
    d->qpos[qadr+6] = res[3];
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

int main(int argc, char *argv[])
{
    // read mujoco scene
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/motoman_ws/src/pracsys_vbnpm/tests/push_trial_4.xml";
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


    std::vector<const char*> joint_names{"torso_joint_b1",
                                        "arm_left_joint_1_s",
                                        "arm_left_joint_1_s",
                                        "arm_left_joint_2_l",
                                        "arm_left_joint_3_e",
                                        "arm_left_joint_4_u",
                                        "arm_left_joint_5_r",
                                        "arm_left_joint_6_b"};

    double q[15] = {0, 1.75, 0.8, 0, -0.66, 0, 0, 0, 
                    1.75, 0.8, 0, -0.66, 0, 0, 0};

    double lb[8] = { -1.58, -3.13, -1.9, -2.95, -2.36, -3.13, -1.9, -3.13 }; /* lower bounds */
    double ub[8] = { 1.58, 3.13, 1.9, 2.95, 2.36, 3.13, 1.9, 3.13 }; /* upper bounds */

    const char* link_name = "arm_left_link_6_b";
    double pose[7];


    // obtain target pose from xml
    int target_bid = mj_name2id(m, mjOBJ_BODY, "target");
    int jnt_idx = m->body_jntadr[target_bid];
    int qadr = m->jnt_qposadr[jnt_idx];


    // inverse_kinematics(joint_names, q, link_name, pose);


    mjtNum total_simstart = d->time;

    inverse_kinematics(joint_names, q, link_name, d->qpos+qadr, lb, ub);

    for (int i=0; i<joint_names.size(); i++)
    {
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        int jnt_qpos = m->jnt_qposadr[joint_idx];
        d->qpos[jnt_qpos] = q[i];
    }
    mj_forward(m, d);



    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        if (d->time - total_simstart < 1)
        {

            // mj_step(m, d);
            mj_forward(m, d);

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
