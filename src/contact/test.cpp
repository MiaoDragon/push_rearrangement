#include "contact.h"

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
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

int t=0;  // 0: rotate around x-axis. 1, 2 similar
const double axis_list[3][3] = {{1,0,0},{0,1,0},{0,0,1}};


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
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


void test_contact()
{
    Contacts contacts(m, d);

    // compare the values
    for (int i=0; i<contacts.contacts.size(); i++)
    {
        std::cout << "contact " << i << std::endl;
        std::cout << "body_id1: " << contacts.contacts[i]->body_id1 << std::endl;
        std::cout << "body_id2: " << contacts.contacts[i]->body_id2 << std::endl;
        std::cout << "body_type1: " << contacts.contacts[i]->body_type1 << std::endl;
        std::cout << "body_type2: " << contacts.contacts[i]->body_type2 << std::endl;
        std::cout << "pos: " << contacts.contacts[i]->pos[0] << "," <<
                                contacts.contacts[i]->pos[1] << "," <<
                                contacts.contacts[i]->pos[2] << std::endl;
        std::cout << "frame: " << contacts.contacts[i]->frame[0] << "," <<
                                  contacts.contacts[i]->frame[1] << "," <<
                                  contacts.contacts[i]->frame[2] << "," <<
                                  contacts.contacts[i]->frame[3] << "," <<
                                  contacts.contacts[i]->frame[4] << "," <<
                                  contacts.contacts[i]->frame[5] << "," <<
                                  contacts.contacts[i]->frame[6] << "," <<
                                  contacts.contacts[i]->frame[7] << "," <<
                                  contacts.contacts[i]->frame[8] << std::endl;


        std::cout << "Mujoco contact: " << std::endl;
        std::cout << "body_id1: " << m->geom_bodyid[d->contact[i].geom1] << std::endl;
        std::cout << "body_id2: " << m->geom_bodyid[d->contact[i].geom2] << std::endl;
        std::cout << "body_type1: " << mj_id2name(m, mjOBJ_BODY, m->body_rootid[m->geom_bodyid[d->contact[i].geom1]]) << std::endl;
        std::cout << "body_type1: " << mj_id2name(m, mjOBJ_BODY, m->body_rootid[m->geom_bodyid[d->contact[i].geom2]]) << std::endl;
        std::cout << "pos: " << d->contact[i].pos[0] << "," <<
                                d->contact[i].pos[1] << "," <<
                                d->contact[i].pos[2] << std::endl;
        std::cout << "frame: " << d->contact[i].frame[0] << "," <<
                                  d->contact[i].frame[1] << "," <<
                                  d->contact[i].frame[2] << "," <<
                                  d->contact[i].frame[3] << "," <<
                                  d->contact[i].frame[4] << "," <<
                                  d->contact[i].frame[5] << "," <<
                                  d->contact[i].frame[6] << "," <<
                                  d->contact[i].frame[7] << "," <<
                                  d->contact[i].frame[8] << std::endl;

    }
}



int main(void)
{

    // read mujoco scene
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/push_mpc/xmls/task1.xml";
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


    mj_forward(m, d);

    test_contact();


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