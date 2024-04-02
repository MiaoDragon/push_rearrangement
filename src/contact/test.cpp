#include "contact.h"

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstring>

#include "../utilities/utilities.h"


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
// const double axis_list[3][3] = {{1,0,0},{0,1,0},{0,0,1}};


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
    // move the point to object


    Contacts contacts(m, d);

    // compare the values
    for (int i=0; i<contacts.contacts.size(); i++)
    {
        // std::cout << "contact " << i << std::endl;
        // std::cout << "body_id1: " << contacts.contacts[i]->body_id1 << std::endl;
        // std::cout << "body_id2: " << contacts.contacts[i]->body_id2 << std::endl;
        // std::cout << "body_type1: " << contacts.contacts[i]->body_type1 << std::endl;
        // std::cout << "body_type2: " << contacts.contacts[i]->body_type2 << std::endl;
        // std::cout << "pos: " << contacts.contacts[i]->pos[0] << "," <<
        //                         contacts.contacts[i]->pos[1] << "," <<
        //                         contacts.contacts[i]->pos[2] << std::endl;
        // std::cout << "frame: " << contacts.contacts[i]->frame[0] << "," <<
        //                           contacts.contacts[i]->frame[1] << "," <<
        //                           contacts.contacts[i]->frame[2] << "," <<
        //                           contacts.contacts[i]->frame[3] << "," <<
        //                           contacts.contacts[i]->frame[4] << "," <<
        //                           contacts.contacts[i]->frame[5] << "," <<
        //                           contacts.contacts[i]->frame[6] << "," <<
        //                           contacts.contacts[i]->frame[7] << "," <<
        //                           contacts.contacts[i]->frame[8] << std::endl;

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


void test_ss_mode()
{
    double ang = 0;
    int n_ss_mode = 2;
    std::vector<int> ss_mode;

    ang = 30.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 60.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 120.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 200.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 300.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;


    n_ss_mode = 3;
    ang = 30.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 100.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 150.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;


    ang = 200.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 280.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

    ang = 320.0;
    ang_to_ss_mode(ang/180*M_PI, n_ss_mode, ss_mode);
    std::cout << "ang: " << ang << std::endl;
    std::cout << "ss_mode: " << std::endl;
    for (int i = 0; i<n_ss_mode; i++) std::cout << ss_mode[i] << ", " << std::endl;

}

void test_focused_contacts()
{

    Contacts contacts(m, d);

    std::cout << "normal contacts:" << std::endl;
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


    std::unordered_set<int> obj_body_indices;
    obj_body_indices.insert(mj_name2id(m, mjOBJ_BODY, "object_0"));
    FocusedContacts focused_contacts(m, d, obj_body_indices, -1);  // -1 for ignoring robot

    std::cout << "focused contacts:" << std::endl;
    // compare the values
    for (int i=0; i<focused_contacts.contacts.size(); i++)
    {
        std::cout << "contact " << i << std::endl;
        std::cout << "body_id1: " << focused_contacts.contacts[i]->body_id1 << std::endl;
        std::cout << "body_id2: " << focused_contacts.contacts[i]->body_id2 << std::endl;
        std::cout << "body_type1: " << focused_contacts.contacts[i]->body_type1 << std::endl;
        std::cout << "body_type2: " << focused_contacts.contacts[i]->body_type2 << std::endl;
        std::cout << "pos: " << focused_contacts.contacts[i]->pos[0] << "," <<
                                focused_contacts.contacts[i]->pos[1] << "," <<
                                focused_contacts.contacts[i]->pos[2] << std::endl;
        std::cout << "frame: " << focused_contacts.contacts[i]->frame[0] << "," <<
                                  focused_contacts.contacts[i]->frame[1] << "," <<
                                  focused_contacts.contacts[i]->frame[2] << "," <<
                                  focused_contacts.contacts[i]->frame[3] << "," <<
                                  focused_contacts.contacts[i]->frame[4] << "," <<
                                  focused_contacts.contacts[i]->frame[5] << "," <<
                                  focused_contacts.contacts[i]->frame[6] << "," <<
                                  focused_contacts.contacts[i]->frame[7] << "," <<
                                  focused_contacts.contacts[i]->frame[8] << std::endl;

    }

    FocusedContacts focused_contacts_2(m, d, obj_body_indices, 2);  // 2 for including ee_position


    std::cout << "focused contacts with robot type 2:" << std::endl;
    // compare the values
    for (int i=0; i<focused_contacts_2.contacts.size(); i++)
    {
        std::cout << "contact " << i << std::endl;
        std::cout << "body_id1: " << focused_contacts_2.contacts[i]->body_id1 << std::endl;
        std::cout << "body_id2: " << focused_contacts_2.contacts[i]->body_id2 << std::endl;
        std::cout << "body_type1: " << focused_contacts_2.contacts[i]->body_type1 << std::endl;
        std::cout << "body_type2: " << focused_contacts_2.contacts[i]->body_type2 << std::endl;
        std::cout << "pos: " << focused_contacts_2.contacts[i]->pos[0] << "," <<
                                focused_contacts_2.contacts[i]->pos[1] << "," <<
                                focused_contacts_2.contacts[i]->pos[2] << std::endl;
        std::cout << "frame: " << focused_contacts_2.contacts[i]->frame[0] << "," <<
                                  focused_contacts_2.contacts[i]->frame[1] << "," <<
                                  focused_contacts_2.contacts[i]->frame[2] << "," <<
                                  focused_contacts_2.contacts[i]->frame[3] << "," <<
                                  focused_contacts_2.contacts[i]->frame[4] << "," <<
                                  focused_contacts_2.contacts[i]->frame[5] << "," <<
                                  focused_contacts_2.contacts[i]->frame[6] << "," <<
                                  focused_contacts_2.contacts[i]->frame[7] << "," <<
                                  focused_contacts_2.contacts[i]->frame[8] << std::endl;

    }
}

void test_vel_to_contact_modes()
{
    Matrix4d start_T, goal_T;
    int obj_bid = mj_name2id(m, mjOBJ_BODY, "object_0");
    int goal_bid = mj_name2id(m, mjOBJ_BODY, "goal");
    pos_mat_to_transform(d->xpos+obj_bid*3, d->xmat+obj_bid*9, start_T);
    pos_mat_to_transform(d->xpos+goal_bid*3, d->xmat+goal_bid*9, goal_T);

    Vector6d unit_twist;
    double twist_theta;

    std::cout << "pose to twist:" << std::endl;
    std::cout << "start_T: " << std::endl;
    std::cout << start_T << std::endl;
    std::cout << "goal_T: " << std::endl;
    std::cout << goal_T << std::endl;

    pose_to_twist(start_T, goal_T, unit_twist, twist_theta);
    Contacts contacts(m, d);
    std::vector<int> cs_modes;
    std::vector<std::vector<int>> ss_modes;

    std::cout << "direct for loop" << std::endl;
    // compare the values
    for (int i=0; i<contacts.contacts.size(); i++)
    {
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

        Vector6d twist1 = VectorXd::Zero(6);
        Vector6d twist2 = VectorXd::Zero(6);
        if (contacts.contacts[i]->body_type1 == BodyType::OBJECT)  twist1 = unit_twist*twist_theta;
        if (contacts.contacts[i]->body_type2 == BodyType::OBJECT)  twist2 = unit_twist*twist_theta;

        int cs_mode;
        std::vector<int> ss_mode;
        vel_to_contact_mode(contacts.contacts[i], twist1, twist2, 2, cs_mode, ss_mode);

        std::cout << "cs_mode: " << cs_mode << std::endl;
        std::cout << "ss_mode: " << std::endl;
        for (int j=0; j<ss_mode.size(); j++)  std::cout << ss_mode[j] << ", " << std::endl;

        cs_modes.push_back(cs_mode);
        ss_modes.push_back(ss_mode);
    }


    std::vector<int> new_cs_modes;
    std::vector<std::vector<int>> new_ss_modes;
    int bid = mj_name2id(m, mjOBJ_BODY, "object_0");
    std::unordered_map<int, Vector6d> twists;
    twists[bid] = unit_twist*twist_theta;
    vel_to_contact_modes(contacts, twists, 2, new_cs_modes, new_ss_modes);

    std::cout << "comparing cs_modes and ss_modes: " << std::endl;
    std::cout << "old cs_modes: " << std::endl;
    for (int i=0; i<cs_modes.size(); i++)
    {
        std::cout << cs_modes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "old ss_modes: " << std::endl;
    for (int i=0; i<ss_modes.size(); i++)
    {
        for (int j=0; j<ss_modes[i].size(); j++)
        {
            std::cout << ss_modes[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    std::cout << "new cs_modes: " << std::endl;
    for (int i=0; i<new_cs_modes.size(); i++)
    {
        std::cout << new_cs_modes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "new ss_modes: " << std::endl;
    for (int i=0; i<new_ss_modes.size(); i++)
    {
        for (int j=0; j<new_ss_modes[i].size(); j++)
        {
            std::cout << new_ss_modes[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;



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


    std::vector<const char*> joint_names{"torso_joint_b1",
                                        "arm_left_joint_1_s",
                                        "arm_left_joint_1_s",
                                        "arm_left_joint_2_l",
                                        "arm_left_joint_3_e",
                                        "arm_left_joint_4_u",
                                        "arm_left_joint_5_r",
                                        "arm_left_joint_6_b"};

    // double q[15] = {0, 1.75, 0.8, 0, -0.66, 0, 0, 0, 
    //                 1.75, 0.8, 0, -0.66, 0, 0, 0};

    // double lb[8] = { -1.58, -3.13, -1.9, -2.95, -2.36, -3.13, -1.9, -3.13 }; /* lower bounds */
    // double ub[8] = { 1.58, 3.13, 1.9, 2.95, 2.36, 3.13, 1.9, 3.13 }; /* upper bounds */

    // const char* link_name = "arm_left_link_6_b";
    // double pose[7];


    // obtain target pose from xml
    // int target_bid = mj_name2id(m, mjOBJ_BODY, "target");
    // int jnt_idx = m->body_jntadr[target_bid];
    // int qadr = m->jnt_qposadr[jnt_idx];


    // inverse_kinematics(joint_names, q, link_name, pose);


    mjtNum total_simstart = d->time;

    mj_forward(m, d);

    int obj_target_id = mj_name2id(m, mjOBJ_BODY, "object_0");
    Vector3d obj_target_pos;
    obj_target_pos[0] = d->xpos[3*obj_target_id];
    obj_target_pos[1] = d->xpos[3*obj_target_id+1];
    obj_target_pos[2] = d->xpos[3*obj_target_id+2];

    Vector3d obj_target_half_size(0.04, 0.1, 0.08);
    Vector6d sol;
    Vector3d robot_pos;

    Vector3d ll(obj_target_pos[0]-0.04, obj_target_pos[1]-0.1, obj_target_pos[2]-obj_target_half_size[2]);
    Vector3d ul(obj_target_pos[0]-0.04, obj_target_pos[1]+0.1, obj_target_pos[2]+obj_target_half_size[2]);

    // uniform_sample_3d(ll, ul, robot_pos);


    test_contact();

    test_ss_mode();



    robot_pos[0] = obj_target_pos[0] - 0.04; //0.7000660902822591; 
    robot_pos[1] = obj_target_pos[1];
    robot_pos[2] = obj_target_pos[2];

    int robot_bid = mj_name2id(m, mjOBJ_BODY, "ee_position");
    std::cout << "joint number: " << m->body_jntnum[robot_bid] << std::endl;
    int qadr1 = m->jnt_qposadr[m->body_jntadr[robot_bid]];
    int qadr2 = m->jnt_qposadr[m->body_jntadr[robot_bid]+1];
    int qadr3 = m->jnt_qposadr[m->body_jntadr[robot_bid]+2];


    // int jnt_idx1 = mj_name2id(m, )

    std::cout << "joint type: " << m->jnt_type[m->body_jntadr[robot_bid]] << std::endl;

    d->qpos[qadr1] = robot_pos[0];
    d->qpos[qadr2] = robot_pos[1];
    d->qpos[qadr3] = robot_pos[2];


    mj_forward(m, d);


    test_focused_contacts();

    test_vel_to_contact_modes();


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