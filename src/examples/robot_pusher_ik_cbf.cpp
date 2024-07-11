/**
 * @file robot_pusher_ik.cpp
 * @author your name (you@domain.com)
 * @brief 
 * added IK to robot pushing task.
 * 1. steer the robot to the start pose, as a single-step IK method and avoid collisions
 * 2. track the position (and velocity) of the end-effector through diff IK
 * @version 0.1
 * @date 2024-06-27
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

#include <mjpc/interface.h>
#include <mjpc/task.h>
#include <mjpc/planners/sampling/planner.h>
#include <mjpc/planners/cross_entropy/planner.h>
#include <mjpc/planners/gradient/planner.h>
#include <mjpc/planners/robust/robust_planner.h>
#include <mjpc/planners/ilqg/planner.h>
#include <mjpc/planners/sample_gradient/planner.h>
#include <mjpc/planners/sampling/disturb_policy.h>
#include <mjpc/planners/sampling/disturb_planner.h>

#include <mjpc/states/state.h>
#include <mjpc/threadpool.h>
#include <mjpc/norm.h>

#include "../control/inverse_kinematics.h"
#include "../task_planner/obj_steer.h"
#include "../utilities/utilities.h"
#include "../utilities/trajectory.h"

#include "../utilities/motoman_utils.h"

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

void sim_step(GLFWwindow* window)
{
    mjrRect viewport = {0,0,0,0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffer
    glfwSwapBuffers(window);

    glfwPollEvents();    
}

int main(void)
{

    // read mujoco scene
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/task1.xml";
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



    std::vector<int> robot_geoms;
    generate_robot_geoms(m, robot_geoms);
    IntPairSet exclude_pairs;
    IntPairVector collision_pairs;
    generate_exclude_pairs(m, exclude_pairs);
    generate_collision_pairs(m, robot_geoms, exclude_pairs, collision_pairs);


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

    d->qpos[m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")]] = 10; // for stable start


    mj_forward(m, d);
    for (int i=0; i<20; i++) mj_step(m, d);  // try to stablize
    for (int i=0; i<select_qpos.size(); i++)
    {
        d->qpos[select_qpos[i]] = q[i];
    }
    mj_forward(m, d);

    std::cout << "after step" << std::endl;

    int obj_bid = mj_name2id(m, mjOBJ_BODY, "object_0");
    int obj_goal_bid = mj_name2id(m, mjOBJ_BODY, "goal");
    Matrix4d obj_start_T, obj_goal_T;
    mj_to_transform(m, d, obj_bid, obj_start_T);
    mj_to_transform(m, d, obj_goal_bid, obj_goal_T);
    
    Vector3d ee_contact_in_obj;
    ee_contact_in_obj[0] = -0.04-0.005/2;//obj_start_pos[0] - 0.04; //0.7000660902822591; 
    ee_contact_in_obj[1] = 0;//obj_start_pos[1];
    ee_contact_in_obj[2] = -0.07;//obj_start_pos[2];

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

    // move the point robot away
    d->qpos[m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_x")]] = 10; // for stable start
    d->qpos[m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_y")]] = 10; // for stable start
    d->qpos[m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")]] = 10; // for stable start
    d->qvel[m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_x")]] = 0; // for stable start
    d->qvel[m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_y")]] = 0; // for stable start
    d->qvel[m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")]] = 0; // for stable start

    std::cout << "ee_position_x: " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_x")] << std::endl;
    std::cout << "ee_position_y: " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_y")] << std::endl;
    std::cout << "ee_position_z: " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")] << std::endl;
    for (int i=0; i<joint_names.size(); i++)
    {
        std::cout << joint_names[i] << std::endl;
        std::cout << select_qpos[i] << std::endl;
    }
    for (int i=0; i<m->nq; i++)
    {
        std::cout << d->qpos[i] << " ";
    }
    std::cout << std::endl;
    mj_forward(m, d);

    // set robot initial position
    ////////////////////////////////
    Vector3d robot_start_position = obj_start_T.block<3,3>(0,0) * ee_contact_in_obj + obj_start_T.block<3,1>(0,3);
    // IK to the goal

    int obs_id = mj_name2id(m, mjOBJ_GEOM, "shelf_bottom");
    double threshold = 0.1;
    VectorXd dgdq = VectorXd::Zero(8);
    VectorXd dgdx = VectorXd::Zero(3);
    MatrixXd jac_collision = MatrixXd::Zero(3,8);

    while (!glfwWindowShouldClose(window))
    {
        Matrix4d ee_T;
        mj_to_transform(m, d, ee_idx, ee_T);
        // Vector6d unit_twist;
        double theta;
        VectorXd dq(8);
        Vector3d dx = robot_start_position - ee_T.block<3,1>(0,3);
        // damped_inv_ik_position_vel(dx, 1e-4, ee_idx, m, d, select_dofs, dq);
        // damped_inv_ik_position_vel_nullspace(dx, 1e-4, ee_idx, dgdq, m, d, select_dofs, dq);
        double max_angvel = 10.0*M_PI/180;
        std::cout << "before ik_position_cbf.." << std::endl;
        ik_position_cbf(robot_start_position, ee_idx, m, d, 
                        select_qpos, select_dofs, collision_pairs, robot_geoms, max_angvel, dq);
        std::cout << "after ik_position_cbf.." << std::endl;

        // change the size of dq
        double max_vel = 0.8;
        // double norm = dq.norm();
        double norm = dq.lpNorm<Eigen::Infinity>();
        if (norm < max_vel) max_vel = norm;
        dq = dq/dq.norm()*max_vel;
        double qpos_i[m->nq], qvel_i[m->nv];
        for (int i=0; i<m->nq; i++)
        {
            qpos_i[i] = d->qpos[i];
        }
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

        mj_integratePos(m, qpos_i, qvel_i, m->opt.timestep);
        // clipping the updated position
        for (int i=0; i<select_qpos.size(); i++)
        {
            qpos_i[select_qpos[i]] = mju_clip(qpos_i[select_qpos[i]], lb[i], ub[i]);
        }
        for (int i=0; i<m->nq; i++)
        {
            d->qpos[i] = qpos_i[i];
        }
        // set the control
        // for (int i=0; i<actuator_names.size(); i++)
        // {
        //     d->act[m->actuator_actadr[select_actid[i]]] = qpos_i[select_qpos[i]];
        //     d->ctrl[select_actid[i]] = dq[i];
        // }
        mj_forward(m, d);
        sim_step(window);
    }

    // free vis storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free mujoco model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;



    double duration = 10; // seconds

    Vector3d robot_ee_start;
    robot_ee_traj->interpolate(0, robot_ee_start);
    Vector3d robot_ee_end;
    robot_ee_traj->interpolate(1, robot_ee_end);

    std::cout << "robot_ee_start: " << std::endl;
    std::cout << robot_ee_start.transpose() << std::endl;
    std::cout << "robot_ee_end: " << std::endl;
    std::cout << robot_ee_end.transpose() << std::endl;


    mjpc::ThreadPool pool(16);
    ////////////////////////////////


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
        // auto duration = std::chrono::system_clock::now()-time_now;
        // std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()/1000.0 << std::endl;
        // time_now = std::chrono::system_clock::now();


        double action[m->nu];
        std::cout << "nominal action: " << std::endl;
        for (int ai=0; ai<m->nu; ai++)
        {
            std::cout << action[ai] << " ";
        }
        std::cout << std::endl;

        // Below is for trying direct control of robot
        // d->ctrl[0] = robot_ee_v[0]/duration;
        // d->ctrl[1] = 0;
        // d->ctrl[2] = 0;
        std::cout << "actuated control: " << std::endl;
        for (int ui=0; ui<m->nu; ui++)
        {
            std::cout << d->ctrl[ui] << " ";
        }
        std::cout << std::endl;


        Vector3d interp_pos, interp_v;
        double interp_t = std::min(1.0,(d->time-start_time)/duration);
        robot_ee_traj->interpolate(interp_t, interp_pos);
        robot_ee_traj->velocity(interp_t, interp_v);

        // d->ctrl[3] = interp_pos[0];
        // d->ctrl[4] = interp_pos[1];
        // d->ctrl[5] = interp_pos[2];

        mj_step(m, d);
        // std::cout << "inside loop, qpos: " << d->qpos[0] << " " << d->qpos[1] << " " << d->qpos[2] << std::endl;
        // int ee_bid = mj_name2id(m, mjOBJ_BODY, "ee_position");
        // std::cout << "xpos of endeffector: " << d->xpos[3*ee_bid] << " " << d->xpos[3*ee_bid+1] << " " << d->xpos[3*ee_bid+2] << std::endl;
        // std::cout << "nq: " << m->nq << std::endl;
        // for (int j=0; j<m->nq; j++)
        // {
        //     std::cout << d->qpos[j] << " ";
        // }
        // std::cout << std::endl;

        std::cout << "step_idx: " << step_idx << std::endl;
        step_idx += 1;

        if (step_idx % 20 != 0) continue;
        sim_step(window);

    }

    // free vis storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free mujoco model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;




}