/**
 * @file ompl_test.cpp
 * @author your name (you@domain.com)
 * @brief 
 * adapt from https://github.com/ompl/ompl/tree/main/demos/PlanarManipulator
 * @version 0.1
 * @date 2024-03-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <chrono>
#include "mjpc/task.h"
#include "mjpc/norm.h"
#include "mujoco/mjmodel.h"
#include <cmath>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <unistd.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <GLFW/glfw3.h>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/config.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

using namespace Eigen;
using namespace std::chrono;

// class ManipulatorStateSpace : public ob::RealVectorStateSpace
// {
//   public:
//     ManipulatorStateSpace(int n) : ob::RealVectorStateSpace(n){}
// };



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


/* planning related */

class TaskMujocoSVC : public ob::StateValidityChecker
{
  public:
    TaskMujocoSVC(const std::vector<const char*>& joint_names, mjModel* m, ob::SpaceInformation* si) : ob::StateValidityChecker(si)
    {
        this->m = mj_copyModel(NULL, m);
        this->d = mj_makeData(m);
        this->nq = joint_names.size();
        mj_forward(this->m, this->d);

        for (int i=0; i<joint_names.size(); i++)
        {
            int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
            int jnt_qpos = m->jnt_qposadr[joint_idx];
            this->qpos_inds.push_back(jnt_qpos);
        }
        root_id = mj_name2id(m, mjOBJ_BODY, "motoman_base");
    
    }
    TaskMujocoSVC(const std::vector<const char*>& joint_names, mjModel* m, const ob::SpaceInformationPtr& si) : ob::StateValidityChecker(si)
    {
        this->m = mj_copyModel(NULL, m);
        this->d = mj_makeData(m);
        mj_forward(this->m, this->d);

        this->nq = joint_names.size();

        for (int i=0; i<joint_names.size(); i++)
        {
            int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
            int jnt_qpos = m->jnt_qposadr[joint_idx];
            this->qpos_inds.push_back(jnt_qpos);
        }
        root_id = mj_name2id(m, mjOBJ_BODY, "motoman_base");
    }

    ~TaskMujocoSVC() override
    {
        delete this->m;
        delete this->d;
    }
    /**
     * @brief 
     * check self-collision and collision with the environment.
     * collision is true if the contact body contains robot
     * @param state 
     * @return true 
     * @return false 
     */
    bool isValid(const ob::State* state) const override // here we should use mutex, since the data pointed to is still changed
    {
        mj_lock.lock();
        const ob::RealVectorStateSpace::StateType* state_ptr = state->as<ob::RealVectorStateSpace::StateType>();
        for (int i=0; i<nq; i++)
        {
            d->qpos[qpos_inds[i]] = state_ptr->values[i];
        }
        mj_forward(m, d);
        // checking contacts
        for (int i=0; i<d->ncon; i++)
        {
            int b1 = m->body_rootid[m->geom_bodyid[d->contact[i].geom[0]]];
            int b2 = m->body_rootid[m->geom_bodyid[d->contact[i].geom[1]]];
            if ((d->contact[i].dist < 0) & ((b1 == root_id) || (b2 == root_id)))
            {
                // in colilsion
                int bid1 = m->geom_bodyid[d->contact[i].geom[0]];
                const char* name1 = mj_id2name(m, mjOBJ_BODY, bid1);
                int bid2 = m->geom_bodyid[d->contact[i].geom[1]];
                const char* name2 = mj_id2name(m, mjOBJ_BODY, bid2);


                std::cout << "body1: " << name1 << std::endl;
                std::cout << "body2: " << name2 << std::endl;
                std::cout << "in collision" << std::endl;
                mj_lock.unlock();
                return false;
            }
        }
        mj_lock.unlock();
        return true;
    }

    double clearance(const ob::State* state) const override
    {
        // TODO: not sure if we should return the minimum clearance or the maximum
        // Also we need to return the distance in the robot joint space, but not the cartesian space.
        return 0.0;
        // const ob::RealVectorStateSpace::StateType* state_ptr = state->as<ob::RealVectorStateSpace::StateType>();
        // for (int i=0; i<nq; i++)
        // {
        //     d->qpos[qpos_inds[i]] = state_ptr->values[i];
        // }
        // mj_forward(m, d);

        // double res = 
        // // checking contacts
        // for (int i=0; i<d->ncon; i++)
        // {
        //     int b1 = m->body_rootid[m->geom_bodyid[d->contact[i].geom1]];
        //     int b2 = m->body_rootid[m->geom_bodyid[d->contact[i].geom2]];
        //     if ((dist < 0) & ((b1 == root_id) || (b2 == root_id)))
        //     {
        //         // in colilsion
        //         return false;
        //     }
        // }
        // return true;
    }
  protected:
    mutable mjModel* m;
    mutable mjData* d;
    mutable std::mutex mj_lock; // ref: https://github.com/mpflueger/mujoco-ompl
    std::vector<int> qpos_inds;
    int nq = 0;
    int root_id = -1;
};



int main(void)
{
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

    mj_forward(m, d);

    /* setting planner params */
    mjModel* plan_m = mj_copyModel(nullptr, m);
    plan_m->opt.timestep = 0.01;
    plan_m->opt.integrator = mjtIntegrator::mjINT_EULER;


    /* TODO: set the horizon etc params */
    mjtNum total_simstart = d->time;


    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize


    /* planning setup */
    // here we want to plan for the torso and all left arms

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

    for (int i=0; i<joint_names.size(); i++)
    {
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        int jnt_qpos = m->jnt_qposadr[joint_idx];
        d->qpos[jnt_qpos] = q[i];
    }
    mj_forward(m, d);

    auto space(std::make_shared<ob::RealVectorStateSpace>(8));

    int nq = 8;
    ob::RealVectorBounds bounds(nq);
    for (int i=0; i<nq; i++)
    {
        bounds.setLow(i, lb[i]);
        bounds.setHigh(i, ub[i]);
    }
    space->setBounds(bounds);

    // space information
    auto si(std::make_shared<ob::SpaceInformation>(space));
    // set state validity checking for the space
    ob::StateValidityCheckerPtr svc(std::make_shared<TaskMujocoSVC>(joint_names, m, si));
    si->setStateValidityChecker(svc);

    // start state from the current mujoco data
    ob::ScopedState<> start(space);
    std::vector<double> init_q = {0, 1.75, 0.8, 0, -0.66, 0, 0, 0};
    start = init_q;

    // set goal
    ob::ScopedState<> goal(space);
    goal.random();

    // set problem description
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));

    pdef->setStartAndGoalStates(start, goal);

    auto planner(std::make_shared<og::RRTConnect>(si));

    planner->setProblemDefinition(pdef);

    planner->setup();

    // si->printSettings(std::cout);
    // pdef->print(std::cout);

    ob::PlannerStatus solved = planner->ob::Planner::solve(1.0);

    if (solved)
    {
        std::cout << "solved!" << std::endl;
    }
    else
    {
        std::cout << "not solved." << std::endl;
    }

    ob::PathPtr path = pdef->getSolutionPath();

    og::PathGeometric path_geom(*(path->as<og::PathGeometric>()));
    path_geom.interpolate();
    const std::vector<ob::State*> solution = path_geom.getStates();

    std::cout << "solution: " << std::endl;
    for (int i=0; i<solution.size(); i++)
    {
        auto solution_i = solution[i]->as<ob::RealVectorStateSpace::StateType>();
        std::cout << "solution[" << i << "]: " << std::endl;
        for (int j=0; j<nq; j++)
        {
            std::cout << solution_i->values[j] << " ";
        }
        std::cout << std::endl;
    }


    /* simulation */
    total_simstart = d->time;
    int step_idx = 0;

    double update_prob = 0.2;
    // double render_prob = 0.2;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);


    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        // if (d->time - total_simstart < 1)
        // 5 time
        if (dis(gen) <= update_prob)
        {
            // set the robot config
            auto solution_i = solution[step_idx]->as<ob::RealVectorStateSpace::StateType>();
            for (int i=0; i<nq; i++)
            {
                int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
                int jnt_qpos = m->jnt_qposadr[joint_idx];
                d->qpos[jnt_qpos] = solution_i->values[i];                
            }
            mj_forward(m, d);
            step_idx += 1;
            step_idx = step_idx % solution.size();
        }

        // skip some frames
        // if (dis(gen) <= (1-render_prob))
        // {
        //     continue;
        // }

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