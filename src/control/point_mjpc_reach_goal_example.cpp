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

#include "mjpc/task.h"
#include "mjpc/norm.h"
#include "mujoco/mjmodel.h"
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <random>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <GLFW/glfw3.h>

#include <mjpc/interface.h>
#include <mjpc/task.h>
#include <mjpc/planners/sampling/planner.h>
#include <mjpc/planners/robust/robust_planner.h>
#include <mjpc/states/state.h>
#include <mjpc/threadpool.h>

#include "../utilities/utilities.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/robust/robust_planner.h"
#include "mjpc/planners/sampling/planner.h"

class PointGoalTask : public mjpc::Task
{
  public:
    std::string Name() const override
    {
        return "PointGoalTask";
    }
    std::string XmlPath() const override
    {
        return "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/point_goal_task1.xml";
    }

    class ResidualFn : public mjpc::BaseResidualFn
    {
      public:
        explicit ResidualFn(const PointGoalTask* task) : mjpc::BaseResidualFn(task){this->target_pos = task->target_pos;}
        void set_target(const Vector3d& target_pos) {this->target_pos = target_pos;}
        void Residual(const mjModel* model, const mjData* data, double* residual) const override
        {
            residual[0] = data->qpos[0] - target_pos[0];
            residual[1] = data->qpos[1] - target_pos[1];
            residual[2] = data->qpos[2] - target_pos[2];
            mju_copy(residual+3, data->ctrl, model->nu);  // ctrl is velocity
        }
      protected:
        Vector3d target_pos;
    };

    PointGoalTask()
    {
        /* set parameters for the cartpole task */
        this->num_residual = 6;
        this->num_term = 4; // each one is a term
        this->num_trace = 0;
        this->dim_norm_residual = {1, 1, 1, 3};
        this->norm = {mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic};
        for (int i=0; i<this->num_term; i++)
        {
            this->num_norm_parameter.push_back(mjpc::NormParameterDimension(this->norm[i]));
        }
        // provide the params
        this->weight = {1, 1, 1, 0.1};
        this->norm_parameter = {};
        this->risk = 0.0;
        this->parameters = {};
        residual_ = new ResidualFn(this);
    }

    PointGoalTask(const Vector3d& target_pos) : PointGoalTask()
    {
        this->target_pos = target_pos;
        residual_->set_target(target_pos);
    }

    ~PointGoalTask()
    {
        delete residual_;
        this->residual_ = nullptr;
    }

  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
        return std::make_unique<ResidualFn>(this);
    }
    ResidualFn* InternalResidual() override { return residual_; }

    Vector3d target_pos;

  private:
    ResidualFn* residual_ = nullptr;

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
    std::string filename = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/point_goal_task1.xml";
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


    // std::shared_ptr<mjpc::Task> task = std::make_shared<CartpoleTask>();
    // mjpc::AgentRunner agent(m, task);

    /* setting planner params */
    mjModel* plan_m = mj_copyModel(nullptr, m);
    plan_m->opt.timestep = 0.001;
    plan_m->opt.integrator = mjtIntegrator::mjINT_IMPLICIT;

    int goal_idx = mj_name2id(m, mjOBJ_BODY, "ee_position_goal");
    Vector3d goal_pos(d->xpos[3*goal_idx+0],d->xpos[3*goal_idx+1],d->xpos[3*goal_idx+2]);
    PointGoalTask task(goal_pos);


    // mjpc::SamplingPlanner planner;
    mjpc::RobustPlanner planner(std::make_unique<mjpc::SamplingPlanner>());

    planner.Initialize(plan_m, task);
    planner.Allocate();
    mjpc::State state;
    state.Initialize(plan_m);
    state.Allocate(plan_m);
    state.Reset();

    mjpc::ThreadPool pool(10);

    /* TODO: set the horizon etc params */
    mjtNum total_simstart = d->time;


    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize

    total_simstart = d->time;
    int step_idx = 0;

    // double render_prob = 0.2;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0., 1.);


    // std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();

    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        // if (d->time - total_simstart < 1)
        // auto duration = std::chrono::system_clock::now()-time_now;
        // std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()/1000.0 << std::endl;
        // time_now = std::chrono::system_clock::now();
        {
            if (step_idx % 1 == 0)
            {
                // compute the residual and pub to sensordata
                task.Residual(plan_m, d, d->sensordata);

                state.Set(plan_m, d);
                planner.SetState(state);
                // try to optimize the trajectroy for multiple times

                for (int opt_iter=0; opt_iter<1; opt_iter++)
                {
                    planner.OptimizePolicy(100, pool);
                }
            }
            planner.ActionFromPolicy(d->ctrl, &state.state()[0], d->time);
            std::cout << "execution: " << std::endl;
            std::cout << d->ctrl[0] << " " << d->ctrl[1] << " " << d->ctrl[2] << std::endl;
            // agent.Step(d);
            mj_step(m, d);
            step_idx += 1;
        }

        // // skip some frames
        // if (dis(gen) <= (1-render_prob))
        // {
        //     continue;
        // }
        if (step_idx % 100 != 0) continue;

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