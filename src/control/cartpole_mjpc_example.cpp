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
#include <mjpc/states/state.h>
#include <mjpc/threadpool.h>

// class CartpoleTask : public mjpc::Task
// {
//   public:
//     std::string Name() const override
//     {
//         return "CartpoleTask";
//     }
//     std::string XmlPath() const override
//     {
//         return "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/cartpole_task.xml";
//     }

//     class ResidualFn : public mjpc::BaseResidualFn
//     {
//       public:
//         explicit ResidualFn(const CartpoleTask* task) : mjpc::BaseResidualFn(task){}
//         void Residual(const mjModel* model, const mjData* data, double* residual) const override
//         {
//             // ---------- Vertical ----------
//             residual[0] = std::cos(data->qpos[1]) - 1;

//             // ---------- Centered ----------
//             residual[1] = -1.0;//data->qpos[0] - parameters_[0];

//             // ---------- Velocity ----------
//             residual[2] = data->qvel[1];

//             // ---------- Control ----------
//             residual[3] = data->ctrl[0];
//             std::cout << "evaluating residual..." << std::endl;
//         }
//     };

//     CartpoleTask() : residual_(this) {}

//   protected:
//     std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
//         return std::make_unique<ResidualFn>(this);
//     }
//     ResidualFn* InternalResidual() override { return &residual_; }

//   private:
//     ResidualFn residual_;

// };

class CartpoleTask : public mjpc::Task
{
  public:
    std::string Name() const override
    {
        return "CartpoleTask";
    }
    std::string XmlPath() const override
    {
        return "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/cartpole_task.xml";
    }

    class ResidualFn : public mjpc::BaseResidualFn
    {
      public:
        explicit ResidualFn(const CartpoleTask* task) : mjpc::BaseResidualFn(task){this->target_pos = task->target_pos;}
        void set_target(double target_pos) {this->target_pos = target_pos;}
        void Residual(const mjModel* model, const mjData* data, double* residual) const override
        {
            // ---------- Vertical ----------
            residual[0] = std::cos(data->qpos[1]) - 1;

            // ---------- Centered ----------
            residual[1] = data->qpos[0] - target_pos;//data->qpos[0] - parameters_[0];

            // ---------- Velocity ----------
            residual[2] = data->qvel[1];

            // ---------- Control ----------
            residual[3] = data->ctrl[0];
        }
      protected:
        double target_pos = 0.0;
    };

    CartpoleTask()
    {
        /* set parameters for the cartpole task */
        this->num_residual = 4;
        this->num_term = 4; // each one is a term
        this->num_trace = 0;
        this->dim_norm_residual = {1, 1, 1, 1};
        this->norm = {mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic, mjpc::NormType::kQuadratic};
        for (int i=0; i<this->num_term; i++)
        {
            this->num_norm_parameter.push_back(mjpc::NormParameterDimension(this->norm[i]));
        }
        // provide the params
        this->weight = {10, 10, 0.1, 0.1};
        this->norm_parameter = {};
        this->risk = 0.0;
        this->parameters = {};
        residual_ = new ResidualFn(this);
    }

    CartpoleTask(double target_pos) : CartpoleTask()
    {
        this->target_pos = target_pos;
        residual_->set_target(target_pos);
    }

    ~CartpoleTask()
    {
        delete residual_;
        this->residual_ = nullptr;
    }

  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
        return std::make_unique<ResidualFn>(this);
    }
    ResidualFn* InternalResidual() override { return residual_; }

    double target_pos = 0.0;

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


    // std::shared_ptr<mjpc::Task> task = std::make_shared<CartpoleTask>();
    // mjpc::AgentRunner agent(m, task);

    /* setting planner params */
    mjModel* plan_m = mj_copyModel(nullptr, m);
    plan_m->opt.timestep = 0.01;
    plan_m->opt.integrator = mjtIntegrator::mjINT_EULER;

    int goal_idx = mj_name2id(m, mjOBJ_BODY, "cart_goal");
    CartpoleTask task(d->xpos[goal_idx*3]);

    mjpc::SamplingPlanner planner;
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

    double render_prob = 0.2;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0., 1.);


    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();

    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        // if (d->time - total_simstart < 1)
        auto duration = std::chrono::system_clock::now()-time_now;
        std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()/1000.0 << std::endl;
        time_now = std::chrono::system_clock::now();
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
            std::cout << d->ctrl[0] << ", " << d->ctrl[1] << std::endl;
            // agent.Step(d);
            mj_step(m, d);
            step_idx += 1;
        }

        // skip some frames
        if (dis(gen) <= (1-render_prob))
        {
            continue;
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