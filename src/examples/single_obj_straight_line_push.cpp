/**
 * push one object from start to goal.
 * Steps:
 * 1. generate object trajectory (position)
 * 2. Option 1: CMGMP. Solve QP to find vel, force for a given contact point
                Assuming robot point contact is sticking.
                Contact point could be selected by heuristics
      Option 2: similar to CMGMP, but also optimize the (x,y,z) contact point location
 *
 * Algos:
 * params: dt, v_max
 * until object reaches target pose:
 * 1. start, goal -> obj vel
 * 2. get contact mode from obj vel. CS of object with workspace is 0, SS depends on vel direction
 * 3. obtain contact point (if not None, then keep using previous contact point)
 *    NOTE: the contact point might need to switch. In this case, we need to find a new contact pt
 * 3. QP(v) -> robot v, obj v
 * 4. step(obj v, robot v)
 *    NOTE: there are two ways to step. (1) make sure contact doesn't change. (2) use vel to simulate
 *
 *
 *
 * Simple algo:
 * 1. start, goal -> obj vel
 * 2. solve for robot sample. Then make sure the robot contact doesn't change, and step
 * This uses the fact that the robot contact is sticking, so every iteration is the same
 */


#include <OsqpEigen/Constants.hpp>
#include <OsqpEigen/Solver.hpp>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>

#include "../constraint/constraint.h"
#include "../contact/contact.h"
#include "../utilities/utilities.h"

#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjvisualize.h>
#include <GLFW/glfw3.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <eiquadprog/eiquadprog.hpp>
// #include "cvxopt.h
// #include <osqp-cpp/osqp.h>
#include "OsqpEigen/OsqpEigen.h"


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

std::vector<Matrix4d> obj_trajectory;
Vector3d robot_in_obj;  // robot contact point
int traj_idx = 0;

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



// Eigen::quaternion: w,x,y,z for init
// mujoco also provides functions such as mju_quat2Mat

double v_max = 0.01;
double r_max = 10*M_PI/180;
double duration = 0;
double dt = 0.01;

/* use the same vel. Solve QP once and use it for all. Keep robot contact unchanged. */
void pose_to_twist(const Matrix4d& start_T, const Matrix4d& goal_T,
                   Vector6d& unit_twist, double& theta)
// TODO: the implementation is wrong. need to use screw theory
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T

    std::cout << "dT: " << std::endl;
    std::cout << goal_T*start_T.inverse() << std::endl;
    SE3_to_twist(goal_T*start_T.inverse(), unit_twist, theta); // unit_twist: [v,w]
    double d_pos = unit_twist.head(3).norm() * theta;
    duration = std::max(d_pos / v_max, theta / r_max);

    theta = theta / duration;  // theta now denotes the velocity
}

void pose_to_twist(const Vector3d& start_p, const Quaterniond& start_r,
                   const Vector3d& goal_p, const Quaterniond& goal_r,
                   Vector6d& unit_twist, double& theta)
// TODO: the implementation is wrong. need to use screw theory
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T
    Vector3d pos, axis;
    double angle;
    Matrix4d start_T, goal_T;
    pos_rot_to_transform(start_p, start_r, start_T);
    pos_rot_to_transform(goal_p, goal_r, goal_T);
    pose_to_twist(start_T, goal_T, unit_twist, theta);

}


int straight_line_plan_1(const Vector3d& robot_pos, Vector6d& sol)
{
    int robot_bid = mj_name2id(m, mjOBJ_BODY, "pointmass");
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

    // set up contacts and modes
    Contacts contacts(m, d);
    std::vector<int> cs_modes;
    std::vector<std::vector<int>> ss_modes;

    // obtain start and goal poses
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

    MatrixXd Ce, Ci;
    VectorXd ce, ci;
    int ce_size, ci_size;

    std::vector<const char*> joint_names{"root_x",
                                        "root_y",
                                        "root_z"};
    std::vector<int> robot_v_indices;
    for (int i=0; i<joint_names.size(); i++)
    {
        int jnt_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        robot_v_indices.push_back(m->jnt_dofadr[jnt_idx]);
    }
    int nobj = 1;
    std::vector<int> obj_body_indices;
    for (int i=0; i<nobj; i++)
    {
        char obj_name[20];
        sprintf(obj_name, "object_%d", i);
        std::cout << "object name: " << obj_name << std::endl;
        obj_body_indices.push_back(mj_name2id(m, mjOBJ_BODY, obj_name));
    }


    MatrixXd Ae, Ai;
    VectorXd ae0, ai0; 
    int ae_size, ai_size;
    total_constraints(m, d, robot_v_indices, obj_body_indices, contacts, cs_modes, ss_modes,
                        Ae, ae0, ae_size, Ai, ai0, ai_size);


    std::cout << "ae_size: " << ae_size << std::endl;
    std::cout << "ai_size: " << ae_size << std::endl;
    std::cout << "Ae: " << std::endl;
    std::cout << Ae << std::endl;
    std::cout << "ae0: " << std::endl;
    std::cout << ae0 << std::endl;
    std::cout << "Ai: " << std::endl;
    std::cout << Ai << std::endl;
    std::cout << "ai0: " << std::endl;
    std::cout << ai0 << std::endl;

    // decision varaible size: Ae.cols()
    int n_vars = Ae.cols();
    // MatrixXd G = MatrixXd::Zero(n_vars, n_vars);
    MatrixXd G = MatrixXd::Identity(n_vars, n_vars);

    VectorXd g0 = VectorXd::Zero(n_vars);
    // MatrixXd Ce = Ae.transpose();
    // VectorXd ce0 = ae0;
    // MatrixXd Ci = Ai.transpose();
    // VectorXd ci0 = ai0;

    std::vector<Vector6d> target_vs;
    // target_vs.push_back(unit_twist*twist_theta);
    Vector6d custom_target_v;
    custom_target_v << 0.01,0,0,0,0,0;
    target_vs.push_back(custom_target_v);

    std::vector<int> active_vs;
    active_vs.push_back(1);

    std::cout << "target_v: " << std::endl;
    std::cout << unit_twist*twist_theta << std::endl;
    vel_objective(m, d, robot_v_indices, target_vs, active_vs, cs_modes.size(), ss_modes[0].size(), G, g0);


    VectorXd x = VectorXd::Zero(n_vars);

    // remove redundant constrs
    remove_linear_redundant_constrs(Ae, ae0);

    // remove_linear_redundant_constrs(Ai, ai0);

    std::cout << "updated AE: " << std::endl;
    std::cout << Ae << std::endl;
    std::cout << "updated ae0: " << std::endl;
    std::cout << ae0 << std::endl;

    eiquadprog::solvers::EiquadprogFast solver;
    int status = solver.solve_quadprog(G, g0, Ae, ae0, Ai, ai0, x);

    // double f = Eigen::solve_quadprog(G, g0, Ae.transpose(), ae0, Ai.transpose(), ai0, x);
    // std::cout << "f: " << f << std::endl;
    // check the result: qdot, v1, v2, ..., vn, C1, C2, ...
    // std::cout << "x: " << std::endl;
    // std::cout << x << std::endl;
    std::cout << "status: " << status << std::endl;

    std::cout << "solution: " << x << std::endl;

    std::cout << "cost: " << 0.5 * x.transpose() * G * x + g0.transpose() * x << std::endl;;

    // obtain the object vel
    sol = x.segment(joint_names.size(), 6);

    /* check if we can solve it by ourselves */
    VectorXd new_x(n_vars);
    int K = 2;
    new_x.setZero();
    new_x[0] = 0.01;
    new_x[3] = 0.01;
    for (int i=0; i<4; i++)
    {
        new_x[3+6+i*2*(1+2*K)+1+1] = 0.01;
        new_x[3+6+i*2*(1+2*K)+1+K*2] = 981/4;
        new_x[3+6+i*2*(1+2*K)+1+K*2+1+1] = 981/4;
    }
    new_x[3+6+4*2*(1+2*K)+1+2*K] = 981;
    // check if this satisfies the constraints
    std::cout << "new_x: " << std::endl;
    std::cout << new_x << std::endl;
    std::cout << "Ae * new_x + ae: " << std::endl;
    std::cout << Ae * new_x + ae0 << std::endl;
    std::cout << "Ai * new_x + ai: " << std::endl;
    std::cout << Ai * new_x + ai0 << std::endl;
    std::cout << "0.5*new_x * G * new_x + g0*new_x: " << std::endl;
    std::cout << 0.5*new_x.transpose()*G*new_x + g0.transpose() * new_x << std::endl;



    /* solve by osqp */
    OsqpEigen::Solver osqp_solver;
    osqp_solver.settings()->setVerbosity(true);
    // osqp_solver.initSolver();
    MatrixXd A(Ae.rows()+Ai.rows(),Ae.cols());
    A << Ae, Ai;

    Eigen::SparseMatrix<double> hessian = G.sparseView();
    Eigen::SparseMatrix<double> linearMatrix = A.sparseView();
    VectorXd lowerBound(Ae.rows()+Ai.rows()), upperBound(Ae.rows()+Ai.rows());
    upperBound.setConstant(OsqpEigen::INFTY);  // a large value
    lowerBound.setZero(); // by default is zero
    lowerBound.head(Ae.rows()) = -ae0;
    upperBound.head(Ae.rows()) = -ae0;
    lowerBound.segment(Ae.rows(),Ai.rows()) = -ai0;

    osqp_solver.data()->setNumberOfVariables(n_vars);
    osqp_solver.data()->setNumberOfConstraints(Ae.rows()+Ai.rows());
    osqp_solver.data()->setHessianMatrix(hessian);
    osqp_solver.data()->setGradient(g0);
    osqp_solver.data()->setLinearConstraintsMatrix(linearMatrix);
    osqp_solver.data()->setLowerBound(lowerBound);
    osqp_solver.data()->setUpperBound(upperBound);
    osqp_solver.initSolver();

    OsqpEigen::ErrorExitFlag flag = osqp_solver.solveProblem();
    if (flag == OsqpEigen::ErrorExitFlag::NoError)
    {
        std::cout << "OSQP no error" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::DataValidationError)
    {
        std::cout << "OSQP data validation error" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::SettingsValidationError)
    {
        std::cout << "OSQP SettingsValidationError" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::LinsysSolverLoadError)
    {
        std::cout << "OSQP LinsysSolverLoadError" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::LinsysSolverInitError)
    {
        std::cout << "OSQP LinsysSolverInitError" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::NonCvxError)
    {
        std::cout << "OSQP NonCvxError" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::MemAllocError)
    {
        std::cout << "OSQP MemAllocError" << std::endl;
    }
    else if (flag == OsqpEigen::ErrorExitFlag::WorkspaceNotInitError)
    {
        std::cout << "OSQP WorkspaceNotInitError" << std::endl;
    }


    VectorXd osqp_sol = osqp_solver.getSolution();
    std::cout << "osqp solution: " << std::endl;
    std::cout << osqp_sol << std::endl;

    return status;


    /* obtain contact information. Set the CS and SS modes  */



    // solve the QP for the twist



}



void sample_robot_pos_loop()
{
    // move the robot to the front of the object
    int obj_target_id = mj_name2id(m, mjOBJ_BODY, "object_0");
    Vector3d obj_target_pos;
    obj_target_pos[0] = d->xpos[3*obj_target_id];
    obj_target_pos[1] = d->xpos[3*obj_target_id+1];
    obj_target_pos[2] = d->xpos[3*obj_target_id+2];

    Vector3d obj_target_half_size(0.04, 0.1, 0.08);
    Vector6d sol;
    Vector3d robot_pos;

    while (true)
    {
        // sample robot position until we find a solution


        // TODO: sample in all faces of the object
        Vector3d ll(obj_target_pos[0]-0.04, obj_target_pos[1]-0.1, obj_target_pos[2]-obj_target_half_size[2]);
        Vector3d ul(obj_target_pos[0]-0.04, obj_target_pos[1]+0.1, obj_target_pos[2]+obj_target_half_size[2]);

        // uniform_sample_3d(ll, ul, robot_pos);


        robot_pos[0] = obj_target_pos[0] - 0.04; //0.7000660902822591; 
        robot_pos[1] = obj_target_pos[1];
        robot_pos[2] = obj_target_pos[2];

        int status = straight_line_plan_1(robot_pos, sol);
        exit(0);

        if (status == 0)
        {
            std::cout << "SUCCESS!" << std::endl;
            std::cout << "robot position: " << std::endl;
            std::cout << robot_pos << std::endl;            
            break;
        }
    }

    // generate trajectory using the object vel
    /**
     * @brief 
     * t time pose: exp([v,w]_x*theta_dot*t) pose_0
     * Need to check: if we iteratively apply exp([v,w]_x*theta_dot*dt) pose, would it be the same?
     */

    Matrix4d obj_start_pose, world_in_obj_start;
    pos_mat_to_transform(d->xpos+3*obj_target_id, d->xmat+9*obj_target_id, obj_start_pose);
    world_in_obj_start = obj_start_pose.inverse();
    robot_in_obj = world_in_obj_start.block<3,3>(0,0) * robot_pos + world_in_obj_start.block<3,1>(0,3);

    obj_trajectory.resize(0);
    obj_trajectory.push_back(obj_start_pose);

    int n_steps = duration / dt;
    dt = duration / n_steps;
    for (int i=0; i<n_steps; i++)
    {
        Vector6d dx = dt * sol;  // [v,w]
        double twist_theta = dx.tail(3).norm();
        // Vector6d unit_twist;
        // if (twist_theta <= 1e-7)
        // {
        //     unit_twist(3) = 0; unit_twist(4) = 0; unit_twist(5) = 1;
        //     // unit_twist;
        // }
        Vector6d unit_twist = dx / twist_theta; // unit twist
        Matrix4d dT;
        twist_to_SE3(unit_twist, twist_theta, dT);
        Matrix4d new_pose = dT*obj_trajectory.back();
        obj_trajectory.push_back(new_pose);
    }
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
    for (int i=0; i<10; i++) mj_step(m, d);  // try to stablize

    // test_contact_constraint();
    // test_total_constraint();
    // test_solve_total_constraint();
    // test_solve_total_constraint_1_loop();
    Vector3d robot_pos(0.7399660902822591, -0.05971188968363312, 1.035);
    sample_robot_pos_loop();

    traj_idx = 0;

    /* visualize the trajectory */
    while (!glfwWindowShouldClose(window))
    {
        if (d->time - total_simstart < 1)
        {
            // set the object trajectory
            Matrix4d obj_pose = obj_trajectory[traj_idx];
            Vector3d robot_pos = obj_pose.block<3,3>(0,0) * robot_in_obj + obj_pose.block<3,1>(0,3);
            std::cout << "robot_pos: " << std::endl;
            std::cout << robot_pos << std::endl;
            int obj_bid = mj_name2id(m, mjOBJ_BODY, "object_0");
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
            int robot_bid = mj_name2id(m, mjOBJ_BODY, "pointmass");
            int robot_jntadr = m->body_jntadr[robot_bid];
            int robot_qposadr = m->jnt_qposadr[robot_jntadr];  // x,y,z

            int qadr1 = m->jnt_qposadr[m->body_jntadr[robot_bid]];
            int qadr2 = m->jnt_qposadr[m->body_jntadr[robot_bid]+1];
            int qadr3 = m->jnt_qposadr[m->body_jntadr[robot_bid]+2];

            d->qpos[qadr1] = robot_pos[0];
            d->qpos[qadr2] = robot_pos[1];
            d->qpos[qadr3] = robot_pos[2];

            // mj_step(m, d);
            mj_forward(m, d);

            traj_idx = (traj_idx + 1) % obj_trajectory.size();

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