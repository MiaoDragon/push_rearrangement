#include "obj_steer.h"
#include "mujoco/mjmodel.h"
#include "mujoco/mujoco.h"
#include <eiquadprog/eiquadprog-fast.hpp>
#include <memory>
#include <unordered_map>

bool single_obj_steer_ee_position(const mjModel* m, mjData* d,  // d is updated for moving end-effector
                                  const int& obj_body_id,
                                  const Matrix4d& start_T, const Matrix4d& goal_T, 
                                  const Vector3d& ee_contact_in_obj,
                                  Vector6d& obj_twist,
                                  std::shared_ptr<PositionTrajectory>& robot_ee_traj,
                                  std::shared_ptr<PoseTrajectory>& obj_pose_traj)
{
    // NOTES: if multi-thread is to be used, then need to make sure d is mutex-ed

    /* set ee position */
    Vector3d robot_pos = start_T.block<3,3>(0,0)*ee_contact_in_obj + start_T.block<3,1>(0,3);
    // int ee_bid = mj_name2id(m, mjOBJ_BODY, "ee_position");
    // int qadr1 = m->jnt_qposadr[m->body_jntadr[ee_bid]];
    // int qadr2 = m->jnt_qposadr[m->body_jntadr[ee_bid]+1];
    // int qadr3 = m->jnt_qposadr[m->body_jntadr[ee_bid]+2];
    int qadr1 = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_x")];
    int qadr2 = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_y")];
    int qadr3 = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")];

    d->qpos[qadr1] = robot_pos[0];
    d->qpos[qadr2] = robot_pos[1];
    d->qpos[qadr3] = robot_pos[2];

    std::cout << "qadr from body_jntadr: " << qadr1 << " " << qadr2 << " " << qadr3 << std::endl;
    std::cout << "qadr from name: " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_x")];
    std::cout << " " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_y")];
    std::cout << " " << m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ee_position_z")] << std::endl;

    // print out robot position
    std::cout << "robot_pos: " << robot_pos.transpose() << std::endl;
    mj_forward(m, d);
    std::cout << "after mj_forward, qpos: " << d->qpos[qadr1] << " " << d->qpos[qadr2] << " " << d->qpos[qadr3] << std::endl;

    /* obtain contacts */
    FocusedContacts contacts(m, d, {obj_body_id}, 2);
    // std::cout << "after contact" << std::endl;


    /* obtain modes given target twist */
    std::unordered_map<int, Vector6d> twists;
    Vector6d unit_twist;
    double theta;
    pose_to_twist(start_T, goal_T, unit_twist, theta);
    twists[obj_body_id] = unit_twist*theta;

    std::vector<int> cs_modes;
    std::vector<std::vector<int>> ss_modes;
    vel_to_contact_modes(contacts, twists, 2, cs_modes, ss_modes);

    /* construct optimization problem to solve for robot and object velocity */
    std::vector<const char*> joint_names{"ee_position_x",
                                        "ee_position_y",
                                        "ee_position_z"};
    std::vector<int> robot_v_indices;
    for (int i=0; i<joint_names.size(); i++)
    {
        int jnt_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        robot_v_indices.push_back(m->jnt_dofadr[jnt_idx]);
    }

    MatrixXd Ae, Ai;
    VectorXd ae0, ai0; 
    int ae_size, ai_size;
    total_constraints(m, d, robot_v_indices, {obj_body_id}, contacts, cs_modes, ss_modes,
                        Ae, ae0, ae_size, Ai, ai0, ai_size);

    // decision varaible size: Ae.cols()
    int n_vars = Ae.cols();

    // MatrixXd G = MatrixXd::Zero(n_vars, n_vars);
    MatrixXd G = MatrixXd::Identity(n_vars, n_vars);

    VectorXd g0 = VectorXd::Zero(n_vars);

    std::vector<Vector6d> target_vs;
    target_vs.push_back(unit_twist*theta); // here we are computing for one object target_v
    // here the target_v makes sure that at time t=1 it will reach the goal

    std::vector<int> active_vs;
    active_vs.push_back(1);  // for one object

    std::cout << "target_v: " << std::endl;
    std::cout << unit_twist*theta << std::endl;

    vel_objective(m, d, robot_v_indices, target_vs, active_vs, cs_modes.size(), ss_modes[0].size(), G, g0);

    VectorXd x = VectorXd::Zero(n_vars);

    // remove redundant constrs
    remove_linear_redundant_constrs(Ae, ae0);

    eiquadprog::solvers::EiquadprogFast solver;
    int status = solver.solve_quadprog(G, g0, Ae, ae0, Ai, ai0, x);
    // x is the solution: qdot, v1, v2, ..., vn, C1, C2, ...

    solver.reset(0, 0, 0);  // this is important to avoid memory issue

    std::cout << "status: " << status << std::endl;

    // check the result: qdot, v1, v2, ..., vn, C1, C2, ...
    std::cout << "solution: " << x << std::endl;

    std::cout << "cost: " << 0.5 * x.transpose() * G * x + g0.transpose() * x << std::endl;;

    /* post processing after solving the problem */
    if (status != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL)
    {
        // problem is not solved.
        obj_twist = VectorXd::Zero(6);
        robot_ee_traj = nullptr; // nothing here
        obj_pose_traj = nullptr;
        return false; // unsolved
    }
    obj_twist = x.segment(3,6);

    // since we assume that the velocity is so that when t=1 goal is achieved, it is just the delta to goal
    /**
     * @brief
     * UPDATE: April 6, changed from LinearPosition and LinearPose to ScrewPosition and ScrewPose
     * this is more general, since after we solve the velocity, it basically means the position should undergo the same
     * screw motion as the object. Hence we need to use screw motion rather than linear motion.
     * 
     */
    robot_ee_traj = std::make_unique<ScrewPositionTrajectory>(robot_pos, obj_twist);
    Vector6d obj_twist_unit;
    double obj_theta;
    twist_to_unit_twist(obj_twist, obj_twist_unit, obj_theta);
    obj_pose_traj = std::make_unique<ScrewPoseTrajectory>(start_T, obj_twist_unit, obj_theta);
    return true;
}