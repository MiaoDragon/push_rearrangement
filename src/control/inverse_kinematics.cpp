#include "inverse_kinematics.h"
#include "mujoco/mjdata.h"
#include "mujoco/mjmodel.h"
#include "mujoco/mujoco.h"
#include <eiquadprog/eiquadprog-fast.hpp>
#include <cmath>

/**
 * @brief 
 * given dx = J * dq
 * we find dq = J^{-1} * dx
 * the current robot configuration is given by mujoco data
 *
 * @param twist [v,w]
 * @param link_idx 
 */
void pseudo_inv_ik_vel(const Vector6d& twist, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq)
{
    double jac[m->nv*6]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, jac+3*m->nv, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<6; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // solving the linear system: dx=J*dq
    dq = jac_m.completeOrthogonalDecomposition().solve(twist);
    std::cout << "dq: " << std::endl;
    std::cout << dq << std::endl;
}

void damped_inv_ik_vel(const Vector6d& twist, const double damp, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq)
{
    // ref: Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods
    double jac[m->nv*6]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, jac+3*m->nv, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<6; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // damped LS: dq = J^T(JJ^T+damp^2 I)^{-1}dx
    // first solve (JJ^T+damp^2 I)f=dx
    // then solve dq=J^Tf
    MatrixXd mat = jac_m*jac_m.transpose()+damp*damp*MatrixXd::Identity(6,6);
    VectorXd f = mat.completeOrthogonalDecomposition().solve(twist);
    dq = jac_m.transpose()*f;
}

void pseudo_inv_ik_position_vel(const Vector3d& linear_v, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq)
{
    double jac[m->nv*3]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, nullptr, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // solving the linear system: dx=J*dq
    dq = jac_m.completeOrthogonalDecomposition().solve(linear_v);
}

void damped_inv_ik_position_vel(const Vector3d& linear_v, const double damp, const int link_idx, const mjModel* m, const mjData* d,
                       const std::vector<int>& select_dofs,
                       VectorXd& dq)
{
    // ref: Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods
    double jac[m->nv*3]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, nullptr, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // damped LS: dq = J^T(JJ^T+damp^2 I)^{-1}dx
    // first solve (JJ^T+damp^2 I)f=dx
    // then solve dq=J^Tf
    MatrixXd mat = jac_m*jac_m.transpose()+damp*damp*MatrixXd::Identity(3,3);
    VectorXd f = mat.completeOrthogonalDecomposition().solve(linear_v);
    dq = jac_m.transpose()*f;
}

void pseudo_inv_ik_position_vel_nullspace(const Vector3d& linear_v, const int link_idx, const VectorXd& nullspace_v,
                                            const mjModel* m, const mjData* d,
                                            const std::vector<int>& select_dofs,
                                            VectorXd& dq)
{
    double jac[m->nv*3]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, nullptr, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // solving the linear system: dx=J*dq
    dq = jac_m.completeOrthogonalDecomposition().solve(linear_v);
    // null_v = (I-J^{-1}J)v
    // null_v = v - J^{-1}(Jv)
    VectorXd null_dq = jac_m*nullspace_v;
    null_dq = nullspace_v - jac_m.completeOrthogonalDecomposition().solve(null_dq);
    dq += null_dq;
}

void damped_inv_ik_position_vel_nullspace(const Vector3d& linear_v, const double damp, const int link_idx, const VectorXd& nullspace_v,
                                            const mjModel* m, const mjData* d,
                                            const std::vector<int>& select_dofs,
                                            VectorXd& dq)
{
    // ref: Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods
    double jac[m->nv*3]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, nullptr, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(i,j) = jac[i*m->nv+j];
        }
    }
    // damped LS: dq = J^T(JJ^T+damp^2 I)^{-1}dx
    // first solve (JJ^T+damp^2 I)f=dx
    // then solve dq=J^Tf
    MatrixXd mat = jac_m*jac_m.transpose()+damp*damp*MatrixXd::Identity(3,3);
    VectorXd f = mat.completeOrthogonalDecomposition().solve(linear_v);
    dq = jac_m.transpose()*f;
    // null_v = (I-J^{-1}J)v
    // null_v = v - J^{-1}(Jv)
    VectorXd null_dq = jac_m*nullspace_v;
    null_dq = nullspace_v - jac_m.completeOrthogonalDecomposition().solve(null_dq);
    dq += null_dq;
}

void cbf_constraints(const mjModel* m, const mjData* d, const std::vector<int>& select_qpos, const std::vector<int>& select_dofs,
                     const IntPairVector& collision_pairs, const std::vector<int>& robot_geom_ids, const double max_angvel,
                     MatrixXd& CI, VectorXd& ci0, int& ci_size)
{
    // the QP is of the form
    /**
     * solves the problem
     * min. x' Hess x + 2 g0' x
     * s.t. CE x + ce0 = 0
     *      CI x + ci0 >= 0
     */

    /* obtain the joint limits */
    VectorXd q_lb(select_dofs.size()), q_ub(select_dofs.size());
    for (int i=0; i<select_dofs.size(); i++)
    {
        q_lb(i) = m->jnt_range[m->dof_jntid[select_dofs[i]]*2];
        q_ub(i) = m->jnt_range[m->dof_jntid[select_dofs[i]]*2+1];
    }

    CI.resize(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4, select_dofs.size());
    ci0.resize(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4);
    CI.setZero();
    ci0.setZero();
    ci_size = 0;


    /* collision */
    double min_col_dist = 0.15;
    double alpha = 0.1;
    for (int i=0; i<robot_geom_ids.size(); i++)
    {
        int geom1 = robot_geom_ids[i];
        int geom2 = mj_name2id(m, mjOBJ_GEOM, "shelf_bottom");
        double from_to[6];
        double dist = mj_geomDistance(m, d, geom1, geom2, min_col_dist, from_to);
        if (dist >= min_col_dist) continue;
        // constraint: dh/dp*dp/dq*u+alpha*h(q) >= 0
        // where h(q) is the distance to the shelf
        int bid = m->geom_bodyid[geom1];
        double jac[3*m->nv];
        mj_jac(m, d, jac, nullptr, from_to, bid);
        Vector3d dhdx;
        dhdx << -(from_to[3]-from_to[0]), -(from_to[4]-from_to[1]), -(from_to[5]-from_to[2]);
        dhdx = dhdx / dhdx.norm();
        MatrixXd jac_m(3,select_dofs.size());

        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m(0,j) = jac[0*m->nv+select_dofs[j]];
            jac_m(1,j) = jac[1*m->nv+select_dofs[j]];
            jac_m(2,j) = jac[2*m->nv+select_dofs[j]];
        }
        jac_m = dhdx*jac_m;        

        // constrs.append(jac_collision @ u + alpha * dist >= 0)
        CI.row(ci_size) = jac_m;
        ci0(ci_size) = alpha*dist;
        ci_size += 1;
    }

    /* self collision */
    for (int i=0; i<collision_pairs.size(); i++)
    {
        int geom1 = collision_pairs[i].first;
        int geom2 = collision_pairs[i].second;
        double from_to[6];
        double dist = mj_geomDistance(m, d, geom1, geom2, min_col_dist, from_to);
        if (dist >= min_col_dist) continue;
        int body1 = m->geom_bodyid[geom1];
        int body2 = m->geom_bodyid[geom2];
        double jac1[3*m->nv], jac2[3*m->nv];
        mj_jac(m, d, jac1, nullptr, from_to, body1);
        mj_jac(m, d, jac2, nullptr, from_to+3, body2);
        Vector3d dhdx1, dhdx2;
        dhdx1 << -(from_to[3]-from_to[0]), -(from_to[4]-from_to[1]), -(from_to[5]-from_to[2]);
        dhdx1 = dhdx1 / dhdx1.norm();
        dhdx2 << -(from_to[0]-from_to[3]), -(from_to[1]-from_to[4]), -(from_to[2]-from_to[5]);
        dhdx2 = dhdx2 / dhdx2.norm();
        MatrixXd jac_m1(3,select_dofs.size()), jac_m2(3,select_dofs.size());
        for (int j=0; j<select_dofs.size(); j++)
        {
            jac_m1(0,j) = jac1[0*m->nv+select_dofs[j]];
            jac_m1(1,j) = jac1[1*m->nv+select_dofs[j]];
            jac_m1(2,j) = jac1[2*m->nv+select_dofs[j]];
            jac_m2(0,j) = jac2[0*m->nv+select_dofs[j]];
            jac_m2(1,j) = jac2[1*m->nv+select_dofs[j]];
            jac_m2(2,j) = jac2[2*m->nv+select_dofs[j]];
        }
        Vector3d dhdq;
        dhdq = dhdx1*jac_m1 + dhdx2*jac_m2;
        // constrs.append(dhdq @ u + alpha * dist >= 0)
        CI.row(ci_size) = dhdq;
        ci0(ci_size) = alpha*dist;
        ci_size += 1;
    }


    /* joint limits */
    VectorXd q_selected(select_qpos.size());
    for (int i=0; i<select_qpos.size(); i++)
    {
        q_selected(i) = d->qpos[select_qpos[i]];
    }
    
    CI.block(ci_size, 0, select_dofs.size(), select_dofs.size()) = -MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    ci0.segment(ci_size, select_dofs.size()) = 1.0*(q_ub-q_selected);
    ci_size += select_dofs.size();
    CI.block(ci_size, 0, select_dofs.size(), select_dofs.size()) = MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    ci0.segment(ci_size, select_dofs.size()) = 1.0*(q_selected-q_lb);
    ci_size += select_dofs.size();

    // constrs.append(-u + 1.0*(np.array(ub)-q_selected)>=0)
    // # h(q) = q - q_lb
    // # dh/dq = I
    // constrs.append(u + 1.0*(q_selected-np.array(lb))>=0)

    /* joint velocity limits */
    // max_angvels = np.zeros((len(qvel_ids))) + max_angvel
    // constrs.append(u >= -max_angvels)
    // constrs.append(u <= max_angvels)

    VectorXd max_angvels(select_dofs.size());
    max_angvels = VectorXd::Zero(select_dofs.size()).array() + max_angvel;
    CI.block(ci_size, 0, select_dofs.size(), select_dofs.size()) = MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    ci0.segment(ci_size, select_dofs.size()) = max_angvels;
    ci_size += select_dofs.size();
    CI.block(ci_size, 0, select_dofs.size(), select_dofs.size()) = -MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    ci0.segment(ci_size, select_dofs.size()) = max_angvels;
    ci_size += select_dofs.size();

    CI.conservativeResize(ci_size, select_dofs.size());
    ci0.conservativeResize(ci_size);

    std::cout << "CI: " << std::endl;
    std::cout << CI << std::endl;
    std::cout << "ci0: " << std::endl;
    std::cout << ci0 << std::endl;

}


void ik_pose_cbf(const Matrix4d& goal, const int link_idx, const mjModel* m, const mjData* d,
                 const std::vector<int>& select_qpos, const std::vector<int>& select_dofs, 
                 const IntPairVector& collision_pairs,
                 const std::vector<int>& robot_geom_ids, const double max_angvel,
                 VectorXd& dq)
{
    // the QP is of the form
    /**
     * solves the problem
     * min. x' Hess x + 2 g0' x
     * s.t. CE x + ce0 = 0
     *      CI x + ci0 >= 0
     */

    MatrixXd G = MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    VectorXd g0 = VectorXd::Zero(select_dofs.size());
    MatrixXd CE = MatrixXd::Zero(0, select_dofs.size());
    VectorXd ce0 = VectorXd::Zero(0);
    MatrixXd CI = MatrixXd::Zero(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4, select_dofs.size());
    VectorXd ci0 = VectorXd::Zero(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4);

    /* construct the error terms */
    Matrix4d current;
    mj_to_transform(m, d, link_idx, current);
    Vector6d error;
    error.head(3) = goal.block<3,1>(0,3) - current.block<3,1>(0,3);
    Matrix3d R = goal.block<3,3>(0,0)*current.block<3,3>(0,0).transpose();
    double angle;
    Vector3d axis;
    rot_to_axis_angle(R, angle, axis);
    error.tail(3) = angle*axis;

    double max_error_val = 10.0/180*M_PI;
    if (error.norm() > max_error_val)
    {
        error = error / error.norm() * max_error_val;
    }
    // objective: ||J*u-error||^2 = u^T J^T J u - 2 error^T J u + error^T error
    double jac[m->nv*6]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, jac+3*m->nv, link_idx);
    MatrixXd jac_m(6,select_dofs.size());
    for (int j=0; j<select_dofs.size(); j++)
    {
        jac_m(0,j) = jac[0*m->nv+select_dofs[j]];
        jac_m(1,j) = jac[1*m->nv+select_dofs[j]];
        jac_m(2,j) = jac[2*m->nv+select_dofs[j]];
        jac_m(3,j) = jac[3*m->nv+select_dofs[j]];
        jac_m(4,j) = jac[4*m->nv+select_dofs[j]];
        jac_m(5,j) = jac[5*m->nv+select_dofs[j]];
    }
    G = jac_m.transpose()*jac_m;
    g0 = -2*error.transpose()*jac_m;


    int ci_size = 0;
    /* construct the CBF constraints */
    cbf_constraints(m, d, select_qpos, select_dofs, collision_pairs, robot_geom_ids, max_angvel, CI, ci0, ci_size);

    /* solve the quadratic program */
    VectorXd x(select_dofs.size());
    eiquadprog::solvers::EiquadprogFast solver;
    int status = solver.solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
    // x is the solution: u (qdot)

    solver.reset(0, 0, 0);  // this is important to avoid memory issue

    std::cout << "status: " << status << std::endl;

    // check the result: u (qdot)
    std::cout << "solution: " << x << std::endl;

    std::cout << "cost: " << x.transpose() * G * x + 2 * g0.transpose() * x << std::endl;;

    /* post processing after solving the problem */
    if (status != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL)
    {
        // problem is not solved.
        dq = VectorXd::Zero(select_dofs.size());
    }
    dq = x;
}

void ik_position_cbf(const Vector3d& goal, const int link_idx, const mjModel* m, const mjData* d,
                    const std::vector<int>& select_qpos, const std::vector<int>& select_dofs, 
                    const IntPairVector& collision_pairs,
                    const std::vector<int>& robot_geom_ids, const double max_angvel,
                    VectorXd& dq)
{
    // the QP is of the form
    /**
     * solves the problem
     * min. x' Hess x + 2 g0' x
     * s.t. CE x + ce0 = 0
     *      CI x + ci0 >= 0
     */

    MatrixXd G = MatrixXd::Identity(select_dofs.size(), select_dofs.size());
    VectorXd g0 = VectorXd::Zero(select_dofs.size());
    MatrixXd CE = MatrixXd::Zero(0, select_dofs.size());
    VectorXd ce0 = VectorXd::Zero(0);
    MatrixXd CI = MatrixXd::Zero(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4, select_dofs.size());
    VectorXd ci0 = VectorXd::Zero(robot_geom_ids.size()+collision_pairs.size()+select_dofs.size()*4);

    /* construct the error terms */
    Matrix4d current;
    mj_to_transform(m, d, link_idx, current);
    Vector6d error;
    error.head(3) = goal.block<3,1>(0,3) - current.block<3,1>(0,3);

    double max_error_val = 0.05;
    if (error.norm() > max_error_val)
    {
        error = error / error.norm() * max_error_val;
    }
    // objective: ||J*u-error||^2 = u^T J^T J u - 2 error^T J u + error^T error
    double jac[m->nv*3]; // 6xnv, [jacp,jacr]
    mj_jacBody(m, d, jac, nullptr, link_idx);
    MatrixXd jac_m(3,select_dofs.size());
    for (int j=0; j<select_dofs.size(); j++)
    {
        jac_m(0,j) = jac[0*m->nv+select_dofs[j]];
        jac_m(1,j) = jac[1*m->nv+select_dofs[j]];
        jac_m(2,j) = jac[2*m->nv+select_dofs[j]];
    }
    G = jac_m.transpose()*jac_m;
    g0 = -2*error.transpose()*jac_m;


    int ci_size = 0;
    /* construct the CBF constraints */
    cbf_constraints(m, d, select_qpos, select_dofs, collision_pairs, robot_geom_ids, max_angvel, CI, ci0, ci_size);

    /* solve the quadratic program */
    VectorXd x(select_dofs.size());
    eiquadprog::solvers::EiquadprogFast solver;
    int status = solver.solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
    // x is the solution: u (qdot)

    solver.reset(0, 0, 0);  // this is important to avoid memory issue

    std::cout << "status: " << status << std::endl;

    // check the result: u (qdot)
    std::cout << "solution: " << x << std::endl;

    std::cout << "cost: " << x.transpose() * G * x + 2 * g0.transpose() * x << std::endl;;

    /* post processing after solving the problem */
    if (status != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL)
    {
        // problem is not solved.
        dq = VectorXd::Zero(select_dofs.size());
    }
    dq = x;
}