#include "inverse_kinematics.h"
#include "mujoco/mjdata.h"
#include "mujoco/mjmodel.h"

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