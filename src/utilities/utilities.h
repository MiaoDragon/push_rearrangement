#pragma once

#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>


typedef Eigen::Vector3d Vector3d;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Matrix4d Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::MatrixXd MatrixXd;


void hat_operator(const Vector3d& a, Matrix3d& res)
{
    res(0,1) = -a(2); res(0,2) = a(1);
    res(1,0) = a(2);  res(1,2) = -a(0);
    res(2,0) = -a(1); res(2,1) = a(0);
}

void adjoint(Matrix4d& g, Matrix6d& res)
{
    Matrix3d R = g.block<3,3>(0,0);
    res.block<3,3>(0,0) = R;
    Matrix3d temp;
    hat_operator(g.block<3,1>(0,3), temp);
    res.block<3,3>(0,3) = temp*R;
    res.block<3,3>(3,3) = R;
}

void pos_mat_to_transform(double* pos, double* mat, Matrix4d& res)
{
    // given position (3) and rotation matrix (9)
    // obtain transformation matrix
    res(0,3) = pos[0]; res(1,3) = pos[1]; res(2,3) = pos[2];
    res(0,0) = mat[0]; res(0,1) = mat[1]; res(0,2) = mat[2];
    res(1,0) = mat[3]; res(1,1) = mat[4]; res(1,2) = mat[5];
    res(2,0) = mat[6]; res(2,1) = mat[7]; res(2,2) = mat[8];
}



/*****************************************************/
/****************** Mujoco utilities******************/
/*****************************************************/

void mj_to_transform(mjModel* m, mjData* d, int bid, Matrix4d& g)
{
    g(0,3) = d->xpos[bid*3+0];
    g(1,3) = d->xpos[bid*3+1];
    g(2,3) = d->xpos[bid*3+2];

    g(0,0) = d->xmat[bid*3+0]; g(0,1) = d->xmat[bid*3+1]; g(0,2) = d->xmat[bid*3+2];
    g(1,0) = d->xmat[bid*3+3]; g(1,1) = d->xmat[bid*3+4]; g(1,2) = d->xmat[bid*3+5];
    g(2,0) = d->xmat[bid*3+6]; g(2,1) = d->xmat[bid*3+7]; g(2,2) = d->xmat[bid*3+8];
}