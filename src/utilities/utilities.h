#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <random>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>


typedef Eigen::Vector3d Vector3d;
typedef Eigen::Vector<double, 6> Vector6d;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXi VectorXi;

typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Matrix4d Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::MatrixXd MatrixXd;

typedef Eigen::Quaterniond Quaterniond;
typedef Eigen::AngleAxisd AngleAxisd;


void hat_operator(const Vector3d& a, Matrix3d& res);
void hat_operator(const Vector6d& a, Matrix4d& res);
void vee_operator(const Matrix3d& so3, Vector3d& res);
void vee_operator(const Matrix4d& se3, Vector6d &res);

void adjoint(Matrix4d& g, Matrix6d& res);

void pos_mat_to_transform(const double* pos, const double* mat, Matrix4d& res);
void pos_rot_to_transform(const Vector3d& pos, const Quaterniond& ori, Matrix4d& res);
void rot_to_axis_angle(const Matrix3d& transform, double& angle, Vector3d& axis);
void twist_to_unit_twist(const Vector6d& twist, Vector6d& unit_twist, double& theta);

void so3_to_SO3(const Matrix3d& unit_so3, const double& theta, Matrix3d& R);

void se3_to_SE3(const Matrix4d& unit_se3, const double& theta, Matrix4d& T);
void w_to_SO3(const Vector3d& unit_w, const double& theta, Matrix3d& R);
void twist_to_SE3(const Vector6d& unit_twist, const double& theta, Matrix4d& T);
void SO3_to_w(const Matrix3d& R, Vector3d& unit_w, double& theta);
void SO3_to_so3(const Matrix3d& R, Matrix3d& unit_so3, double& theta);

void SE3_to_twist(const Matrix4d& T, Vector6d& unit_twist, double& theta);
void SE3_to_se3(const Matrix4d& T, Matrix4d& unit_se3, double& theta);
void pose_to_rel_transform(const Matrix4d& start_T, const Matrix4d& goal_T,
                           Matrix4d& dT);
void pose_to_rel_transform(const Vector3d& start_p, const Quaterniond& start_r,
                           const Vector3d& goal_p, const Quaterniond& goal_r,
                           Matrix4d& dT);

/* assuming time is in [0,1] to interpolate between start_T and goal_T */
void pose_to_twist(const Matrix4d& start_T, const Matrix4d& goal_T,
                   Vector6d& unit_twist, double& theta);
void pose_to_twist(const Vector3d& start_p, const Quaterniond& start_r,
                   const Vector3d& goal_p, const Quaterniond& goal_r,
                   Vector6d& unit_twist, double& theta);

/*****************************************************/
/****************** Mujoco utilities******************/
/*****************************************************/

void mj_to_transform(const mjModel* m, const mjData* d, int bid, Matrix4d& g);
/* sampling points on an object */
void sample_point(const mjModel* m, const mjData* d, const int gid, Vector3d& pos);
/*****************************************************/
/*************** helper function for opt  ************/
/*****************************************************/

// citation: ChatGPT
void remove_redundant_constrs(Eigen::MatrixXd& A, Eigen::VectorXd& b);
void remove_linear_redundant_constrs(Eigen::MatrixXd& A, Eigen::VectorXd& b);



/*****************************************************/
/****************** handling Eigen  ******************/
/*****************************************************/

bool compare_in_dic_order(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2);
void sort_eigen_matrix(Eigen::MatrixXd& matrix);
void unique_row_matrix(Eigen::MatrixXd& matrix);
// ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
void get_col_space_span(const MatrixXd& A, MatrixXd& res);

/* sampling distribution */
void uniform_sample_3d(const Vector3d& ll, const Vector3d& ul, Vector3d& res);
// void uniform_samples(const VectorXd& ll, const VectorXd& ul,)

bool compare_vector_smaller_eq(const VectorXd& a, const VectorXd& b);