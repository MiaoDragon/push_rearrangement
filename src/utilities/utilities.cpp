#include <algorithm>
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

#include "utilities.h"

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


void hat_operator(const Vector3d& a, Matrix3d& res)
{
    res.setZero();
    res(0,1) = -a(2); res(0,2) = a(1);
    res(1,0) = a(2);  res(1,2) = -a(0);
    res(2,0) = -a(1); res(2,1) = a(0);
}

void hat_operator(const Vector6d& a, Matrix4d& res)
/* a = [v,w] */
{
    res.setZero();

    Matrix3d R;
    hat_operator(a.tail(3), R);
    res.block<3,3>(0,0) = R;
    res.block<3,1>(0,3) = a.head(3);
}

void vee_operator(const Matrix3d& so3, Vector3d& res)
{
    res[0] = so3(2,1); res[1] = so3(0,2); res[2] = so3(1,0);
}

void vee_operator(const Matrix4d& se3, Vector6d &res)
/* se3 = [v,w]_x */
{
    Vector3d w;
    vee_operator(se3.block<3,3>(0,0), w);
    res.tail(3) = w;
    res.head(3) = se3.block<3,1>(0,3);
}

void adjoint(Matrix4d& g, Matrix6d& res)
{
    Matrix3d R = g.block<3,3>(0,0);
    res.setZero();
    res.block<3,3>(0,0) = R;
    Matrix3d temp;
    hat_operator(g.block<3,1>(0,3), temp);
    res.block<3,3>(0,3) = temp*R;
    res.block<3,3>(3,3) = R;
}

void pos_mat_to_transform(const double* pos, const double* mat, Matrix4d& res)
{
    // given position (3) and rotation matrix (9)
    // obtain transformation matrix
    res.setZero();
    res(0,3) = pos[0]; res(1,3) = pos[1]; res(2,3) = pos[2];
    res(0,0) = mat[0]; res(0,1) = mat[1]; res(0,2) = mat[2];
    res(1,0) = mat[3]; res(1,1) = mat[4]; res(1,2) = mat[5];
    res(2,0) = mat[6]; res(2,1) = mat[7]; res(2,2) = mat[8];
    res(3,3) = 1;
}

void pos_rot_to_transform(const Vector3d& pos, const Quaterniond& ori, Matrix4d& res)
{
    res.setZero();
    res.block<3,3>(0,0) = ori.toRotationMatrix();
    res(3,3) = 1;
    res.block<3,1>(0,3) = pos;
}

void rot_to_axis_angle(const Matrix3d& transform, double& angle, Vector3d& axis)
{
    AngleAxisd axis_angle(transform);
    if (axis_angle.angle() < 0)
    {
        angle = -axis_angle.angle();
        axis = -axis_angle.axis();
    }
    else
    {
        angle = axis_angle.angle();
        axis = axis_angle.axis();
    }
}

void twist_to_unit_twist(const Vector6d& twist, Vector6d& unit_twist, double& theta)
{
    unit_twist.setZero();
    double norm = twist.tail(3).norm();
    if (norm == 0.0)
    {
        // pure translation
        theta = twist.head(3).norm();
        if (theta == 0.0)
        {
            // even the translation has zero value, set theta to zero
        }
        else
        {
            // otherwise obtain the unit translation vector
            unit_twist.head(3) = twist.head(3) / theta;
        }
    }
    else // angular velocity term is nonzero. normalize the angular velocity
    {
        theta = norm;
        unit_twist = twist / theta;
    }
}


void so3_to_SO3(const Matrix3d& unit_so3, const double& theta, Matrix3d& R)
/* obtain exp([w]_x * theta) = R */
// unit_so3 could be zero, when w=0
{
    R = Matrix3d::Identity() + unit_so3*sin(theta) + unit_so3*unit_so3*(1-cos(theta));
}

void se3_to_SE3(const Matrix4d& unit_se3, const double& theta, Matrix4d& T)
/**
 * @brief
 * unit_se3 = [w]_x v
 *            0     0
 * 
 * T = exp([w]_x*theta) p
 *     0                1
 *
 * unit_so3 = 0, when w = 0
 */
{
    Matrix3d R;
    so3_to_SO3(unit_se3.block<3,3>(0,0), theta, R);  // R = exp([w]_x*theta)
    // p = (I-exp([w]_x*theta)) (w x v) + w w^T v theta
    Vector3d w;
    vee_operator(unit_se3.block<3,3>(0,0), w);
    Vector3d v = unit_se3.block<3,1>(0,3);

    if (w.isApproxToConstant(0))
    {
        T.setZero();
        T.block<3,3>(0,0) = Matrix3d::Identity();
        T.block<3,1>(0,3) = v*theta;
        T(3,3) = 1;
    }
    else
    {
        // w != 0
        Vector3d p = (Matrix3d::Identity()- R)*(w.cross(v)) + w*w.transpose()*v*theta;
        T.setZero();
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = p;
        T(3,3) = 1;
    }

}

void w_to_SO3(const Vector3d& unit_w, const double& theta, Matrix3d& R)
/* R = exp(w_x * theta) */
// unit_w could be zero
{
    Matrix3d w_hat;
    hat_operator(unit_w, w_hat);
    so3_to_SO3(w_hat, theta, R);
}

void twist_to_SE3(const Vector6d& unit_twist, const double& theta, Matrix4d& T)
/* unit_twist: [v,w] */
{
    Matrix4d unit_se3;
    hat_operator(unit_twist, unit_se3); // this could handle when unit_twist = 0
    se3_to_SE3(unit_se3, theta, T);
}

void SO3_to_w(const Matrix3d& R, Vector3d& unit_w, double& theta)
{
    rot_to_axis_angle(R, theta, unit_w);
    // when theta = 0, unit_w is still nonzero. 
    // For the sake of SE3, we set unit_w to zero too.

    if (theta == 0.0)
    {
        unit_w.setZero();
    }
}


void SO3_to_so3(const Matrix3d& R, Matrix3d& unit_so3, double& theta)
/* given exp([w]_x * theta), obtain [w]_x * theta */
// unit_so3 = 0 when w = 0
{
    Vector3d w;
    SO3_to_w(R, w, theta);
    hat_operator(w, unit_so3);
}


void SE3_to_twist(const Matrix4d& T, Vector6d& unit_twist, double& theta)
// twist is represented by: unit_twist * theta
// unit twist: [v,w]
// two cases:
// (1) w = 0
// (2) w != =
{
    unit_twist.setZero();
    Vector3d w;
    Matrix3d R = T.block<3,3>(0,0);
    Vector3d d = T.block<3,1>(0,3);
    rot_to_axis_angle(R, theta, w);
    if (theta == 0.0)
    {
        // w = 0
        // T = [I,p;0,1]
        // p = v*theta
        // theta = p.norm()
        // unit_v = p / theta
        theta = d.norm();
        if (theta == 0.0)
        {
            // if p.norm() == 0, then we set twist to zero
        }
        else
        {
            // otherwise, normalize the translation part
            unit_twist.head(3) = d/theta;
        }
    }
    else
    {
        // w != 0
        // p = (I-exp([w]_x * theta)) (w x v) + w w^T v theta
        // v = G^{-1}p, G = (I-exp([w]_x * theta)) [w]_x + ww^T theta
        Matrix3d hat_w;
        hat_operator(w, hat_w);
        Matrix3d G = (Matrix3d::Identity()-R)*hat_w + w*w.transpose()*theta;
        Vector3d v = G.inverse()*T.block<3,1>(0,3);

        unit_twist.head(3) = v;
        unit_twist.tail(3) = w;
    }
}

void SE3_to_se3(const Matrix4d& T, Matrix4d& unit_se3, double& theta)
{
    /* given exp([v,w]_x * theta), obtain [v,w]_x * theta */
    Vector6d unit_twist;
    SE3_to_twist(T, unit_twist, theta);  // twist: [v,w]
    // w could be 0, in which case, v is unit. Otherwise w is unit.
    Matrix3d unit_so3;
    hat_operator(unit_twist.tail(3), unit_so3);
    unit_se3.setZero();
    unit_se3.block<3,3>(0,0) = unit_so3;
    unit_se3.block<3,1>(0,3) = unit_twist.head(3);
}

void pose_to_rel_transform(const Matrix4d& start_T, const Matrix4d& goal_T,
                           Matrix4d& dT)
{
    /* obtain the transformation from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T
    dT = goal_T * start_T.inverse();
}

void pose_to_rel_transform(const Vector3d& start_p, const Quaterniond& start_r,
                           const Vector3d& goal_p, const Quaterniond& goal_r,
                           Matrix4d& dT)
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T
    Matrix4d start_T, goal_T;
    pos_rot_to_transform(start_p, start_r, start_T);
    pos_rot_to_transform(goal_p, goal_r, goal_T);
    pose_to_rel_transform(start_T, goal_T, dT);
}

void pose_to_twist(const Matrix4d& start_T, const Matrix4d& goal_T,
                   Vector6d& unit_twist, double& theta)
// TODO: the implementation is wrong. need to use screw theory
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T

    SE3_to_twist(goal_T*start_T.inverse(), unit_twist, theta); // unit_twist: [v,w]
}

void pose_to_twist(const Vector3d& start_p, const Quaterniond& start_r,
                   const Vector3d& goal_p, const Quaterniond& goal_r,
                   Vector6d& unit_twist, double& theta)
{
    Matrix4d start_T, goal_T;
    pos_rot_to_transform(start_p, start_r, start_T);
    pos_rot_to_transform(goal_p, goal_r, goal_T);
    pose_to_twist(start_T, goal_T, unit_twist, theta);
}


/*****************************************************/
/****************** Mujoco utilities******************/
/*****************************************************/

void mj_to_transform(const mjModel* m, const mjData* d, int bid, Matrix4d& g)
{
    g.setZero();

    g(0,3) = d->xpos[bid*3+0];
    g(1,3) = d->xpos[bid*3+1];
    g(2,3) = d->xpos[bid*3+2];

    g(0,0) = d->xmat[bid*9+0]; g(0,1) = d->xmat[bid*9+1]; g(0,2) = d->xmat[bid*9+2];
    g(1,0) = d->xmat[bid*9+3]; g(1,1) = d->xmat[bid*9+4]; g(1,2) = d->xmat[bid*9+5];
    g(2,0) = d->xmat[bid*9+6]; g(2,1) = d->xmat[bid*9+7]; g(2,2) = d->xmat[bid*9+8];

    g(3,3) = 1;
}

/* sampling points on an object */
void sample_point(mjModel* m, mjData* d, const int gid, Vector3d& pos)
{
    // sample points on the geom gid, and return the postiion as pos in the *world frame*
    if (m->geom_type[gid] == mjGEOM_BOX)
    {
        // obtain each face

    }
}


/*****************************************************/
/*************** helper function for opt  ************/
/*****************************************************/

// citation: ChatGPT
void remove_redundant_constrs(Eigen::MatrixXd& A, Eigen::VectorXd& b)
{
    /**
     * @brief 
     * constraints could be Av+b=0, or Av+b>=0
     * this function removes rows that are identical.
     */
     // put A and b together
    Eigen::MatrixXd P = A;
    P.conservativeResize(Eigen::NoChange, A.cols()+1);
    P.col(A.cols()) = b;
    sort_eigen_matrix(P);
    // P = P.unique();
    unique_row_matrix(P);
    // update A and b
    A = P.block(0,0,P.rows(),P.cols()-1);
    b = P.col(P.cols()-1);
}

void remove_linear_redundant_constrs(Eigen::MatrixXd& A, Eigen::VectorXd& b)
{
    /**
     * @brief 
     * constraints could only be equality: Av+b=0
     * shouldn't work on ineq constraints
     * Given Av+b = 0, we have [A,b][v,1]^T = 0
     * assume ei is the span of the rows of [A,b]
     * then we have the following:
     * [A,b][v,1]^T = 0  =>  ei^T[v,1]^T = 0
     * ei^T[v,1]^T = 0  =>  [A,b][v,1]^T = 0
     * hence we just need to find the span of the rows of [A,b]
     * we can achieve this through QR decomposition
     * ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
     */
     // put A and b together
    Eigen::MatrixXd P = A;
    P.conservativeResize(Eigen::NoChange, A.cols()+1);
    P.col(A.cols()) = b;
    MatrixXd res;
    get_col_space_span(P.transpose(), res);
    P = res.transpose();
    // update A and b
    A = P.block(0,0,P.rows(),P.cols()-1);
    b = P.col(P.cols()-1);
    std::cout << "after removing linear redundant constraints" << std::endl;
}




/*****************************************************/
/****************** handling Eigen  ******************/
/*****************************************************/

bool compare_in_dic_order(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    for (int i = 0; i < vec1.size() && i < vec2.size(); ++i) {
        if (vec1(i) < vec2(i)) {
            return true;
        } else if (vec1(i) > vec2(i)) {
            return false;
        }
    }
    // If all elements are equal up to the length of the shorter vector,
    // the shorter vector is considered less in dictionary order
    return vec1.size() < vec2.size();
}

void sort_eigen_matrix(Eigen::MatrixXd& matrix) 
{
    std::vector<Eigen::VectorXd> vec;
    for (int i=0; i<matrix.rows(); i++)
    {
        vec.push_back(matrix.row(i));
    }

    std::sort(vec.begin(), vec.end(), compare_in_dic_order);
    for (int i=0; i<matrix.rows(); i++)
    {
    matrix.row(i) = vec[i];
    }
}

void unique_row_matrix(Eigen::MatrixXd& matrix)
{
    // return the "unique" of matrix rows. remove consecutive redundant rows.
    int nrows=0, ncols=matrix.cols();
    Eigen::MatrixXd unique_matrix(matrix.rows(), matrix.cols());

    unique_matrix.row(0) = matrix.row(0);
    nrows += 1;
    for (int i=1; i<matrix.rows();i++)
    {
        // if the new row is the same as last one, then do nothing
        if (matrix.row(i) != unique_matrix.row(nrows-1))
        {
            unique_matrix.row(nrows) = matrix.row(i);
            nrows += 1;
        }
    }
    unique_matrix.conservativeResize(nrows, ncols);
    matrix = unique_matrix;
}

// ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
void get_col_space_span(const MatrixXd& A, MatrixXd& res)
{
    // given a matrix A, get the span for the col space of A
    // through QR decomposition
    // std::cout << "A: " << std::endl;
    // std::cout << A << std::endl;
    Eigen::ColPivHouseholderQR<MatrixXd> qr(A);  // AP = QR (P: permutation of cols of A)
    int rank = qr.rank();
    std::cout << "rank: " << rank << std::endl;
    MatrixXd Q = qr.householderQ();
    // MatrixXd R = qr.matrixR().triangularView<Eigen::Upper>(); //qr.matrixR();
    // obtain the first "rank" columns of Q as the span of the col space of mat_interest
    // ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
    MatrixXd span = Q.block(0,0,Q.rows(),rank);
    std::cout << "after getting col_space_span" << std::endl;
    res = span;
}


/* sampling distribution */
void uniform_sample_3d(const Vector3d& ll, const Vector3d& ul, Vector3d& res)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // randomly sample a vector uniformly in range [ll, ul]
    for (int i=0; i<3; i++)
    {
        std::uniform_real_distribution<double> dis(ll[i], ul[i]);
        res[i] = dis(gen);
    }
}
// void uniform_samples(const VectorXd& ll, const VectorXd& ul,)


bool compare_vector_smaller_eq(const VectorXd& a, const VectorXd& b)
{
    for (int i=0; i<std::min(a.size(),b.size()); i++)
    {
        if (a[i] > b[i]) return false;
    }
    return true;
}