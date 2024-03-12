#pragma once

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


void hat_operator(const Vector3d& a, Matrix3d& res)
{
    res.setZero();
    res(0,1) = -a(2); res(0,2) = a(1);
    res(1,0) = a(2);  res(1,2) = -a(0);
    res(2,0) = -a(1); res(2,1) = a(0);
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

void pos_mat_to_transform(double* pos, double* mat, Matrix4d& res)
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


void pose_to_transform(const Matrix4d& start_T, const Matrix4d& goal_T,
                       Vector3d& pos, double& angle, Vector3d& axis)
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T
    Matrix4d dT = goal_T * start_T.inverse();
    // make sure bounds on vel are met
    pos = dT.block<3,1>(0,3);
    double angle;
    Vector3d axis;
    rot_to_axis_angle(dT.block<3,3>(0,0), angle, axis);
}

void pose_to_transform(const Vector3d& start_p, const Quaterniond& start_r,
                       const Vector3d& goal_p, const Quaterniond& goal_r,
                       Vector3d& pos, double& angle, Vector3d& axis)
{
    /* obtain the vel from start to goal, expressed in the world frame */
    // R(vel*dt) * start_T = goal_T
    Matrix4d start_T, goal_T;
    pos_rot_to_transform(start_p, start_r, start_T);
    pos_rot_to_transform(goal_p, goal_r, goal_T);
    pose_to_transform(start_T, goal_T, pos, angle, axis);
}


/*****************************************************/
/****************** Mujoco utilities******************/
/*****************************************************/

void mj_to_transform(mjModel* m, mjData* d, int bid, Matrix4d& g)
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
void sort_eigen_matrix(Eigen::MatrixXd& matrix);
void unique_row_matrix(Eigen::MatrixXd& matrix);
void get_col_space_span(const MatrixXd& A, MatrixXd& res);

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
}


/**
 * @brief 
 * 
 * TODO: unit test
 * @param ang_in 
 * @param n_ss_mode 
 * @param ss_mode 
 */

void ang_to_ss_mode(const double ang_in, const int n_ss_mode,
                    std::vector<int>& ss_mode)
{
    ss_mode.resize(n_ss_mode);
    for (int i=0; i<ss_mode.size(); i++) ss_mode[i] = 0;

    // decide on axis. the region is pi/Nc. index in [0,2Nc)
    // obtain the angle of vel
    double ang = ang_in % (2*M_PI);
    // [-pi, pi] -> [0,2pi]
    double ss_ang = M_PI / ss_mode.size();
    // first idx
    int axis0 = ang / (ss_ang);
    int idx0 = axis0 / ss_mode.size();
    int sign0 = 1;
    if (idx0 == 1) sign0 = -1;
    axis0 = axis0 % (ss_mode.size());

    // second idx
    int axis1 = ang / (ss_ang) + 1;
    axis1 = axis1 % (2*ss_mode.size());
    idx1 = axis1 / ss_mode.size();
    int sign1 = 1;
    if (idx1 == 1) sign1 = -1;
    axis1 = axis1 % (ss_mode.size());

    ss_mode[axis0] = sign0;  ss_mode[axis1] = sign1;
}


void vel_to_contact_mode(const Contact* contact,
                         const Vector6d& twist1, const Vector6d& twist2,
                         const int n_ss_mode,
                         int& cs_mode, std::vector<int>& ss_mode)
{
    // if the contact is obj with workspace, then set cs modes and ss modes
    if (((contact->body_type1 == BodyType::OBJECT) && (contact->body_type2 == BodyType::ENV)) ||
        ((contact->body_type1 == BodyType::ENV) && (contact->body_type2 == BodyType::OBJECT)))
    {
        Vector6d twist = twist1 - twist2; // the relative twist in the world frame
        // since the relative twist is in the world frame, the relative vel at the contact point is:
        // w_omega cross g_wc + w_v
        Vector3d w_vel = twist.head<3>() + twist.tail<3>().cross(contact->eigen_pos);
        // obtain the velocity in the contact frame. g_cw * w_vel
        Vector3d c_vel = contact->eigen_frame.inverse() * w_vel;
        ang_to_ss_mode(std::atan2(c_vel[1], c_vel[0]), n_ss_mode, ss_mode);

        cs_mode = 0;
        return;
    }
    // TODO: obj-obj case?

    ss_mode.resize(n_ss_mode);
    for (int i=0; i<ss_mode.size(); i++) ss_mode[i] = 0;
    // robot-obj case: cs_mode = 0, ss_mode=sticking
    if (((contact->body_type1 == BodyType::ROBOT) && (contact->body_type2 == BodyType::OBJECT)) ||
        ((contact->body_type1 == BodyType::OBJECT) && (contact->body_type2 == BodyType::ROBOT)))
    {
        cs_mode = 0;
    }    

    // otherwise cs_mode = 1
    cs_mode = 1;

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
    std::cout << "A: " << std::endl;
    std::cout << A << std::endl;
    Eigen::ColPivHouseholderQR<MatrixXd> qr(A);  // AP = QR (P: permutation of cols of A)
    int rank = qr.rank();
    std::cout << "rank: " << rank << std::endl;
    MatrixXd Q = qr.householderQ();
    // MatrixXd R = qr.matrixR().triangularView<Eigen::Upper>(); //qr.matrixR();
    // obtain the first "rank" columns of Q as the span of the col space of mat_interest
    // ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
    MatrixXd span = Q.block(0,0,Q.rows(),rank);
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