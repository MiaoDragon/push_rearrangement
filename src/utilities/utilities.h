#pragma once

#include <vector>
#include <iostream>

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
    res(0,3) = pos[0]; res(1,3) = pos[1]; res(2,3) = pos[2];
    res(0,0) = mat[0]; res(0,1) = mat[1]; res(0,2) = mat[2];
    res(1,0) = mat[3]; res(1,1) = mat[4]; res(1,2) = mat[5];
    res(2,0) = mat[6]; res(2,1) = mat[7]; res(2,2) = mat[8];
    res(3,3) = 1;
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
    Eigen::ColPivHouseholderQR<MatrixXd> qr(A);  // AP = QR (P: permutation of cols of A)
    int rank = qr.rank();
    MatrixXd Q = qr.householderQ();
    // MatrixXd R = qr.matrixR().triangularView<Eigen::Upper>(); //qr.matrixR();
    // obtain the first "rank" columns of Q as the span of the col space of mat_interest
    // ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
    MatrixXd span = Q.block(0,0,Q.rows(),rank);
    res = span;
}