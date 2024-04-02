#include "utilities.h"

#include <ostream>
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <algorithm>

void test_adjoint()
{
    Matrix4d g;
    g << 1, 0, 0, 0,
         0, 1, 0, 2,
         0, 0, 1, 0,
         0, 0, 0, 0;
    Matrix6d res;
    adjoint(g, res);
}


bool compareInDictionaryOrder(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
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

int test_sort() {
  // Create a matrix
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(3, 4);

  // Sort the rows of the matrix
  std::cout << matrix << std::endl;

  std::vector<Eigen::VectorXd> vec;
  for (int i=0; i<matrix.rows(); i++)
  {
    vec.push_back(matrix.row(i));
  }

  std::sort(vec.begin(), vec.end(), compareInDictionaryOrder);

  for (int i=0; i<matrix.rows(); i++)
  {
    matrix.row(i) = vec[i];
  }
  

    // matrix.rowwise().stableSort(compareInDictionaryOrder);
  // Print the sorted matrix
  std::cout << matrix << std::endl;

  return 0;
}

int test_unique_mat()
{
    Eigen::MatrixXd matrix(5,3);
    matrix << 0, 1, 2,
              0, 1, 2,
              3, 4, 5,
              3, 4, 5,
              3, 4, 5;
    unique_row_matrix(matrix);
    std::cout << "matrix: " << std::endl;
    std::cout << matrix << std::endl;

    return 0;
}

void test_remove_redundant_rows()
{
    Eigen::MatrixXd matrix(5,3);
    matrix << 0, 1, 2,
              3, 4, 5,
              0, 1, 2,
              3, 4, 5,
              3, 4, 5;

    Eigen::VectorXd vec(5);
    vec << 0, 1, 0, 2, 1;
    remove_redundant_constrs(matrix, vec);
    std::cout << "matrix: " << std::endl;
    std::cout << matrix << std::endl;
    std::cout << "vec: " << std::endl;
    std::cout << vec << std::endl;
}

void test_qr_col_pivot()
{
    Eigen::MatrixXd matrix(5,3);
    matrix << 0, 1, 2,
              3, 4, 5,
              0, 1, 2,
              3, 4, 5,
              3, 4, 5;

    Eigen::VectorXd vec(5);
    vec << 0, 1, 0, 2, 1;
    Eigen::MatrixXd P = matrix;
    P.conservativeResize(P.rows(),P.cols()+1);
    P.col(matrix.cols()) = vec;
    MatrixXd mat_interest = P.transpose();
    // use qr
    Eigen::ColPivHouseholderQR<MatrixXd> qr(mat_interest);
    MatrixXd Perm = qr.colsPermutation();
    int rank = qr.rank();
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixR().triangularView<Eigen::Upper>(); //qr.matrixR();
    // obtain the first "rank" columns of Q as the span of the col space of mat_interest
    // ref: https://en.wikipedia.org/wiki/QR_decomposition#Rectangular_matrix
    MatrixXd span = Q.block(0,0,Q.rows(),rank);

    std::cout << "Q: " << std::endl;
    std::cout << Q << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;    
    std::cout << "QR: " << std::endl;
    std::cout << Q*R << std::endl;
    std::cout << "QR from obj: " << std::endl;
    std::cout << qr.matrixQR() << std::endl;  // it seems this is equiv to qr.matrixR()
    std::cout << "Perm: " << std::endl;
    std::cout << Perm << std::endl;
    std::cout << "QRPT: " << std::endl;
    std::cout << Q*R*Perm.transpose() << std::endl;
    std::cout << "matrix: " << std::endl;
    std::cout << mat_interest << std::endl;
    std::cout << "col span of mat_interest: " << std::endl;
    std::cout << span << std::endl;

    // checking if vectors in span lies in the col space of A: check if Ax=v
    for (int i =0; i<span.cols(); i++)
    {
        VectorXd x = qr.solve(span.col(i));
        // check whether it is solved
        span.col(i).isApprox(mat_interest*x);
        std::cout << "mat*x: " << std::endl;
        std::cout << mat_interest*x << std::endl;
        std::cout << "b: " << std::endl;
        std::cout << span.col(i) << std::endl;
    }
}

int main(void)
{
    test_adjoint();

    test_sort();

    test_unique_mat();

    test_remove_redundant_rows();
    
    // test_remove_linear_dep_row();

    test_qr_col_pivot();
}