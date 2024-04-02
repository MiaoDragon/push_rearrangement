#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std::chrono;

void resize1()
{
    int rows = 1000;
    int cols = 1000;

    MatrixXd matrix(rows, cols);
    matrix.setRandom();

    // Resize the matrix to a smaller size
    int new_rows = 500;
    int new_cols = 500;

    // matrix.resize(new_rows, new_cols);
    matrix.conservativeResize(new_rows, new_cols);
    // matrix = matrix.block(0,0,new_rows,new_cols);  // this fails too.
    // matrix = matrix.block(0,0,new_rows,new_cols).eval();  // this fails too.
}

void resize2()
{
    int rows = 1000;
    int cols = 1000;

    MatrixXd matrix(rows, cols);
    matrix.setRandom();


    // Resize the matrix to a smaller size
    int new_rows = 500;
    int new_cols = 500;

    // matrix.resize(new_rows, new_cols);
    // matrix.conservativeResize(new_rows, new_cols);
    // matrix = matrix.block(0,0,new_rows,new_cols);  // this fails too.
    matrix = matrix.block(0,0,new_rows,new_cols).eval();
}


int main() {
    // Define initial size and create MatrixXd
    int rows = 3;
    int cols = 4;
    MatrixXd matrix(rows, cols);

    // Store some data in the matrix
    matrix << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;

    std::cout << "Original matrix:\n" << matrix << std::endl;

    // Resize the matrix to a smaller size
    // int new_rows = 2;
    // int new_cols = 2;

    // matrix.resize(new_rows, new_cols);
    // matrix.conservativeResize(new_rows, new_cols);
    // matrix = matrix.block(0,0,new_rows,new_cols);  // this fails too.
    // matrix = matrix.block(0,0,new_rows,new_cols).eval();
    auto start = high_resolution_clock::now();
    resize1();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "\nResized matrix:\n" << matrix << std::endl;
    std::cout << "resize 1 Resizing took: " << duration.count() << " microseconds" << std::endl;

    start = high_resolution_clock::now();
    resize2();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    std::cout << "\nResized matrix:\n" << matrix << std::endl;
    std::cout << "resize 2 Resizing took: " << duration.count() << " microseconds" << std::endl;


    return 0;
}