#include "utilities.h"

#include <ostream>
#include <vector>
#include <deque>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <algorithm>

void create_deque(std::deque<int>& res)
{
    std::deque<int> b;
    b.push_back(1);
    b.push_back(2);
    res = b;
}

void test_vector_of_deque()
{
    std::vector<std::deque<int>> as;
    as.resize(5);
    for (int i=0; i<5; i++)
    {
        std::deque<int> b;
        create_deque(b);
        as[i] = b;
    }
    // access data
    for (int i=0; i<as.size(); i++)
    {
        for (int j=0; j<as[i].size(); j++)
        {
            std::cout << "as[" << i << "][" << "j" << "]: " << as[i][j] << std::endl;
        }
    }
}

int main(void)
{
    test_vector_of_deque();


}