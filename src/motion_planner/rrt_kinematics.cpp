/**
 * @file rrt_kinematics.cpp
 * @author your name (you@domain.com)
 * @brief 
 * implement RRT with kinematics constraints, where the goal is specified as operational space (or task space)
 * constraints on the end-effector. 
 * The goal is defined by a StateSpace of SE(3), which can sample states within that region.
 * Examples of TaskStateSpace includes:
 * - position goal
 * - position + orientation goal
 * - other general goal
 * (the implementation could be to steer from goal, or sample a state in the goal region and steer)
 * RRT needs to be adapted as follows:
 * - during goal sampling, the sampled state is in the task space. Hence need to steer the task space
 * to tree node based on diff IK.
 * - to achieve the above, we also need to store the FK results for each robot joint
 *
 * @version 0.1
 * @date 2024-04-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <ompl/base/SpaceInformation.h>
// #include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/config.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;