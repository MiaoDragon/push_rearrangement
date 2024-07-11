/**
 * @file motoman_utils.h
 * @author your name (you@domain.com)
 * @brief 
 * provide helper functions for motoman.
 * @version 0.1
 * @date 2024-07-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include "utilities.h"

void generate_robot_geoms(const mjModel* m, std::vector<int>& robot_geoms);
void generate_exclude_pairs(const mjModel* m, IntPairSet& exclude_pairs);
void generate_collision_pairs(const mjModel* m,
                              const std::vector<int>& robot_geoms, const IntPairSet& exclude_pairs,
                              IntPairVector& collision_pairs);
