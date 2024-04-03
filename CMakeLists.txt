cmake_minimum_required(VERSION 3.16.3)

set (CMAKE_CXX_STANDARD 17)

project(push_rearrangement)
# set(CMAKE_VERBOSE_MAKEFILE ON)


# find_package(mujoco REQUIRED)
# find_package(glfw3 3.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")

add_subdirectory(src/external)

message(STATUS "compile options: ${AVX_COMPILE_OPTIONS}")

add_compile_options(${MJPC_COMPILE_OPTIONS})  # include the external libraries compile option
add_link_options(${MJPC_LINK_OPTIONS})


set(mjpc_libs libmjpc)
set(project_required_libraries
    "${mjpc_libs}"
    "Eigen3::Eigen"
    "osqp"
    OsqpEigen::OsqpEigen
    "eiquadprog"

)

set(project_include_dirs
    "{PROJECT_SOURCE_DIR}/src"
    "{PROJECT_SOURCE_DIR}/src/external/mujoco_mpc"
    )

include_directories(${project_include_dirs})
link_libraries(${project_required_libraries})


# set(MJ_INCLUDE_DIR "$ENV{MJ_PATH}/include")


# find_library(nlopt /usr/local/lib/)
# find_package(OsqpEigen REQUIRED)


# set(example_required_libraries 
#     "${mujoco_LIBRARIES}"
#     "Eigen3::Eigen"
#     # "glfw"
#     "nlopt"
#     "eiquadprog"
#     # "osqp"
#     # "cvxopt"
#     OsqpEigen::OsqpEigen
#     "${mujoco_mpc_libs}"
#     )

# set(example_include_dirs 
#     "${PROJECT_SOURCE_DIR}"
#     # "${MJ_INCLUDE_DIR}"
#     "/usr/local/include/"
#     "${PROJECT_SOURCE_DIR}/src/external/mujoco_mpc"
#     )

# set(example_link_dirs 
#     "${PROJECT_BINARY_DIR}" 
#     )




# include_directories(${example_include_dirs})
# link_libraries(${example_required_libraries})


add_subdirectory(src/utilities)
add_subdirectory(src/contact)
add_subdirectory(src/constraint)
add_subdirectory(src/control)
add_subdirectory(src/motion_planner)
add_subdirectory(src/task_planner)
add_subdirectory(src/tests)
add_subdirectory(src/examples)

# add_executable(nlopt_test src/examples/nlopt_test.cpp)
# add_executable(mujoco_ik_test src/examples/mujoco_ik_test.cpp)
# add_executable(mujoco_mpc_test src/examples/mujoco_mpc_test.cpp)
# add_executable(single_obj_straight_line_push src/examples/single_obj_straight_line_push.cpp)

# add_executable(multi_step_push src/multi_step_push.cpp)

# set(project_link_libraries 
#     push-utils
#     truncated_normal
#     push-contact
#     push-constraint
# )


# target_link_libraries(single_obj_straight_line_push
#                     PUBLIC
#                     ${project_required_libraries}
#                     ${project_link_libraries}
# )