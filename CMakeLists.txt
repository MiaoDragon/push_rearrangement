cmake_minimum_required(VERSION 3.16.3)

project(push_rearrangement)


find_package(mujoco REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Eigen3 REQUIRED)

if (mujoco_FOUND)
    message("find mujoco")
else()
    message("cannot find mujoco")
endif()
link_libraries(mujoco::mujoco Eigen3::Eigen)
set(MJ_INCLUDE_DIR "$ENV{MJ_PATH}/include")


find_library(nlopt /usr/local/lib/)
find_package(OsqpEigen REQUIRED)


set(example_required_libraries 
    "${mujoco_LIBRARIES}"
    "glfw"
    "nlopt"
    "eiquadprog"
    # "osqp"
    # "cvxopt"
    OsqpEigen::OsqpEigen
    )

set(example_include_dirs 
    "${PROJECT_SOURCE_DIR}"
    "${MJ_INCLUDE_DIR}"
    "/usr/local/include/"
    )

set(example_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    )




include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})


add_subdirectory(src/utilities)
add_subdirectory(src/contact)
add_subdirectory(src/constraint)
add_subdirectory(src/tests)



add_executable(nlopt_test src/examples/nlopt_test.cpp)
add_executable(mujoco_ik_test src/examples/mujoco_ik_test.cpp)
add_executable(mujoco_mpc_test src/examples/mujoco_mpc_test.cpp)
add_executable(single_obj_straight_line_push src/examples/single_obj_straight_line_push.cpp)


# add_executable(multi_step_push src/multi_step_push.cpp)


