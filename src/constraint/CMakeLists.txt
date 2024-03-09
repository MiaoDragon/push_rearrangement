find_package(Eigen3 REQUIRED)

set(example_required_libraries 
    "${mujoco_LIBRARIES}"
    "glfw"
    "eiquadprog"
    )

set(example_include_dirs 
    "${PROJECT_SOURCE_DIR}" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/eigen3" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/dartsim"
    )

set(example_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )


include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})



add_library(push-mpc-constraint 
    constraint.h
    )

add_executable(constraint_test test.cpp)

set_target_properties(push-mpc-constraint PROPERTIES LINKER_LANGUAGE CXX)

