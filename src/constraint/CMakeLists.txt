# find_package(Eigen3 REQUIRED)

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

set(project_link_libraries 
    push-utils
    truncated_normal
    push-contact
    push-constraint
)


include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})



add_library(push-constraint 
    constraint.cpp
    )

add_executable(constraint_test test.cpp)

set_target_properties(push-constraint PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(constraint_test
                    PUBLIC
                    ${project_link_libraries}
)