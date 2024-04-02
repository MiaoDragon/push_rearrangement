# find_package(Eigen3 REQUIRED)

set(tp_required_libraries 
    "${mujoco_LIBRARIES}"
    "glfw"
    "eiquadprog"
    )

set(tp_include_dirs 
    "${PROJECT_SOURCE_DIR}" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/eigen3" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/dartsim"
    )

set(tp_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )

set(tp_link_libraries 
    push-utils
    truncated_normal
    push-contact
    push-constraint
)


include_directories(${tp_include_dirs})
link_libraries(${tp_required_libraries})



add_library(push-obj-steer 
    obj_steer.cpp
    )

add_executable(obj_steer_test obj_steer_test.cpp)

set_target_properties(push-obj-steer PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(obj_steer_test
                    PUBLIC
                    ${tp_link_libraries}
                    push-obj-steer
)