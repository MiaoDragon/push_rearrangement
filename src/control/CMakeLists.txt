# find_package(Eigen3 REQUIRED)
# find_package(OpenMP)

set(control_required_libraries 
    "glfw"
    "${mjpc_libs}"
    # "libmjpc"
    )

set(control_include_dirs 
    "${PROJECT_SOURCE_DIR}/src/control"
    )

set(control_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )

# set(control_link_libraries 
#     push-mpc-utils
#     truncated_normal
#     push-mpc-contact
#     push-mpc-constraint
# )

include_directories(${control_include_dirs})
link_libraries(${control_required_libraries})

add_executable(cartpole_example cartpole_mjpc_example.cpp)
target_link_libraries(cartpole_example
                    PUBLIC
                    ${project_required_libraries}
                    ${control_required_libraries}
)


# add_library(mujoco-mppi 
#     mujoco_mppi.cpp
#     mujoco_mppi_intvel.cpp
#     )


# set_target_properties(mujoco-mppi PROPERTIES LINKER_LANGUAGE CXX)





# add_library(mujoco-mppi-policies 
#     policies.cpp
#     )

# add_library(mujoco-mppi-task 
#     task.cpp
#     )

# add_library(mujoco-mppi-new
#     mppi.cpp
#     )



# set_target_properties(mujoco-mppi-policies PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(mujoco-mppi-policies
#                     PUBLIC
#                     ${project_link_libraries}
# )

# set_target_properties(mujoco-mppi-task PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(mujoco-mppi-task
#                     PUBLIC
#                     ${project_link_libraries}
# )
# set_target_properties(mujoco-mppi-new PROPERTIES LINKER_LANGUAGE CXX)
# target_link_libraries(mujoco-mppi-new
#                     PUBLIC
#                     ${project_link_libraries}
# )




# add_executable(policy_test policy_test.cpp)
# target_link_libraries(policy_test
#                     PUBLIC
#                     ${project_link_libraries}
#                     mujoco-mppi-policies
# )


# add_executable(pendulum_test pendulum_example.cpp)

# # add_executable(cartpole_example cartpole_example.cpp)


# target_link_libraries(pendulum_test
#                     PUBLIC
#                     push-mpc-utils
#                     truncated_normal
#                     mujoco-mppi
#                     mujoco-mppi-policies                    
#                     mujoco-mppi-task                    
#                     mujoco-mppi-new                    
# )

# target_link_libraries(pendulum_test
#                     PUBLIC
#                     ${project_link_libraries}
# )

# add_executable(cartpole_example cartpole_example.cpp)

# target_link_libraries(cartpole_example
#                     PUBLIC
#                     push-mpc-utils
#                     truncated_normal
#                     mujoco-mppi
#                     mujoco-mppi-policies                    
#                     mujoco-mppi-task                    
#                     mujoco-mppi-new                    
# )

# target_link_libraries(cartpole_example
#                     PUBLIC
#                     ${project_link_libraries}
# )
