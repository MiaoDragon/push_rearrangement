# add_executable(nlopt_test src/examples/nlopt_test.cpp)
# add_executable(mujoco_ik_test src/examples/mujoco_ik_test.cpp)
# add_executable(mujoco_mpc_test src/examples/mujoco_mpc_test.cpp)
add_executable(single_obj_straight_line_push single_obj_straight_line_push.cpp)

# add_executable(multi_step_push src/multi_step_push.cpp)

set(project_link_libraries 
    push-utils
    truncated_normal
    push-contact
    push-constraint
)


target_link_libraries(single_obj_straight_line_push
                    PUBLIC
                    ${project_required_libraries}
                    ${project_link_libraries}
                    push-obj-steer

)


add_executable(point_pusher_mpc point_pusher_mpc.cpp)

target_link_libraries(point_pusher_mpc
                    PUBLIC
                    ${project_required_libraries}
                    ${project_link_libraries}
                    push-obj-steer
)

add_executable(robot_pusher_ik robot_pusher_ik.cpp)

target_link_libraries(robot_pusher_ik
                    PUBLIC
                    ${project_required_libraries}
                    ${project_link_libraries}
                    push-obj-steer
                    push-iks
)

