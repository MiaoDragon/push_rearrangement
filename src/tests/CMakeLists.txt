find_package(mujoco REQUIRED)
find_package(Eigen3 REQUIRED)


if (mujoco_FOUND)
    message("find mujoco")
else()
    message("cannot find mujoco")
endif()
# link_libraries(mujoco::mujoco Eigen3::Eigen)


set(example_required_libraries
    mujoco::mujoco
    "${mujoco_LIBRARIES}"
    )

set(example_include_dirs 
    "${PROJECT_SOURCE_DIR}"
    "${MJ_INCLUDE_DIR}"
    "/usr/local/include/"
    )


set(example_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )


include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})



add_executable(resize_test resize_mat.cpp)

# set_target_properties(push-mpc-test1 PROPERTIES LINKER_LANGUAGE CXX)
