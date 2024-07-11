# find_package(mujoco REQUIRED)
# find_package(Eigen3 REQUIRED)

# link_libraries(mujoco::mujoco Eigen3::Eigen)


# # find python libraries, for matplotlib
# find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# find_package(PythonLibs 3.0 REQUIRED)
# include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})


set(utilities_required_libraries
    mujoco::mujoco
    "${mujoco_LIBRARIES}"
    "eiquadprog"
    )

set(utilities_include_dirs 
    "${PROJECT_SOURCE_DIR}"
    "${MJ_INCLUDE_DIR}"
    "/usr/local/include/"
    )


set(utilities_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )


include_directories(${utilities_include_dirs})
link_libraries(${utilities_required_libraries})
    
add_library(truncated_normal
    truncated_normal.cpp
)


add_library(push-utils 
    utilities.cpp
    sample.cpp
    spline.cpp
    trajectory.cpp
    )

add_library(motoman-utils 
    motoman_utils.cpp
    )


add_executable(vector_of_vector_test vector_of_vector_test.cpp)
add_executable(utility_test test.cpp)
add_executable(screw_test test2.cpp)
add_executable(sample_test sample_test.cpp)
add_executable(utils_spline_test spline_test.cpp)
add_executable(traj_test traj_test.cpp)



set_target_properties(push-utils PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(truncated_normal PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(motoman-utils PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(utility_test
                    PUBLIC
                    push-utils
                    truncated_normal
)

target_link_libraries(screw_test
                    PUBLIC
                    push-utils
                    truncated_normal
)

target_link_libraries(sample_test
                    PUBLIC
                    push-utils
                    truncated_normal
)

target_link_libraries(utils_spline_test
                    PUBLIC
                    push-utils
                    truncated_normal
)

target_link_libraries(traj_test
                    PUBLIC
                    push-utils
                    truncated_normal
)