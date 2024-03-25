find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(absl)

set(MUJOCO_DEP_VERSION_abseil
    fb3621f4f897824c0dbe0615fa94543df6192f30 # LTS 20230802.1
    CACHE STRING "Version of `abseil` to be fetched."
    FORCE
)

# list(APPEND CMAKE_MODULE_PATH "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/src/control")

# include(FindOrFetch)

# findorfetch(
#   USE_SYSTEM_PACKAGE
#   OFF
#   PACKAGE_NAME
#   absl
#   LIBRARY_NAME
#   abseil-cpp
#   GIT_REPO
#   https://github.com/abseil/abseil-cpp.git
#   GIT_TAG
#   ${MUJOCO_DEP_VERSION_abseil}
#   TARGETS
#   absl::core_headers
#   EXCLUDE_FROM_ALL
# )

if (absl_FOUND)
    message("find absl")
else()
    message("cannot find absl")
endif()

set(example_required_libraries 
    "${mujoco_LIBRARIES}"
    "glfw"
    OpenMP::OpenMP_CXX
    absl::base
    # "libmjpc"
    )

set(example_include_dirs 
    "${PROJECT_SOURCE_DIR}" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/eigen3" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/dartsim"
    "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/mujoco_mpc"
    )

set(example_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/mujoco_mpc/build/lib"
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )

set(project_link_libraries 
    push-mpc-utils
    truncated_normal
    push-mpc-contact
    push-mpc-constraint
)

include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})



add_library(mujoco-mppi 
    mujoco_mppi.cpp
    mujoco_mppi_intvel.cpp
    )



add_executable(pendulum_test pendulum_example.cpp)

# add_executable(cartpole_example cartpole_example.cpp)

set_target_properties(mujoco-mppi PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(pendulum_test
                    PUBLIC
                    push-mpc-utils
                    truncated_normal
                    mujoco-mppi
)

target_link_libraries(pendulum_test
                    PUBLIC
                    ${project_link_libraries}
)



add_library(mujoco-mppi-policies 
    policies.cpp
    )
    
set_target_properties(mujoco-mppi-policies PROPERTIES LINKER_LANGUAGE CXX)
