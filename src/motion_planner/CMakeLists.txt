find_package(ompl REQUIRED)
if (OMPL_FOUND)
    message("find ompl")
else()
    message("cannot find ompl")
endif()


set(mp_required_libraries 
    "${mujoco_LIBRARIES}"
    "glfw"
    "${OMPL_LIBRARIES}"
    )

set(mp_include_dirs 
    "${PROJECT_SOURCE_DIR}" 
    "${OMPL_INCLUDE_DIRS}"
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/eigen3" 
    # "${PROJECT_SOURCE_DIR}/cmgmp/external/dartsim"
    )

set(mp_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    # "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )


include_directories(${mp_include_dirs})
link_libraries(${mp_required_libraries})



add_executable(ompl_test ompl_test.cpp)

# set_target_properties(push-contact PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(ompl_test
                    PUBLIC
                    ${project_link_libraries}
)