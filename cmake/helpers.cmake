#different functions, etc

function(check_compiler_version)

    set(GCC_MIN_SUPPORTED "4.8")
    set(ICC_MIN_SUPPORTED "15.0")

    if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_MIN_SUPPORTED})
            message(FATAL_ERROR "gcc min supported version is ${GCC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_MIN_SUPPORTED})
            message(FATAL_ERROR "icc min supported version is ${ICC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    else()
        message(WARNING "Compilation with ${CMAKE_C_COMPILER_ID} was not tested, no warranty")
    endif()

endfunction(check_compiler_version)


function(get_vcs_properties VCS)

    if(${VCS} STREQUAL "git")
        # Get the current working branch
        execute_process(COMMAND git rev-parse --abbrev-ref HEAD
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        OUTPUT_VARIABLE GIT_BRANCH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )

        # Get the latest abbreviated commit hash of the working branch
        execute_process(COMMAND git log -1 --format=%h
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        OUTPUT_VARIABLE GIT_COMMIT_HASH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )
        message("-- Git branch: ${GIT_BRANCH}, commit: ${GIT_COMMIT_HASH}")
        set(VCS_INFO "(${GIT_BRANCH}/${GIT_COMMIT_HASH})" PARENT_SCOPE)
    endif()
endfunction(get_vcs_properties)


function(activate_compute_runtime MODULES_PATH COMPUTE_RUNTIME)

    string( TOLOWER "${COMPUTE_RUNTIME}" COMPUTE_RUNTIME)

    set(CCL_ENABLE_SYCL_V 0 PARENT_SCOPE)
    set(CCL_ENABLE_SYCL_L0 0 PARENT_SCOPE)

    message("Search Compute Runtime by MODULES_PATH: ${MODULES_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${MODULES_PATH}")

    if(COMPUTE_RUNTIME STREQUAL "computecpp")
        message ("COMPUTE_RUNTIME=${COMPUTE_RUNTIME} requested. Using ComputeCpp provider")
        SET (COMPUTE_RUNTIME_LOAD_MODULE "ComputeCpp"
                CACHE STRING
             "COMPUTE_RUNTIME=${COMPUTE_RUNTIME} requested. Using ComputeCpp provider")

        find_package(${COMPUTE_RUNTIME_LOAD_MODULE} REQUIRED)

        if(NOT ComputeCpp_FOUND)
            message(FATAL_ERROR "Failed to find ComputeCpp")
        endif()

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_RUNTIME_CXXFLAGS_LOCAL "${COMPUTE_RUNTIME_CXXFLAGS_LOCAL} ${COMPUTECPP_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_RUNTIME_TARGET_NAME Codeplay::ComputeCpp)
        set (COMPUTE_RUNTIME_TARGET_NAME Codeplay::ComputeCpp PARENT_SCOPE)
    endif()

    if(COMPUTE_RUNTIME STREQUAL "dpcpp")
        message ("COMPUTE_RUNTIME=${COMPUTE_RUNTIME} requested. Using DPC++ provider")
        SET (COMPUTE_RUNTIME_LOAD_MODULE "IntelSYCL"
                CACHE STRING
             "COMPUTE_RUNTIME=${COMPUTE_RUNTIME} requested. Using DPC++ provider")

        find_package(${COMPUTE_RUNTIME_LOAD_MODULE} REQUIRED)

        if(NOT IntelSYCL_FOUND)
            message(FATAL_ERROR "Failed to find IntelSYCL")
        endif()

        if(LevelZero_FOUND)
            set(CCL_ENABLE_SYCL_L0 1 PARENT_SCOPE)
        endif()

        set(CCL_ENABLE_SYCL_V 1 PARENT_SCOPE)

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_RUNTIME_CXXFLAGS_LOCAL "${COMPUTE_RUNTIME_CXXFLAGS_LOCAL} ${INTEL_SYCL_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_RUNTIME_TARGET_NAME Intel::SYCL)
        set (COMPUTE_RUNTIME_TARGET_NAME Intel::SYCL PARENT_SCOPE)
    endif()

    if(COMPUTE_RUNTIME STREQUAL "l0")
        SET (COMPUTE_RUNTIME_LOAD_MODULE "L0"
                CACHE STRING
             "COMPUTE_RUNTIME=${COMPUTE_RUNTIME} requested")

        find_package(${COMPUTE_RUNTIME_LOAD_MODULE} REQUIRED)

        if(NOT LevelZero_FOUND)
            message(STATUS "Can not find level-zero")
            return()
        endif()

        # No compiler flags
        set (COMPUTE_RUNTIME_CXXFLAGS_LOCAL "")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_RUNTIME_TARGET_NAME ze_loader)
        set (COMPUTE_RUNTIME_TARGET_NAME ze_loader PARENT_SCOPE)
    endif()

    if (NOT COMPUTE_RUNTIME_TARGET_NAME)
        message(FATAL_ERROR "Failed to find requested compute runtime: ${COMPUTE_RUNTIME}")
    endif()

    # extract target properties
    get_target_property(COMPUTE_RUNTIME_INCLUDE_DIRS_LOCAL
                        ${COMPUTE_RUNTIME_TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(COMPUTE_RUNTIME_LIBRARIES_LOCAL
                        ${COMPUTE_RUNTIME_TARGET_NAME} INTERFACE_LINK_LIBRARIES)

    # set output variables in the parent scope:
    # Only `COMPUTE_RUNTIME_FLAGS` is actually required, because  the other flags are derived from
    # 'target_link_libraries'.
    # For simplicity, set all variables
    set(COMPUTE_RUNTIME_FLAGS        ${COMPUTE_RUNTIME_CXXFLAGS_LOCAL}      PARENT_SCOPE)
    set(COMPUTE_RUNTIME_LIBRARIES    ${COMPUTE_RUNTIME_LIBRARIES_LOCAL}     PARENT_SCOPE)
    set(COMPUTE_RUNTIME_INCLUDE_DIRS ${COMPUTE_RUNTIME_INCLUDE_DIRS_LOCAL}  PARENT_SCOPE)

endfunction(activate_compute_runtime)
