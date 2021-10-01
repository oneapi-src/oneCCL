#different functions, etc

function(set_lp_env)

    set(GCC_BF16_MIN_SUPPORTED "4.9.0")
    set(GCC_BF16_AVX512BF_MIN_SUPPORTED "10.0.0")
    set(ICC_BF16_AVX512BF_MIN_SUPPORTED "19.1.0")
    set(CLANG_BF16_MIN_SUPPORTED "9.0.0")
    set(CLANG_BF16_AVX512BF_MIN_SUPPORTED "9.3.0")

    if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_BF16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BF16_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_BF16_COMPILER)
        set(CCL_BF16_COMPILER ON)
    else()
        set(CCL_BF16_COMPILER OFF)
    endif()
    message(STATUS "BF16 compiler: ${CCL_BF16_COMPILER}")

    if ((${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_BF16_AVX512BF_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_BF16_AVX512BF_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BF16_AVX512BF_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_BF16_AVX512BF_COMPILER)
        set(CCL_BF16_AVX512BF_COMPILER ON)
    else()
        set(CCL_BF16_AVX512BF_COMPILER OFF)
    endif()
    message(STATUS "BF16 AVX512BF compiler: ${CCL_BF16_AVX512BF_COMPILER}")

    if (CCL_BF16_COMPILER)
        if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
            add_definitions(-DCCL_BF16_TARGET_ATTRIBUTES)
            set(CCL_BF16_TARGET_ATTRIBUTES ON)
        else()
            set(CCL_BF16_TARGET_ATTRIBUTES OFF)
        endif()
        message(STATUS "BF16 target attributes: ${CCL_BF16_TARGET_ATTRIBUTES}")
    endif()

    option(CCL_BF16_GPU_TRUNCATE "Truncate BF16 in GPU operations" ON)
    if (CCL_BF16_GPU_TRUNCATE)
        add_definitions(-DCCL_BF16_GPU_TRUNCATE)
    endif()
    message(STATUS "BF16 GPU truncate: ${CCL_BF16_GPU_TRUNCATE}")


    set(GCC_FP16_MIN_SUPPORTED "4.9.0")
    set(CLANG_FP16_MIN_SUPPORTED "9.0.0")

    if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel"
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_FP16_MIN_SUPPORTED})
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"
            AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_FP16_MIN_SUPPORTED})
        )
        add_definitions(-DCCL_FP16_COMPILER)
        set(CCL_FP16_COMPILER ON)
    else()
        set(CCL_FP16_COMPILER OFF)
    endif()
    message(STATUS "FP16 compiler: ${CCL_FP16_COMPILER}")

    if (CCL_FP16_COMPILER)
        if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
            add_definitions(-DCCL_FP16_TARGET_ATTRIBUTES)
            set(CCL_FP16_TARGET_ATTRIBUTES ON)
        else()
            set(CCL_FP16_TARGET_ATTRIBUTES OFF)
        endif()
        message(STATUS "FP16 target attributes: ${CCL_FP16_TARGET_ATTRIBUTES}")
    endif()

    option(CCL_FP16_GPU_TRUNCATE "Truncate FP16 in GPU operations" OFF)
    if (CCL_FP16_GPU_TRUNCATE)
        add_definitions(-DCCL_FP16_GPU_TRUNCATE)
    endif()
    message(STATUS "FP16 GPU truncate: ${CCL_FP16_GPU_TRUNCATE}")

    set(LP_ENV_DEFINED 1 PARENT_SCOPE)

endfunction(set_lp_env)

function(check_compiler_version)

    set(GCC_MIN_SUPPORTED "4.8")
    set(ICC_MIN_SUPPORTED "15.0")
    set(CLANG_MIN_SUPPORTED "9.0")

    if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_MIN_SUPPORTED})
            message(FATAL_ERROR "gcc min supported version is ${GCC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${ICC_MIN_SUPPORTED})
            message(FATAL_ERROR "icc min supported version is ${ICC_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
        endif()
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
        if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${CLANG_MIN_SUPPORTED})
            message(FATAL_ERROR "clang min supported version is ${CLANG_MIN_SUPPORTED}, current version ${CMAKE_C_COMPILER_VERSION}")
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


function(activate_compute_backend MODULES_PATH COMPUTE_BACKEND)
    string( TOLOWER "${COMPUTE_BACKEND}" COMPUTE_BACKEND)

    set(CCL_ENABLE_SYCL_V 0 PARENT_SCOPE)
    set(CCL_ENABLE_SYCL_L0 0 PARENT_SCOPE)

    message("Search Compute Runtime by MODULES_PATH: ${MODULES_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${MODULES_PATH}")

    if(COMPUTE_BACKEND STREQUAL "computecpp")
        message ("COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using ComputeCpp provider")
        SET (COMPUTE_BACKEND_LOAD_MODULE "ComputeCpp"
                CACHE STRING
             "COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using ComputeCpp provider")

        find_package(${COMPUTE_BACKEND_LOAD_MODULE} REQUIRED)

        if(NOT ComputeCpp_FOUND)
            message(FATAL_ERROR "Failed to find ComputeCpp")
        endif()

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_BACKEND_CXXFLAGS_LOCAL "${COMPUTE_BACKEND_CXXFLAGS_LOCAL} ${COMPUTECPP_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_BACKEND_TARGET_NAME Codeplay::ComputeCpp)
        set (COMPUTE_BACKEND_TARGET_NAME Codeplay::ComputeCpp PARENT_SCOPE)
    endif()

    if(COMPUTE_BACKEND STREQUAL "dpcpp_level_zero")
        message ("COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")
        SET (COMPUTE_BACKEND_LOAD_MODULE "IntelSYCL_level_zero"
                CACHE STRING
             "COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")

        find_package(${COMPUTE_BACKEND_LOAD_MODULE} REQUIRED)

        if(NOT IntelSYCL_level_zero_FOUND)
            message(FATAL_ERROR "Failed to find IntelSYCL_level_zero")
        endif()

        set(CCL_ENABLE_SYCL_V 1 PARENT_SCOPE)

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_BACKEND_CXXFLAGS_LOCAL "${COMPUTE_BACKEND_CXXFLAGS_LOCAL} ${INTEL_SYCL_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL_level_zero)
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL_level_zero PARENT_SCOPE)
        message ("COMPUTE_BACKEND_TARGET_NAME=${COMPUTE_BACKEND_TARGET_NAME} requested. Using DPC++ provider")

    elseif(COMPUTE_BACKEND STREQUAL "level_zero")
        SET (COMPUTE_BACKEND_LOAD_MODULE "level_zero"
                CACHE STRING
             "COMPUTE_BACKEND=${COMPUTE_BACKEND} requested")

        find_package(${COMPUTE_BACKEND_LOAD_MODULE} REQUIRED)

        if(NOT LevelZero_FOUND)
            message(STATUS "Can not find level-zero")
            return()
        endif()

        # No compiler flags
        set (COMPUTE_BACKEND_CXXFLAGS_LOCAL "")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_BACKEND_TARGET_NAME ze_loader)
        set (COMPUTE_BACKEND_TARGET_NAME ze_loader PARENT_SCOPE)

    elseif(COMPUTE_BACKEND STREQUAL "dpcpp")
        message ("COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")
        SET (COMPUTE_BACKEND_LOAD_MODULE "IntelSYCL"
                CACHE STRING
             "COMPUTE_BACKEND=${COMPUTE_BACKEND} requested. Using DPC++ provider")

        find_package(${COMPUTE_BACKEND_LOAD_MODULE} REQUIRED)

        if(NOT IntelSYCL_FOUND)
            message(FATAL_ERROR "Failed to find IntelSYCL")
        endif()

        # remember compilation flags, because flag required for OBJECTS target
        # but if we use `target_link_libraries`, then these flags applied to all compiler options
        # for c & cxx. But we need special flags for cxx only
        # So set it manually
        set (COMPUTE_BACKEND_CXXFLAGS_LOCAL "${COMPUTE_BACKEND_CXXFLAGS_LOCAL} ${INTEL_SYCL_FLAGS}")

        # remember current target for `target_link_libraries` in ccl
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL)
        set (COMPUTE_BACKEND_TARGET_NAME Intel::SYCL PARENT_SCOPE)
    # elseif(COMPUTE_BACKEND STREQUAL "host")
        # message ("COMPUTE_BACKEND=${COMPUTE_BACKEND} requested.")
    # else()
         # message(FATAL_ERROR "Please provide one of the following compute runtime: dpcpp, level_zero, dpcpp_level_zero, host")
    endif()

    # extract target properties
    get_target_property(COMPUTE_BACKEND_INCLUDE_DIRS_LOCAL
                        ${COMPUTE_BACKEND_TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(COMPUTE_BACKEND_LIBRARIES_LOCAL
                        ${COMPUTE_BACKEND_TARGET_NAME} INTERFACE_LINK_LIBRARIES)

    # When we use dpcpp compiler(dpcpp/dpcpp_level_zero backends), use c++17 to be aligned with compiler
    if (${COMPUTE_BACKEND_TARGET_NAME} MATCHES "^Intel::SYCL.*")
        set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    # And use c++11 for all other cases
    else()
        set(CMAKE_CXX_STANDARD 11 PARENT_SCOPE)
    endif()

    # set output variables in the parent scope:
    # Only `COMPUTE_BACKEND_FLAGS` is actually required, because  the other flags are derived from
    # 'target_link_libraries'.
    # For simplicity, set all variables
    set(COMPUTE_BACKEND_FLAGS        ${COMPUTE_BACKEND_CXXFLAGS_LOCAL}      PARENT_SCOPE)
    set(COMPUTE_BACKEND_LIBRARIES    ${COMPUTE_BACKEND_LIBRARIES_LOCAL}     PARENT_SCOPE)
    set(COMPUTE_BACKEND_INCLUDE_DIRS ${COMPUTE_BACKEND_INCLUDE_DIRS_LOCAL}  PARENT_SCOPE)

endfunction(activate_compute_backend)

function(set_compute_backend COMMON_CMAKE_DIR)
    activate_compute_backend("${COMMON_CMAKE_DIR}" ${COMPUTE_BACKEND})

    # When we use dpcpp compiler(dpcpp/dpcpp_level_zero backends), use c++17 to be aligned with compiler
    # Although the same thing is done in activate_compute_backend we need to set the variable here as
    # well bacause both set_compute_backend and activate_compute_backend can be called directly
    if (${COMPUTE_BACKEND_TARGET_NAME} MATCHES "^Intel::SYCL.*")
        set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
    # And use c++11 for all other cases
    else()
        set(CMAKE_CXX_STANDARD 11 PARENT_SCOPE)
    endif()

    if (NOT COMPUTE_BACKEND_TARGET_NAME)
        message(FATAL_ERROR "Failed to find requested compute runtime: ${COMPUTE_BACKEND} in ${COMMON_CMAKE_DIR}")
    endif()
    message(STATUS "COMPUTE_BACKEND_TARGET_NAME: ${COMPUTE_BACKEND_TARGET_NAME}")

    if (${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "Intel::SYCL" OR ${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "Intel::SYCL_level_zero")
        option (CCL_ENABLE_SYCL "Enable CCL SYCL runtime" ON)
        message(STATUS "Enable CCL SYCL runtime")
        execute_process(COMMAND dpcpp -v
            OUTPUT_VARIABLE DPCPP_VERSION
            ERROR_VARIABLE DPCPP_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "DPC++ compiler version:\n" "${DPCPP_VERSION}")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_BACKEND_FLAGS}")
    if (${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "Intel::SYCL_level_zero" OR ${COMPUTE_BACKEND_TARGET_NAME} STREQUAL "ze_loader")
        set(MULTI_GPU_SUPPORT ON PARENT_SCOPE)
        set(MULTI_GPU_SUPPORT ON)
    endif()
    if (MULTI_GPU_SUPPORT)
        message(STATUS "Enable GPU support using level-zero")
    endif()

    # need to pass these variables to overlying function
    set (COMPUTE_BACKEND_TARGET_NAME ${COMPUTE_BACKEND_TARGET_NAME} PARENT_SCOPE)
    set (COMPUTE_BACKEND_FLAGS ${COMPUTE_BACKEND_FLAGS} PARENT_SCOPE)
    set (COMPUTE_BACKEND_LIBRARIES ${COMPUTE_BACKEND_LIBRARIES} PARENT_SCOPE)
    set (COMPUTE_BACKEND_FLAGS ${COMPUTE_BACKEND_FLAGS} PARENT_SCOPE)
endfunction(set_compute_backend)
