#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# Prioritize ZE_ROOT
list(APPEND l0_root_hints
            ${ZE_ROOT}
            $ENV{ZE_ROOT})

set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(NOT l0_root_hints)
    set(l0_root_hints "/usr")
    message("ZE_ROOT prefix path hint is not defined, use default: ${l0_root_hints}")
endif()

list(INSERT CMAKE_PREFIX_PATH 0 ${l0_root_hints})

if (TARGET ze_loader)
    set(LevelZero_FOUND ON)
endif()

if(NOT TARGET ze_loader)
    find_path(LevelZero_INCLUDE_DIR
      NAMES ze_api.h
      PATHS
            ENV ZE_ROOT
            ${l0_root_hints}
      PATH_SUFFIXES
            include
            include/level_zero
            local/include
            local/include/level_zero
      NO_DEFAULT_PATH
    )

    find_library(LevelZero_LIBRARY
      NAMES ze_intel_gpu level_zero ze_loader ze_loader32 ze_loader64
      PATHS
            ENV ZE_ROOT
            ${l0_root_hints}
      PATH_SUFFIXES
            lib
            lib/x86_64-linux-gnu
            lib/level_zero
            local/lib
            local/lib/level_zero
      NO_DEFAULT_PATH
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(LevelZero
      REQUIRED_VARS
        LevelZero_INCLUDE_DIR
        LevelZero_LIBRARY
      HANDLE_COMPONENTS
    )
    mark_as_advanced(LevelZero_LIBRARY LevelZero_INCLUDE_DIR)

    if(LevelZero_FOUND)
        list(APPEND LevelZero_LIBRARIES ${LevelZero_LIBRARY} ${CMAKE_DL_LIBS})
        list(APPEND LevelZero_INCLUDE_DIRS ${LevelZero_INCLUDE_DIR})
        find_package(OpenCL)
        if(OpenCL_FOUND)
            message("L0 is using OpenCL interoperability")
            list(APPEND LevelZero_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
        endif()
        add_library(ze_loader INTERFACE IMPORTED)
        set_target_properties(ze_loader
          PROPERTIES INTERFACE_LINK_LIBRARIES "${LevelZero_LIBRARIES}"
        )
        set_target_properties(ze_loader
          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIRS}"
        )
    endif()
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
