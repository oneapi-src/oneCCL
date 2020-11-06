#
# Copyright 2016-2020 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required (VERSION 2.8)

set(PROJECT_NAME "oneCCL examples")

project(${PROJECT_NAME})

include(../cmake/helpers.cmake)

if (DEFINED ENV{CCL_CONFIGURATION})
    set(CCL_CONFIGURATION "$ENV{CCL_CONFIGURATION}")
    if(${CCL_CONFIGURATION} STREQUAL "cpu_gpu_dpcpp")
        set(COMPUTE_RUNTIME_TARGET_NAME "Intel::SYCL")
    endif()
endif()

if (DEFINED ENV{CCL_ROOT})
    set(CCL_ROOT "$ENV{CCL_ROOT}")
else()
    message(FATAL_ERROR "Please define CCL_ROOT environment variable")
endif()

if (DEFINED ENV{I_MPI_ROOT})
    set(I_MPI_ROOT "$ENV{I_MPI_ROOT}")
else()
    message(FATAL_ERROR "Please define I_MPI_ROOT environment variable")
endif()

message(STATUS "CCL_ROOT: ${CCL_ROOT}")
message(STATUS "I_MPI_ROOT: ${I_MPI_ROOT}")
message(STATUS "CCL_CONFIGURATION: ${CCL_CONFIGURATION}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if (NOT DEFINED ${CCL_INSTALL_EXAMPLES})
    set(CCL_INSTALL_EXAMPLES "$ENV{PWD}")
endif()

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
    set(C_COMPILER_FLAGS "-std=gnu99")
endif()

#common release/debug compilation settings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_COMPILER_FLAGS} -Wall -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${C_COMPILER_FLAGS} -O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_COMPILER_FLAGS} -O2 -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMPILER_FLAGS} -Wall -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_COMPILER_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_COMPILER_FLAGS} -O2 -g")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (COMPUTE_RUNTIME)
    activate_compute_runtime("${CCL_ROOT}/lib;${PROJECT_SOURCE_DIR}/../cmake" ${COMPUTE_RUNTIME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_RUNTIME_FLAGS}")
    list(APPEND CCL_INC_DIRS ${COMPUTE_RUNTIME_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBS ${COMPUTE_RUNTIME_LIBRARIES})
endif()

set(GCC_BF16_MIN_SUPPORTED "4.9.0")

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BF16_MIN_SUPPORTED}))
    add_definitions(-DCCL_BF16_COMPILER)
    set(CCL_BF16_COMPILER ON)
    message(STATUS "BF16 compiler: yes")
else()
    set(CCL_BF16_COMPILER OFF)
    message(STATUS "BF16 compiler: no")
endif()

if (CCL_BF16_COMPILER)
    if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        add_definitions(-DCCL_BF16_TARGET_ATTRIBUTES)
        message(STATUS "BF16 target attributes: yes")
    else()
        message(STATUS "BF16 target attributes: no")
    endif()
endif()

include_directories(include)
include_directories(${I_MPI_ROOT}/include/)
include_directories(${CCL_ROOT}/include/${CCL_CONFIGURATION})
link_directories(${I_MPI_ROOT}/lib/release_mt)
link_directories(${I_MPI_ROOT}/libfabric/lib)
link_directories(${CCL_ROOT}/lib/${CCL_CONFIGURATION})

add_subdirectory(cpu)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_subdirectory(sycl)
endif()
add_subdirectory(common)
add_subdirectory(benchmark)
