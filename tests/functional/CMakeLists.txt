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
file(GLOB sources "*_test.c" "*_test.cpp")

if (NOT DEFINED LP_ENV_DEFINED)
    include(${PROJECT_SOURCE_DIR}/../../cmake/helpers.cmake)
    set_lp_env()
endif()

set(SERVICE_SRC
    conf.cpp
    lp.cpp)

if (DEFINED ENV{CCL_CONFIGURATION})
    set(CCL_CONFIGURATION "$ENV{CCL_CONFIGURATION}")
    if(${CCL_CONFIGURATION} STREQUAL "cpu_gpu_dpcpp")
        set(COMPUTE_BACKEND "dpcpp_level_zero")
    endif()
endif()

if (DEFINED ENV{CCL_ROOT})
    set(CCL_ROOT "$ENV{CCL_ROOT}")
endif()

set(CCL_INSTALL_TESTS "$ENV{PWD}")
enable_testing()
ADD_SUBDIRECTORY (googletest-release-1.8.1/googletest/)

set(INC_DIRS
     ${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/include
     ${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/src
     ${PROJECT_SOURCE_DIR}/include)

include_directories(${INC_DIRS})

message(STATUS "CCL_ROOT: ${CCL_ROOT}")
message(STATUS "CCL_CONFIGURATION: ${CCL_CONFIGURATION}")
message(STATUS "tests/functional LIBFABRIC_LIB_DIR: ${LIBFABRIC_LIB_DIR}")
message(STATUS "tests/functional LIBFABRIC_INCLUDE_DIR: ${LIBFABRIC_INCLUDE_DIR}")
message(STATUS "INC_DIRS: ${INC_DIRS}")

#include_directories(${CCL_ROOT}/include/${CCL_CONFIGURATION})
#link_directories(${CCL_ROOT}/lib/${CCL_CONFIGURATION})

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

if (COMPUTE_BACKEND)
    activate_compute_backend("${CCL_ROOT}/lib;${PROJECT_SOURCE_DIR}/cmake" ${COMPUTE_BACKEND})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_BACKEND_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COMPUTE_BACKEND_LIBRARIES}")
endif()

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src} ${SERVICE_SRC})
    if (${executable} MATCHES ".*reduce.*")
        target_compile_definitions(${executable} PRIVATE TEST_CCL_REDUCE)
    endif()
    if (${executable} MATCHES ".*bcast.*")
        target_compile_definitions(${executable} PRIVATE TEST_CCL_BCAST)
    endif()
    target_link_libraries(${executable} PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest)
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PRIVATE m)
    target_link_libraries(${executable} PUBLIC mpi)
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS} OPTIONAL)
    add_test (NAME ${executable} CONFIGURATIONS default COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/${executable} --gtest_output=xml:${CCL_INSTALL_TESTS}/${executable}_default_report.junit.xml)
endforeach()

foreach(algo direct; naive; flat; multi_bcast)
add_test (NAME allgatherv_${algo} CONFIGURATIONS allgatherv_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allgatherv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allgatherv_${algo}_report.junit.xml)
endforeach()

add_test (NAME allreduce_fusion CONFIGURATIONS allreduce_fusion COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_fusion_report.junit.xml)

foreach(algo direct; rabenseifner; starlike; ring; ring_rma; double_tree; recursive_doubling; 2d)
add_test (NAME allreduce_${algo} CONFIGURATIONS allreduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_${algo}_report.junit.xml)
endforeach()

foreach(algo starlike; ring; 2d)
add_test (NAME allreduce_${algo}_chunked CONFIGURATIONS allreduce_${algo}_chunked COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_${algo}_chunked_report.junit.xml)
endforeach()

foreach(algo direct; naive; scatter; scatter_barrier)
add_test (NAME alltoall_${algo} CONFIGURATIONS alltoall_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoall_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoall_${algo}_report.junit.xml)
endforeach()

foreach(algo scatter_barrier)
add_test (NAME alltoall_${algo}_chunked CONFIGURATIONS alltoall_${algo}_chunked COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoall_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoall_${algo}_chunked_report.junit.xml)
endforeach()

foreach(algo direct; naive; scatter; scatter_barrier)
add_test (NAME alltoallv_${algo} CONFIGURATIONS alltoallv_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoallv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoallv_${algo}_report.junit.xml)
endforeach()

foreach(algo scatter_barrier)
add_test (NAME alltoallv_${algo}_chunked CONFIGURATIONS alltoallv_${algo}_chunked COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoallv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoallv_${algo}_chunked_report.junit.xml)
endforeach()

foreach(algo direct; ring; double_tree; naive)
add_test (NAME bcast_${algo} CONFIGURATIONS bcast_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/bcast_test --gtest_output=xml:${CCL_INSTALL_TESTS}/bcast_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; rabenseifner; tree; double_tree)
add_test (NAME reduce_${algo} CONFIGURATIONS reduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/reduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; ring)
add_test (NAME reduce_scatter_${algo} CONFIGURATIONS reduce_scatter_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/reduce_scatter_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_scatter_${algo}_report.junit.xml)
endforeach()
