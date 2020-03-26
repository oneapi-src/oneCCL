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
file(GLOB sources "*.c" "*.cpp")

set(CCL_INSTALL_TESTS "$ENV{PWD}")
enable_testing()
ADD_SUBDIRECTORY (googletest-release-1.8.1/googletest/)
include_directories(${PROJECT_SOURCE_DIR}/tests/googletest-release-1.8.1/googletest/include ${PROJECT_SOURCE_DIR}/tests/googletest-release-1.8.1/googletest/src)
include_directories($ENV{CCL_ROOT}/include/cpu_icc)
link_directories($ENV{CCL_ROOT}/lib/cpu_icc)

# Check SYCL support
if (NOT CMAKE_DISABLE_SYCL)
    set(CMAKE_DISABLE_SYCL 0)
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

if( ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if (${CMAKE_DISABLE_SYCL} EQUAL 0)
        set(CMAKE_CLANG_FLAGS "-fsycl")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lsycl")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CLANG_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CLANG_FLAGS}")
endif()


foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    target_link_libraries(${executable} PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest)
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS})
    # FIXME: enable allreduce_custom for non-direct case only
    if(NOT ${executable} STREQUAL "allreduce_custom_test")
        add_test (NAME ${executable} CONFIGURATIONS Default COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/${executable} --gtest_output=xml:${CCL_INSTALL_TESTS}/${executable}_default_report.junit.xml)
    endif()
endforeach()

foreach(algo direct; naive; flat; multi_bcast)
add_test (NAME allgatherv_test_mpi_${algo} CONFIGURATIONS mpi_allgatherv_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allgatherv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allgatherv_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; rabenseifner; starlike; ring; ring_rma; double_tree; recursive_doubling; 2d)
add_test (NAME allreduce_test_mpi_${algo} CONFIGURATIONS mpi_allreduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo starlike; ring; 2d)
add_test (NAME allreduce_test_mpi_${algo}_chunked CONFIGURATIONS mpi_allreduce_${algo}_chunked COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_test_mpi_runtime_adjust_${algo}_chunked_report.junit.xml)
endforeach()

add_test (NAME allreduce_custom CONFIGURATIONS Custom COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_custom_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_custom_report.junit.xml)

foreach(algo direct; naive)
add_test (NAME alltoall_test_mpi_${algo} CONFIGURATIONS mpi_alltoall_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoall_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoall_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; naive)
add_test (NAME alltoallv_test_mpi_${algo} CONFIGURATIONS mpi_alltoallv_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoallv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoallv_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; ring; double_tree; naive)
add_test (NAME bcast_test_mpi_${algo} CONFIGURATIONS mpi_bcast_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/bcast_test --gtest_output=xml:${CCL_INSTALL_TESTS}/bcast_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; rabenseifner; tree; double_tree)
add_test (NAME reduce_test_mpi_${algo} CONFIGURATIONS mpi_reduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/reduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()
