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
set (executable "device_communicator_suite")


enable_testing()
message ("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

if(MULTI_GPU_SUPPORT)
    option(CCL_GPU_DEVICES_AFFINITY_ENABLE "Enable L0" ON)
    if(CCL_GPU_DEVICES_AFFINITY_ENABLE)
        set(CCL_GPU_DEVICES_AFFINITY_MASK_SIZE 4)
        message ("Set L0 device mask affinity size: ${CCL_GPU_DEVICES_AFFINITY_MASK_SIZE}")
    endif()
endif()


if(API_UT)
    set(PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../..")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/helpers.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
if(COMPUTE_RUNTIME)
    activate_compute_runtime("${PROJECT_SOURCE_DIR}/cmake" ${COMPUTE_RUNTIME})

    if (${CCL_ENABLE_SYCL_V} STREQUAL 1)
        option (CCL_ENABLE_SYCL "Enable CCL SYCL runtime" ON)
        message(STATUS "Enable CCL SYCL runtime")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPUTE_RUNTIME_FLAGS}")
    add_definitions(${DEFINITIONS})
endif()

if(API_UT)
    set(CCL_MAJOR_VERSION     "0")
    set(CCL_MINOR_VERSION     "0")
    set(CCL_UPDATE_VERSION    "0")
    set(CCL_PRODUCT_STATUS    "epsilon")
    string(TIMESTAMP CCL_PRODUCT_BUILD_DATE "%Y-%m-%dT %H:%M:%SZ")
    set(CCL_PRODUCT_FULL "${CCL_PRODUCT_STATUS}-${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION} ${CCL_PRODUCT_BUILD_DATE} ${VCS_INFO}")
    configure_file(${PROJECT_SOURCE_DIR}/include/oneapi/ccl/ccl_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/oneapi/ccl/ccl_config.h")
endif(API_UT)


include_directories(${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/include
                    ${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/src
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/src)

file(GLOB sources "*.c" "*.cpp")
if(MULTI_GPU_SUPPORT)
set (dependencies_sources
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_utils.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_empty_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_empty_coll_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_empty_stream.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_app_api_comm_split_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_app_api_coll_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_stream.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_event.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/coll_common_attributes.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_allgather_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_allreduce_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_alltoall_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_alltoallv_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_bcast_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_reduce_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_reduce_scatter_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_sparse_allreduce_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/coll/ccl_barrier_op_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/log/log.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/event/event.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/stream/stream.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/comm/comm.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/comm/comm_interface.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/comm/l0/comm_context_storage.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/comm/l0/comm_context.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/comm/l0/gpu_comm_attr.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/base.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/device.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/driver.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/context.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/platform.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/subdevice.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/primitives.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_utils.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/spinlock.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/kvs.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/request.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/host_communicator.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/single_device_communicator.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/device_topology_communicator.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/thread_topology_communicator.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/process_topology_communicator.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/process_group_context.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/thread_group_context.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/global.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/env.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/cache.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/ccl.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/datatype.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/native_runtime_utils.cpp")

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src} ${dependencies_sources})
    add_dependencies(${executable} gtest)
        if(COMPUTE_RUNTIME)
            target_include_directories(${executable} PRIVATE
                                                    $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
            set_target_properties(${executable} PROPERTIES
                                                    INTERFACE_INCLUDE_DIRECTORIES
                                                    $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)

            target_link_libraries(${executable} PUBLIC ${COMPUTE_RUNTIME_TARGET_NAME})
        endif(COMPUTE_RUNTIME)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PUBLIC pthread)
    target_link_libraries(${executable} PUBLIC stdc++)
    target_link_libraries(${executable} PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest)
endforeach()
endif()
