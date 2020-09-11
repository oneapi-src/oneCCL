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
set (executable "stream_suite")


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


#add_subdirectory (${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/ ${PROJECT_SOURCE_DIR}/tests/units)
include_directories(${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/include
                    ${PROJECT_SOURCE_DIR}/tests/functional/googletest-release-1.8.1/googletest/src
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/src)

file(GLOB sources "*.c" "*.cpp")

set (dependencies_sources
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_utils.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/log/log.cpp"
             "${PROJECT_SOURCE_DIR}/src/common/stream/stream.cpp"
             "${PROJECT_SOURCE_DIR}/src/ccl_cpp_stream.cpp"
             "${PROJECT_SOURCE_DIR}/tests/unit/api/stubs/spinlock.cpp")

if(MULTI_GPU_SUPPORT)
    list( APPEND dependencies_sources
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/base.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/device.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/driver.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/context.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/platform.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/subdevice.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/primitives.cpp"
             "${PROJECT_SOURCE_DIR}/src/native_device_api/l0/utils.cpp"
             )
endif()


foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src} ${dependencies_sources})
    add_dependencies(${executable} gtest)
    #if(API_UT)
        if(COMPUTE_RUNTIME)
            target_include_directories(${executable} PRIVATE
                                                    $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
            set_target_properties(${executable} PROPERTIES
                                                    INTERFACE_INCLUDE_DIRECTORIES
                                                    $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)

            target_link_libraries(${executable} PUBLIC ${COMPUTE_RUNTIME_TARGET_NAME})
        endif(COMPUTE_RUNTIME)
    #endif(API_UT)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PUBLIC pthread)
    target_link_libraries(${executable} PUBLIC stdc++)
    target_link_libraries(${executable} PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest)
endforeach()
