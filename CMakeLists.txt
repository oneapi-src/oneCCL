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
#common CMakeList.txt to build CCL, ATL, tests

cmake_minimum_required (VERSION 2.8)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(PROJECT_NAME "oneCCL")
set(PROJECT_FULL_NAME "oneAPI Collective Communications Library")

project(${PROJECT_NAME})

include(${PROJECT_SOURCE_DIR}/cmake/helpers.cmake)

check_compiler_version()

#set default build type
#available build types are: Debug, Release, RelWithDebInfo and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#make build variable case insensitive
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_CASE_INSENSITIVE)

#predefined options
option(USE_SECURITY_FLAGS "Use security flags" TRUE)
if (${CMAKE_BUILD_TYPE_CASE_INSENSITIVE} STREQUAL "debug")
    set(USE_SECURITY_FLAGS FALSE)
endif()

option(BUILD_EXAMPLES "Build examples" TRUE)
option(BUILD_FT "Build functional tests" TRUE)
option(BUILD_UT "Build unit tests" FALSE)
option(BUILD_CONFIG "Build cmake configs" TRUE)

option(USE_CODECOV_FLAGS "Calculate code coverage" FALSE)
option(WITH_ASAN "Use address sanitizer, can only be used in Debug build" FALSE)

#installation path variables
include(GNUInstallDirs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

#show build info
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE}")
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build functional tests: ${BUILD_FT}")
message(STATUS "Build unit tests: ${BUILD_UT}")
message(STATUS "Build cmake configs: ${BUILD_CONFIG}")

add_definitions(-DCCL_C_COMPILER="${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
add_definitions(-DCCL_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

SET(MULTI_GPU_SUPPORT OFF CACHE BOOL "Enable Multi GPU extension support")

set(CCL_COMMON_INSTALL_PREFIX "intel64")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CCL_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CCL_INSTALL_INCLUDE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(CCL_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(CCL_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(CCL_INSTALL_ENV "${CMAKE_INSTALL_PREFIX}/env")
set(CCL_INSTALL_ETC "${CMAKE_INSTALL_PREFIX}/etc")
set(CCL_INSTALL_LICENSE "${CMAKE_INSTALL_PREFIX}/licensing")
set(CCL_INSTALL_MODULE "${CMAKE_INSTALL_PREFIX}/modulefiles")
set(CCL_INSTALL_EXAMPLES "${CMAKE_INSTALL_PREFIX}/examples")
set(CCL_INSTALL_TESTS "${CMAKE_INSTALL_PREFIX}/tests")
set(CCL_INSTALL_UNIT_TESTS "${CMAKE_INSTALL_PREFIX}/tests/unit")
set(CCL_INSTALL_KERNELS "${CMAKE_INSTALL_PREFIX}/lib/kernels")

set(CCL_UNIT_TESTS_BUILD "${CMAKE_BINARY_DIR}/tests/unit")


# setup dependency directories

set(DEPS_DIR "${PROJECT_SOURCE_DIR}/deps")

set(MPI_INCLUDE_DIR "${DEPS_DIR}/mpi/include/")
set(MPI_LIB_DIR "${DEPS_DIR}/mpi/lib/")
if ( "${LIBFABRIC_DIR}" STREQUAL "")
    set(LIBFABRIC_INCLUDE_DIR "${DEPS_DIR}/ofi/include")
    set(LIBFABRIC_LIB_DIR "${DEPS_DIR}/ofi/lib/")
else()
    set(LIBFABRIC_INCLUDE_DIR "${LIBFABRIC_DIR}/include/")
    set(LIBFABRIC_LIB_DIR "${LIBFABRIC_DIR}/lib")
endif()
set(HWLOC_INCLUDE_DIR "${DEPS_DIR}/hwloc/include/")
set(HWLOC_LIB_DIR "${DEPS_DIR}/hwloc/lib/")

message(STATUS "MPI_INCLUDE_DIR: ${MPI_INCLUDE_DIR}")
message(STATUS "MPI_LIB_DIR: ${MPI_LIB_DIR}")
message(STATUS "LIBFABRIC_LIB_DIR: ${LIBFABRIC_LIB_DIR}")
message(STATUS "LIBFABRIC_INCLUDE_DIR: ${LIBFABRIC_INCLUDE_DIR}")
message(STATUS "HWLOC_INCLUDE_DIR: ${HWLOC_INCLUDE_DIR}")
message(STATUS "HWLOC_LIB_DIR: ${HWLOC_LIB_DIR}")

include_directories(${MPI_INCLUDE_DIR})
include_directories(${LIBFABRIC_INCLUDE_DIR})

link_directories(${MPI_LIB_DIR})
link_directories(${LIBFABRIC_LIB_DIR})


set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_RPATH TRUE)

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
    set(C_COMPILER_FLAGS "-std=gnu99")
endif()

# special flags for CCL library only
set(SRC_C_FLAGS "")
set(SRC_CXX_FLAGS "")
set(SRC_SHARED_LINKER_FLAGS "")

#common settings of security options
if(USE_SECURITY_FLAGS)
    set(SRC_C_FLAGS "${SRC_C_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -fPIE -fPIC -z noexecstack -z relro -z now")
    if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(SRC_C_FLAGS "${SRC_C_FLAGS} -fstack-protector-strong")
            set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -fstack-protector-strong")
        endif()
    endif()
endif()

set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -Wl,--version-script=${PROJECT_SOURCE_DIR}/ccl.map")

if(${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    if (USE_CODECOV_FLAGS)
        set(SRC_C_FLAGS "${SRC_C_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
        set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
    endif()
endif()

#TODO: add -Wextra to c/cxx flags

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

set(TRY_ENABLE_SYCL_L0 ON)

set(COMMON_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
if (COMPUTE_BACKEND)
    message(STATUS "COMPUTE_BACKEND: ${COMPUTE_BACKEND}")
    set_compute_backend(${COMMON_CMAKE_DIR})
endif()

if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        #c++17 introduces algined new operator, use it
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
    endif()
endif()

# Clang doesn't automatically detects ninja processes as supporting colored output
# due to the way they are spawned. In order to fix the issue we need to use the option
# to force colored output
if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

if(WITH_ASAN AND ${CMAKE_BUILD_TYPE_CASE_INSENSITIVE} STREQUAL "debug")
    message(STATUS "Compiling with address sanitizer")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lasan")
endif()

set_lp_env()

set(CCL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)

enable_testing()

set(EXTERNAL_LIBS "")
set(EXAMPLES_INC_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/include ${MPI_INCLUDE_DIR})

# allow `deprecated`
set(CMAKE_CLANG_FLAGS "${CMAKE_CLANG_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

#generate & install vars.sh
configure_file(cmake/vars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/vars.sh @ONLY)
configure_file(cmake/setvars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh @ONLY)
configure_file(cmake/ccl ${CMAKE_CURRENT_BINARY_DIR}/ccl @ONLY)
configure_file(third-party-programs.txt ${CMAKE_CURRENT_BINARY_DIR}/third-party-programs.txt COPYONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ccl DESTINATION ${CCL_INSTALL_MODULE})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/third-party-programs.txt DESTINATION ${CCL_INSTALL_LICENSE})
install(PROGRAMS ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CCL_INSTALL_LICENSE})

# copy kernels
if(COMPUTE_BACKEND AND EXISTS "${PROJECT_SOURCE_DIR}/src/kernels")
file(GLOB spv_kernels "${PROJECT_SOURCE_DIR}/src/kernels/ring_*.spv")
    install(PROGRAMS ${spv_kernels}
            DESTINATION ${CCL_INSTALL_KERNELS}
            PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
endif()

set(CCL_MAJOR_VERSION     "2021")
set(CCL_MINOR_VERSION     "3")
set(CCL_UPDATE_VERSION    "0")
set(CCL_PRODUCT_STATUS    "Gold")
string(TIMESTAMP CCL_PRODUCT_BUILD_DATE "%Y-%m-%dT %H:%M:%SZ")
get_vcs_properties("git")
set(CCL_PRODUCT_FULL "${CCL_PRODUCT_STATUS}-${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION} ${CCL_PRODUCT_BUILD_DATE} ${VCS_INFO}")
configure_file(${PROJECT_SOURCE_DIR}/include/oneapi/ccl/config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/oneapi/ccl/config.h")
file(COPY "${CMAKE_CURRENT_BINARY_DIR}/include/oneapi/ccl/config.h" DESTINATION ${PROJECT_SOURCE_DIR}/include/oneapi/ccl)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

set(PROJECT_VERSION "${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION}")

if (BUILD_CONFIG)
    configure_file("cmake/templates/oneCCLConfig.cmake.in"
                   "${CCL_INSTALL_LIB}/cmake/oneCCL/oneCCLConfig.cmake"
                   COPYONLY)
    configure_file("cmake/templates/oneCCLConfigVersion.cmake.in"
                   "${CCL_INSTALL_LIB}/cmake/oneCCL/oneCCLConfigVersion.cmake"
                   @ONLY)
endif()

#include other CMakeLists

add_subdirectory(src)

if (BUILD_EXAMPLES)
    add_subdirectory(examples/benchmark)
    add_subdirectory(examples/common)
    add_subdirectory(examples/cpu)
    add_subdirectory(examples/external_launcher)
    if (CCL_ENABLE_SYCL)
        add_subdirectory(examples/sycl)
    endif()
endif()

if (BUILD_FT)
    add_subdirectory(tests/functional)
endif()

if (BUILD_UT)
    #add_subdirectory(tests/unit)
endif()
