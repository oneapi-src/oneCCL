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
# Default installation path: <oneccl_root>/lib/cmake/oneCCL/
get_filename_component(_oneccl_root "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_oneccl_root "${_oneccl_root}/../../../" ABSOLUTE)

if ("$ENV{CCL_CONFIGURATION}" STREQUAL "cpu")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" _fsycl_option)
    if (_fsycl_option)
        message(STATUS "STATUS: -fsycl not supported for CCL_CONFIGURATION=cpu")
    endif()

    get_filename_component(_oneccl_headers "${_oneccl_root}/include" ABSOLUTE)
    get_filename_component(_oneccl_lib "${_oneccl_root}/lib/ccl/cpu/lib/libccl.so" ABSOLUTE)
else()
    get_filename_component(_oneccl_headers "${_oneccl_root}/include" ABSOLUTE)
    get_filename_component(_oneccl_lib "${_oneccl_root}/lib/libccl.so" ABSOLUTE)
endif()

if (EXISTS "${_oneccl_headers}" AND EXISTS "${_oneccl_lib}")
    if (NOT TARGET oneCCL)
        add_library(oneCCL SHARED IMPORTED)
        set_target_properties(oneCCL PROPERTIES
                             INTERFACE_INCLUDE_DIRECTORIES "${_oneccl_headers}"
                             IMPORTED_LOCATION "${_oneccl_lib}")
        unset(_oneccl_headers)
        unset(_oneccl_lib)

        find_package(MPI QUIET)
        if (NOT MPI_FOUND)
            message(STATUS "oneCCL: MPI is not found")
        else()
            set_target_properties(oneCCL PROPERTIES INTERFACE_LINK_LIBRARIES MPI)
            message(STATUS "oneCCL: MPI found")
        endif()
    endif()
else()
    if (NOT EXISTS "${_oneccl_headers}")
        message(STATUS "oneCCL: headers do not exist - ${_oneccl_headers}")
    endif()
    if (NOT EXISTS "${_oneccl_lib}")
        message(STATUS "oneCCL: lib do not exist - ${_oneccl_lib}")
    endif()
    set(oneCCL_FOUND FALSE)
endif()
