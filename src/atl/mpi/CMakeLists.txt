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

set(MPI_SRC
    atl_mpi.c)

set(COMMON_MPI_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/mpi/include/
    ${PROJECT_SOURCE_DIR}/src/)

# special library that holds objects only
add_library(ccl_atl_mpi-objects OBJECT ${MPI_SRC})
set_target_properties(ccl_atl_mpi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(ccl_atl_mpi-objects PRIVATE ${COMMON_MPI_INC_DIRS})

# add library search directory
link_directories(${PROJECT_SOURCE_DIR}/mpi/lib)

# shared
add_library(ccl_atl_mpi SHARED $<TARGET_OBJECTS:ccl_atl_mpi-objects>)
target_include_directories(ccl_atl_mpi PRIVATE ${COMMON_MPI_INC_DIRS})

# target_link_libraries(ccl_atl_mpi PRIVATE fabric m)
target_link_libraries(ccl_atl_mpi PRIVATE mpi)

# link with release_mt libmpi.so for oneAPI Base toolkit
set_target_properties(ccl_atl_mpi PROPERTIES LINK_FLAGS "-Wl,-rpath,../../../mpi/latest/lib/release_mt/")
if (NOT LIB_ATL_MPI_SO_VERSION AND NOT LIB_ATL_MPI_MAJOR_VERSION)
        set_target_properties(ccl_atl_mpi PROPERTIES VERSION 1 SOVERSION 1.0)
else()
        set_target_properties(ccl_atl_mpi PROPERTIES VERSION ${LIB_ATL_MPI_SO_VERSION} SOVERSION ${LIB_ATL_MPI_MAJOR_VERSION})
endif()

set_target_properties(ccl_atl_mpi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl_atl_mpi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

# static
add_library(ccl_atl_mpi-static STATIC $<TARGET_OBJECTS:ccl_atl_mpi-objects>)
set_target_properties(ccl_atl_mpi-static PROPERTIES OUTPUT_NAME ccl_atl_mpi)
set_target_properties(ccl_atl_mpi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl_atl_mpi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
