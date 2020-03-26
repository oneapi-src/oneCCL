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
#builds resizable_pmi

set(KVS_SRC
    helper.c
    rank_list.c
    shift_list.c
    kvs_keeper.c
    resizable_pmi.c
    listener.c
    request_wrappers_k8s.c
    kvs/kvs.c)

set(COMMON_KVS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_resizable_rt/pmi_resizable
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs
    ${PROJECT_SOURCE_DIR}/ofi/include)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pthread")

#special library that holds objects only
add_library(resizable_pmi-objects OBJECT ${KVS_SRC})
set_target_properties(resizable_pmi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(resizable_pmi-objects PUBLIC ${COMMON_KVS_INC_DIRS})
target_compile_definitions(resizable_pmi-objects PRIVATE HAVE_UNISTD_H HAVE_STDLIB_H HAVE_STRING_H HAVE_STRINGS_H)

#shared lib
add_library(resizable_pmi SHARED $<TARGET_OBJECTS:resizable_pmi-objects>)
target_include_directories(resizable_pmi PUBLIC ${COMMON_KVS_INC_DIRS})
set_target_properties(resizable_pmi PROPERTIES VERSION 1 SOVERSION 1.0)
set_target_properties(resizable_pmi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})

install(TARGETS resizable_pmi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

#static lib
add_library(resizable_pmi-static STATIC $<TARGET_OBJECTS:resizable_pmi-objects>)
set_target_properties(resizable_pmi-static PROPERTIES OUTPUT_NAME resizable_pmi)
set_target_properties(resizable_pmi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS resizable_pmi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
