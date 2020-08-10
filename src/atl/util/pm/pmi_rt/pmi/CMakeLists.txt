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
#builds pmi

set(PMI_SRC
    simple_pmiutil.c
    simple_pmi.c)

set(COMMON_PMI_INC_DIRS
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_rt/pmi)

#special library that holds objects only
add_library(pmi-objects OBJECT ${PMI_SRC})
set_target_properties(pmi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(pmi-objects PUBLIC ${COMMON_PMI_INC_DIRS})
target_compile_definitions(pmi-objects PRIVATE HAVE_UNISTD_H HAVE_STDLIB_H HAVE_STRING_H HAVE_STRINGS_H)

#shared lib
add_library(pmi SHARED $<TARGET_OBJECTS:pmi-objects>)
target_include_directories(pmi PUBLIC INTERFACE ${COMMON_PMI_INC_DIRS})
if (NOT LIB_PMI_SO_VERSION AND NOT LIB_PMI_MAJOR_VERSION)
        set_target_properties(pmi PROPERTIES VERSION 1 SOVERSION 1.0)
else()
        set_target_properties(pmi PROPERTIES VERSION ${LIB_PMI_SO_VERSION} SOVERSION ${LIB_PMI_MAJOR_VERSION})
endif()
set_target_properties(pmi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})

install(TARGETS pmi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

#static lib
add_library(pmi-static STATIC $<TARGET_OBJECTS:pmi-objects>)
set_target_properties(pmi-static PROPERTIES OUTPUT_NAME pmi)
set_target_properties(pmi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS pmi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
