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
file(GLOB sources "*.c" "*.cpp")

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    target_include_directories(${executable} PRIVATE ${EXAMPLES_INC_DIRS})
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PRIVATE ${COMPUTE_RUNTIME_TARGET_NAME})
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_EXAMPLES}/sycl)
endforeach()
