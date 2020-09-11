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
add_subdirectory(coll_attrs)
add_subdirectory(comm_split_attrs)
add_subdirectory(host_communicator)
add_subdirectory(datatype_attrs)

if (MULTI_GPU_SUPPORT OR CCL_ENABLE_SYCL)
    add_subdirectory(device_communicator)
    add_subdirectory(event)
    add_subdirectory(stream)
endif()
