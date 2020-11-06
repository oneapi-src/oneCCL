/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "oneapi/ccl/ccl_aliases.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/l0/comm_context_impl.hpp"
#include "common/utils/spinlock.hpp"
#include "common/comm/atl_tag.hpp"

namespace ccl {
comm_group::comm_group(shared_communicator_t parent_comm,
                       size_t threads_count,
                       size_t on_process_ranks_count,
                       group_unique_key id)
        : pimpl(new gpu_comm_attr(parent_comm, threads_count, on_process_ranks_count, id)){};

bool comm_group::sync_group_size(size_t device_group_size) {
    return pimpl->sync_group_size(device_group_size);
}

comm_group::~comm_group() {}

const group_unique_key& comm_group::get_unique_id() const {
    return pimpl->get_unique_id();
}
/*
std::string comm_group::to_string() const
{
    pimpl->to_string();
}*/
} // namespace ccl
// container-based method force-instantiation will trigger ALL other methods instantiations
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(ccl::vector_class<ccl::device_index_type>, typename ccl::unified_device_context_type::ccl_native_t);
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(ccl::list_class<ccl::device_index_type>, typename ccl::unified_device_context_type::ccl_native_t);
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(ccl::device_indices_t, typename ccl::unified_device_context_type::ccl_native_t);
COMM_CREATOR_INDEXED_INSTANTIATION_TYPE(ccl::device_index_type, typename ccl::unified_device_context_type::ccl_native_t);

COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(ccl::vector_class<typename ccl::unified_device_type::ccl_native_t>, typename ccl::unified_device_context_type::ccl_native_t);
COMM_CREATOR_INDEXED_INSTANTIATION_TYPE(typename ccl::unified_device_type::ccl_native_t, typename ccl::unified_device_context_type::ccl_native_t);
