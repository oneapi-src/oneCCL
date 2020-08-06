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
#pragma once
#include <map>
#include <memory>
#include <list>
#include <set>
#include <vector>

#include "common/comm/l0/devices/ccl_gpu_base_comm.hpp"

namespace native {

//Adapter for different thread devices
template <class device_t>
class ccl_thread_comm : public ccl_gpu_base_comm<ccl_thread_comm<device_t>,
                                                 gpu_types::CONCURRENT_GPU + device_t::type_idx()> {
public:
    using base = ccl_gpu_base_comm<ccl_thread_comm<device_t>,
                                   gpu_types::CONCURRENT_GPU + device_t::type_idx()>;
    using typename base::comm_rank_t;
    using impl_t = device_t;

    template <ccl_coll_type algo_type,
              ccl::device_group_split_type group,
              ccl::device_topology_type mode>
    using gpu_module_t =
        typename device_t::template gpu_module_t<algo_type, group, mode>; //same as in-process GPU

    template <ccl_coll_type algo_type,
              ccl::device_group_split_type group,
              ccl::device_topology_type mode,
              class native_data_type>
    using gpu_kernel_t =
        typename gpu_module_t<algo_type, group, mode>::template kernel<native_data_type>;

    static constexpr const char* name_impl() {
        return "CONCURRENT_GPU";
    }

    ccl_thread_comm(ccl_device& assigned_device,
                    typename base::comm_rank_t idx,
                    device_t& next_thread_device)
            : base(assigned_device, idx),
              next_thread_gpu_comm(next_thread_device) {}

    ~ccl_thread_comm() = default;

    std::string to_string_impl() const {
        std::string ret(name_impl());
        ret = ret + "(" + next_thread_gpu_comm.to_string_impl() + ")";
        return ret;
    }

    template <ccl::device_group_split_type group_id, ccl::device_topology_type class_id>
    topology_addr<group_id, class_id> get_comm_data() const {
        return next_thread_gpu_comm.template get_comm_data<group_id, class_id>();
    }

    template <ccl_coll_type module_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id,
              class native_data_type>
    gpu_kernel_t<module_type, group_id, class_id, native_data_type>& get_gpu_kernel() {
        return next_thread_gpu_comm
            .template get_gpu_kernel<module_type, group_id, class_id, native_data_type>();
    }

    device_t& get_impl_device() {
        return next_thread_gpu_comm;
    }

private:
    device_t& next_thread_gpu_comm;
};
} // namespace native
