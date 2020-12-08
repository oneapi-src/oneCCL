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
class ccl_ipc_source_gpu_comm
        : public ccl_gpu_base_comm<ccl_ipc_source_gpu_comm<device_t>,
                                   gpu_types::IPC_GPU + device_t::type_idx()> {
public:
    using base = ccl_gpu_base_comm<ccl_ipc_source_gpu_comm<device_t>,
                                   gpu_types::IPC_GPU + device_t::type_idx()>;
    using typename base::comm_rank_t;
    using impl_t = device_t;
    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_module_t =
        typename device_t::template gpu_module_t<algo_type, group, mode>; //same as in-process GPU

    template <ccl_coll_type algo_type,
              ccl::group_split_type group,
              ccl::device_topology_type mode,
              class native_data_type>
    using gpu_kernel_t =
        typename gpu_module_t<algo_type, group, mode>::template kernel<native_data_type>;

    static constexpr const char* name_impl() {
        return "SOURCE_IPC_GPU";
    }

    ccl_ipc_source_gpu_comm(ccl_device& assigned_device,
                            typename base::comm_rank_t idx,
                            device_t& process_device,
                            ccl::group_split_type group_id,
                            ccl::device_topology_type class_id)
            : base(assigned_device, idx),
              inprocess_gpu_comm(process_device) {
        //register in topology
        switch (group_id) {
            case ccl::group_split_type::cluster: {
                switch (class_id) {
                    case ccl::device_topology_type::ring: {
                        const auto& original_rank =
                            inprocess_gpu_comm
                                .template get_comm_data<ccl::group_split_type::cluster,
                                                        ccl::device_topology_type::ring>();
                        base::template reset_rank<ccl::group_split_type::cluster,
                                                  ccl::device_topology_type::ring>(
                            original_rank.rank, original_rank.size);
                        break;
                    }
                    case ccl::device_topology_type::a2a: {
                        const auto& original_rank =
                            inprocess_gpu_comm
                                .template get_comm_data<ccl::group_split_type::cluster,
                                                        ccl::device_topology_type::a2a>();
                        base::template reset_rank<ccl::group_split_type::cluster,
                                                  ccl::device_topology_type::a2a>(
                            original_rank.rank, original_rank.size);
                        break;
                    }
                    default: {
                        throw std::runtime_error(
                            std::string("ccl_ipc_source_gpu_comm must be created") +
                            " unknown topology class: " + std::to_string(class_id));
                    }
                }
                break;
            }
            default: {
                throw std::runtime_error(
                    std::string("ccl_ipc_source_gpu_comm must be created") +
                    "for process-based topology, but requested: " +
                    std::to_string(
                        static_cast<typename std::underlying_type<ccl::group_split_type>::type>(
                            group_id)));
            }
        }
    }

    ~ccl_ipc_source_gpu_comm() = default;

    //TODO L0 work
    device_t& get_impl() {
        return inprocess_gpu_comm;
    }

    std::string to_string_impl() const {
        std::string ret(name_impl());
        ret = ret + "(" + inprocess_gpu_comm.to_string_impl() + ")";
        return ret;
    }
    /*
    template<ccl::group_split_type group_id>
    topology_addr<group_id> get_comm_data() const
    {
        return inprocess_gpu_comm.template get_comm_data<group_id>();
    }
*/
    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class native_data_type>
    gpu_kernel_t<module_type, group_id, class_id, native_data_type>& get_gpu_kernel() {
        return inprocess_gpu_comm
            .template get_gpu_kernel<module_type, group_id, class_id, native_data_type>();
    }

    template <class native_data_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry,
              class = typename std::enable_if<group_id == ccl::group_split_type::cluster>::type>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id, native_data_type>& register_entry(
        gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr =
            base::template get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id, native_data_type>();
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size);
        return main_func;
    }

private:
    device_t& inprocess_gpu_comm;
};
} // namespace native
