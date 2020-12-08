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

#include "common/comm/l0/communicator/base_communicator.hpp"
#include "common/comm/l0/device_community_holder.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

/*
namespace native
{
    template<ccl::device_topology_type class_id>
    struct device_community;

    template<ccl::device_topology_type class_id>
    device_community_container;
}
*/
template <class comm_impl,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          class communicator_traits>
class typed_base_communicator : public base_communicator {
public:
    using base_t = base_communicator;
    using impl_t = comm_impl;
    using self_t = typed_base_communicator<comm_impl, group_id, class_id, communicator_traits>;
    using traits = communicator_traits;

    // Topologies
    static constexpr ccl::group_split_type topology_type() {
        return group_id;
    }

    static constexpr ccl::device_topology_type topology_class() {
        return class_id;
    }

    // traits
    bool is_host() const noexcept override {
        return traits::is_host();
    }

    bool is_cpu() const noexcept override {
        return traits::is_cpu();
    }

    bool is_gpu() const noexcept override {
        return traits::is_gpu();
    }

    bool is_accelerator() const noexcept override {
        return traits::is_accelerator();
    }

    ccl::communicator_interface_ptr split(const ccl::comm_split_attr& attr) override;

    typed_base_communicator(ccl::unified_device_type&& device,
                            ccl::unified_context_type&& ctx,
                            size_t thread_idx,
                            size_t process_idx,
                            const ccl::comm_split_attr& attr);

    ccl::group_split_type get_topology_type() const override;
    ccl::device_topology_type get_topology_class() const override;

    void initialize_comm_addr(const ccl::device_index_type& device_id,
                              native::device_community_container<class_id>& community);

    bool is_ready() const override;

    native::ccl_driver_context_ptr get_native_context() {
        return native::get_runtime_context(context.get());
    }

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif /* CCL_ENABLE_SYCL */

    // Device community interface
    /*    template<class device_t>
    size_t get_device_count() const;

    template<class device_t>
    native::indexed_device_container<device_t>& get_devices();
*/
    // troubleshooting
    std::string to_string() const;

    native::device_community_container<class_id> device_community_impl;

    impl_t* get_impl() {
        return static_cast<impl_t*>(this);
    }
};
