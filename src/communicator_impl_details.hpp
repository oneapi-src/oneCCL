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
#include "oneapi/ccl/exception.hpp"
#include "common/comm/comm_interface.hpp"

#include "common/env/env.hpp"

namespace ccl {

/**
 * Base dispatcher for conversion vector to map
 */

template <class Context>
struct context_extractor {
    static const Context& extract(const Context& ctx) {
        return ctx;
    }
};

template <>
struct context_extractor<ccl::context> {
    static const typename unified_context_type::ccl_native_t& extract(const ccl::context& ctx) {
        return ctx.get_native();
    }
};

template <class impl>
struct comm_impl_base_dispatch {
    static void validate_contract(const size_t cluster_devices_size, const size_t table_size) {
        if (table_size == 0) {
            throw ccl::invalid_argument(
                "API", "create_communicators", "`local_rank_device_map` cannot be empty");
        }

        if (table_size > cluster_devices_size) {
            throw ccl::invalid_argument(
                "API",
                "create_communicators",
                "`local_rank_device_map` size: " + std::to_string(table_size) +
                    " must not exceed total size: " + std::to_string(cluster_devices_size));
        }

        /*
           Indicate that multiple devices are not supported
           Don't throw anything if comm_kernels=1 to enable our testing with partial functionality.
        */
        // if (table_size > 1 && !ccl::global_data::env().enable_comm_kernels) {
        //     throw ccl::unimplemented("API", "create_communicators", "for multiple devices");
        // }
    }
};

template <cl_backend_type type>
struct comm_impl_dispatch_selector {};

#if !defined(CCL_ENABLE_SYCL) and !defined(CCL_ENABLE_ZE)
template <>
struct comm_impl_dispatch_selector<cl_backend_type::empty_backend>
        : public comm_impl_base_dispatch<
              comm_impl_dispatch_selector<cl_backend_type::empty_backend>> {
    using base_t =
        comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::empty_backend>>;

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators_selector(
        const size_t cluster_devices_size, /*global devices count*/
        const map_class<int, DeviceType>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<ikvs_wrapper> kvs) {
        base_t::validate_contract(cluster_devices_size, local_rank_device_map.size());
        if (local_rank_device_map.size() != 1) {
            throw ccl::unsupported(
                "API",
                "create_communicators",
                "`local_rank_device_map` size: " + std::to_string(local_rank_device_map.size()) +
                    " but must be: 1 for your configuration: " + backend_traits::name());
        }

        LOG_TRACE("Create host communicator");

        ccl::communicator_interface_ptr impl =
            ccl::communicator_interface::create_communicator_impl(
                cluster_devices_size, local_rank_device_map.begin()->first, kvs);
        ccl::vector_class<ccl::communicator> ret;
        ret.push_back(ccl::communicator(std::move(impl)));
        return ret;
    }
};
#endif

#if defined(CCL_ENABLE_SYCL) and !defined(CCL_ENABLE_ZE)
template <>
struct comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl>
        : public comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl>> {
    using base_t =
        comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl>>;

    template <class ContextType>
    static vector_class<communicator> create_communicators_selector(
        const size_t cluster_devices_size, /*global devices count*/
        const map_class<int, cl::sycl::device>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<ikvs_wrapper> kvs) {
        base_t::validate_contract(cluster_devices_size, local_rank_device_map.size());

        auto it = local_rank_device_map.begin();
        const cl::sycl::device& dev = it->second;

        if (dev.is_host()) {
            it = std::find_if_not(
                local_rank_device_map.begin(),
                local_rank_device_map.end(),
                [](const typename map_class<int, cl::sycl::device>::value_type& val) {
                    return val.second.is_host();
                });
        }
        else if (dev.is_cpu()) {
            it = std::find_if_not(
                local_rank_device_map.begin(),
                local_rank_device_map.end(),
                [](const typename map_class<int, cl::sycl::device>::value_type& val) {
                    return val.second.is_cpu();
                });
        }
        else if (dev.is_gpu()) {
            it = std::find_if_not(
                local_rank_device_map.begin(),
                local_rank_device_map.end(),
                [](const typename map_class<int, cl::sycl::device>::value_type& val) {
                    return val.second.is_gpu();
                });
        }
        else if (dev.is_accelerator()) {
            it = std::find_if_not(
                local_rank_device_map.begin(),
                local_rank_device_map.end(),
                [](const typename map_class<int, cl::sycl::device>::value_type& val) {
                    return val.second.is_accelerator();
                });
        }
        else {
            throw ccl::invalid_argument(
                "API", "create_communicators", "invalid `cl::sycl::device` type");
        }

        if (it != local_rank_device_map.end()) {
            throw ccl::invalid_argument("API",
                                        "create_communicators",
                                        "mixed collection of `cl::sycl::device` are not supported");
        }

        if (local_rank_device_map.size() != 1) {
            throw ccl::unsupported(
                "API",
                "create_communicators",
                "`local_rank_device_map` size: " + std::to_string(local_rank_device_map.size()) +
                    " but must be: 1 for your configuration: " + backend_traits::name());
        }

        // if (dev.is_host() || dev.is_cpu())
        // {
        //     LOG_DEBUG("Create host communicator");

        //     ccl::communicator_interface_ptr impl =
        //         ccl::communicator_interface::create_communicator_impl(cluster_devices_size, local_rank_device_map.begin()->first, kvs);

        //     ccl::vector_class<ccl::communicator> ret;
        //     ret.push_back(ccl::communicator(std::move(impl)));
        //     return ret;
        // }

        /* reset iterator after std::find_if_not */
        it = local_rank_device_map.begin();
        int rank = it->first;
        auto& device = it->second;

        LOG_DEBUG(
            "Create single device communicator from SYCL device (sycl and !mgpu), after find_if rank ",
            rank);

        std::shared_ptr<atl_base_comm> atl =
            atl_comm_manager::create_atl_comm(cluster_devices_size, { rank }, kvs);

        ccl::communicator_interface_ptr impl =
            ccl::communicator_interface::create_communicator_impl(device,
                                                                  context,
                                                                  rank,
                                                                  cluster_devices_size,
                                                                  preview::create_comm_split_attr(),
                                                                  atl,
                                                                  ccl::group_split_type::single);

        ccl::vector_class<ccl::communicator> ret;
        ret.push_back(ccl::communicator(std::move(impl)));
        return ret;
    }
};
#endif //defined(CCL_ENABLE_SYCL) and !defined(CCL_ENABLE_ZE)

#if defined(CCL_ENABLE_ZE) and !defined(CCL_ENABLE_SYCL)
template <>
struct comm_impl_dispatch_selector<cl_backend_type::l0>
        : public comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::l0>> {
    using base_t = comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::l0>>;
};
#endif //defined(CCL_ENABLE_ZE) and !defined(CCL_ENABLE_SYCL)

#if defined(CCL_ENABLE_ZE) and defined(CCL_ENABLE_SYCL)
template <>
struct comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl_l0>
        : public comm_impl_base_dispatch<
              comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl_l0>> {
    using base_t =
        comm_impl_base_dispatch<comm_impl_dispatch_selector<cl_backend_type::dpcpp_sycl_l0>>;
};
#endif //defined(CCL_ENABLE_ZE) and defined(CCL_ENABLE_SYCL)
} // namespace ccl
