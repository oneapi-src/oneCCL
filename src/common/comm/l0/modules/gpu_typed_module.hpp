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
#include <memory>
#include "common/comm/l0/modules/base_entry_module.hpp"
#include "common/comm/l0/modules/modules_utils.hpp"

namespace native {

template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl>
struct real_gpu_typed_module : private gpu_module_base {
    template <class native_data_type>
    using kernel = kernel_function_impl<native_data_type>;

    template <class... native_data_types>
    using kernels = std::tuple<kernel<native_data_types>...>;

    template <class native_data_type>
    using kernel_numa = kernel_numa_function_impl<native_data_type>;

    template <class... native_data_types>
    using kernels_numa = std::tuple<kernel_numa<native_data_types>...>;

    using handle = gpu_module_base::handle;

    real_gpu_typed_module(handle module_handle) : gpu_module_base(module_handle) {
        LOG_DEBUG("Real gpu module created: ",
                  ccl_coll_type_to_str(type),
                  ", modules handle: ",
                  (void*)module);
        ccl_tuple_for_each(kernel_main_functions,
                           detail::kernel_entry_initializer(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        ccl_tuple_for_each(kernel_numa_functions,
                           detail::kernel_entry_initializer(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        LOG_DEBUG("Imported functions count: ", functions.size());
    }

    handle get() const {
        return module;
    }

    template <class native_data_type>
    kernel<native_data_type>& get_main_function() {
        return const_cast<kernel<native_data_type>&>(
            static_cast<const real_gpu_typed_module*>(this)->get_main_function<native_data_type>());
    }

    template <class native_data_type>
    const kernel<native_data_type>& get_main_function() const {
        return ccl_tuple_get<kernel<native_data_type>>(kernel_main_functions);
    }

    ~real_gpu_typed_module() {
        LOG_DEBUG("Real gpu module destroyed: ",
                  ccl_coll_type_to_str(type),
                  ", modules handle: ",
                  (void*)module);
    }

protected:
    kernels<SUPPORTED_KERNEL_NATIVE_DATA_TYPES> kernel_main_functions;
    kernels_numa<SUPPORTED_KERNEL_NATIVE_DATA_TYPES> kernel_numa_functions;
};

template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl>
struct ipc_gpu_typed_module : private gpu_module_base {
    template <class native_data_type>
    using kernel = kernel_function_impl<native_data_type>;

    template <class... native_data_types>
    using kernels = std::tuple<kernel<native_data_types>...>;

    template <class native_data_type>
    using kernel_numa = kernel_numa_function_impl<native_data_type>;

    template <class... native_data_types>
    using kernels_numa = std::tuple<kernel_numa<native_data_types>...>;

    using handle = gpu_module_base::handle;

    ipc_gpu_typed_module(handle module_handle) : gpu_module_base(nullptr) {
        LOG_DEBUG("Remote gpu module created: ", ccl_coll_type_to_str(type));
        ccl_tuple_for_each(kernel_main_functions,
                           detail::kernel_entry_initializer(
                               [](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return nullptr;
                               }));
        LOG_DEBUG("No need to import functions");
    }

    template <class native_data_type>
    kernel<native_data_type>& get_main_function() {
        return const_cast<kernel<native_data_type>&>(
            static_cast<const ipc_gpu_typed_module*>(this)->get_main_function<native_data_type>());
    }

    template <class native_data_type>
    const kernel<native_data_type>& get_main_function() const {
        return ccl_tuple_get<kernel<native_data_type>>(kernel_main_functions);
    }
    ~ipc_gpu_typed_module() = default;

protected:
    kernels<SUPPORTED_KERNEL_NATIVE_DATA_TYPES> kernel_main_functions;
};

//3) virtual gpu module
template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl>
struct virtual_gpu_typed_module : private gpu_module_base {
    using real_referenced_module =
        real_gpu_typed_module<type, kernel_function_impl, kernel_numa_function_impl>;

    template <class native_data_type>
    using kernel =
        typename real_referenced_module::template kernel<native_data_type>; //The same as real

    template <class... native_data_types>
    using kernels = std::tuple<kernel<native_data_types>...>;

    template <class native_data_type>
    using kernel_numa =
        typename real_referenced_module::template kernel_numa<native_data_type>; //The same as real

    template <class... native_data_types>
    using kernels_numa = std::tuple<kernel_numa<native_data_types>...>;

    using handle = typename real_referenced_module::handle;

    virtual_gpu_typed_module(std::shared_ptr<real_referenced_module> real_module)
            : gpu_module_base(real_module->get()),
              real_module_ref(real_module) {
        LOG_DEBUG("Virtual gpu module created:", ccl_coll_type_to_str(type));
        ccl_tuple_for_each(kernel_main_functions,
                           detail::kernel_entry_initializer(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));
        ccl_tuple_for_each(kernel_numa_functions,
                           detail::kernel_entry_initializer(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        LOG_DEBUG("Linked functions count: ", functions.size());
    }

    template <class native_data_type>
    kernel<native_data_type>& get_main_function() {
        return const_cast<kernel<native_data_type>&>(
            static_cast<const virtual_gpu_typed_module*>(this)
                ->get_main_function<native_data_type>());
    }

    template <class native_data_type>
    const kernel<native_data_type>& get_main_function() const {
        return ccl_tuple_get<kernel<native_data_type>>(kernel_main_functions);
    }

    std::shared_ptr<real_referenced_module> real_module_ref;

    ~virtual_gpu_typed_module() {
        LOG_DEBUG("Virtual gpu module destroyed: ",
                  ccl_coll_type_to_str(type),
                  ", modules handle: ",
                  (void*)module);
        module = nullptr; //real module owner will destroy it
        release();
    }

protected:
    kernels<SUPPORTED_KERNEL_NATIVE_DATA_TYPES> kernel_main_functions;
    kernels_numa<SUPPORTED_KERNEL_NATIVE_DATA_TYPES> kernel_numa_functions;
};

#define DEFINE_SPECIFIC_GPU_MODULE_CLASS( \
    module_type, base_module_type, coll_type, mode, export_function, export_numa_function) \
    template <ccl::device_group_split_type topology> \
    struct module_type<coll_type, topology, mode> \
            : public base_module_type<coll_type, export_function, export_numa_function> { \
        using base = base_module_type<coll_type, export_function, export_numa_function>; \
        using base::kernel; \
        using base::kernel_numa; \
        using base::handle; \
        static constexpr ccl_coll_type get_coll_type() { \
            return coll_type; \
        } \
        static constexpr ccl::device_group_split_type get_topology_type() { \
            return topology; \
        } \
        static constexpr ccl::device_topology_type get_topology_class() { \
            return mode; \
        } \
\
        module_type<coll_type, topology, mode>(handle module_handle) : base(module_handle) {} \
    }
/*

#define DEFINE_SPECIFIC_GPU_MODULE_CLASS(module_type, base_module_type, export_function, export_numa_function)     \
template<ccl_coll_type type, ccl::device_group_split_type topology>                                   \
using module_type = base_module_type<type, topology, export_function, export_numa_function>;
*/

#define DEFINE_VIRTUAL_GPU_MODULE_CLASS(coll_type, mode, export_function, export_numa_function) \
    template <ccl::device_group_split_type topology> \
    struct virtual_device_coll_module<coll_type, topology, mode> \
            : public virtual_gpu_typed_module<coll_type, export_function, export_numa_function> { \
        using base = virtual_gpu_typed_module<coll_type, export_function, export_numa_function>; \
        using base::kernel; \
        using base::kernel_numa; \
        using base::handle; \
        using real_referenced_module = typename base::real_referenced_module; \
        static constexpr ccl_coll_type get_coll_type() { \
            return coll_type; \
        } \
        static constexpr ccl::device_group_split_type get_topology_type() { \
            return topology; \
        } \
        static constexpr ccl::device_topology_type get_topology_class() { \
            return mode; \
        } \
\
        virtual_device_coll_module<coll_type, topology, mode>( \
            std::shared_ptr<real_referenced_module> real_module) \
                : base(real_module) {} \
    }

} // namespace native
