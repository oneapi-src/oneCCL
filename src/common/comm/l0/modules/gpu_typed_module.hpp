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

#include <type_traits>
#include <memory>
#include "common/comm/l0/modules/base_entry_module.hpp"
#include "common/comm/l0/modules/modules_utils.hpp"
#include "common/comm/l0/communicator/base_communicator.hpp"

#include "common/comm/l0/modules/kernel_class.hpp"

namespace native {

template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl,
          template <typename>
          class kernel_scale_out_cpu_gw_function_impl>
struct real_gpu_typed_module : private gpu_module_base,
                               public kernel_class<type, kernel_function_impl>,
                               public kernel_class<type, kernel_numa_function_impl>,
                               public kernel_class<type, kernel_scale_out_cpu_gw_function_impl> {
    using handle = gpu_module_base::handle;

    using main_class = kernel_class<type, kernel_function_impl>;
    using numa_class = kernel_class<type, kernel_numa_function_impl>;
    using scale_out_cpu_gw_class = kernel_class<type, kernel_scale_out_cpu_gw_function_impl>;

    using self_t = real_gpu_typed_module;

    real_gpu_typed_module(handle module_handle) : gpu_module_base(module_handle) {
        LOG_DEBUG("Real gpu module created: ",
                  ccl_coll_type_to_str(type),
                  ", modules handle: ",
                  (void*)module);
        ccl_tuple_for_each(main_class::value,
                           detail::kernel_entry_initializer<type>(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        ccl_tuple_for_each(numa_class::value,
                           detail::kernel_entry_initializer<type>(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        LOG_DEBUG("Imported functions count: ", functions.size());
    }

    handle get() const {
        return module;
    }

    template <class specific_kernel_class>
    specific_kernel_class& get_class() {
        static_assert(
            std::is_base_of<specific_kernel_class, self_t>::value,
            "Relationship IS-A `specific_kernel_class` for `real_gpu_typed_module` failed");
        return static_cast<specific_kernel_class&>(*this);
    }

    ~real_gpu_typed_module() {
        LOG_DEBUG("Real gpu module destroyed: ",
                  ccl_coll_type_to_str(type),
                  ", modules handle: ",
                  (void*)module);
    }
};

//2) virtual ipc_gpu_typed_module
template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl,
          template <typename>
          class kernel_scale_out_cpu_gw_function_impl>
struct ipc_gpu_typed_module : private gpu_module_base,
                              public kernel_class<type, kernel_function_impl> {
    using main_class = kernel_class<type, kernel_function_impl>;

    using self_t = ipc_gpu_typed_module;

    using handle = gpu_module_base::handle;

    ipc_gpu_typed_module(handle module_handle) : gpu_module_base(nullptr) {
        LOG_DEBUG("Remote gpu module created: ", ccl_coll_type_to_str(type));
        ccl_tuple_for_each(main_class::value,
                           detail::kernel_entry_initializer<type>(
                               [](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return nullptr;
                               }));
        LOG_DEBUG("No need to import functions");
    }

    template <class specific_kernel_class>
    specific_kernel_class& get_class() {
        static_assert(
            std::is_base_of<specific_kernel_class, self_t>::value,
            "Relationship IS-A `specific_kernel_class` for `ipc_gpu_typed_module` failed");
        return static_cast<specific_kernel_class&>(*this);
    }

    ~ipc_gpu_typed_module() = default;
};

//3) virtual gpu module
template <ccl_coll_type type,
          template <typename>
          class kernel_function_impl,
          template <typename>
          class kernel_numa_function_impl,
          template <typename>
          class kernel_scale_out_cpu_gw_function_impl>
struct virtual_gpu_typed_module : private gpu_module_base,
                                  public kernel_class<type, kernel_function_impl>,
                                  public kernel_class<type, kernel_numa_function_impl>,
                                  public kernel_class<type, kernel_scale_out_cpu_gw_function_impl> {
    // TODO: use real_referenced_module to reduce given params
    using real_referenced_module = real_gpu_typed_module<type,
                                                         kernel_function_impl,
                                                         kernel_numa_function_impl,
                                                         kernel_scale_out_cpu_gw_function_impl>;

    using main_class = kernel_class<type, kernel_function_impl>;
    using numa_class = kernel_class<type, kernel_numa_function_impl>;
    using scale_out_cpu_gw_class = kernel_class<type, kernel_scale_out_cpu_gw_function_impl>;

    using self_t = virtual_gpu_typed_module;

    using handle = typename real_referenced_module::handle;

    virtual_gpu_typed_module(std::shared_ptr<real_referenced_module> real_module)
            : gpu_module_base(real_module->get()),
              real_module_ref(real_module) {
        LOG_DEBUG("Virtual gpu module created:", ccl_coll_type_to_str(type));
        ccl_tuple_for_each(main_class::value,
                           detail::kernel_entry_initializer<type>(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));
        ccl_tuple_for_each(numa_class::value,
                           detail::kernel_entry_initializer<type>(
                               [this](const std::string& name) -> gpu_module_base::kernel_handle {
                                   return this->import_kernel(name);
                               }));

        LOG_DEBUG("Linked functions count: ", functions.size());
    }

    template <class specific_kernel_class>
    specific_kernel_class& get_class() {
        static_assert(
            std::is_base_of<specific_kernel_class, self_t>::value,
            "Relationship IS-A `specific_kernel_class` for `virtual_gpu_typed_module` failed");
        return static_cast<specific_kernel_class&>(*this);
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
};

#define DEFINE_SPECIFIC_GPU_MODULE_CLASS(module_type, \
                                         base_module_type, \
                                         coll_type, \
                                         mode, \
                                         export_function, \
                                         export_numa_function, \
                                         export_scale_out_cpu_gw_function) \
    template <ccl::group_split_type topology> \
    struct module_type<coll_type, topology, mode> \
            : public base_module_type<coll_type, \
                                      export_function, \
                                      export_numa_function, \
                                      export_scale_out_cpu_gw_function> { \
        using base = base_module_type<coll_type, \
                                      export_function, \
                                      export_numa_function, \
                                      export_scale_out_cpu_gw_function>; \
        using base::handle; \
\
        module_type<coll_type, topology, mode>(handle module_handle) : base(module_handle) {} \
    }

#define DEFINE_VIRTUAL_GPU_MODULE_CLASS( \
    coll_type, mode, export_function, export_numa_function, export_scale_out_cpu_gw_function) \
    template <ccl::group_split_type topology> \
    struct virtual_device_coll_module<coll_type, topology, mode> \
            : public virtual_gpu_typed_module<coll_type, \
                                              export_function, \
                                              export_numa_function, \
                                              export_scale_out_cpu_gw_function> { \
        using base = virtual_gpu_typed_module<coll_type, \
                                              export_function, \
                                              export_numa_function, \
                                              export_scale_out_cpu_gw_function>; \
        using base::handle; \
        using real_referenced_module = typename base::real_referenced_module; \
\
        virtual_device_coll_module<coll_type, topology, mode>( \
            std::shared_ptr<real_referenced_module> real_module) \
                : base(real_module) {} \
    }

} // namespace native
