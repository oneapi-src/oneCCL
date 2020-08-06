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
#include "common/comm/l0/modules/kernel_argument_types.hpp"

namespace native {
// kernel with its argument collection
template <class... arguments>
struct kernel_data_storage {
    using func_args_t = std::tuple<arguments...>;
    func_args_t args;
    ze_kernel_handle_t handle;

    // generic getter
    template <class kernel_argument>
    typename kernel_argument::return_t get_arg() const {
        return std::get<kernel_argument::index>(args).load();
    }

    // generic test value
    template <class... kernel_argument>
    bool test_args() const {
        std::array<bool, sizeof...(kernel_argument)> ready{
            std::get<kernel_argument::index>(args).test()...
        };

        return std::all_of(ready.begin(), ready.end(), [](bool v) {
            return v;
        });
    }

    // generic setter
    template <class kernel_argument,
              class = typename std::enable_if<
                  not std::is_pointer<typename kernel_argument::arg_type>::value>::type>
    void set_arg(typename kernel_argument::arg_type& new_val) {
        std::get<kernel_argument::index>(args).store(new_val);
    }

    template <class kernel_argument,
              class = typename std::enable_if<
                  std::is_pointer<typename kernel_argument::arg_type>::value>::type>
    void set_arg(typename kernel_argument::arg_type new_val) {
        std::get<kernel_argument::index>(args).store(new_val);
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "handle: " << handle << "\n{\n";
        details::args_printer func(ss);
        ccl_tuple_for_each(args, func);
        ss << "}" << std::endl;
        return ss.str();
    }
};

// major kernel args
enum main_kernel_args { rank_index = 0, size_index = 1, args_start_index };

//main kernel - used for GPU program execution
template <class Impl, class... arguments>
struct execution_kernel : public kernel_data_storage<arg<main_kernel_args::rank_index, size_t>,
                                                     arg<main_kernel_args::size_index, size_t>,
                                                     arguments...> {
    using base = kernel_data_storage<arg<main_kernel_args::rank_index, size_t>,
                                     arg<main_kernel_args::size_index, size_t>,
                                     arguments...>;
    using base::args;
    using base::handle;

    using rank_type = size_t;
    using size_type = size_t;

    static constexpr const char* name() {
        return Impl::specific_name();
    }

    void set_rank(rank_type rank) {
        ze_result_t result = zeKernelSetArgumentValue(
            handle, main_kernel_args::rank_index, sizeof(rank_type), &rank);
        if (result != ZE_RESULT_SUCCESS) {
            CCL_THROW("Cannot set rank_argument argument in kernel: ", name());
        }

        std::get<main_kernel_args::rank_index>(args).store(rank);
    }

    rank_type get_rank() const {
        return std::get<main_kernel_args::rank_index>(args).load().second;
    }

    void set_size(size_type size) {
        ze_result_t result = zeKernelSetArgumentValue(
            handle, main_kernel_args::size_index, sizeof(size_type), &size);
        if (result != ZE_RESULT_SUCCESS) {
            CCL_THROW("Cannot set size_argument argument in kernel: ", name());
        }

        std::get<main_kernel_args::size_index>(args).store(size);
    }

    size_type get_size() const {
        return std::get<main_kernel_args::size_index>(args).load().second;
    }

    // modified setter
    template <class kernel_argument,
              class = typename std::enable_if<
                  not std::is_pointer<typename kernel_argument::arg_type>::value>::type>
    void set_arg(typename kernel_argument::arg_type& new_val) {
        LOG_TRACE("Function: ",
                  name(),
                  ", handle: ",
                  handle,
                  " - set_arg() index: ",
                  kernel_argument::index,
                  ", value: ",
                  new_val);
        ze_result_t result = zeKernelSetArgumentValue(
            handle, kernel_argument::index, sizeof(typename kernel_argument::arg_type), &new_val);
        if (result != ZE_RESULT_SUCCESS) {
            CCL_THROW("Cannot set kernel argument by index: ",
                      kernel_argument::index,
                      ", kernel: ",
                      name(),
                      ", handle: ",
                      handle);
        }

        base::template set_arg<kernel_argument>(new_val);
    }

    template <class kernel_argument,
              class = typename std::enable_if<
                  std::is_pointer<typename kernel_argument::arg_type>::value>::type>
    void set_arg(typename kernel_argument::arg_type new_val) {
        LOG_TRACE("Function: ",
                  name(),
                  ", handle: ",
                  handle,
                  " - set_arg(pointer) index: ",
                  kernel_argument::index,
                  ", value: ",
                  new_val);
        ze_result_t result = zeKernelSetArgumentValue(handle,
                                                      kernel_argument::index,
                                                      sizeof(typename kernel_argument::arg_type),
                                                      &new_val); //& from pointer
        if (result != ZE_RESULT_SUCCESS) {
            CCL_THROW("Cannot set kernel argument by pointer index: ",
                      kernel_argument::index,
                      ", kernel: ",
                      name(),
                      ", handle: ",
                      handle);
        }

        base::template set_arg<kernel_argument>(new_val);
    }
};

// ipc_kernel - used for GPU data synchronization only
template <class Impl, class... arguments>
struct ipc_kernel : public kernel_data_storage<arg<main_kernel_args::rank_index, size_t>,
                                               arg<main_kernel_args::size_index, size_t>,
                                               arguments...> {
    using base = kernel_data_storage<arg<main_kernel_args::rank_index, size_t>,
                                     arg<main_kernel_args::size_index, size_t>,
                                     arguments...>;
    using base::args;
    using base::handle;
    static constexpr const char* name() {
        return Impl::specific_name();
    }

    template <class kernel_argument,
              class = typename std::enable_if<not std::is_pointer<kernel_argument>::value>::type>
    void set_arg(typename kernel_argument::arg_type& new_val) {
        LOG_TRACE("Function: ",
                  name(),
                  ", handle: ",
                  handle,
                  " - set_arg() index: ",
                  kernel_argument::index,
                  ", value: ",
                  new_val);
        base::template set_arg<kernel_argument>(new_val);
    }

    template <class kernel_argument,
              class = typename std::enable_if<std::is_pointer<kernel_argument>::value>::type>
    void set_arg(typename kernel_argument::arg_type new_val) {
        LOG_TRACE("Function: ",
                  name(),
                  ", handle: ",
                  handle,
                  " - set_arg(pointer) index: ",
                  kernel_argument::index,
                  ", value: ",
                  new_val);
        base::template set_arg<kernel_argument>(new_val);
    }
};

} // namespace native
