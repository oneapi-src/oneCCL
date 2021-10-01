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
#include "common/comm/l0/modules/kernel_functions.hpp"

namespace native {

namespace ring {

namespace bcast {

/**
 * Common args for all kernel types
 */

template <class native_t>
using buf_arg = thread_exchangable_arg<main_kernel_args::args_start_index, native_t*>;

template <class native_t>
using right_buf_arg = thread_exchangable_arg<main_kernel_args::args_start_index + 1, native_t*>;

using root_arg = arg<main_kernel_args::args_start_index + 2, size_t>;
using root_arg_type = typename root_arg::arg_type;

// IMPORTANT: the number and types of arguments must be the same in all classes,
// excluding arguments specific for numa/scaleout etc.
struct main_kernel
        : public execution_kernel<main_kernel, buf_arg<void>, right_buf_arg<void>, root_arg> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "bcast_execution";
    }

    using common_entry_buf_arg = buf_arg<processing_type>;

    using base = execution_kernel<main_kernel,
                                  buf_arg<processing_type>,
                                  right_buf_arg<processing_type>,
                                  root_arg>;

    using base::base;
};

struct numa_kernel
        : public execution_kernel<numa_kernel, buf_arg<void>, right_buf_arg<void>, root_arg> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "bcast_execution_numa";
    }

    using common_entry_buf_arg = buf_arg<processing_type>;

    using base = execution_kernel<numa_kernel,
                                  buf_arg<processing_type>,
                                  right_buf_arg<processing_type>,
                                  root_arg>;

    template <class ctx_params_t>
    void bind_data(const ctx_params_t& out_ctx_params) {
        // TODO not implemented
        (void)out_ctx_params;
        throw ccl::exception(std::string(__FUNCTION__) + " - not implemented for that kernel type");
    }

    using base::base;
};

struct ipc_kernel : public base_ipc_kernel<ipc_kernel,
                                           buf_arg<void>,
                                           stub_arg<main_kernel_args::args_start_index + 1>,
                                           stub_arg<main_kernel_args::args_start_index + 2>> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "ring_bcast_ipc";
    }

    using common_entry_buf_arg = buf_arg<processing_type>;

    using base = base_ipc_kernel<ipc_kernel,
                                 buf_arg<processing_type>,
                                 stub_arg<main_kernel_args::args_start_index + 1>,
                                 stub_arg<main_kernel_args::args_start_index + 2>>;

    template <class ipc_handles_t>
    void bind_data(const ipc_handles_t& ipc_handles) {
        auto recv_buf = reinterpret_cast<typename buf_arg<processing_type>::arg_type>(
            ipc_handles.at(0).get().pointer);
        this->template set_arg<buf_arg<processing_type>>(recv_buf);
    }

    using base::base;
};

struct scale_out_cpu_gw_kernel : public execution_kernel<scale_out_cpu_gw_kernel,
                                                         buf_arg<void>,
                                                         right_buf_arg<void>,
                                                         root_arg> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "bcast_execution_scale_out_cpu_gw";
    }

    using common_entry_buf_arg = buf_arg<processing_type>;

    using base = execution_kernel<scale_out_cpu_gw_kernel,
                                  buf_arg<processing_type>,
                                  right_buf_arg<processing_type>,
                                  root_arg>;

    template <class ctx_params_t>
    void bind_data(const ctx_params_t& out_ctx_params) {
        // TODO not implemented
        (void)out_ctx_params;
        throw ccl::exception(std::string(__FUNCTION__) + " - not implemented for that kernel type");
    }

    using base::base;
};

} // namespace bcast
} // namespace ring
} // namespace native
