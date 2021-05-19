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

namespace allgatherv {

/**
 * Common args for all kernel types
 */

using send_buf_size_arg = arg<main_kernel_args::args_start_index, size_t>;
using send_buf_size_arg_type = typename send_buf_size_arg::arg_type;

using recv_elem_counts_buf_arg = arg<main_kernel_args::args_start_index + 1, size_t*>;
using recv_elem_counts_buf_arg_type = typename recv_elem_counts_buf_arg::arg_type;

using recv_elem_offsets_buf_arg = arg<main_kernel_args::args_start_index + 2, size_t*>;
using recv_elem_offsets_buf_arg_type = typename recv_elem_offsets_buf_arg::arg_type;

template <class native_t>
using send_buf_arg = arg<main_kernel_args::args_start_index + 3, native_t*>;

template <class native_t>
using recv_buf_arg = external_arg<main_kernel_args::args_start_index + 4, native_t*>;

template <class native_t>
using right_output_buf_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 5, native_t*>;

using income_data_flag_arg = external_arg<main_kernel_args::args_start_index + 6, int*>;
using income_data_flag_arg_type = typename income_data_flag_arg::arg_type;

using ready_to_recv_flag_arg = external_arg<main_kernel_args::args_start_index + 7, int*>;
using ready_to_recv_flag_arg_type = typename ready_to_recv_flag_arg::arg_type;

using right_income_data_flag_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 8, int*>;

using right_ready_to_recv_flag_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 9, int*>;

// IMPORTANT: the number and types of arguments must be the same in all classes,
// excluding arguments specific for numa/scaleout etc.
struct main_kernel
        : public execution_kernel<main_kernel,
                                  send_buf_size_arg, // elems_count
                                  recv_elem_counts_buf_arg, // recv_elem_counts_buf
                                  recv_elem_offsets_buf_arg, // recv_elem_offsets_buf
                                  send_buf_arg<void>, // send_buf
                                  recv_buf_arg<void>, // recv_buf (output_buffer)
                                  right_output_buf_arg<void>, // right_output_buffer
                                  income_data_flag_arg, // left_wrote_to_me_flag
                                  ready_to_recv_flag_arg, // i_ready_to_receive_flag
                                  right_income_data_flag_arg, // i_send_to_right_flag
                                  right_ready_to_recv_flag_arg> // right_ready_to_recv_flag
{
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allgatherv_execution";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = execution_kernel<main_kernel,
                                  send_buf_size_arg,
                                  recv_elem_counts_buf_arg,
                                  recv_elem_offsets_buf_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  right_output_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg>;

    using base::base;
};

// IMPORTANT: the params order is default, see *algatherv*.cl for that
struct numa_kernel
        : public execution_kernel<numa_kernel,
                                  send_buf_size_arg, // elems_count
                                  recv_elem_counts_buf_arg, // recv_elem_counts_buf
                                  recv_elem_offsets_buf_arg, // recv_elem_offsets_buf
                                  send_buf_arg<void>, // send_buf
                                  recv_buf_arg<void>, // recv_buf (output_buffer)
                                  right_output_buf_arg<void>, // right_output_buffer
                                  income_data_flag_arg, // left_wrote_to_me_flag
                                  ready_to_recv_flag_arg, // i_ready_to_receive_flag
                                  right_income_data_flag_arg, // i_send_to_right_flag
                                  right_ready_to_recv_flag_arg> // right_ready_to_recv_flag
{
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allgatherv_execution_numa";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = execution_kernel<numa_kernel,
                                  send_buf_size_arg,
                                  recv_elem_counts_buf_arg,
                                  recv_elem_offsets_buf_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  right_output_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg>;

    template <class ctx_params_t>
    void bind_data(const ctx_params_t& out_ctx_params) {
        // TODO not implemented
        (void)out_ctx_params;
        throw ccl::exception(std::string(__FUNCTION__) + " - not implemented for that kernel type");
    }

    using base::base;
};

struct ipc_kernel : public base_ipc_kernel<ipc_kernel,
                                           send_buf_size_arg, // elems_count
                                           recv_elem_counts_buf_arg, // recv_elem_counts_buf
                                           recv_elem_offsets_buf_arg, // recv_elem_offsets_buf
                                           send_buf_arg<void>, // send_buf
                                           recv_buf_arg<void>, // recv_buf (output_buffer)
                                           right_output_buf_arg<void>, // right_output_buffer
                                           income_data_flag_arg, // left_wrote_to_me_flag
                                           ready_to_recv_flag_arg, // i_ready_to_receive_flag
                                           right_income_data_flag_arg, // i_send_to_right_flag
                                           right_ready_to_recv_flag_arg> // right_ready_to_recv_flag
{
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "ring_allgatherv_ipc";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = base_ipc_kernel<ipc_kernel,
                                 send_buf_size_arg,
                                 recv_elem_counts_buf_arg,
                                 recv_elem_offsets_buf_arg,
                                 send_buf_arg<processing_type>,
                                 recv_buf_arg<processing_type>,
                                 right_output_buf_arg<processing_type>,
                                 income_data_flag_arg,
                                 ready_to_recv_flag_arg,
                                 right_income_data_flag_arg,
                                 right_ready_to_recv_flag_arg>;

    template <class ipc_handles_t>
    void bind_data(const ipc_handles_t& ipc_handles) {
        auto recv_buf = reinterpret_cast<typename recv_buf_arg<processing_type>::arg_type>(
            ipc_handles.at(0).get().pointer);
        this->template set_arg<recv_buf_arg<processing_type>>(recv_buf);

        auto income_data_flag =
            reinterpret_cast<income_data_flag_arg_type>(ipc_handles.at(1).get().pointer);
        this->template set_arg<income_data_flag_arg>(income_data_flag);

        auto ready_to_recv_flag =
            reinterpret_cast<ready_to_recv_flag_arg_type>(ipc_handles.at(2).get().pointer);
        this->template set_arg<ready_to_recv_flag_arg>(ready_to_recv_flag);
    }

    using base::base;
};

struct scale_out_cpu_gw_kernel
        : public execution_kernel<scale_out_cpu_gw_kernel,
                                  send_buf_size_arg, // elems_count
                                  recv_elem_counts_buf_arg, // recv_elem_counts_buf
                                  recv_elem_offsets_buf_arg, // recv_elem_offsets_buf
                                  send_buf_arg<void>, // send_buf
                                  recv_buf_arg<void>, // recv_buf (output_buffer)
                                  right_output_buf_arg<void>, // right_output_buffer
                                  income_data_flag_arg, // left_wrote_to_me_flag
                                  ready_to_recv_flag_arg, // i_ready_to_receive_flag
                                  right_income_data_flag_arg, // i_send_to_right_flag
                                  right_ready_to_recv_flag_arg> // right_ready_to_recv_flag
{
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allgatherv_execution_scale_out_cpu_gw";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = execution_kernel<scale_out_cpu_gw_kernel,
                                  send_buf_size_arg,
                                  recv_elem_counts_buf_arg,
                                  recv_elem_offsets_buf_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  right_output_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg>;

    template <class ctx_params_t>
    void bind_data(const ctx_params_t& out_ctx_params) {
        // TODO not implemented
        (void)out_ctx_params;
        throw ccl::exception(std::string(__FUNCTION__) + " - not implemented for that kernel type");
    }

    using base::base;
};

} // namespace allgatherv
} // namespace ring
} // namespace native
