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

namespace a2a {

namespace allreduce {

/**
 * Common args for all kernel types
 */

// own
using send_buf_size_arg = arg<main_kernel_args::args_start_index, size_t>;
using send_buf_size_arg_type = typename send_buf_size_arg::arg_type;

template <class native_t>
using send_buf_arg = arg<main_kernel_args::args_start_index + 1, native_t*>;

template <class native_t>
using recv_buf_arg = arg<main_kernel_args::args_start_index + 2, native_t*>;

template <class native_t>
using tmp_recv_buf_arg = external_arg<main_kernel_args::args_start_index + 3, native_t*>;

using income_data_flag_arg = external_arg<main_kernel_args::args_start_index + 4, int*>;
using income_data_flag_arg_type = typename income_data_flag_arg::arg_type;

using ready_to_recv_flag_arg = external_arg<main_kernel_args::args_start_index + 5, int*>;
using ready_to_recv_flag_arg_type = typename ready_to_recv_flag_arg::arg_type;

using local_barrier_flag_arg = arg<main_kernel_args::args_start_index + 6, int*>;
using local_barrier_flag_arg_type = typename local_barrier_flag_arg::arg_type;

// right
template <class native_t>
using right_tmp_recv_buf_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 7, native_t*>;

using right_income_data_flag_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 8, int*>;

using right_ready_to_recv_flag_arg =
    thread_exchangable_arg<main_kernel_args::args_start_index + 9, int*>;

// IMPORTANT: the number and types of arguments must be the same in all classes,
// excluding arguments specific for numa/scaleout etc.
struct main_kernel : public execution_kernel<main_kernel,
                                             send_buf_size_arg,
                                             send_buf_arg<void>,
                                             recv_buf_arg<void>,
                                             tmp_recv_buf_arg<void>,
                                             income_data_flag_arg,
                                             ready_to_recv_flag_arg,
                                             local_barrier_flag_arg,
                                             right_tmp_recv_buf_arg<void>,
                                             right_income_data_flag_arg,
                                             right_ready_to_recv_flag_arg> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allreduce_execution";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = execution_kernel<main_kernel,
                                  send_buf_size_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  tmp_recv_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  local_barrier_flag_arg,
                                  right_tmp_recv_buf_arg<processing_type>,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg>;

    using base::base;
};

struct numa_kernel
        : public execution_kernel<numa_kernel,
                                  send_buf_size_arg,
                                  send_buf_arg<void>,
                                  recv_buf_arg<void>,
                                  tmp_recv_buf_arg<void>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  local_barrier_flag_arg,
                                  right_tmp_recv_buf_arg<void>,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg,

                                  // numa-specific args
                                  permanent_arg<main_kernel_args::args_start_index + 10, void*>,
                                  permanent_arg<main_kernel_args::args_start_index + 11, int*>> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allreduce_execution_numa";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    // event data
    using event_prod_chunk_mem_arg =
        permanent_arg<main_kernel_args::args_start_index + 10, processing_type*>;
    using event_prod_chunk_mem_arg_type = typename event_prod_chunk_mem_arg::arg_type;

    using event_prod_bytes_arg = permanent_arg<main_kernel_args::args_start_index + 11, int*>;
    using event_prod_bytes_arg_type = typename event_prod_bytes_arg::arg_type;

    using base = execution_kernel<numa_kernel,
                                  send_buf_size_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  tmp_recv_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  local_barrier_flag_arg,
                                  right_tmp_recv_buf_arg<processing_type>,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg,
                                  event_prod_chunk_mem_arg,
                                  event_prod_bytes_arg>;

    using base::base;
};

struct ipc_kernel : public base_ipc_kernel<ipc_kernel,
                                           stub_arg<main_kernel_args::args_start_index>,
                                           stub_arg<main_kernel_args::args_start_index + 1>,
                                           stub_arg<main_kernel_args::args_start_index + 2>,
                                           tmp_recv_buf_arg<void>,
                                           income_data_flag_arg,
                                           ready_to_recv_flag_arg,
                                           stub_arg<main_kernel_args::args_start_index + 6>,
                                           stub_arg<main_kernel_args::args_start_index + 7>,
                                           stub_arg<main_kernel_args::args_start_index + 8>,
                                           stub_arg<main_kernel_args::args_start_index + 9>> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "a2a_allreduce_ipc";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    using base = base_ipc_kernel<ipc_kernel,
                                 stub_arg<main_kernel_args::args_start_index>,
                                 stub_arg<main_kernel_args::args_start_index + 1>,
                                 stub_arg<main_kernel_args::args_start_index + 2>,
                                 tmp_recv_buf_arg<processing_type>,
                                 income_data_flag_arg,
                                 ready_to_recv_flag_arg,
                                 stub_arg<main_kernel_args::args_start_index + 6>,
                                 stub_arg<main_kernel_args::args_start_index + 7>,
                                 stub_arg<main_kernel_args::args_start_index + 8>,
                                 stub_arg<main_kernel_args::args_start_index + 9>>;

    using base::base;
};

struct scale_out_cpu_gw_kernel
        : public execution_kernel<scale_out_cpu_gw_kernel,
                                  send_buf_size_arg,
                                  send_buf_arg<void>,
                                  recv_buf_arg<void>,
                                  tmp_recv_buf_arg<void>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  local_barrier_flag_arg,
                                  right_tmp_recv_buf_arg<void>,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg,

                                  // scaleout-specific args
                                  permanent_arg<main_kernel_args::args_start_index + 10, void*>,
                                  permanent_arg<main_kernel_args::args_start_index + 11, int*>> {
    using processing_type = void;

    static constexpr const char* specific_name() {
        return "allreduce_execution_scale_out_cpu_gw";
    }

    using common_entry_buf_size_arg = send_buf_size_arg;
    using common_entry_buf_arg = send_buf_arg<processing_type>;

    // event data
    using event_prod_chunk_mem_arg =
        permanent_arg<main_kernel_args::args_start_index + 10, processing_type*>;
    using event_prod_chunk_mem_arg_type = typename event_prod_chunk_mem_arg::arg_type;

    using event_prod_bytes_arg = permanent_arg<main_kernel_args::args_start_index + 11, int*>;
    using event_prod_bytes_arg_type = typename event_prod_bytes_arg::arg_type;

    using base = execution_kernel<scale_out_cpu_gw_kernel,
                                  send_buf_size_arg,
                                  send_buf_arg<processing_type>,
                                  recv_buf_arg<processing_type>,
                                  tmp_recv_buf_arg<processing_type>,
                                  income_data_flag_arg,
                                  ready_to_recv_flag_arg,
                                  local_barrier_flag_arg,
                                  right_tmp_recv_buf_arg<processing_type>,
                                  right_income_data_flag_arg,
                                  right_ready_to_recv_flag_arg,
                                  event_prod_chunk_mem_arg,
                                  event_prod_bytes_arg>;

    using base::base;
};

} // namespace allreduce
} // namespace a2a
} // namespace native
