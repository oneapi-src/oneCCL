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
#include <initializer_list>
#include <atomic>

#include "sched/entry/l0/l0_entry.hpp"

//TODO L0 Workaround

namespace native {
template <class gpu_comm_impl, ccl::group_split_type topology>
class l0_allgatherv_typed_entry : public base_gpu_entry<gpu_comm_impl,
                                                        topology,
                                                        ccl::device_topology_type::ring,
                                                        ccl_coll_allgatherv> {
public:
    friend class ccl_gpu_comm;
    friend class ccl_virtual_gpu_comm;

    using base = base_gpu_entry<gpu_comm_impl,
                                topology,
                                ccl::device_topology_type::ring,
                                ccl_coll_allgatherv>;
    using base::parent_communicator;
    using base::comm_addr;
    using base::req;
    using base::status;
    using base::launch_args;
    using base::kernel_router;
    using base::get_ctx;
    using base::get_local_kernel;
    using kernel_main_typed = ring::allgatherv::main_kernel;
    using processing_type = void;

    using income_data_flag_gpu_type =
        typename std::remove_pointer<typename ring::allgatherv::income_data_flag_arg_type>::type;
    using ready_to_recv_flag_gpu_type =
        typename std::remove_pointer<typename ring::allgatherv::ready_to_recv_flag_arg_type>::type;
    using recv_counts_typed_entry_type = typename std::remove_pointer<
        typename ring::allgatherv::recv_elem_counts_buf_arg_type>::type;
    using recv_offsets_typed_entry_type = typename std::remove_pointer<
        typename ring::allgatherv::recv_elem_offsets_buf_arg_type>::type;

    static constexpr const char* class_name() noexcept {
        return "L0_ALLGATHERV_TYPED";
    }

    static constexpr ccl_coll_type type() noexcept {
        return ccl_coll_allgatherv;
    }

    l0_allgatherv_typed_entry() = delete;
    l0_allgatherv_typed_entry(
        ccl_sched* sched,
        std::shared_ptr<gpu_comm_impl> comm,
        specific_indexed_device_storage& available_devices,
        ccl_driver_context_ptr in_ctx,
        const ccl_buffer send_buf,
        size_t send_count,
        ccl_buffer recv_buf,
        const size_t* recv_counts,
        const coll_param_gpu& params,
        std::shared_ptr<ccl_stream> device_stream = std::shared_ptr<ccl_stream>())
            : base(sched, comm, in_ctx, send_buf, params, device_stream),
              // left_wrote_to_me_flag
              income_data_flag(this->template alloc_memory_wrap(
                  typename ring::allgatherv::income_data_flag_arg{},
                  parent_communicator,
                  1,
                  get_ctx())),
              // ready_to_recv_flag_arg
              ready_to_recv_flag(this->template alloc_memory_wrap(
                  typename ring::allgatherv::ready_to_recv_flag_arg{},
                  parent_communicator,
                  1,
                  get_ctx())),
              recv_counts_buf(parent_communicator->get_device()
                                  .template alloc_memory<recv_counts_typed_entry_type>(
                                      comm_addr.size,
                                      sizeof(recv_counts_typed_entry_type),
                                      get_ctx())),

              recv_offsets_buf(parent_communicator->get_device()
                                   .template alloc_memory<recv_offsets_typed_entry_type>(
                                       comm_addr.size,
                                       sizeof(recv_offsets_typed_entry_type),
                                       get_ctx()))

    {
        // copy recv_buf into alloced recv_buf_entry
        recv_buf_entry = recv_buf;
        cnt_entry = send_count;
        // same as parent_communicator->template
        //                    get_comm_data<base::get_topology(),
        //                    base::get_topology_class()>().size;
        int local_topology_size = comm_addr.size;
        std::vector<size_t> recv_offsets_v(local_topology_size, 0);

        for (int idx = 0; idx < local_topology_size; idx++) {
            if (idx > 0)
                recv_offsets_v[idx] += recv_offsets_v[idx - 1] + recv_counts[idx - 1];
        }

        recv_counts_buf.enqueue_write_sync(recv_counts, local_topology_size);
        recv_offsets_buf.enqueue_write_sync(recv_offsets_v);

        int next_rank = (comm_addr.rank + 1) % comm_addr.size;
        kernel_router = base::template create_kernel_router_for_rank<
            l0_allgatherv_typed_entry<gpu_comm_impl, topology>>(
            *this, next_rank, available_devices, base::get_params());

        ENTRY_LOG_DEBUG("Init phase of current entry for ext_rank:", next_rank);

        // Once we filled our local parameters, we go wait for another entry to set its
        // parameters so we can use them
        this->set_state(gpu_entry_state::created);
    }

    ~l0_allgatherv_typed_entry() {
        // TODO: remove the memory once the entry is destroyed if it's not cleared automatically
        // TODO: should we destroy handles here?
    }

    void start() override {
        ENTRY_LOG_DEBUG("Start entry, cnt ", cnt_entry);
        //Create base primitives
        base::start();

        auto& main_entry_function = get_local_kernel();

        auto recv_buf_ptr = reinterpret_cast<processing_type*>(recv_buf_entry.get_ptr());

        //create implementation specified primitives
        main_entry_function
            .template set_args<typename ring::allgatherv::income_data_flag_arg,
                               typename ring::allgatherv::ready_to_recv_flag_arg,
                               typename ring::allgatherv::recv_buf_arg<processing_type>,
                               typename ring::allgatherv::recv_elem_counts_buf_arg,
                               typename ring::allgatherv::recv_elem_offsets_buf_arg,
                               typename kernel_main_typed::common_entry_buf_size_arg>(
                income_data_flag.get(),
                ready_to_recv_flag.get(),
                recv_buf_ptr,
                recv_counts_buf.get(),
                recv_offsets_buf.get(),
                cnt_entry);

        // Once we filled our local parameters, we go wait for another entry to set its
        // parameters so we can use them
        this->set_state(gpu_entry_state::wait_for_entry);

        //make sure, that kernel ready for launch
        this->submit_for_execution();
        status = ccl_sched_entry_status_started;
    }

    const char* name() const override {
        return class_name();
    }

    std::vector<ccl_device::device_ipc_memory_handle> get_ipc_data() override {
        ccl_device& owned_device = parent_communicator->get_device();

        auto recv_buf_ptr = reinterpret_cast<processing_type*>(recv_buf_entry.get_ptr());

        std::vector<ccl_device::device_ipc_memory_handle> ret;
        ret.reserve(3);
        ret.push_back(owned_device.create_ipc_memory_handle(recv_buf_ptr, get_ctx()));
        ret.push_back(owned_device.create_ipc_memory_handle(income_data_flag.get(), get_ctx()));
        ret.push_back(owned_device.create_ipc_memory_handle(ready_to_recv_flag.get(), get_ctx()));
        return ret;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        base::dump_detail(str);
    }

private:
    ccl_device::device_memory<income_data_flag_gpu_type> income_data_flag;
    ccl_device::device_memory<ready_to_recv_flag_gpu_type> ready_to_recv_flag;
    ccl_buffer recv_buf_entry;
    ccl_device::device_memory<recv_counts_typed_entry_type> recv_counts_buf;
    ccl_device::device_memory<recv_offsets_typed_entry_type> recv_offsets_buf;
    size_t cnt_entry;
    std::shared_ptr<ccl_context> ctx;

public:
    template <class left_kernel_t, class right_kernel_t>
    bool execute(left_kernel_t& left_kernel, right_kernel_t& right_kernel) {
        bool is_right_kernel_ready =
            right_kernel
                .template test_args<typename ring::allgatherv::recv_buf_arg<processing_type>,
                                    typename ring::allgatherv::income_data_flag_arg,
                                    typename ring::allgatherv::ready_to_recv_flag_arg>();

        // Once we're sure that the parameters ready read them from the right kernel
        // Note: we not only read the parameters but also reset their 'ready' flag
        // (since we're using a destructive-copying policy) meaning that they must be stored
        // in order to be read again.
        // This is a protection to a case of multiple kernel launches
        // (i.e. the collective is ran multiple times) where we might read not up-to-date
        // values from the previous run.

        if (is_right_kernel_ready) {
            auto right_recv_buf_arg =
                right_kernel
                    .template get_arg<typename ring::allgatherv::recv_buf_arg<processing_type>>();
            auto right_income_data_flag_arg =
                right_kernel.template get_arg<typename ring::allgatherv::income_data_flag_arg>();
            auto right_ready_to_recv_flag_arg =
                right_kernel.template get_arg<typename ring::allgatherv::ready_to_recv_flag_arg>();

            // ENTRY_LOG_DEBUG("Bind right arguments from ",
            //                 right_kernel_t::name(),
            //                 " kernel",
            //                 " to ",
            //                 left_kernel_t::name(),
            //                 " kernel. "
            //                 "Right arguments:\n{ ",
            //                 right_recv_buf_arg.first,
            //                 ", ",
            //                 right_recv_buf_arg.second,
            //                 "}\n",
            //                 "{ ",
            //                 right_income_data_flag_arg.first,
            //                 ", ",
            //                 right_income_data_flag_arg.second,
            //                 "}\n",
            //                 "{ ",
            //                 right_ready_to_recv_flag_arg.first,
            //                 ", ",
            //                 right_ready_to_recv_flag_arg.second,
            //                 "}\n");

            left_kernel
                .template set_args<typename ring::allgatherv::right_output_buf_arg<processing_type>,
                                   typename ring::allgatherv::right_income_data_flag_arg,
                                   typename ring::allgatherv::right_ready_to_recv_flag_arg>(
                    right_recv_buf_arg.second,
                    right_income_data_flag_arg.second,
                    right_ready_to_recv_flag_arg.second);

            ENTRY_LOG_DEBUG("Binding arguments between kernels is complete. ",
                            "Arguments of the left kernel after binding:\n",
                            left_kernel.to_string());
        }
        return is_right_kernel_ready;
    }
};
} // namespace native
