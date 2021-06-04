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
class l0_alltoallv_typed_entry : public base_gpu_entry<gpu_comm_impl,
                                                       topology,
                                                       ccl::device_topology_type::ring,
                                                       ccl_coll_alltoallv> {
public:
    friend class ccl_gpu_comm;
    friend class ccl_virtual_gpu_comm;

    using base = base_gpu_entry<gpu_comm_impl,
                                topology,
                                ccl::device_topology_type::ring,
                                ccl_coll_alltoallv>;
    using base::parent_communicator;
    using base::comm_addr;
    using base::req;
    using base::status;
    using base::launch_args;
    using base::kernel_router;
    using base::get_ctx;
    using base::get_local_kernel;
    using kernel_main_typed = ring::alltoallv::main_kernel;

    using income_data_flag_gpu_type =
        typename std::remove_pointer<typename ring::alltoallv::income_data_flag_arg_type>::type;
    using ready_to_recv_flag_gpu_type =
        typename std::remove_pointer<typename ring::alltoallv::ready_to_recv_flag_arg_type>::type;

    using recv_counts_typed_entry_type =
        typename std::remove_pointer<typename ring::alltoallv::recv_elem_counts_buf_arg_type>::type;
    using recv_offsets_typed_entry_type = typename std::remove_pointer<
        typename ring::alltoallv::recv_elem_offsets_buf_arg_type>::type;

    using proxy_size_flag_gpu_type =
        typename std::remove_pointer<typename ring::alltoallv::proxy_size_flag_arg_type>::type;

    using send_counts_typed_entry_type =
        typename std::remove_pointer<typename ring::alltoallv::send_buf_size_arg_type>::type;
    using send_offsets_typed_entry_type = typename std::remove_pointer<
        typename ring::alltoallv::send_elem_offsets_buf_arg_type>::type;

    static constexpr const char* class_name() noexcept {
        return "L0_ALLTOALLV_TYPED";
    }

    static constexpr ccl_coll_type type() noexcept {
        return ccl_coll_alltoallv;
    }

    l0_alltoallv_typed_entry() = delete;
    l0_alltoallv_typed_entry(
        ccl_sched* sched,
        std::shared_ptr<gpu_comm_impl> comm,
        specific_indexed_device_storage& available_devices,
        ccl_driver_context_ptr in_ctx,
        const ccl_buffer send_buf,
        const size_t* send_counts,
        size_t total_send_counts,
        ccl_buffer recv_buf,
        const size_t* recv_counts,
        size_t total_recv_counts,
        const coll_param_gpu& params,
        std::shared_ptr<ccl_stream> device_stream = std::shared_ptr<ccl_stream>())
            : base(sched, comm, in_ctx, send_buf, params, device_stream),
              temp_buffer(this->template alloc_memory_wrap(
                  typename ring::alltoallv::tmp_recv_buf_arg<uint8_t>{},
                  parent_communicator,
                  total_recv_counts,
                  get_ctx())),
              // left_wrote_to_me_flag
              income_data_flag(
                  this->template alloc_memory_wrap(typename ring::alltoallv::income_data_flag_arg{},
                                                   parent_communicator,
                                                   1,
                                                   get_ctx())),
              // ready_to_recv_flag_arg
              ready_to_recv_flag(this->template alloc_memory_wrap(
                  typename ring::alltoallv::ready_to_recv_flag_arg{},
                  parent_communicator,
                  1,
                  get_ctx())),
              proxy_size_flag_entry(
                  this->template alloc_memory_wrap(typename ring::alltoallv::proxy_size_flag_arg{},
                                                   parent_communicator,
                                                   1,
                                                   get_ctx())),
              recv_counts_buf(parent_communicator->get_device()
                                  .template alloc_memory<recv_counts_typed_entry_type>(
                                      total_recv_counts,
                                      sizeof(recv_counts_typed_entry_type),
                                      get_ctx())),
              recv_offsets_buf(parent_communicator->get_device()
                                   .template alloc_memory<recv_offsets_typed_entry_type>(
                                       comm_addr.size,
                                       sizeof(recv_offsets_typed_entry_type),
                                       get_ctx())),
              send_counts_buf(parent_communicator->get_device()
                                  .template alloc_memory<recv_counts_typed_entry_type>(
                                      total_send_counts,
                                      sizeof(recv_counts_typed_entry_type),
                                      get_ctx())),
              send_offsets_buf(parent_communicator->get_device()
                                   .template alloc_memory<send_offsets_typed_entry_type>(
                                       comm_addr.size,
                                       sizeof(send_offsets_typed_entry_type),
                                       get_ctx())) {
        // copy recv_buf into recv_buf_entry
        recv_buf_entry = recv_buf;

        // same as parent_communicator->template
        //                    get_comm_data<base::get_topology(),
        //                    base::get_topology_class()>().size;
        int local_topology_size = comm_addr.size;
        std::vector<size_t> recv_offsets_v(local_topology_size, 0);

        for (int idx = 0; idx < local_topology_size; idx++) {
            if (idx > 0)
                recv_offsets_v[idx] += recv_offsets_v[idx - 1] + recv_counts[idx - 1];
        }

        std::vector<size_t> send_offsets_v(local_topology_size, 0);
        for (int idx = 0; idx < local_topology_size; idx++) {
            if (idx > 0)
                send_offsets_v[idx] += send_offsets_v[idx - 1] + send_counts[idx - 1];
        }
        // recv
        recv_counts_buf.enqueue_write_sync(recv_counts, local_topology_size);
        recv_offsets_buf.enqueue_write_sync(recv_offsets_v);
        // send
        send_counts_buf.enqueue_write_sync(send_counts, local_topology_size);
        send_offsets_buf.enqueue_write_sync(send_offsets_v);
        // flag
        proxy_size_flag_entry.enqueue_write_sync({ (int)0 });

        int next_rank = (comm_addr.rank + 1) % comm_addr.size;
        kernel_router = base::template create_kernel_router_for_rank<
            l0_alltoallv_typed_entry<gpu_comm_impl, topology>>(
            *this, next_rank, available_devices, base::get_params());

        ENTRY_LOG_DEBUG("Init phase of current entry for ext_rank:", next_rank);

        // Once we filled our local parameters, we go wait for another entry to set its
        // parameters so we can use them
        this->set_state(gpu_entry_state::created);
    }

    ~l0_alltoallv_typed_entry() {
        // TODO: remove the memory once the entry is destroyed if it's not cleared automatically
        // TODO: should we destroy handles here?
    }

    void start() override {
        LOG_DEBUG(class_name(), " entry req ", &req, ", rank: ", comm_addr.to_string());

        //Create base primitives
        base::start();

        auto& main_entry_function = get_local_kernel();

        auto recv_buf_ptr = reinterpret_cast<void*>(recv_buf_entry.get_ptr());

        //create implementation specified primitives
        main_entry_function.template set_args<typename ring::alltoallv::tmp_recv_buf_arg<void>,
                                              typename ring::alltoallv::income_data_flag_arg,
                                              typename ring::alltoallv::ready_to_recv_flag_arg,
                                              typename ring::alltoallv::recv_buf_arg<void>,
                                              typename ring::alltoallv::recv_elem_counts_buf_arg,
                                              typename ring::alltoallv::recv_elem_offsets_buf_arg,
                                              typename ring::alltoallv::proxy_size_flag_arg,
                                              typename ring::alltoallv::send_buf_size_arg>(
            temp_buffer.get(),
            income_data_flag.get(),
            ready_to_recv_flag.get(),
            recv_buf_ptr,
            recv_counts_buf.get(),
            recv_offsets_buf.get(),
            proxy_size_flag_entry.get(),
            send_counts_buf.get());

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

        std::vector<ccl_device::device_ipc_memory_handle> ret;
        ret.reserve(4);
        ret.push_back(owned_device.create_ipc_memory_handle(temp_buffer.get(), get_ctx()));
        ret.push_back(owned_device.create_ipc_memory_handle(income_data_flag.get(), get_ctx()));
        ret.push_back(owned_device.create_ipc_memory_handle(ready_to_recv_flag.get(), get_ctx()));
        ret.push_back(
            owned_device.create_ipc_memory_handle(proxy_size_flag_entry.get(), get_ctx()));

        return ret;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        base::dump_detail(str);
    }

private:
    ccl_device::device_memory<uint8_t> temp_buffer;
    ccl_device::device_memory<income_data_flag_gpu_type> income_data_flag;
    ccl_device::device_memory<ready_to_recv_flag_gpu_type> ready_to_recv_flag;
    ccl_device::device_memory<proxy_size_flag_gpu_type> proxy_size_flag_entry;
    ccl_buffer recv_buf_entry;
    ccl_device::device_memory<recv_counts_typed_entry_type> recv_counts_buf;
    ccl_device::device_memory<recv_offsets_typed_entry_type> recv_offsets_buf;
    ccl_device::device_memory<send_counts_typed_entry_type> send_counts_buf;
    ccl_device::device_memory<send_offsets_typed_entry_type> send_offsets_buf;
    std::shared_ptr<ccl_context> ctx;

public:
    template <class left_kernel_t, class right_kernel_t>
    bool execute(left_kernel_t& left_kernel, right_kernel_t& right_kernel) {
        bool is_right_kernel_ready =
            right_kernel.template test_args<typename ring::alltoallv::tmp_recv_buf_arg<void>,
                                            typename ring::alltoallv::income_data_flag_arg,
                                            typename ring::alltoallv::ready_to_recv_flag_arg,
                                            typename ring::alltoallv::proxy_size_flag_arg>();

        // Once we're sure that the parameters ready read them from the right kernel
        // Note: we not only read the parameters but also reset their 'ready' flag
        // (since we're using a destructive-copying policy) meaning that they must be stored
        // in order to be read again.
        // This is a protection to a case of multiple kernel launches
        // (i.e. the collective is ran multiple times) where we might read not up-to-date
        // values from the previous run.

        if (is_right_kernel_ready) {
            auto right_tmp_recv_buf_arg =
                right_kernel.template get_arg<typename ring::alltoallv::tmp_recv_buf_arg<void>>();
            auto right_income_data_flag_arg =
                right_kernel.template get_arg<typename ring::alltoallv::income_data_flag_arg>();
            auto right_ready_to_recv_flag_arg =
                right_kernel.template get_arg<typename ring::alltoallv::ready_to_recv_flag_arg>();
            auto right_proxy_size_flag_arg =
                right_kernel.template get_arg<typename ring::alltoallv::proxy_size_flag_arg>();

            // ENTRY_LOG_DEBUG("Bind right arguments from ",
            //                 right_kernel_t::name(),
            //                 " kernel",
            //                 " to ",
            //                 left_kernel_t::name(),
            //                 " kernel. "
            //                 "Right arguments:\n{ ",
            //                 right_tmp_recv_buf_arg.first,
            //                 ", ",
            //                 right_tmp_recv_buf_arg.second,
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
            //                 "}\n",
            //                 "{ ",
            //                 right_proxy_size_flag_arg.first,
            //                 ", ",
            //                 right_proxy_size_flag_arg.second,
            //                 "}\n");

            left_kernel.template set_args<typename ring::alltoallv::right_tmp_recv_buf_arg<void>,
                                          typename ring::alltoallv::right_income_data_flag_arg,
                                          typename ring::alltoallv::right_ready_to_recv_flag_arg,
                                          typename ring::alltoallv::right_proxy_size_flag_arg>(
                right_tmp_recv_buf_arg.second,
                right_income_data_flag_arg.second,
                right_ready_to_recv_flag_arg.second,
                right_proxy_size_flag_arg.second);

            ENTRY_LOG_DEBUG("Binding arguments between kernels is complete. ",
                            "Arguments of the left kernel after binding:\n",
                            left_kernel.to_string());
        }
        return is_right_kernel_ready;
    }
};
} // namespace native
