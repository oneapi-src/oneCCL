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
template <class native_type, class gpu_comm_impl, ccl::group_split_type topology>
class l0_allgatherv_typed_entry : public base_gpu_entry<native_type,
                                                        gpu_comm_impl,
                                                        topology,
                                                        ccl::device_topology_type::ring,
                                                        ccl_coll_allgatherv> {
public:
    friend class ccl_gpu_comm;
    friend class ccl_virtual_gpu_comm;

    using base = base_gpu_entry<native_type,
                                gpu_comm_impl,
                                topology,
                                ccl::device_topology_type::ring,
                                ccl_coll_allgatherv>;
    using base::parent_communicator;
    using base::comm_addr;
    using base::req;
    using base::status;
    using base::launch_args;
    using base::kernel_router;
    using kernel_main_typed = ring_allgatherv_kernel<native_type>;
    using kernel_ipc_typed = ring_allgatherv_ipc<native_type>;

    using income_data_flag_gpu_type =
        typename std::remove_pointer<typename kernel_main_typed::income_data_flag_arg_type>::type;
    using ready_to_recv_flag_gpu_type =
        typename std::remove_pointer<typename kernel_main_typed::ready_to_recv_flag_arg_type>::type;
    using recv_counts_typed_entry_type = typename std::remove_pointer<
        typename kernel_main_typed::recv_elem_counts_buf_arg_type>::type;
    using recv_offsets_typed_entry_type = typename std::remove_pointer<
        typename kernel_main_typed::recv_elem_offsets_buf_arg_type>::type;

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
        const ccl_buffer send_buf,
        size_t send_count,
        ccl_buffer recv_buf,
        const size_t* recv_counts,
        std::shared_ptr<ccl_stream> device_stream = std::shared_ptr<ccl_stream>())
            : base(sched,
                   comm,
                   send_buf,
                   ccl::native_type_info<native_type>::ccl_type_value,
                   device_stream),
              // left_wrote_to_me_flag
              income_data_flag(parent_communicator->get_device()
                                   .template alloc_memory<income_data_flag_gpu_type>(
                                       1,
                                       sizeof(income_data_flag_gpu_type), std::shared_ptr<ccl_context> { })),
              // ready_to_recv_flag_arg
              ready_to_recv_flag(parent_communicator->get_device()
                                     .template alloc_memory<ready_to_recv_flag_gpu_type>(
                                         1,
                                         sizeof(ready_to_recv_flag_gpu_type), std::shared_ptr<ccl_context> { })),
              recv_counts_buf(parent_communicator->get_device()
                                  .template alloc_memory<recv_counts_typed_entry_type>(
                                      send_count,
                                      sizeof(recv_counts_typed_entry_type), std::shared_ptr<ccl_context> { })),

              recv_offsets_buf(parent_communicator->get_device()
                                   .template alloc_memory<recv_offsets_typed_entry_type>(
                                       comm_addr.size,
                                       sizeof(recv_offsets_typed_entry_type), std::shared_ptr<ccl_context> { }))

    {
        // copy recv_buf into alloced recv_buf_entry
        recv_buf_entry = recv_buf;
        cnt_entry = send_count;
        // same as parent_communicator->template
        //                    get_comm_data<base::get_topology(),
        //                    base::get_topology_class()>().size;
        size_t local_topology_size = comm_addr.size;
        std::vector<size_t> recv_offsets_v(local_topology_size, 0);

        for (size_t idx = 0; idx < local_topology_size; idx++) {
            if (idx > 0)
                recv_offsets_v[idx] += recv_offsets_v[idx - 1] + recv_counts[idx - 1];
        }

        recv_counts_buf.enqueue_write_sync(recv_counts, local_topology_size);
        recv_offsets_buf.enqueue_write_sync(recv_offsets_v);

        LOG_DEBUG(class_name(),
                  " entry req ",
                  &req,
                  ", cnt ",
                  cnt_entry,
                  ", rank: ",
                  "local topology size:",
                  local_topology_size,
                  comm_addr.to_string());
        size_t next_rank = (comm_addr.rank + 1) % comm_addr.size;
        kernel_router = base::template create_kernel_router_for_rank<
            l0_allgatherv_typed_entry<native_type, gpu_comm_impl, topology>>(
            *this, next_rank, available_devices);

        //TODO L0 workaround
        //if(std::is_same<gpu_comm_impl::type_idx() ==, ccl_gpu_comm>::value)
        if (gpu_comm_impl::type_idx() == ccl_gpu_comm::type_idx()) {
            wait_count = reinterpret_cast<ccl_gpu_comm*>(parent_communicator.get())
                             ->get_virtual_gpu_count() +
                         1 /*+ own*/;
        }
        else if (gpu_comm_impl::type_idx() == ccl_ipc_source_gpu_comm<ccl_gpu_comm>::type_idx()) {
            wait_count =
                reinterpret_cast<ccl_ipc_source_gpu_comm<ccl_gpu_comm>*>(parent_communicator.get())
                    ->get_impl()
                    .get_virtual_gpu_count() +
                1 /*+ own*/;
        }
        {
            std::unique_lock<std::mutex> lock(global_mutex);
            registered_thread.insert(std::this_thread::get_id());
        }
    }

    ~l0_allgatherv_typed_entry() {
        /* std::stringstream ss;
        ss << "native_type: " << std::endl;
        std::vector<native_type> mem_get(right_output_buffer.count());
        std::copy(right_output_buffer.handle, right_output_buffer.handle + right_output_buffer.count(), mem_get.begin());
        std::copy(mem_get.begin(), mem_get.end(), std::ostream_iterator<native_type>(ss, ","));
        ss << std::endl;

        ss << "income_data_flag: "  << std::endl;
        ss << *(income_data_flag.handle) << std::endl;

        ss << "ready_to_recv_flag: "  << std::endl;
        ss << *(ready_to_recv_flag.handle) << std::endl;

        LOG_INFO("Kernel results: ", ss.str());
        */
    }

    void start() override {
        LOG_DEBUG(class_name(),
                  " entry req ",
                  &req,
                  ", rank: ",
                  comm_addr.to_string(),
                  ", cnt ",
                  cnt_entry);

        //Create base primitives
        base::start();

        //ccl_device &device = parent_communicator->get_device();
        kernel_main_typed& main_entry_function =
            parent_communicator->template get_gpu_kernel<base::type(),
                                                         topology,
                                                         ccl::device_topology_type::ring,
                                                         native_type>();

        auto recv_buf_ptr = reinterpret_cast<native_type*>(recv_buf_entry.get_ptr());
        //create implementation specified primitives
        main_entry_function
            .template set_args<typename kernel_main_typed::income_data_flag_arg,
                               typename kernel_main_typed::ready_to_recv_flag_arg,
                               typename kernel_main_typed::recv_buf_arg,
                               typename kernel_main_typed::recv_elem_counts_buf_arg,
                               typename kernel_main_typed::recv_elem_offsets_buf_arg,
                               typename kernel_main_typed::common_entry_buf_size_arg>(
                income_data_flag.get(),
                ready_to_recv_flag.get(),
                recv_buf_ptr,
                recv_counts_buf.get(),
                recv_offsets_buf.get(),
                cnt_entry);

        /* TRY To APPEND Kernel HERE!!! Not in update
         *
         * ze_result_t result = zeCommandListAppendLaunchKernel(exec_cmd_list->handle, main_entry_function.handle, &launch_args, nullptr, 0, nullptr);
        if(result != ZE_RESULT_SUCCESS)
        {
            LOG_ERROR("zeCommandListAppendLaunchKernel failed, error: ", to_string(result));
            throw std::runtime_error("zeCommandListAppendLaunchKernel failed");
        }

        / * result = zeCommandListClose(exec_cmd_list->handle);
        if(result != ZE_RESULT_SUCCESS)
        {
            LOG_ERROR("zeCommandListClose failed, error: ", to_string(result));
            throw std::runtime_error("zeCommandListClose failed");
        }*/

        //make sure, that kernel ready for launch
        this->submit_for_execution();
        status = ccl_sched_entry_status_started;
    }

    const char* name() const override {
        return class_name();
    }

    std::vector<ccl_device::device_ipc_memory_handle> get_ipc_data() {
        ccl_device& owned_device = parent_communicator->get_device();

        //TODO
        std::vector<ccl_device::device_ipc_memory_handle> ret;
        ret.reserve(3);
        ret.push_back(owned_device.create_ipc_memory_handle(income_data_flag.get(), ctx));
        ret.push_back(owned_device.create_ipc_memory_handle(ready_to_recv_flag.get(), ctx));
        return ret;
    }

protected:
    bool finalize_entry() override {
        LOG_TRACE("entry: ", class_name(), ", rank: ", comm_addr.to_string());
        ccl_device& device = parent_communicator->get_device();

        kernel_main_typed& main_entry_function =
            parent_communicator->template get_gpu_kernel<base::type(),
                                                         topology,
                                                         ccl::device_topology_type::ring,
                                                         native_type>();
        if ((*kernel_router)(main_entry_function)) {
            ze_result_t result;
            //TODO L0 Workaround
            if (!is_kernel_added) {
                std::unique_lock<std::mutex> lock(global_mutex);
                exec_count++;
                cur_index = exec_count;
                result = zeCommandListAppendLaunchKernel(device.get_cmd_list(ctx).get(),
                                                         main_entry_function.handle,
                                                         &launch_args,
                                                         nullptr,
                                                         0,
                                                         nullptr);
                if (result != ZE_RESULT_SUCCESS) {
                    LOG_ERROR("zeCommandListAppendLaunchKernel failed, error: ", to_string(result));
                    throw std::runtime_error("zeCommandListAppendLaunchKernel failed");
                }
                is_kernel_added = true;

                LOG_DEBUG("entry: ",
                          class_name(),
                          ", rank: ",
                          comm_addr.to_string(),
                          ". Kernel added: ",
                          main_entry_function.to_string(),
                          " in list");
            }

            while (exec_count < registered_thread.size()) {
            }

            //TODO L0 workaround
            LOG_INFO("Check L0 Workaround: WaitCount: ",
                     wait_count,
                     ", ExecCount: ",
                     exec_count,
                     ", CurIndex: ",
                     cur_index);
            if (cur_index == wait_count /*std::is_same<gpu_comm_impl, ccl_gpu_comm>::value*/) {
                if (topology == ccl::group_split_type::cluster) {
                    // TODO: implement process communicator case
                    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + "TODO: implement process communicator case");
                    // auto c = ccl::environment::instance().create_communicator();
                    // if (c.rank() == 0) {
                        // LOG_INFO("L0 Workaround: one device close list!!!",
                        //          "WaitCount: ",
                        //          wait_count,
                        //          ", ExecCount: ",
                        //          exec_count,
                        //          ", CurIndex: ",
                        //          cur_index);
                        // result = zeCommandListClose(device.get_cmd_list().get());
                        // if (result != ZE_RESULT_SUCCESS) {
                        //     LOG_ERROR("zeCommandListClose failed, error: ",
                        //               native::to_string(result));
                        //     throw std::runtime_error("zeCommandListClose failed");
                        // }
                    // }
                }
                else {
                    LOG_INFO("L0 Workaround: one device close list!!!",
                             "WaitCount: ",
                             wait_count,
                             ", ExecCount: ",
                             exec_count,
                             ", CurIndex: ",
                             cur_index);
                    result = zeCommandListClose(device.get_cmd_list(ctx).get());
                    if (result != ZE_RESULT_SUCCESS) {
                        LOG_ERROR("zeCommandListClose failed, error: ", native::to_string(result));
                        throw std::runtime_error("zeCommandListClose failed");
                    }
                }

                LOG_INFO("entry: ", class_name(), ", rank: ", comm_addr.to_string(), " finalized!");
                return true;
            }
            else if (cur_index > wait_count) {
                LOG_INFO("L0 Workaround: one device should close list before!!! ",
                         "WaitCount: ",
                         wait_count,
                         ", ExecCount: ",
                         exec_count,
                         ", CurIndex: ",
                         cur_index);
                LOG_INFO("entry: ", class_name(), ", rank: ", comm_addr.to_string(), " finalized!");
                return true;
            }
        }
        return false;
    }

    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, class_name(), "TODO\n");
    }

private:
    bool is_kernel_added = false; //TODO L0 workaround - one dev close list

    ccl_device::device_memory<income_data_flag_gpu_type> income_data_flag;
    ccl_device::device_memory<ready_to_recv_flag_gpu_type> ready_to_recv_flag;
    ccl_buffer recv_buf_entry;
    ccl_device::device_memory<recv_counts_typed_entry_type> recv_counts_buf;
    ccl_device::device_memory<recv_offsets_typed_entry_type> recv_offsets_buf;
    size_t cnt_entry;
    std::shared_ptr<ccl_context> ctx;
public:
    bool execute(kernel_main_typed& main_entry_function, kernel_main_typed& right_kernel) {
        //Check argument binding in kernels for next rank
        bool is_right_kernel_ready =
            right_kernel.template test_args<typename kernel_main_typed::recv_buf_arg,
                                            typename kernel_main_typed::income_data_flag_arg,
                                            typename kernel_main_typed::ready_to_recv_flag_arg>();
        if (is_right_kernel_ready) {
            if (is_kernel_added) {
                LOG_DEBUG("entry: ",
                          class_name(),
                          ", rank: ",
                          comm_addr.to_string(),
                          ". Function: ",
                          main_entry_function.to_string(),
                          " - binded already");
                return true;
            }

            //TODO do not get arguments sequencially - use array version instead
            typename kernel_main_typed::recv_buf_arg::return_t right_output_buf =
                right_kernel.template get_arg<typename kernel_main_typed::recv_buf_arg>();
            typename kernel_main_typed::income_data_flag_arg::return_t right_income_data_flag_arg =
                right_kernel.template get_arg<typename kernel_main_typed::income_data_flag_arg>();
            typename kernel_main_typed::ready_to_recv_flag_arg::return_t
                right_ready_to_recv_flag_arg =
                    right_kernel
                        .template get_arg<typename kernel_main_typed::ready_to_recv_flag_arg>();

            LOG_DEBUG("entry: ",
                      class_name(),
                      ", rank: ",
                      comm_addr.to_string(),
                      ", bind elapsed arguments for kernel: ",
                      kernel_main_typed::name());
            LOG_TRACE("Args: \n{ ",
                      right_output_buf.first,
                      ", ",
                      right_output_buf.second,
                      "}\n",
                      "{ ",
                      right_income_data_flag_arg.first,
                      ", ",
                      right_income_data_flag_arg.second,
                      "}\n",
                      "{ ",
                      right_ready_to_recv_flag_arg.first,
                      ", ",
                      right_ready_to_recv_flag_arg.second,
                      "}\n");

            //TODO register argument for current device kernel: use array-version
            main_entry_function
                .template set_args<typename kernel_main_typed::right_output_buf_arg,
                                   typename kernel_main_typed::right_income_data_flag_arg,
                                   typename kernel_main_typed::right_ready_to_recv_flag_arg>(
                    right_output_buf.second,
                    right_income_data_flag_arg.second,
                    right_ready_to_recv_flag_arg.second);
            LOG_TRACE("Set right_output_buf",
                      "Set right_income_data_flag_arg",
                      "Set right_ready_to_recv_flag_arg");
            LOG_DEBUG("entry: ",
                      class_name(),
                      ", rank: ",
                      comm_addr.to_string(),
                      ". Function: ",
                      main_entry_function.to_string());
        }
        return is_right_kernel_ready;
    }

    bool execute(kernel_main_typed& main_entry_function, kernel_ipc_typed& right_kernel) {
        //Check argument binding in kernels for next rank
        bool is_right_kernel_ready =
            right_kernel.template test_args< //typename kernel_ipc_typed::right_output_buf_arg,
                typename kernel_ipc_typed::income_data_flag_arg,
                typename kernel_ipc_typed::ready_to_recv_flag_arg>();
        if (is_right_kernel_ready) {
            if (is_kernel_added) {
                LOG_DEBUG("entry: ",
                          class_name(),
                          ", rank: ",
                          comm_addr.to_string(),
                          ". Function: ",
                          main_entry_function.to_string(),
                          " - binded already");
                return true;
            }

            //TODO do not get arguments sequencially - use array version instead
            typename kernel_main_typed::right_output_buf_arg::return_t right_output_buf_arg =
                right_kernel.template get_arg<typename kernel_ipc_typed::right_output_buf_arg>();
            typename kernel_main_typed::income_data_flag_arg::return_t right_income_data_flag_arg =
                right_kernel.template get_arg<typename kernel_ipc_typed::income_data_flag_arg>();
            typename kernel_main_typed::ready_to_recv_flag_arg::return_t
                right_ready_to_recv_flag_arg =
                    right_kernel
                        .template get_arg<typename kernel_ipc_typed::ready_to_recv_flag_arg>();

            LOG_DEBUG("entry: ",
                      class_name(),
                      ", rank: ",
                      comm_addr.to_string(),
                      ", bind elapsed arguments for kernel: ",
                      kernel_main_typed::name());
            LOG_TRACE("Args: \n{ ",
                      right_output_buf_arg.first,
                      ", ",
                      right_output_buf_arg.second,
                      "}\n",
                      "{ ",
                      right_income_data_flag_arg.first,
                      ", ",
                      right_income_data_flag_arg.second,
                      "}\n",
                      "{ ",
                      right_ready_to_recv_flag_arg.first,
                      ", ",
                      right_ready_to_recv_flag_arg.second,
                      "}\n");

            //TODO register argument for current device kernel: user array version
            main_entry_function
                .template set_args< //typename kernel_main_typed::right_output_buf_arg,
                    typename kernel_main_typed::right_income_data_flag_arg,
                    typename kernel_main_typed::right_ready_to_recv_flag_arg>(
                    //right_output_buf_arg.second,
                    right_income_data_flag_arg.second,
                    right_ready_to_recv_flag_arg.second);
            LOG_TRACE("Set right_output_buf_arg",
                      "Set right_income_data_flag_arg",
                      "Set right_ready_to_recv_flag_arg");
            LOG_DEBUG("entry: ",
                      class_name(),
                      ", rank: ",
                      comm_addr.to_string(),
                      ". Function: ",
                      main_entry_function.to_string());
        }
        return is_right_kernel_ready;
    }
};
} // namespace native
