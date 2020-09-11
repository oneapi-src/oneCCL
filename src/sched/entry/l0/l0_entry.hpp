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
#include "ccl_types.hpp"
#include "common/datatype/datatype.hpp"
#include "ccl_type_traits.hpp"
#include "comp/comp.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "sched/entry/coll/direct/base_coll_entry.hpp"

#include "common/comm/l0/modules_connector.hpp"
#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
//TODO L0 Workaround
#include <unistd.h>
static std::mutex global_fence_mutex;

namespace native {
template <class native_type,
          class gpu_comm_impl,
          ccl::device_group_split_type group_id,
          ccl::device_topology_type class_id>
class base_gpu_entry : public sched_entry {
public:
    using gpu_comm = gpu_comm_impl;
    using processing_type = native_type;
    using kernel_main_typed = ring_allreduce_kernel<processing_type>;
    using kernel_ipc_typed = ring_allreduce_ipc<native_type>;

    friend class ccl_gpu_comm;
    friend class ccl_virtual_gpu_comm;
    static constexpr const char *class_name() noexcept {
        return "L0_ALLREDUCE";
    }
    static constexpr ccl_coll_type type() noexcept {
        return ccl_coll_allreduce;
    }

    static constexpr ccl::device_group_split_type get_topology() {
        return group_id;
    }

    static constexpr ccl::device_topology_type get_topology_class() {
        return class_id;
    }

    base_gpu_entry() = delete;
    base_gpu_entry(ccl_sched *sched,
                   std::shared_ptr<gpu_comm> comm,
                   const ccl_buffer send_buf,
                   ccl_buffer recv_buf,
                   size_t cnt,
                   ccl_datatype_t dtype_in,
                   ccl_reduction_t op,
                   std::shared_ptr<ccl_stream> &stream)
            : sched_entry(sched),
              parent_communicator(comm),
              comm_addr(parent_communicator
                            ->template get_comm_data<get_topology(), get_topology_class()>()),
              send_buf(send_buf),
              recv_buf(recv_buf),
              elem_count(cnt),
              dtype(),
              op(op),
              device_stream(stream) {
        dtype = ccl::global_data::get().dtypes->get(dtype_in);
    }

    virtual ~base_gpu_entry() {}

    virtual void start() override {
        LOG_DEBUG(class_name(),
                  " entry req ",
                  &req,
                  ", elem_count ",
                  elem_count,
                  ", rank: ",
                  comm_addr.to_string());

        ccl_device &device = parent_communicator->get_device();
        {
            LOG_DEBUG(class_name(), " entry req ", &req, " - create initial gpu primitives");

            //TODO make check, that device_stream belong to the device
            auto queue_prop = ccl_device::get_default_queue_desc();
            auto &cmd_queue = device.get_cmd_queue(queue_prop);
            fence = device.create_or_get_fence(cmd_queue);
        }
        //else
        //{
        //zeCommandListReset(copy_send_cmd_list->handle); // ready for command
        //zeCommandListReset(copy_recv_cmd_list->handle); // ready for command
        //zeCommandListReset(exec_cmd_list->handle); // ready for command
        //}

        //set kernel args for main kernel on current device
        kernel_main_typed &main_entry_function =
            parent_communicator->template register_entry<native_type, group_id, class_id>(*this);

        auto recv_buf_ptr = reinterpret_cast<native_type *>(recv_buf.get_ptr());
        auto send_buf_ptr = reinterpret_cast<native_type *>(send_buf.get_ptr());

        main_entry_function.template set_arg<typename kernel_main_typed::send_buf_size_arg>(
            elem_count);
        main_entry_function.template set_arg<typename kernel_main_typed::send_buf_arg>(
            send_buf_ptr);
        main_entry_function.template set_arg<typename kernel_main_typed::recv_buf_arg>(
            recv_buf_ptr);

        /*ze_result_t result = zeCommandListAppendLaunchKernel(exec_cmd_list->handle, main_entry_function.handle, &launch_args, nullptr, 0, nullptr);
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

        status = ccl_sched_entry_status_started;
    }

    bool submit_for_execution() {
        ready_to_exec = finalize_entry();
        if (ready_to_exec) {
            //TODO L0 workaround
            //if(std::is_same<gpu_comm_impl, ccl_gpu_comm>::value)
            if (gpu_comm_impl::type_idx() == ccl_gpu_comm::type_idx() or
                gpu_comm_impl::type_idx() == ccl_ipc_source_gpu_comm<ccl_gpu_comm>::type_idx()) {
                if (group_id == ccl::device_group_split_type::cluster) {
                    //auto c = ccl::environment::instance().create_communicator();
                    //(void)c;
                    //if(c->rank() == 0)
                    {
                        // Execute command list in command queue
                        ccl_device &device = parent_communicator->get_device();
                        auto queue_prop = ccl_device::get_default_queue_desc();
                        //TODO SPECIAL FOR VIRTUAL
                        /*
                    if(std::is_same<gpu_comm, ccl_virtual_gpu_comm>::value)
                    {
                        queue_prop.ordinal = parent_communicator->get_rank(); //TODO SPECIAL FOR VIRTUAL
                    }
                    queue_prop.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;*/
                        auto &cmd_queue = device.get_cmd_queue(queue_prop);

                        LOG_DEBUG(class_name(),
                                  " entry req ",
                                  &req,
                                  ", rank: ",
                                  comm_addr.to_string(),
                                  " - ready for execution on: ",
                                  device.handle,
                                  ", queue:",
                                  cmd_queue.get(),
                                  ", list: ",
                                  device.get_cmd_list().get());
                        ze_result_t ret = zeCommandQueueExecuteCommandLists(
                            cmd_queue.get(), 1, device.get_cmd_list().get_ptr(), fence);
                        if (ret != ZE_RESULT_SUCCESS) {
                            throw ccl::ccl_error(
                                std::string("cannot execute command list, error: ") +
                                std::to_string(ret));
                        }
                    }
                }
                else {
                    // Execute command list in command queue
                    ccl_device &device = parent_communicator->get_device();
                    auto queue_prop = ccl_device::get_default_queue_desc();
                    //TODO SPECIAL FOR VIRTUAL
                    /*
            if(std::is_same<gpu_comm, ccl_virtual_gpu_comm>::value)
            {
                queue_prop.ordinal = parent_communicator->get_rank(); //TODO SPECIAL FOR VIRTUAL
            }
            queue_prop.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;*/
                    auto &cmd_queue = device.get_cmd_queue(queue_prop);

                    LOG_DEBUG(class_name(),
                              " entry req ",
                              &req,
                              ", rank: ",
                              comm_addr.to_string(),
                              " - ready for execution on: ",
                              device.handle,
                              ", queue:",
                              cmd_queue.get(),
                              ", list: ",
                              device.get_cmd_list().get());
                    ze_result_t ret = zeCommandQueueExecuteCommandLists(
                        cmd_queue.get(), 1, device.get_cmd_list().get_ptr(), fence);
                    if (ret != ZE_RESULT_SUCCESS) {
                        throw ccl::ccl_error(std::string("cannot execute command list, error: ") +
                                             std::to_string(ret));
                    }
                }
            }
        }
        return ready_to_exec;
    }

    virtual void update() override {
        if (!ready_to_exec) {
            submit_for_execution();
        }
        else {
            //wait execution
            ccl_device &device = parent_communicator->get_device();
            auto queue_prop = ccl_device::get_default_queue_desc();
            //TODO SPECIAL FOR VIRTUAL
            /*
            if(std::is_same<gpu_comm, ccl_virtual_gpu_comm>::value)
            {
                queue_prop.ordinal = parent_communicator->get_rank(); //TODO SPECIAL FOR VIRTUAL
            }*/
            //queue_prop.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            auto &cmd_queue = device.get_cmd_queue(queue_prop);

            LOG_TRACE(class_name(),
                      " entry req ",
                      &req,
                      ", rank: ",
                      comm_addr.to_string(),
                      " waiting for finished execution, queue: ",
                      cmd_queue.get());
            /* TODO fence!
            ze_result_t ret = zeCommandQueueSynchronize(cmd_queue.handle,
                                                          std::numeric_limits<uint32_t>::max());*/
            ze_result_t ret = zeFenceQueryStatus(fence);
            if (ret != ZE_RESULT_SUCCESS) {
                if (ret != ZE_RESULT_NOT_READY) {
                    //TODO L0 workaround: Virtual Device may execute this part before fence actually queued
                    //Therefore, ZE_RESULT_ERROR_INVALID_ARGUMENT is normal for virtual device, but not for real
                    if (gpu_comm_impl::type_idx() == ccl_gpu_comm::type_idx() or
                        gpu_comm_impl::type_idx() ==
                            ccl_ipc_source_gpu_comm<ccl_gpu_comm>::type_idx()) {
                        if (group_id == ccl::device_group_split_type::cluster) {
                            auto c = ccl::environment::instance().create_communicator();
                            if (c->rank() == 0) {
                                throw ccl::ccl_error(
                                    std::string("cannot sync queue from real device, error: ") +
                                    native::to_string(ret));
                            }
                        }
                        else {
                            throw ccl::ccl_error(
                                std::string("cannot sync queue from real device, error: ") +
                                native::to_string(ret));
                        }
                    }
                    else {
                        if (ret != ZE_RESULT_ERROR_INVALID_ARGUMENT) {
                            throw ccl::ccl_error(
                                std::string("cannot sync queue from virtual device, error: ") +
                                native::to_string(ret));
                        }
                    }
                }
                LOG_TRACE(class_name(),
                          " entry req ",
                          &req,
                          ", rank: ",
                          comm_addr.to_string(),
                          " not completed yet, reason: ",
                          native::to_string(ret));
            }
            else {
                status = ccl_sched_entry_status_complete;
                LOG_DEBUG(class_name(),
                          " entry req ",
                          &req,
                          ", rank: ",
                          comm_addr.to_string(),
                          " completed");
            }
        }
    }

    virtual const char *name() const override {
        return class_name();
    }

    ze_fence_handle_t get_fence() {
        return fence;
    }

protected:
    virtual bool finalize_entry() = 0;
    virtual void dump_detail(std::stringstream &str) const override {
        ccl_logger::format(str,
                           class_name(),
                           ", dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", elem_count ",
                           elem_count,
                           ", send_buf ",
                           send_buf,
                           ", recv_buf ",
                           recv_buf,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", comm_id ",
                           sched->coll_param.comm->id(),
                           ", req ",
                           &req,
                           "\n");
    }

protected:
    std::shared_ptr<gpu_comm> parent_communicator;
    topology_addr<group_id, class_id> comm_addr;
    ccl_buffer send_buf;
    ccl_buffer recv_buf;
    size_t elem_count;
    ccl_datatype dtype;
    ccl_reduction_t op;
    atl_req_t req{};
    std::shared_ptr<ccl_stream> device_stream;

    // GPU
    bool ready_to_exec = false;

    //std::unique_ptr<ccl_device::device_cmd_list> copy_send_cmd_list;
    //std::unique_ptr<ccl_device::device_cmd_list> copy_recv_cmd_list;
    //std::unique_ptr<ccl_device::device_cmd_list> exec_cmd_list;
    ze_fence_handle_t fence;

    //TODO
    ze_group_count_t launch_args = { 1, 1, 1 };

    template <class executor>
    static std::unique_ptr<base_connector_interface<kernel_main_typed>>
    create_kernel_router_for_rank(executor &exec,
                                  size_t next_rank,
                                  specific_indexed_device_storage &group_devices) {
        std::unique_ptr<base_connector_interface<kernel_main_typed>> kernel_router;
        while (!kernel_router) {
            //Gather data from in-process GPU
            auto &map_devices = std::get<ccl_gpu_comm::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_gpu_comm> gpu = it->second;
            kernel_main_typed &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();

            //communicate with real device
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, kernel_main_typed>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //concurrent GPU
            using thread_group_comm_device = ccl_thread_comm<ccl_gpu_comm>;
            native::indexed_device_container<thread_group_comm_device> &map_devices =
                std::get<thread_group_comm_device::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_thread_comm<ccl_gpu_comm>> gpu = it->second;

            /*std::shared_ptr<thread_group_comm_device> gpu = map_devices.find(next_rank);
            if(gpu == nullptr)
            {
                break; // not ready yet!
            }*/
            kernel_main_typed &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();

            //communicate with real device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, kernel_main_typed>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //concurrent GPU
            using thread_group_comm_device = ccl_thread_comm<ccl_virtual_gpu_comm>;
            native::indexed_device_container<thread_group_comm_device> &map_devices =
                std::get<thread_group_comm_device::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_thread_comm<ccl_virtual_gpu_comm>> gpu = it->second;
            /*
            std::shared_ptr<thread_group_comm_device> gpu = map_devices.find(next_rank);
            if(gpu == nullptr)
            {
                break; // not ready yet!
            }*/
            kernel_main_typed &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();

            //communicate with virtual device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, kernel_main_typed>(
                    exec, right_main_func));
        }
        /*
        while(!kernel_router)
        {
            //ipc-source GPU
            using thread_group_comm_device = ccl_ipc_source_gpu_comm<ccl_gpu_comm>;
            native::indexed_device_container<thread_group_comm_device>& map_devices = std::get<thread_group_comm_device::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if(it == map_devices.end())
            {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_ipc_source_gpu_comm<ccl_gpu_comm>> gpu = it->second;
            kernel_main_typed &right_main_func = gpu->get_gpu_kernel<type(), native_type>();

            //communicate with real device from another thread
            kernel_router.reset(new router<kernel_main_typed,
                                           executor,
                                           kernel_main_typed>(exec, right_main_func));
        }

        while(!kernel_router)
        {
            //concurrent GPU
            using thread_group_comm_device = ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>;
            native::indexed_device_container<thread_group_comm_device>& map_devices = std::get<thread_group_comm_device::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if(it == map_devices.end())
            {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>> gpu = it->second;
            kernel_main_typed &right_main_func = gpu->get_gpu_kernel<type(), native_type>();

            //communicate with virtual device from another thread
            kernel_router.reset(new router<kernel_main_typed,
                                           executor,
                                           kernel_main_typed>(exec, right_main_func));
        }
*/
        while (!kernel_router) {
            //Virtual GPU
            auto &map_devices = std::get<ccl_virtual_gpu_comm::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_virtual_gpu_comm> gpu = it->second;
            kernel_main_typed &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, kernel_main_typed>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //gather data for ipc-GPU
            auto &map_devices = std::get<ccl_ipc_gpu_comm::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<ccl_ipc_gpu_comm> gpu = it->second;
            kernel_ipc_typed &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();
            kernel_router.reset(new kernel_connector<kernel_main_typed, executor, kernel_ipc_typed>(
                exec, right_main_func));
        }

        //sanity
        if (!kernel_router) {
            LOG_ERROR("Cannot bind communicators in group for next rank: ", next_rank);
        }
        return kernel_router;
    }

    std::unique_ptr<base_connector_interface<kernel_main_typed>> kernel_router;
};

} // namespace native
