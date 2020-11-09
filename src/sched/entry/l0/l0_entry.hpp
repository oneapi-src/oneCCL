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
#include "oneapi/ccl/types.hpp"
#include "common/datatype/datatype.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives.hpp"
#include "common/comm/l0/modules/kernel_functions.hpp"

#include "oneapi/ccl.hpp"

#include "comp/comp.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "sched/entry/coll/direct/base_coll_entry.hpp"

#include "common/comm/l0/modules_connector.hpp"
#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
//TODO L0 Workaround
#include <unistd.h>
static std::mutex global_fence_mutex;

#define ENTRY_LOG_TRACE(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::TRACE)) { \
        do { \
            std::stringstream ss; \
            this->dump_detail(ss); \
            logger.trace("|TRACE| ", \
                         basedir_static(__FILE__), \
                         ":", \
                         __LINE__, \
                         "  ", \
                         ss.str(), \
                         " - ", \
                         ##__VA_ARGS__); \
        } while (0); \
    }

#define ENTRY_LOG_DEBUG(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::DEBUG)) { \
        do { \
            std::stringstream ss; \
            this->dump_detail(ss); \
            logger.debug("|DEBUG| ", \
                         basedir_static(__FILE__), \
                         ":", \
                         __LINE__, \
                         "  ", \
                         ss.str(), \
                         " - ", \
                         ##__VA_ARGS__); \
        } while (0); \
    }

#define ENTRY_LOG_INFO(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::INFO)) { \
        do { \
            std::stringstream ss; \
            this->dump_detail(ss); \
            logger.info("|INFO| ", \
                        basedir_static(__FILE__), \
                        ":", \
                        __LINE__, \
                        "  ", \
                        ss.str(), \
                        " - ", \
                        ##__VA_ARGS__); \
        } while (0); \
    }

namespace native {
template <class native_type,
          class gpu_comm_impl,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          ccl_coll_type type_op>
class base_gpu_entry : public sched_entry {
public:
    using gpu_comm = gpu_comm_impl;
    using processing_type = native_type;
    using kernel_main_typed =
        typename gpu_comm::template gpu_kernel_t<type_op, group_id, class_id, native_type>;
    // using kernel_ipc_typed = ring_allreduce_ipc<native_type>;

    template <class elem_t>
    using device_memory = memory<elem_t, ccl_device, ccl_context>;

    friend class ccl_gpu_comm;
    friend class ccl_virtual_gpu_comm;
    static constexpr const char *class_name() noexcept {
        return ccl_coll_type_to_str(type_op);
    }
    static constexpr ccl_coll_type type() noexcept {
        return type_op;
    }

    static constexpr ccl::group_split_type get_topology() {
        return group_id;
    }

    static constexpr ccl::device_topology_type get_topology_class() {
        return class_id;
    }

    base_gpu_entry() = delete;
    base_gpu_entry(ccl_sched *sched,
                   std::shared_ptr<gpu_comm> comm,
                   ccl_driver_context_ptr in_ctx,
                   const ccl_buffer send_buf,
                   ccl::datatype dtype_in,
                   std::shared_ptr<ccl_stream> &stream)
            : sched_entry(sched),
              parent_communicator(comm),
              comm_addr(parent_communicator
                            ->template get_comm_data<get_topology(), get_topology_class()>()),
              send_buf(send_buf),
              dtype(dtype_in),
              device_stream(stream),
              ctx(in_ctx) {}

    virtual ~base_gpu_entry() {}

    virtual void start() override {
        ccl_device &device = parent_communicator->get_device();
        {
            //TODO make check, that device_stream belong to the device
            auto queue_prop = ccl_device::get_default_queue_desc();
            auto &cmd_queue = device.get_cmd_queue(queue_prop, ctx);
            fence = device.get_fence(cmd_queue, ctx).get();

            ENTRY_LOG_DEBUG("start base entry initialization, ctx: ",
                            ctx.get(),
                            ", queue: ",
                            cmd_queue.get(),
                            ", fence: ",
                            fence);
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

        auto send_buf_ptr = reinterpret_cast<native_type *>(send_buf.get_ptr());

        main_entry_function.template set_args<typename kernel_main_typed::common_entry_buf_arg>(
            send_buf_ptr);

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
        ENTRY_LOG_DEBUG("started");
    }

    bool submit_for_execution() {
        ready_to_exec = finalize_entry();
        if (ready_to_exec) {
            //TODO L0 workaround
            //if(std::is_same<gpu_comm_impl, ccl_gpu_comm>::value)
            if (gpu_comm_impl::type_idx() == ccl_gpu_comm::type_idx() or
                gpu_comm_impl::type_idx() == ccl_ipc_source_gpu_comm<ccl_gpu_comm>::type_idx()) {
                ccl_device &device = parent_communicator->get_device();
                auto queue_prop = ccl_device::get_default_queue_desc();
                auto &cmd_queue = device.get_cmd_queue(queue_prop, ctx);
                auto &cmd_list = device.get_cmd_list(ctx);
                ENTRY_LOG_DEBUG("Start submit for execution: main device: ",
                                parent_communicator->to_string(),
                                ", queue: ",
                                cmd_queue.get(),
                                ", list: ",
                                cmd_list.get());
                if (group_id == ccl::group_split_type::cluster) {
                    //auto c = ccl::detail::environment::instance().create_communicator();
                    //(void)c;
                    //if(c->rank() == 0)
                    {
                        // Execute command list in command queue
                        //TODO SPECIAL FOR VIRTUAL
                        /*
                    if(std::is_same<gpu_comm, ccl_virtual_gpu_comm>::value)
                    {
                        queue_prop.ordinal = parent_communicator->get_rank(); //TODO SPECIAL FOR VIRTUAL
                    }
                    queue_prop.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;*/

                        ze_result_t ret = zeCommandQueueExecuteCommandLists(
                            cmd_queue.get(), 1, cmd_list.get_ptr(), fence);
                        if (ret != ZE_RESULT_SUCCESS) {
                            throw ccl::exception(
                                std::string("cannot execute command list, error: ") +
                                std::to_string(ret));
                        }
                    }
                }
                else {
                    /*-S-
                    ze_result_t ret = zeCommandQueueExecuteCommandLists(
                        cmd_queue.get(), 1, cmd_list.get_ptr(), fence);
                    if (ret != ZE_RESULT_SUCCESS) {
                        throw ccl::exception(std::string("cannot execute command list, error: ") +
                                             std::to_string(ret));
                    }
                    */
                }
            }
        }
        ENTRY_LOG_TRACE("submission result: ", ready_to_exec);
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
            auto &cmd_queue = device.get_cmd_queue(queue_prop, ctx);

            ENTRY_LOG_TRACE(" waiting for finished execution, queue: ", cmd_queue.get());
            /* TODO fence!
            ze_result_t ret = zeCommandQueueSynchronize(cmd_queue.handle,
                                                          std::numeric_limits<uint32_t>::max());*/
            ze_result_t ret = zeFenceQueryStatus(fence);
            ENTRY_LOG_TRACE(
                "Fence query status: ", native::to_string(ret), ", queue: ", cmd_queue.get());
            if (ret != ZE_RESULT_SUCCESS) {
                if (ret != ZE_RESULT_NOT_READY) {
                    //TODO L0 workaround: Virtual Device may execute this part before fence actually queued
                    //Therefore, ZE_RESULT_ERROR_INVALID_ARGUMENT is normal for virtual device, but not for real
                    if (gpu_comm_impl::type_idx() == ccl_gpu_comm::type_idx() or
                        gpu_comm_impl::type_idx() ==
                            ccl_ipc_source_gpu_comm<ccl_gpu_comm>::type_idx()) {
                        if (group_id == ccl::group_split_type::cluster) {
                            // TODO: implement process communicator case
                            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                                 "TODO: implement process communicator case");
                            // auto c = ccl::detail::environment::instance().create_communicator();
                            // if (c.rank() == 0) {
                            // throw ccl::exception(
                            //     std::string("cannot sync queue from real device, error: ") +
                            //     native::to_string(ret));
                            // }
                        }
                        else {
                            throw ccl::exception(
                                std::string("cannot sync queue from real device, error: ") +
                                native::to_string(ret));
                        }
                    }
                    else {
                        if (ret != ZE_RESULT_ERROR_INVALID_ARGUMENT) {
                            throw ccl::exception(
                                std::string("cannot sync queue from virtual device, error: ") +
                                native::to_string(ret));
                        }
                    }
                }
            }
            else {
                status = ccl_sched_entry_status_complete;
                ENTRY_LOG_DEBUG(" Completed on queue: ", cmd_queue.get());
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
        ccl_logger::format(str, "{", name(), ", addr: ", comm_addr.to_string(), "}");
    }

protected:
    ccl_driver_context_ptr get_ctx() const {
        return ctx;
    }

    template <template <size_t pos, class Policy> class KernelArg, size_t POS, class POL>
    device_memory<typename std::remove_pointer<typename KernelArg<POS, POL>::arg_type>::type>
    alloc_memory_wrap(const KernelArg<POS, POL> &arg,
                      std::shared_ptr<gpu_comm> parent_communicator,
                      size_t cnt,
                      std::shared_ptr<ccl_context> ctx) {
        using alloc_type =
            typename std::remove_pointer<typename KernelArg<POS, POL>::arg_type>::type;
        auto memory = parent_communicator->get_device().template alloc_memory<alloc_type>(
            cnt, sizeof(alloc_type), ctx);
        LOG_DEBUG("Allocation memory by default: ",
                  POS,
                  ", ctx: ",
                  (void *)ctx.get(),
                  ", memory: ",
                  (void *)memory.get());
        return memory;
    }

    template <template <size_t pos, class> class KernelArg, size_t POS, class Type, bool B>
    device_memory<typename std::remove_pointer<
        typename KernelArg<POS, arg_access_policy_atomic_uncached<POS, Type, B>>::arg_type>::type>
    alloc_memory_wrap(const KernelArg<POS, arg_access_policy_atomic_uncached<POS, Type, B>> &arg,
                      std::shared_ptr<gpu_comm> parent_communicator,
                      size_t cnt,
                      std::shared_ptr<ccl_context> ctx) {
        using alloc_type = typename std::remove_pointer<
            typename KernelArg<POS,
                               arg_access_policy_atomic_uncached<POS, Type, B>>::arg_type>::type;
        ze_device_mem_alloc_desc_t mem_descr{
            .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            .pNext = NULL,
            .flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED,
            .ordinal = 0,
        };
        auto memory = parent_communicator->get_device().template alloc_memory<alloc_type>(
            cnt, sizeof(alloc_type), ctx, mem_descr);
        LOG_DEBUG("Allocation memory with bias uncached flag: ",
                  POS,
                  ", ctx: ",
                  (void *)ctx.get(),
                  ", memory: ",
                  (void *)memory.get(),
                  " mem_descr: ",
                  native::to_string(mem_descr));
        return memory;
    }

    std::shared_ptr<gpu_comm> parent_communicator;
    topology_addr<group_id, class_id> comm_addr;
    ccl_buffer send_buf;
    ccl::datatype dtype;
    atl_req_t req{};
    std::shared_ptr<ccl_stream> device_stream;
    // GPU
    bool ready_to_exec = false;
    ze_fence_handle_t fence;

    //TODO
    ze_group_count_t launch_args = { 1, 1, 1 };

    template <class executor>
    static std::unique_ptr<base_connector_interface<kernel_main_typed>>
    create_kernel_router_for_rank(executor &exec,
                                  int next_rank,
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

        // TODO: check for launching ipc kernel
        // while (!kernel_router) {
        //     //gather data for ipc-GPU
        //     auto &map_devices = std::get<ccl_ipc_gpu_comm::type_idx()>(group_devices);
        //     auto it = map_devices.find(next_rank);
        //     if (it == map_devices.end()) {
        //         break; // not ready yet!
        //     }
        //     std::shared_ptr<ccl_ipc_gpu_comm> gpu = it->second;
        //     kernel_ipc_typed &right_main_func =
        //         gpu->get_gpu_kernel<type(), get_topology(), get_topology_class(), native_type>();
        //     kernel_router.reset(new kernel_connector<kernel_main_typed, executor, kernel_ipc_typed>(
        //         exec, right_main_func));
        // }

        //sanity
        if (!kernel_router) {
            LOG_ERROR("Cannot bind communicators in group for next rank: ", next_rank);
        }
        return kernel_router;
    }

    std::unique_ptr<base_connector_interface<kernel_main_typed>> kernel_router;

private:
    ccl_driver_context_ptr ctx;
};

} // namespace native
