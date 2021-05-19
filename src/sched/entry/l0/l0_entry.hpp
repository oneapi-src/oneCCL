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

#include "common/comm/l0/context/scale/ipc/ipc_session_key.hpp"
#include "common/comm/l0/context/scale/base/base_session.hpp"

//TODO L0 Workaround
#include <unistd.h>
static std::mutex global_fence_mutex;

#define ENTRY_LOG_TRACE(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::trace)) { \
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
    if (unlikely(logger.get_log_level() >= ccl_log_level::debug)) { \
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
    if (unlikely(logger.get_log_level() >= ccl_log_level::info)) { \
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

#define ENTRY_LOG_WARN(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::warn)) { \
        do { \
            std::stringstream ss; \
            this->dump_detail(ss); \
            logger.info("|WARN| ", \
                        basedir_static(__FILE__), \
                        ":", \
                        __LINE__, \
                        "  ", \
                        ss.str(), \
                        " - ", \
                        ##__VA_ARGS__); \
        } while (0); \
    }

#define ENTRY_LOG_ERROR(...) \
    if (unlikely(logger.get_log_level() >= ccl_log_level::error)) { \
        do { \
            std::stringstream ss; \
            this->dump_detail(ss); \
            logger.info("|ERROR| ", \
                        basedir_static(__FILE__), \
                        ":", \
                        __LINE__, \
                        "  ", \
                        ss.str(), \
                        " - ", \
                        ##__VA_ARGS__); \
        } while (0); \
    }

// This is a internal gpu entry state to keep track of the progress
// filling and submitting a kernel as well as its execution
enum class gpu_entry_state {
    // default state
    initial,
    // Entry is fully constructed
    created,
    // Local parameters are filled and the entry is waiting for
    // parameters from the neighbour entry. No further progress
    // until these parameters are received. After that
    // the entry appends it's kernel. After that the kernel is
    // submited to the queue(only one rank does that in case of
    // virtual device.
    wait_for_entry,
    // The kernel is submited and the entry is waiting for kernel
    // completion by checking fence status.
    wait_for_completion,
    // Execution is done, it's possible to reuse the entry by
    // moving the entry to created state
    completed,
    // Last element in the enum, not used as state
    last
};

inline std::string to_string(gpu_entry_state state) {
    return utils::enum_to_str<static_cast<int>(gpu_entry_state::last)>{
        "initial", "created", "wait_for_entry", "wait_for_completion", "completed"
    }
        .choose(state);
}

namespace native {
template <class gpu_comm_impl,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          ccl_coll_type type_op>
class base_gpu_entry : public sched_entry {
public:
    using gpu_comm = gpu_comm_impl;
    using kernel_main_typed = typename gpu_comm::template gpu_kernel_t<type_op, group_id, class_id>;
    using kernel_ipc_typed =
        typename ccl_ipc_gpu_comm::template gpu_kernel_t<type_op, group_id, class_id>;

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
                   const coll_param_gpu &params,
                   std::shared_ptr<ccl_stream> &stream)
            : sched_entry(sched),
              parent_communicator(comm),
              comm_addr(parent_communicator
                            ->template get_comm_data<get_topology(), get_topology_class()>()),
              send_buf(send_buf),
              params(params),
              device_stream(stream),
              ctx(in_ctx),
              entry_state(gpu_entry_state::initial),
              queue_descr(init_queue_descr(parent_communicator->get_device())),
              list_descr(init_list_descr(parent_communicator->get_device())),
              dev_queue(init_default_queue(parent_communicator->get_device())),
              dev_cmd_list(init_default_cmd_list()) {
        // TODO: remove once all the child entries are refactored to not
        // use fence field directly
        fence = get_fence();
    }

    kernel_main_typed &get_local_kernel() noexcept {
        return parent_communicator
            ->template get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);
    }

    virtual ~base_gpu_entry() {}

    virtual void start() override {
        {
            //TODO make check, that device_stream belong to the device
            auto &cmd_queue = get_dev_queue();

            auto fence = parent_communicator->get_fence(cmd_queue, get_ctx());

            ENTRY_LOG_DEBUG("start base entry initialization, ctx: ",
                            ctx.get(),
                            ", queue: ",
                            cmd_queue.get(),
                            ", fence: ",
                            fence.get());
        }

        //set kernel args for main kernel on current device
        kernel_main_typed &main_entry_function =
            parent_communicator->template register_entry<group_id, class_id>(*this);

        auto send_buf_ptr = send_buf.get_ptr();

        //bind data
        main_entry_function.template set_args<typename kernel_main_typed::common_entry_buf_arg>(
            send_buf_ptr);

        status = ccl_sched_entry_status_started;
        ENTRY_LOG_DEBUG("started");
    }

    bool submit_for_execution() {
        ready_to_exec = finalize_entry();
        ENTRY_LOG_TRACE("submission result: ", ready_to_exec);
        return ready_to_exec;
    }

    virtual void update() override {
        if (!ready_to_exec) {
            // TODO: what if submit_for_execution() return false?
            submit_for_execution();
        }
        else {
            //wait execution
            auto &cmd_queue = get_dev_queue();

            ENTRY_LOG_TRACE(" waiting for finished execution, queue: ", cmd_queue.get());

            ze_result_t ret;

            // Quering fence doesn't sync kernel output with the host, so if we need this
            // we use QuerySyncronize API.
            if (ccl::global_data::env().comm_kernels_debug == 0) {
                ret = get_fence_impl().query_status();
            }
            else {
                ret = zeCommandQueueSynchronize(cmd_queue.get(), 0);
            }

            ENTRY_LOG_TRACE(
                "Fence query status: ", native::to_string(ret), ", queue: ", cmd_queue.get());
            if (ret == ZE_RESULT_SUCCESS) {
                this->set_state(gpu_entry_state::completed);

                // Once all the ranks in the group got the notification, reset the state for further launches
                reset_state();

                status = ccl_sched_entry_status_complete;
                ENTRY_LOG_DEBUG(" Completed on queue: ", cmd_queue.get());
            }
            else if (ret == ZE_RESULT_NOT_READY) {
                // Just return in case if the kernel is not ready yet, will check again on the next iteration
                return;
            }
        }
    }

    virtual const char *name() const override {
        return class_name();
    }

    // getters
    ccl_device::device_queue &get_dev_queue() const {
        return dev_queue;
    }

    ze_fence_handle_t get_fence() {
        return get_fence_impl().get();
    }

    ze_command_queue_desc_t &get_queue_descr() {
        return queue_descr;
    }

    //USE GPU cache binding
    virtual std::vector<ccl_device::device_ipc_memory_handle> get_ipc_data() = 0;
    virtual observer::invoke_params<type()> get_numa_data() {
        //TODO make pure-virtual
        ENTRY_LOG_ERROR("NOT implemented for that collective type");
        abort();
    }

    virtual observer::invoke_params<type()> get_scaleout_data() {
        //TODO make pure-virtual
        ENTRY_LOG_ERROR("NOT implemented for that collective type");
        abort();
    }

    virtual native::ipc_session_key get_ipc_session_key() const {
        return native::ipc_session_key{ this };
    }

    virtual native::observer::session_key get_numa_session_key() const {
        return native::observer::session_key{ this };
    }

    virtual native::observer::session_key get_scaleout_session_key() const {
        return native::observer::session_key{ this };
    }

    const coll_param_gpu &get_params() const {
        return params;
    }

protected:
    size_t get_work_group_size(size_t buffer_size, ccl_device &device) {
        size_t group_size;
        size_t val_vector_size;
        auto dtype = params.get_datatype();

        if (ccl::global_data::env().gpu_thread_count != CCL_ENV_SIZET_NOT_SPECIFIED) {
            group_size = ccl::global_data::env().gpu_thread_count;

            ENTRY_LOG_DEBUG(
                "Set group size for x dimension by CCL_GPU_THREAD_COUNT=", group_size, " by user");
        }
        else {
            if (dtype == ccl::datatype::bfloat16) {
                val_vector_size = 1;
            }
            else {
                // For comm kernels, we have float4 {float x, float y, float z, float w};
                // data type, that's why we set a divider for group_size, wchich equals to 4.
                // The vecsize of 4 goes with all data types except bfloat16
                val_vector_size = 4;
            }

            group_size = buffer_size / val_vector_size;

            ENTRY_LOG_DEBUG("Set group size for x dimension: ", group_size);
        }
        if (group_size > device.get_compute_properties().maxGroupSizeX || group_size == 0) {
            group_size = device.get_compute_properties().maxGroupSizeX;
            ENTRY_LOG_DEBUG(
                "Group size is limited by compute_properties.maxGroupSizeX and should NOT equal to 0, set group_size: ",
                group_size,
                " by default");
        }

        //TODO: remove 'return 1' and retrun 'group_size', when fix small msg sizes issue
        return 1; //group_size;
    }

    void get_suggested_group_size(ze_kernel_handle_t &kernel, size_t buffer_size) {
        // zeKernelSuggestGroupSize ignores the group size that is set using zeKernelSetGroupSize
        uint32_t group_size_x = 1u;
        uint32_t group_size_y = 1u;
        uint32_t group_size_z = 1u;
        ze_result_t result = zeKernelSuggestGroupSize(
            kernel, buffer_size, 1u, 1u, &group_size_x, &group_size_y, &group_size_z);
        if (result != ZE_RESULT_SUCCESS) {
            throw std::runtime_error(
                std::string(__FUNCTION__) +
                " - zeKernelSuggestGroupSize failed. Result: " + native::to_string(result));
        }
        ENTRY_LOG_DEBUG("Suggested kernel group sizes, which is based on buffer_size: ",
                        buffer_size,
                        ", are: groupSizeX: ",
                        group_size_x,
                        " groupSizeY: ",
                        group_size_y,
                        " groupSizeZ: ",
                        group_size_z);
    }

    void set_group_size(ze_kernel_handle_t &kernel, size_t buffer_size) {
        // setting the group size to control resource consumption
        // assuming that group_size_x can be adjusted by changing the value or CCL_GPU_THREAD_COUNT knob
        // group_size_y / group_size_z shouldn't be > 1
        uint32_t group_size_x = get_work_group_size(buffer_size, parent_communicator->get_device());
        uint32_t group_size_y = 1u;
        uint32_t group_size_z = 1u;

        ze_result_t result = zeKernelSetGroupSize(kernel, group_size_x, group_size_y, group_size_z);
        if (result != ZE_RESULT_SUCCESS) {
            throw std::runtime_error(
                std::string(__FUNCTION__) +
                " - zeKernelSetGroupSize failed. Result: " + native::to_string(result) +
                " groupSizeX: " + std::to_string(static_cast<uint32_t>(group_size_x)) +
                " groupSizeY: " + std::to_string(static_cast<uint32_t>(group_size_y)) +
                " groupSizeZ: " + std::to_string(static_cast<uint32_t>(group_size_z)));
        }

        ENTRY_LOG_DEBUG("Set kernel group size successfully: groupSizeX: ",
                        group_size_x,
                        " groupSizeY: ",
                        group_size_y,
                        " groupSizeZ: ",
                        group_size_z);
    }

    bool finalize_entry() {
        kernel_main_typed &main_entry_function = get_local_kernel();

        if (this->get_state() == gpu_entry_state::wait_for_entry) {
            if (!(*kernel_router)(main_entry_function)) {
                // Parameters are not ready yet, will try again later
                return false;
            }
        }

        ENTRY_LOG_TRACE("Try to finalize");

        auto &&cmd_list = get_dev_cmd_list();

        // setting group size
        set_group_size(main_entry_function.handle, send_buf.get_size());

        // get suggested group size for info usage only
        get_suggested_group_size(main_entry_function.handle, send_buf.get_size());

        cmd_list.append_kernel(main_entry_function.handle, &launch_args);

        ENTRY_LOG_DEBUG("Append kernel successfully: ",
                        main_entry_function.to_string(),
                        " in list: ",
                        cmd_list.get());

        assert(this->get_state() != gpu_entry_state::wait_for_completion);

        if (get_topology() == ccl::group_split_type::cluster) {
            // TODO: in case of (vitual device + IPC) we can get the data race here
            // How we can detect such case?
            // In the case when we use one GPU queue per process, everything should be ok
            // throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
            //                      "TODO: implement process communicator case");
            cmd_list.close_and_execute(get_ctx(), this->get_fence());
        }
        else {
            // TODO: how to ensure that fence update is thread safe?
            cmd_list.close_and_execute(get_ctx(), this->get_fence());
        }

        ENTRY_LOG_INFO("List closed:", cmd_list.get(), ", go to submit entry");
        this->set_state(gpu_entry_state::wait_for_completion);
        return true;
    }

    virtual void dump_detail(std::stringstream &str) const override {
        ccl_logger::format(str, "{", name(), ", addr: ", comm_addr.to_string(), "}");
    }

    void reset_state() {
        // Reset the state of the used handles
        get_fence_impl().reset();
        get_dev_cmd_list().reset();
    }

protected:
    ccl_driver_context_ptr get_ctx() const {
        return ctx;
    }

    ze_command_list_desc_t get_list_descr() const {
        return list_descr;
    }

    template <class options>
    ze_device_mem_alloc_desc_t get_mem_descr(options opt) {
        ze_device_mem_alloc_desc_t mem_descr = ccl_device::get_default_mem_alloc_desc();
        // Explicitly reset flags to avoid potential conflicts with the default value
        mem_descr.flags = 0;
        mem_descr.flags |= (opt.is_uncached() ? ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED
                                              : ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED);

        return mem_descr;
    }

    // Wrapper to handle memory allocation with different options
    template <class kernel_arg,
              // Get the actual underlying type to specify the type for the allocation
              class arg_type = typename std::remove_pointer<typename kernel_arg::arg_type>::type>
    device_memory<arg_type> alloc_memory_wrap(const kernel_arg &arg,
                                              std::shared_ptr<gpu_comm> parent_communicator,
                                              size_t cnt,
                                              std::shared_ptr<ccl_context> ctx) {
        auto mem_descr = get_mem_descr(typename kernel_arg::options_t{});
        auto memory = parent_communicator->get_device().template alloc_memory<arg_type>(
            cnt, sizeof(arg_type), ctx, mem_descr);

        LOG_DEBUG("Allocation memory by default: ",
                  kernel_arg::index,
                  ", ctx: ",
                  (void *)ctx.get(),
                  ", memory: ",
                  (void *)memory.get(),
                  ", mem_descr: ",
                  native::to_string(mem_descr));

        return memory;
    }

    std::shared_ptr<gpu_comm> parent_communicator;
    topology_addr<group_id, class_id> comm_addr;
    ccl_buffer send_buf;
    coll_param_gpu params;

    // TODO: we don't need dtype anymore?
    // ccl::datatype dtype;
    atl_req_t req{};
    std::shared_ptr<ccl_stream> device_stream;
    // GPU
    bool ready_to_exec = false;
    ze_fence_handle_t fence;

    auto get_fence_impl() -> decltype(parent_communicator->get_fence(get_dev_queue(), get_ctx())) {
        return parent_communicator->get_fence(get_dev_queue(), get_ctx());
    }

    auto get_dev_cmd_list()
        -> decltype(parent_communicator->get_cmd_list(get_ctx(), get_list_descr())) {
        return dev_cmd_list;
    }

    void set_state(gpu_entry_state new_state) noexcept {
        ENTRY_LOG_DEBUG(
            "switching entry state from ", to_string(entry_state), " to ", to_string(new_state));
        entry_state = new_state;
    }

    gpu_entry_state get_state() const noexcept {
        return entry_state;
    }

    //TODO
    ze_group_count_t launch_args = { 1, 1, 1 };

    template <class executor>
    static std::unique_ptr<base_connector_interface<kernel_main_typed>>
    create_kernel_router_for_rank(executor &exec,
                                  int next_rank,
                                  specific_indexed_device_storage &group_devices,
                                  const coll_param_gpu &params) {
        std::unique_ptr<base_connector_interface<kernel_main_typed>> kernel_router;
        while (!kernel_router) {
            //Gather data from in-process GPU
            using right_gpu_type = ccl_gpu_comm;
            auto &map_devices = std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }

            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;

            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with real device
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //Virtual GPU
            using right_gpu_type = ccl_virtual_gpu_comm;
            auto &map_devices = std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }

            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;

            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //concurrent GPU
            using right_gpu_type = ccl_thread_comm<ccl_gpu_comm>;
            native::indexed_device_container<right_gpu_type> &map_devices =
                std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;
            /*std::shared_ptr<thread_group_comm_device> gpu = map_devices.find(next_rank);
            if(gpu == nullptr)
            {
                break; // not ready yet!
            }*/
            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with real device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //concurrent GPU
            using right_gpu_type = ccl_thread_comm<ccl_virtual_gpu_comm>;
            native::indexed_device_container<right_gpu_type> &map_devices =
                std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;
            /*
            std::shared_ptr<thread_group_comm_device> gpu = map_devices.find(next_rank);
            if(gpu == nullptr)
            {
                break; // not ready yet!
            }*/
            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with virtual device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //ipc-source GPU REAL
            using right_gpu_type = ccl_ipc_source_gpu_comm<ccl_gpu_comm>;
            native::indexed_device_container<right_gpu_type> &map_devices =
                std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;
            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with real device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //ipc-source GPU VIRTUAL
            using right_gpu_type = ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>;
            native::indexed_device_container<right_gpu_type> &map_devices =
                std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }

            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;
            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with virtual device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        while (!kernel_router) {
            //ipc-source GPU VIRTUAL
            using right_gpu_type = ccl_ipc_gpu_comm;
            native::indexed_device_container<right_gpu_type> &map_devices =
                std::get<right_gpu_type::type_idx()>(group_devices);
            auto it = map_devices.find(next_rank);
            if (it == map_devices.end()) {
                break; // not ready yet!
            }
            std::shared_ptr<right_gpu_type> gpu = it->second;
            using right_kernel_main_type = typename right_gpu_type::
                template gpu_kernel_t<type(), get_topology(), get_topology_class()>;
            right_kernel_main_type &right_main_func =
                gpu->get_gpu_kernel<type(), get_topology(), get_topology_class()>(params);

            //communicate with virtual device from another thread
            kernel_router.reset(
                new kernel_connector<kernel_main_typed, executor, right_kernel_main_type>(
                    exec, right_main_func));
        }

        //sanity
        if (!kernel_router) {
            LOG_ERROR("Cannot bind communicators in group for next rank: ", next_rank);
        }
        return kernel_router;
    }

    std::unique_ptr<base_connector_interface<kernel_main_typed>> kernel_router;

private:
    ccl_driver_context_ptr ctx;
    // Internal gpu entry state to keep track of kernel status, it's not directly related to status field
    gpu_entry_state entry_state;
    ze_command_queue_desc_t queue_descr;
    ze_command_list_desc_t list_descr;
    ccl_device::device_queue &dev_queue;
    decltype(parent_communicator->get_cmd_list(ctx, list_descr)) dev_cmd_list;

    // initialize
    ze_command_queue_desc_t init_queue_descr(ccl_device &device) {
        native::ccl_device::queue_group_properties queue_props = device.get_queue_group_prop();

        queue_descr = device.get_default_queue_desc();

        // find compute ordinal
        uint32_t computeOrdinal = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < queue_props.size(); i++) {
            // Prefer CCS
            if (queue_props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE &&
                queue_props[i].numQueues > 1) {
                queue_descr.ordinal = i;
                break;
            }
        }
        // if CCS not found, look for RCS/CCS
        if (computeOrdinal == std::numeric_limits<uint32_t>::max()) {
            for (uint32_t i = 0; i < queue_props.size(); i++) {
                if (queue_props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                    queue_descr.ordinal = i;
                    break;
                }
            }
        }

        //calculate rank (remember it is a local rank)
        queue_descr.index = comm_addr.rank % queue_props[queue_descr.ordinal].numQueues;
        ENTRY_LOG_DEBUG("Rank to calculate for queue idx:",
                        comm_addr.rank,
                        ", queue : ",
                        to_string(queue_descr));
        return queue_descr;
    }

    ze_command_list_desc_t init_list_descr(ccl_device &device) {
        list_descr = parent_communicator->get_device().get_default_list_desc();
        return list_descr;
    }

    ccl_device::device_queue &init_default_queue(ccl_device &device) {
        return device.get_cmd_queue(queue_descr, ctx);
    }

    auto init_default_cmd_list() -> decltype(parent_communicator->get_cmd_list(ctx, list_descr)) {
        list_descr.commandQueueGroupOrdinal = queue_descr.ordinal;
        ENTRY_LOG_DEBUG("cmd_list: ", to_string(list_descr));
        return parent_communicator->get_cmd_list(ctx, list_descr);
    }
};

} // namespace native
