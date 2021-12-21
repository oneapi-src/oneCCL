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

#include "common/comm/comm.hpp"
#include "common/global/global.hpp"
#include "sched/sched.hpp"
#include "sched/entry/entry.hpp"

#include <ze_api.h>

using namespace ccl::ze;

struct cmd_primitives {
    ze_command_queue_handle_t queue{};
    ze_command_queue_desc_t queue_desc{ default_cmd_queue_desc };
    ze_command_list_handle_t list{};
    ze_command_list_desc_t list_desc{ default_cmd_list_desc };
};

class ze_base_entry : public sched_entry {
public:
    ze_base_entry() = delete;
    ze_base_entry(const ze_base_entry &) = delete;
    virtual ~ze_base_entry();

    static ze_event_handle_t create_event(ze_event_pool_handle_t event_pool,
                                          ze_event_desc_t event_desc);
    static bool is_event_completed(ze_event_handle_t event);

    ze_event_handle_t entry_event{};

protected:
    explicit ze_base_entry(ccl_sched *sched,
                           init_mode mode = init_mode::compute,
                           ccl_comm *comm = nullptr,
                           uint32_t add_event_count = 0,
                           std::vector<ze_event_handle_t> wait_events = {});

    void init() override;
    void finalize() override;

    /* ze hooks which can be implemented in derived entry */
    virtual void init_ze_hook(){};
    virtual void finalize_ze_hook(){};

    virtual void start() override;
    virtual void update() override;

    void init_entries();
    void finalize_entries();

    ze_command_list_handle_t get_comp_list();
    ze_command_list_handle_t get_copy_list();

    virtual std::string name_ext() const;

    void init_primitives(cmd_primitives &cmd_primitives);
    void get_copy_primitives(const ze_queue_properties_t &queue_props,
                             cmd_primitives &copy_primitives,
                             init_mode mode);
    void get_comp_primitives(const ze_queue_properties_t &queue_props,
                             cmd_primitives &comp_primitives);

    ze_event_handle_t create_event();
    void reset_events();
    void destroy_events();

    void close_lists();

    init_mode mode;

    ccl_comm *comm{};
    int comm_rank{};
    int comm_size{};

    size_t worker_idx{};

    bool is_initialized{};

    ze_module_handle_t module{};

    ze_device_handle_t device{};
    ze_context_handle_t context{};

    cmd_primitives comp_primitives{};
    cmd_primitives copy_primitives{};

    const bool use_single_list;

private:
    uint32_t event_counter{};
    ze_event_pool_desc_t event_pool_desc{};
    ze_event_pool_handle_t event_pool{};
    std::vector<ze_event_handle_t> events;

    std::vector<ze_event_handle_t> wait_events;

    bool is_queue_completed(ze_command_queue_handle_t queue);
    void append_wait_on_events();
};

class ze_kernel {
public:
    ze_kernel(ze_module_handle_t module, const std::string &kernel_name, size_t worker_idx = 0);
    ze_kernel(const ze_kernel &) = delete;
    ze_kernel(ze_kernel &&other) noexcept;
    ~ze_kernel();

    void set_args(ze_kernel_args_t kernel_args);
    void calculate_group_size(size_t elem_count);
    ze_kernel_handle_t get_kernel() const;
    const ze_group_count_t *get_group_count() const;

private:
    ze_module_handle_t module{};
    std::string kernel_name{};
    size_t worker_idx{};
    ze_group_count_t group_count{};
    ze_group_size_t group_size{};
    ze_kernel_handle_t kernel{};
};
