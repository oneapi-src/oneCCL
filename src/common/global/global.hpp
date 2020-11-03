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

//#include "ccl.h"
#include "coll/algorithms/algorithms_enum.hpp"
#include "comp/bf16/bf16_utils.hpp"
#include "common/env/env.hpp"
#include "common/utils/utils.hpp"
#include "common/comm/l0/comm_context_storage.hpp"

#include <memory>
#include <thread>

#define COMMON_CATCH_BLOCK() \
    catch (ccl::exception & ccl_e) { \
        LOG_ERROR("ccl internal error: ", ccl_e.what()); \
        return ccl_status_invalid_arguments; \
    } \
    catch (std::exception & e) { \
        LOG_ERROR("error: ", e.what()); \
        return ccl_status_runtime_error; \
    } \
    catch (...) { \
        LOG_ERROR("general error"); \
        return ccl_status_runtime_error; \
    }

class ccl_comm;
class ccl_stream;
class ccl_comm_id_storage;
class ccl_datatype_storage;
class ccl_executor;
class ccl_sched_cache;
class ccl_parallelizer;
class ccl_fusion_manager;
struct ccl_group_context;

template <ccl_coll_type... registered_types_id>
class ccl_algorithm_selector_wrapper;

namespace ccl {

// class comm_group;
// using comm_group_t = std::shared_ptr<comm_group>;

// struct ccl_group_context {
//      TODO
//      * In multithreading scenario we use different comm_group_t objects in different threads.
//      * But we need to match different groups created for the same world in different threads
//      * The assumption is done: if different groups created from the same communicator color, than they
//      * should be interpreted as the same groups in the same world.
//      *
//      *
//      * In the final solution the 'group_unique_key' should be equal to unique KVS idenditifier

//     using group_unique_key = typename ccl::ccl_host_attributes_traits<ccl_host_color>::type;
//     std::map<group_unique_key, comm_group_t> communicator_group_map;
//     ccl_spinlock mutex;
// };

class global_data {
public:
    global_data(const global_data&) = delete;
    global_data(global_data&&) = delete;

    global_data& operator=(const global_data&) = delete;
    global_data& operator=(global_data&&) = delete;

    ~global_data();

    ccl_status_t init();
    ccl_status_t reset();

    static global_data& get();
    static env_data& env();

    /* public methods to have access from listener thread function */
    void init_resize_dependent_objects();
    void reset_resize_dependent_objects();

    std::unique_ptr<ccl_comm_id_storage> comm_ids;
    std::shared_ptr<ccl_comm> comm;
    std::unique_ptr<ccl_datatype_storage> dtypes;
    std::unique_ptr<ccl_executor> executor;
    std::unique_ptr<ccl_coll_attr_t> default_coll_attr; // TODO: use ccl_coll_attr
    std::unique_ptr<ccl_sched_cache> sched_cache;
    std::unique_ptr<ccl_parallelizer> parallelizer;
    std::unique_ptr<ccl_fusion_manager> fusion_manager;
    std::unique_ptr<ccl_algorithm_selector_wrapper<CCL_COLL_LIST>> algorithm_selector;
    std::unique_ptr<group_context> global_ctx;

    static thread_local bool is_worker_thread;
    bool is_ft_enabled;
    ccl_bf16_impl_type bf16_impl_type;

    //TODO new_api configure thread wait timeout
    size_t thread_barrier_wait_timeout_sec = 5;

private:
    global_data();

    void init_resize_independent_objects();
    void reset_resize_independent_objects();

    env_data env_object;
};

#define CCL_CHECK_IS_BLOCKED() \
    { \
        do { \
            if (unlikely(ccl::global_data::get().executor->is_locked)) { \
                return ccl_status_blocked_due_to_resize; \
            } \
        } while (0); \
    }

} /* namespace ccl */
