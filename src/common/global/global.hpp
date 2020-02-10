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

#include "ccl.h"
#include "common/utils/utils.hpp"
#include "coll/algorithms/algorithms_enum.hpp"

#include <memory>
#include <thread>

#define COMMON_CATCH_BLOCK()                             \
    catch (ccl::ccl_error& ccl_e)                        \
    {                                                    \
        LOG_ERROR("ccl internal error: ", ccl_e.what()); \
        return ccl_status_invalid_arguments;             \
    }                                                    \
    catch (std::exception& e)                            \
    {                                                    \
        LOG_ERROR("error: ", e.what());                  \
        return ccl_status_runtime_error;                 \
    }                                                    \
    catch (...)                                          \
    {                                                    \
        LOG_ERROR("general error");                      \
        return ccl_status_runtime_error;                 \
    }                                                    \

class ccl_comm;
class ccl_stream;
class ccl_atl_tag;
class ccl_comm_id_storage;
class ccl_executor;
class ccl_sched_cache;
class ccl_parallelizer;
class ccl_fusion_manager;
class ccl_unordered_coll_manager;
class ccl_allreduce_2d_builder;

template<ccl_coll_type... registered_types_id>
class ccl_algorithm_selector_wrapper;

struct alignas(CACHELINE_SIZE) ccl_global_data
{
    std::unique_ptr<ccl_comm_id_storage> comm_ids;
    std::shared_ptr<ccl_comm> comm;
    std::unique_ptr<ccl_atl_tag> atl_tag;
    std::unique_ptr<ccl_executor> executor;
    std::unique_ptr<ccl_coll_attr_t> default_coll_attr;
    std::unique_ptr<ccl_sched_cache> sched_cache;
    std::unique_ptr<ccl_parallelizer> parallelizer;
    std::unique_ptr<ccl_fusion_manager> fusion_manager;
    std::unique_ptr<ccl_unordered_coll_manager> unordered_coll_manager;
    std::unique_ptr<ccl_algorithm_selector_wrapper<CCL_COLL_LIST>> algorithm_selector;
    std::unique_ptr<ccl_allreduce_2d_builder> allreduce_2d_builder;
    static thread_local bool is_worker_thread;
    bool is_ft_enabled;
};

extern ccl_global_data global_data;

#define CCL_CHECK_IS_BLOCKED()                         \
  {                                                    \
    do                                                 \
    {                                                  \
        if (unlikely(global_data.executor->is_locked)) \
        {                                              \
            return ccl_status_blocked_due_to_resize;   \
        }                                              \
    } while (0);                                       \
  }

void ccl_reset_for_size_update(ccl_global_data* gl_data);

void ccl_init_global_objects(ccl_global_data& gl_data);
