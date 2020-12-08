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
#include "coll/selection/selection.hpp"
#include "common/comm/atl_tag.hpp"
#include "common/comm/comm_id_storage.hpp"
#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/tree.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/cache/cache.hpp"

namespace ccl {

thread_local bool global_data::is_worker_thread = false;

global_data::global_data() {
    /* create ccl_logger before ccl::global_data
       to ensure static objects construction/destruction rule */
    LOG_INFO("create global_data object");

    //TODO new_api configure thread wait timeout
    thread_barrier_wait_timeout_sec = 5;
}

global_data::~global_data() {
    reset();
}

global_data& global_data::get() {
    static global_data data;
    return data;
}

env_data& global_data::env() {
    return get().env_object;
}

ccl::status global_data::reset() {
    /*
        executor is resize_dependent object but out of regular reset procedure
        executor is responsible for resize logic and has own multi-step reset
     */
    executor.reset();
    reset_resize_dependent_objects();
    reset_resize_independent_objects();

    return ccl::status::success;
}

ccl::status global_data::init() {
    env_object.parse();
    env_object.set_internal_env();

    init_resize_dependent_objects();
    init_resize_independent_objects();

    return ccl::status::success;
}

void global_data::init_resize_dependent_objects() {
    dtypes = std::unique_ptr<ccl_datatype_storage>(new ccl_datatype_storage());

    sched_cache = std::unique_ptr<ccl_sched_cache>(new ccl_sched_cache());

    if (env_object.enable_fusion) {
        /* create fusion_manager before executor because service_worker uses fusion_manager */
        fusion_manager = std::unique_ptr<ccl_fusion_manager>(new ccl_fusion_manager());
    }

    executor = std::unique_ptr<ccl_executor>(new ccl_executor());

    comm_ids =
        std::unique_ptr<ccl_comm_id_storage>(new ccl_comm_id_storage(ccl_comm::max_comm_count));
}

void global_data::init_resize_independent_objects() {
    parallelizer = std::unique_ptr<ccl_parallelizer>(new ccl_parallelizer(env_object.worker_count));

    algorithm_selector = std::unique_ptr<ccl_algorithm_selector_wrapper<CCL_COLL_LIST>>(
        new ccl_algorithm_selector_wrapper<CCL_COLL_LIST>());

    algorithm_selector->init();

    bf16_impl_type = ccl_bf16_get_impl_type();
}

void global_data::reset_resize_dependent_objects() {
    comm_ids.reset();
    fusion_manager.reset();
    sched_cache.reset();
    dtypes.reset();
}

void global_data::reset_resize_independent_objects() {
    parallelizer.reset();
    algorithm_selector.reset();
}

} /* namespace ccl */
