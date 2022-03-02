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
#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/buffer/buffer_cache.hpp"
#include "sched/cache/cache.hpp"

#include <sys/utsname.h>

namespace ccl {

thread_local bool global_data::is_worker_thread = false;

std::string os_information::to_string() {
    std::stringstream ss;
    ss << " { " << sysname << " " << nodename << " " << release << " " << version << " " << machine
       << " }";
    return ss.str();
}

void os_information::fill() {
    struct utsname os;
    uname(&os);
    sysname = os.sysname;
    nodename = os.nodename;
    release = os.release;
    version = os.version;
    machine = os.machine;
}

global_data::global_data() {
    /* create ccl_logger before ccl::global_data
       to ensure static objects construction/destruction rule */
    LOG_INFO("create global_data object");
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

os_information& global_data::get_os_info() {
    return get().os_info;
}

ccl::status global_data::reset() {
    /*
        executor is resize_dependent object but out of regular reset procedure
        executor is responsible for resize logic and has own multi-step reset
     */
    executor.reset();
    reset_resize_dependent_objects();
    reset_resize_independent_objects();

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    ze_data.reset();
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    return ccl::status::success;
}

ccl::status global_data::init() {
    env_object.parse();
    env_object.set_internal_env();

    os_info.fill();
    LOG_INFO("OS info:", os_info.to_string());
    if (os_info.release.find("WSL2") != std::string::npos) {
        env_object.enable_topo_algo = 0;
    }

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    if (ccl::global_data::env().backend == backend_mode::native &&
        ccl::global_data::env().ze_enable) {
        LOG_INFO("initializing level-zero api");
        if (ze_api_init()) {
            try {
                ze_data.reset(new ze::global_data_desc);
            }
            catch (const ccl::exception& e) {
                LOG_INFO("could not initialize level-zero: ", e.what());
            }
            catch (...) {
                LOG_INFO("could not initialize level-zero: unknown error");
            }
        }
        else {
            LOG_INFO("could not initialize level-zero api");
        }
    }
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    init_resize_dependent_objects();
    init_resize_independent_objects();

    return ccl::status::success;
}

void global_data::init_resize_dependent_objects() {
    dtypes.reset(new ccl_datatype_storage());

    sched_cache.reset(new ccl_sched_cache());
    buffer_cache.reset(new ccl::buffer_cache(env_object.worker_count));

    if (env_object.enable_fusion) {
        /* create fusion_manager before executor because service_worker uses fusion_manager */
        fusion_manager.reset(new ccl_fusion_manager());
    }

    executor.reset(new ccl_executor());
}

void global_data::init_resize_independent_objects() {
    parallelizer.reset(new ccl_parallelizer(env_object.worker_count));

    algorithm_selector.reset(new ccl_algorithm_selector_wrapper<CCL_COLL_LIST>());
    algorithm_selector->init();

    hwloc_wrapper.reset(new ccl_hwloc_wrapper());
}

void global_data::reset_resize_dependent_objects() {
    fusion_manager.reset();
    sched_cache.reset();
    buffer_cache.reset();
    dtypes.reset();
}

void global_data::reset_resize_independent_objects() {
    parallelizer.reset();
    algorithm_selector.reset();
    hwloc_wrapper.reset();
}

} // namespace ccl
