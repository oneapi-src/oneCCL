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

#include "sched/entry/factory/entry_factory.h"

#include "sched/entry/send_entry.hpp"
#include "sched/entry/recv_entry.hpp"
#include "sched/entry/write_entry.hpp"
#include "sched/entry/reduce_local_entry.hpp"
#include "sched/entry/recv_reduce_entry.hpp"
#include "sched/entry/copy_entry.hpp"
#include "sched/entry/sync_entry.hpp"
#include "sched/entry/prologue_entry.hpp"
#include "sched/entry/epilogue_entry.hpp"
#include "sched/entry/sparse_allreduce_completion_entry.hpp"
#include "sched/entry/wait_value_entry.hpp"
#include "sched/entry/function_entry.hpp"
#include "sched/entry/probe_entry.hpp"
#include "sched/entry/register_entry.hpp"
#include "sched/entry/deregister_entry.hpp"
#include "sched/entry/subsched_entry.hpp"
#include "sched/entry/coll/coll_entry.hpp"
#include "sched/entry/coll/direct/allgatherv_entry.hpp"
#include "sched/entry/coll/direct/allreduce_entry.hpp"
#include "sched/entry/coll/direct/alltoall_entry.hpp"
#include "sched/entry/coll/direct/alltoallv_entry.hpp"
#include "sched/entry/coll/direct/barrier_entry.hpp"
#include "sched/entry/coll/direct/bcast_entry.hpp"
#include "sched/entry/coll/direct/reduce_entry.hpp"

#ifdef CCL_ENABLE_SYCL
#include "sched/entry/sycl_copy_device_to_host_entry.hpp"
#include "sched/entry/sycl_copy_host_to_device_entry.hpp"
#endif /* CCL_ENABLE_SYCL */

#include "sched/sched.hpp"

namespace entry_factory {
/* generic interface for entry creation */
template <class EntryType, class... Arguments>
EntryType* make_entry(ccl_sched* sched, Arguments&&... args) {
    LOG_DEBUG("creating ", EntryType::class_name(), " entry");
    EntryType* new_entry = detail::entry_creator<EntryType>::template create<
        ccl_sched_add_mode::ccl_sched_add_mode_last_value>(sched, std::forward<Arguments>(args)...);
    LOG_DEBUG("created: ", EntryType::class_name(), ", entry: ", new_entry, ", for sched: ", sched);
    return new_entry;
}

template <class EntryType, ccl_sched_add_mode mode, class... Arguments>
EntryType* make_ordered_entry(ccl_sched* sched, Arguments&&... args) {
    LOG_DEBUG("creating ", EntryType::class_name(), " entry, use mode: ", to_string(mode));
    return detail::entry_creator<EntryType>::template create<mode>(
        sched, std::forward<Arguments>(args)...);
}

/* Example for non-standard entry 'my_non_standard_entry' creation
    namespace detail
    {
        template <>
        class entry_creator<my_non_standard_entry>
        {
            public:
            static my_non_standard_entry* create(/ *** specific parameters for construction *** /)
            {
                auto &&new_entry = std::unique_ptr<my_non_standard_entry>(
                            new my_non_standard_entry(/ *** specific parameters for construction *** /));

                //Add custom contruction/registration logic, if needed

                return static_cast<my_non_standard_entry*>(sched->add_entry(std::move(new_entry)));
            }
        };
    }*/
} // namespace entry_factory
