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
#include "sched/gpu_sched.hpp"
#include "sched/entry/l0/l0_allreduce_typed_entry.hpp"
//#include "sched/entry/l0/l0_allgather_handles_entry.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#include "common/comm/l0/device_community.hpp"
namespace native {

//template<ccl_coll_type entry_type>
struct device_group_scheduler {
    using schedule_ptr = std::unique_ptr<ccl_gpu_sched>;
    //create schedule
    /*
     static constexpr ccl_coll_type type() noexcept
    {
        return entry_type;
    }
  */
    template <class EntryType,
              ccl_sched_add_mode mode,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class device_t,
              class... Arguments>
    schedule_ptr submit_entry(device_community<class_id>& device_topology,
                              device_t& device,
                              native::ccl_driver_context_ptr ctx,
                              Arguments&&... args) {
        //create schedule
        size_t group_size =
            device_topology.template get_device_count<native::ccl_gpu_comm>() +
            device_topology.template get_device_count<native::ccl_virtual_gpu_comm>();

        //make entry
        if (!current_schedule) {
            current_schedule.reset(
                new ccl_gpu_sched(device_topology.get_device_storage(), group_size));
        }

        auto created_entry =
            entry_factory::make_ordered_entry<EntryType, mode>(current_schedule.get(),
                                                               device,
                                                               device_topology.get_device_storage(),
                                                               ctx,
                                                               std::forward<Arguments>(args)...);
        LOG_DEBUG("do initial progress");

        created_entry->start();
        current_schedule->set_fence(created_entry->get_fence()); //TODO temporary

        //active_group_sched->add_entry(std::move(created_entry));
        current_schedule->do_progress();

        LOG_DEBUG("Device group filled for: ", current_schedule->entries_count(), "/", group_size);
        if (current_schedule->entries_count() == group_size) {
            LOG_DEBUG("Device group finalized");
            schedule_ptr ret(new ccl_gpu_sched(device_topology.get_device_storage(), group_size));
            ret.swap(current_schedule);

            return ret;
        }
        //if sched is not ready - send NULL
        return std::unique_ptr<ccl_gpu_sched>();
    }

private:
    schedule_ptr current_schedule;
};

} // namespace native
