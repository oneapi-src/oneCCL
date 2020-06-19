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
#include <mutex>
#include "common/utils/spinlock.hpp"
#include "common/comm/l0/gpu_device_types.hpp"

namespace native
{
struct device_storage
{
    size_t get_storage_size() const;

    template<class device_t>
    size_t get_size() const
    {
        return ccl_tuple_get<device_container<device_t>>(gpu_device_storage).size();
    }

    std::shared_ptr<specific_plain_device_storage> create_devices_by_indices(size_t thread_id, const ccl::device_indices_t& indices);

    template<class device_t, class ...Args>
    device_t_ptr<device_t> create_gpu_device(ccl_device& device, size_t ranks, Args&&...args)
    {
        auto& gpus = ccl_tuple_get<device_container<device_t>>(gpu_device_storage);
        std::shared_ptr<device_t> gpu_instance = std::make_shared<device_t>(device,
                                                                            ranks,
                                                                            std::forward<Args>(args)...);
        gpus.emplace(std::piecewise_construct,
                     std::forward_as_tuple(device.handle),
                     std::forward_as_tuple(gpu_instance));

        {
            auto acc = get_node_storage();
            specific_indexed_device_storage& global_storage = acc.get();
            indexed_device_container<device_t>& device_cont = ccl_tuple_get<indexed_device_container<device_t>>(global_storage);
            device_cont.insert({ranks, gpu_instance});
        }
        return gpu_instance;
    }

    specific_device_storage gpu_device_storage;
    using thread_plain_device_map = std::map<size_t, std::shared_ptr<specific_plain_device_storage>>;
    thread_plain_device_map thread_gpu_comms;

    struct accessor
    {
        accessor(ccl_spinlock& mutex, specific_indexed_device_storage& storage)
         : lock(mutex),
           inner_data(storage)
        {
        }
        accessor(accessor&&src) = default;

        specific_indexed_device_storage& get()
        {
            return inner_data;
        }
    private:
        std::unique_lock<ccl_spinlock> lock;
        specific_indexed_device_storage& inner_data;
    };

    accessor get_node_storage()
    {
        return accessor(node_storage_mutex, node_rank_device_storage);
    }
private:
    ccl_spinlock node_storage_mutex;
    specific_indexed_device_storage node_rank_device_storage;
};

}
