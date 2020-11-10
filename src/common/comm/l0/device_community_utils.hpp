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

namespace native {
namespace detail {

/**
 *
 */
template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
struct rank_getter {
    rank_getter(const ccl::device_index_type& device_idx,
                std::multiset<ccl::device_index_type>& registered_ids);

    template <class device_t>
    void operator()(const native::indexed_device_container<device_t>& container);

    template <class device_t>
    void operator()(const native::plain_device_container<device_t>& container);

    int get_assigned_rank() const;
    int get_assigned_size() const;

private:
    ccl::device_index_type device_id;
    std::multiset<ccl::device_index_type>& registered_device_id;
    int rank = 0;
    int size = 0;
    bool find = false;
    size_t enumerator = 0;
};

/**
 *
 */
template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
rank_getter<group_id, class_id>::rank_getter(const ccl::device_index_type& device_idx,
                                             std::multiset<ccl::device_index_type>& registered_ids)
        : device_id(device_idx),
          registered_device_id(registered_ids) {}

template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
template <class device_t>
void rank_getter<group_id, class_id>::operator()(
    const native::indexed_device_container<device_t>& container) {
    if (find) {
        return;
    }

    for (const auto& dev : container) {
        ccl_device& device = dev.second->get_device();
        const ccl::device_index_type& find_id = device.get_device_path();
        if (find_id == device_id) {
            if (enumerator == registered_device_id.count(device_id)) {
                rank = dev.first; //dev.second->template get_comm_data<group_id>().rank;
                //size = dev.second->template get_comm_data<group_id, class_id>().size;
                find = true;

                registered_device_id.insert(device_id);
            }
            enumerator++;
        }

        if (find) {
            return;
        }
    }
}

template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
template <class device_t>
void rank_getter<group_id, class_id>::operator()(
    const native::plain_device_container<device_t>& container) {
    if (find) {
        return;
    }

    for (const auto& dev : container) {
        ccl_device& device = dev.second->get_device();
        ccl::device_index_type find_id = device.get_device_path();
        if (find_id == device_id) {
            if (enumerator == registered_device_id.count(device_id)) {
                rank = dev.second->template get_comm_data<group_id, class_id>().rank;
                //size = dev.second->template get_comm_data<group_id, class_id>().size;
                find = true;

                registered_device_id.insert(device_id);
            }
            enumerator++;
        }
        if (find) {
            return;
        }
    }
}

template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
int rank_getter<group_id, class_id>::get_assigned_rank() const {
    if (!find) {
        throw std::runtime_error(
            std::string(__FUNCTION__) +
            "rank_getter doesn't assign rank for device: " + ccl::to_string(device_id));
    }
    return rank;
}
} // namespace detail
} // namespace native
