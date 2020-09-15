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
#include <memory>
#include <tuple>
#include <vector>

#include "ccl_types.hpp"
#include "common/comm/l0/device_community.hpp"

namespace native {

template <ccl::device_topology_type class_id>
struct device_community_container {
    using element_type = std::shared_ptr<device_community<class_id>>;
    using container_type = element_type;

    container_type storage;

    element_type get_topology() {
        return storage;
    }
    void set_topology(element_type item) {
        storage = item;
    }

    template <ccl::device_group_split_type group_id>
    void register_device_by_id(const ccl::device_index_type& device_id,
                               ccl::context_comm_addr& registered_addr);
};

template <>
struct device_community_container<ccl::device_topology_type::ring> {
    using element_type = std::shared_ptr<device_community<ccl::device_topology_type::ring>>;

    using container_type = std::vector<element_type>;
    container_type closed_rings;
    container_type torn_apart_rings;

    const element_type get_topology(size_t ring_index = 0) const {
        return closed_rings.at(ring_index);
    }

    element_type get_topology(size_t ring_index) {
        return closed_rings.at(ring_index);
    }
    void set_topology(element_type item) {
        closed_rings.push_back(std::move(item));
    }

    const element_type get_additiona_topology(size_t ring_index) const {
        return torn_apart_rings.at(ring_index);
    }
    element_type get_additiona_topology(size_t ring_index) {
        return torn_apart_rings.at(ring_index);
    }

    void set_additiona_topology(element_type item) {
        torn_apart_rings.push_back(std::move(item));
    }

    template <ccl::device_group_split_type group_id>
    void register_device_by_id(const ccl::device_index_type& device_id,
                               ccl::context_comm_addr& registered_addr);
};

template <ccl::device_group_split_type group_id, ccl::device_topology_type... class_id>
class device_group_community_holder {
public:
    using device_topologies_t = std::tuple<device_community_container<class_id>...>;
    using self_t = device_group_community_holder<group_id, class_id...>;

    template <ccl::device_topology_type requested_id>
    const device_community_container<requested_id>& get_community() const;

    template <ccl::device_topology_type requested_id>
    device_community_container<requested_id>& get_community();

    template <ccl::device_topology_type requested_id>
    void register_device_by_id(const ccl::device_index_type& device_id,
                               ccl::context_comm_addr& registered_addr);

    std::string to_string() const;

private:
    device_topologies_t typed_communities;
};
} // namespace native
