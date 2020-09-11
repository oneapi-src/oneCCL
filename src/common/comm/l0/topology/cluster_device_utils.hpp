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
#include "common/comm/l0/topology/topology_construction_utils.hpp"
namespace native {
namespace details {
namespace cluster_utils {

inline global_sorted_plain_graphs extract_full_node_plain_graphs(
    std::ostream& out,
    const ccl::cluster_device_indices_t& cluster_indices,
    const std::string& hostname,
    const details::global_sorted_plain_graphs& cluster_graphs) {
    details::global_sorted_plain_graphs ret;

    out << "Find host: " << hostname << " in cluster size: " << cluster_indices.size() << std::endl;
    auto node_it = cluster_indices.find(hostname);
    if (node_it == cluster_indices.end()) {
        out << "Cannot find node with: " << hostname << std::endl;
        return ret;
    }

    //iterate over all allied processes on the same host
    const ccl::process_device_indices_t& processes = node_it->second;
    out << "Find processes count: " << processes.size() << " on node: " << hostname << std::endl;
    for (const auto& process_val : processes) {
        auto process_id = process_val.first;
        auto process_graph_list_it = cluster_graphs.find(process_id);
        if (process_graph_list_it == cluster_graphs.end()) {
            out << "There are cluster topology for process: " << process_id << std::endl;
            std::stringstream ss;
            ss << out.rdbuf();
            throw std::runtime_error(std::string(__FUNCTION__) + " - log:\n" + ss.str());
        }

        // remember allied process and it topology
        ret.insert({ process_id, process_graph_list_it->second });
    }

    return ret;
}
} // namespace cluster_utils
} // namespace details
} // namespace native
