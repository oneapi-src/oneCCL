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
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <set>
#include <unistd.h>
#include <limits.h>
#include <gnu/libc-version.h>

#include "oneapi/ccl.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"

#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"
#include "common/comm/l0/device_community_holder_impl.hpp"
#include "common/comm/l0/topology/ring/cluster_group_device_creator_impl.hpp"
#include "common/comm/l0/topology/topology_serializer.hpp"
#include "common/comm/l0/context/device_storage.hpp"
#include "common/comm/l0/scheduler/thread_group_scheduler.hpp"
#include "common/comm/l0/scheduler/allied_process_group_scheduler.hpp"

#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/l0/context/scaling_ctx/numa_ctx_impl.hpp"
#include "common/comm/l0/context/scaling_ctx/scale_up_ctx_impl.hpp"
#include "common/comm/l0/context/scaling_ctx/scale_out_ctx_impl.hpp"
namespace native {

process_group_context::process_group_context(std::shared_ptr<ccl::host_communicator> comm)
        : ccl_communicator(comm),
          thread_group_ctx(new thread_group_context),
          gpu_device_storage(new device_storage) {
    if (!ccl_communicator) {
        LOG_ERROR("Process context need non-empty communicator");
        throw std::runtime_error("Process context need non-empty communicator");
    }
    process_idx = ccl_communicator->rank();
    process_count = ccl_communicator->size();

    //Get current hostname
    char hostname[HOST_NAME_MAX];
    int ret = gethostname(hostname, HOST_NAME_MAX);
    if (ret == -1 && (errno == ENAMETOOLONG || errno == EINVAL)) {
        assert(std::string(gnu_get_libc_version()) == "2.2" && "Cannot gethostname");
        hostname[HOST_NAME_MAX - 1] = '\0';
        std::cerr << "Hostname truncated: " << hostname << std::endl;
    }
    my_host_name = hostname;
}

process_group_context::~process_group_context() {}

bool process_group_context::delegate_sync(const ccl::device_indices_type& thread_device_indices,
                                          ccl::context_comm_addr& comm_addr) {
    // set thread id sequencially
    //comm_addr.thread_idx = process_device_topology.size();

    // prepare device communities
    auto& ring_container = process_device_topology[comm_addr.thread_idx]
                               .get_community<ccl::device_topology_type::ring>();
    (void)ring_container;

    auto& a2a_container = process_device_topology[comm_addr.thread_idx]
                              .get_community<ccl::device_topology_type::a2a>();
    a2a_container.set_topology(
        std::make_shared<device_community<ccl::device_topology_type::a2a>>(comm_addr));

    // sync all threads at first - blocking operation
    return thread_group_ctx->sync_barrier(thread_device_indices, comm_addr, *gpu_device_storage);
}

bool process_group_context::sync_barrier(const ccl::device_mask_t& thread_device_mask,
                                         ccl::context_comm_addr& comm_addr) {
    return sync_barrier(ccl_device_driver::get_device_indices(thread_device_mask), comm_addr);
}

bool process_group_context::sync_barrier(const ccl::device_indices_type& thread_device_indices,
                                         ccl::context_comm_addr& comm_addr) {
    // sync all threads at first - blocking operation
    if (!delegate_sync(thread_device_indices, comm_addr)) {
        return false;
    }

    //barrie mutex is locked by MASTER thread
    const ccl::process_device_indices_type& thread_indices =
        thread_group_ctx->get_thread_group_device_indices();

    LOG_INFO("Process (",
             process_idx,
             "/",
             process_count,
             ") reached process group communicator barrier");

    ccl::device_indices_type process_aggregated_device_indices =
        std::accumulate(thread_indices.begin(),
                        thread_indices.end(),
                        ccl::device_indices_type(),
                        [](ccl::device_indices_type& partial_indices,
                           const typename ccl::process_device_indices_type::value_type& val) {
                            partial_indices.insert(val.second.begin(), val.second.end());
                            return partial_indices;
                        });
    build_cluster_affinity_table(process_aggregated_device_indices);

    //iterate over allied processes(on the same host)
    //find possible IPC device with P2P capability
    LOG_INFO("Process (", process_idx, "/", process_count, ") starts hardware topologies creation");

    cluster_group_device_creator ally_process_topology(
        process_idx, process_count, *this, *gpu_device_storage);

    {
        const ccl::process_device_indices_type& node_mask = get_node_afinity_indices(get_host_id());
        std::stringstream ss;
        detail::adjacency_matrix p2p_dependency_graph =
            ally_process_topology.build_p2p_capability_matrix(ss, node_mask);
        ss << "\nMatrix\n" << p2p_dependency_graph << std::endl;
        if (!ally_process_topology.build_all(ss,
                                             comm_addr,
                                             thread_group_ctx->get_thread_group_device_indices(),
                                             p2p_dependency_graph)) {
            LOG_ERROR(ss.str(), "\nCannot build ipc ring! Abort. Build Log:\n", ss.str());
            abort();
        }
        LOG_DEBUG("Build IPC ring succesfully. Log:\n", ss.str());
    }

    {
        //TODO Create A2A topology
        LOG_INFO("Process Context Topologies A2A TODO");
    }

    //create scheduler
    scheduler_impl.reset(new allied_process_group_scheduler(
        process_count, comm_addr.thread_count, ccl_communicator, *gpu_device_storage));

    std::stringstream out;
    dump_process_topologies(out);

    LOG_INFO("Thread (MASTER): ",
             comm_addr.thread_idx,
             " finalized process topology creation:\n",
             out.str());
    return true;
}

std::shared_ptr<thread_group_context> process_group_context::get_thread_context(size_t process_id) {
    (void)process_id;
    return thread_group_ctx;
}
/*
std::shared_ptr<process_group_context::ring_topology>& process_group_context::get_process_ring_topology(size_t process_id, size_t thread_id)
{
    (void)process_id;
    auto per_thread_top = process_ring_topology.find(thread_id);
    if(per_thread_top == process_ring_topology.end())
    {
        LOG_ERROR("No process topologies for ",thread_id, ".Empty topology");
        static std::shared_ptr<process_group_context::ring_topology> empty;
        return empty;
    }
    return per_thread_top->second;
}
*/

std::shared_ptr<ccl::host_communicator> process_group_context::get_communicator() {
    return ccl_communicator;
}

bool process_group_context::build_cluster_affinity_table(
    const ccl::device_indices_type& process_aggregated_device_indices) {
    LOG_INFO("Node: ", my_host_name, " start build affinity table for process idx: ", process_idx);

    //create cluster mask affinity
    //1) request hostname & device indices count
    size_t send_hostname_size = my_host_name.size();
    std::vector<size_t> receive_hostname_sizes(ccl_communicator->size());
    std::vector<size_t> recv_counts(ccl_communicator->size(), 1);

    size_t send_process_indices_count = process_aggregated_device_indices.size();
    std::vector<size_t> receive_process_indices_sizes(ccl_communicator->size());
    std::vector<size_t> recv_process_indices_counts(ccl_communicator->size(), 1);

    constexpr size_t hostname_indices_requests_count = 2;
    std::vector<ccl::event> requests;
    requests.reserve(hostname_indices_requests_count);
    {
        ccl::stream::impl_value_t empty_stream{};
        requests.push_back(ccl_communicator->allgatherv_impl(&send_hostname_size,
                                                             1,
                                                             receive_hostname_sizes.data(),
                                                             recv_counts,
                                                             empty_stream,
                                                             ccl::default_allgatherv_attr,
                                                             {}));
        LOG_TRACE("Request hostname sizes, process (",
                  ccl_communicator->rank(),
                  "/",
                  ccl_communicator->size(),
                  ") has own hostname: ",
                  my_host_name,
                  ", size: ",
                  send_hostname_size);

        requests.push_back(ccl_communicator->allgatherv_impl(&send_process_indices_count,
                                                             1,
                                                             receive_process_indices_sizes.data(),
                                                             recv_process_indices_counts,
                                                             empty_stream,
                                                             ccl::default_allgatherv_attr,
                                                             {}));
        LOG_TRACE("Request device indices sizes, process (",
                  ccl_communicator->rank(),
                  "/",
                  ccl_communicator->size(),
                  ") has own indices count: ",
                  send_process_indices_count);
    }

    //wait for completion
    for (auto& req : requests) {
        req.wait();
    }

    size_t total_hostname_size =
        std::accumulate(receive_hostname_sizes.begin(), receive_hostname_sizes.end(), 0);
    LOG_DEBUG("Memory required for hostnames size: ", total_hostname_size, " bytes");

    size_t total_device_indices_count = std::accumulate(
        receive_process_indices_sizes.begin(), receive_process_indices_sizes.end(), 0);
    LOG_DEBUG("Memory required for device indices size: ", total_device_indices_count, " count");

    //Serialize own devices path data
    auto serialized_indices = detail::serialize::device_path_serializer::serialize_indices(
        process_aggregated_device_indices);
    // TODO assert(serialized_indices.size() == receive_process_indices_sizes[process_idx] && "Indices unexpected count");

    decltype(serialized_indices) affinity_indices;
    std::vector<char> hostnames;
    auto indices_count_to_bytes_converter = [](size_t elements) -> size_t {
        return elements * detail::serialize::device_path_serializable::device_index_size();
    };

    try {
        requests.clear();
        hostnames.resize(total_hostname_size);

        ccl::stream::impl_value_t empty_stream{};
        requests.push_back(ccl_communicator->allgatherv_impl((int8_t*)my_host_name.data(),
                                                             send_hostname_size,
                                                             (int8_t*)hostnames.data(),
                                                             receive_hostname_sizes,
                                                             empty_stream,
                                                             ccl::default_allgatherv_attr,
                                                             {}));
        LOG_TRACE("Submit request for hostnames. Process (",
                  ccl_communicator->rank(),
                  "/",
                  ccl_communicator->size(),
                  ")"
                  " has own hostname: ",
                  my_host_name);

        //TODO Reorder requests!

        //need to convert to bytes to satisfy serialized data type
        affinity_indices.resize(indices_count_to_bytes_converter(total_device_indices_count));
        std::transform(receive_process_indices_sizes.begin(),
                       receive_process_indices_sizes.end(),
                       receive_process_indices_sizes.begin(),
                       indices_count_to_bytes_converter);
        requests.push_back(ccl_communicator->allgatherv_impl(
            reinterpret_cast<const int8_t*>(serialized_indices.data()),
            serialized_indices.size(),
            reinterpret_cast<int8_t*>(affinity_indices.data()),
            receive_process_indices_sizes,
            empty_stream,
            ccl::default_allgatherv_attr,
            {}));
        LOG_TRACE("Submit request for affinity masks. Process (",
                  ccl_communicator->rank(),
                  "/",
                  ccl_communicator->size(),
                  ")"
                  " has own mask size: ",
                  serialized_indices.size());
    }
    catch (std::exception& ex) {
        LOG_ERROR("Cannot submit requests: ", ex.what());
        LOG_INFO("Memory required for hostnames size: ", total_hostname_size, " bytes");
        LOG_INFO("Memory required for device indices size: ", total_device_indices_count, " count");
        abort();
    }

    //wait for completion
    for (auto& req : requests) {
        req.wait();
    }

    //parse hostnames
    size_t rank_index = 0;
    auto name_from_iterator = hostnames.begin();
    auto affinity_mask_from_iterator = affinity_indices.begin();
    for (auto rank_hostname_size = receive_hostname_sizes.begin();
         rank_hostname_size != receive_hostname_sizes.end();
         ++rank_hostname_size) {
        //check hostnames
        if ((size_t)std::distance(name_from_iterator, hostnames.end()) < *rank_hostname_size) {
            LOG_ERROR("Received hostnames data is too short: ",
                      hostnames.size(),
                      " expected: ",
                      std::distance(name_from_iterator, hostnames.end()) + *rank_hostname_size);
            abort();
        }

        //get hostaname
        std::string hostname(name_from_iterator, name_from_iterator + *rank_hostname_size);
        //shift hostname data
        std::advance(name_from_iterator, *rank_hostname_size);

        //check affinity
        if ((size_t)std::distance(affinity_mask_from_iterator, affinity_indices.end()) <
            receive_process_indices_sizes[rank_index]) {
            LOG_ERROR("Received affinity_masks data is too short: ",
                      affinity_indices.size(),
                      " expected at least: ",
                      receive_process_indices_sizes[rank_index]);
            abort();
        }

        //get affinity
        ccl::device_indices_type rank_indices = detail::serialize::device_path_deserializer::
            deserialize_indices<std::multiset, ccl::device_index_type>(
                affinity_mask_from_iterator,
                affinity_mask_from_iterator + receive_process_indices_sizes[rank_index]);
        std::advance(affinity_mask_from_iterator, receive_process_indices_sizes[rank_index]);

        {
            std::stringstream ss;
            for (const auto& path : rank_indices) {
                ss << path << ", ";
            }
            LOG_DEBUG(
                "Collected hostname: ", hostname, ", rank: ", rank_index, ", affinity: ", ss.str());
        }

        //fill global mask
        set_node_afinity_indices(hostname, rank_index, rank_indices);
        LOG_DEBUG("Global affinity mask nodes count: ", cluster_gpu_indices.size());
        //next
        rank_index++;
    }

    {
        std::stringstream ss;
        process_group_context::dump_cluster_affinity_indices(cluster_gpu_indices, ss);
        LOG_INFO("Cluster device affinity indices table: ", ss.str());
    }

    return true;
}

const ccl::host_id process_group_context::get_host_id() const {
    return my_host_name;
}

const ccl::cluster_aggregated_device_mask_t& process_group_context::get_afinity_mask() const {
    return global_mask;
}
const ccl::cluster_device_indices_type& process_group_context::get_affinity_indices() const {
    return cluster_gpu_indices;
}

const ccl::process_aggregated_device_mask_t& process_group_context::get_node_afinity_mask(
    const ccl::host_id& host) const {
    auto it = global_mask.find(host);
    if (it == global_mask.end()) {
        LOG_ERROR("Cannot get affinity mask for node: ", host);
        static const ccl::process_aggregated_device_mask_t empty;
        return empty;
    }
    return it->second;
}

const ccl::process_device_indices_type& process_group_context::get_node_afinity_indices(
    const ccl::host_id& host) const {
    auto it = cluster_gpu_indices.find(host);
    if (it == cluster_gpu_indices.end()) {
        LOG_ERROR("Cannot get affinity indices for node: ", host);
        static const ccl::process_device_indices_type empty;
        return empty;
    }
    return it->second;
}

void process_group_context::set_node_afinity_indices(const ccl::host_id& host,
                                                     int rank_id,
                                                     const ccl::device_indices_type& indices) {
    /*
    ccl::device_mask_t rank_mask = ccl_device_driver::get_device_mask(indices);
    auto& per_host_mask = global_mask[host];
    auto process_it = per_host_mask.find(rank_id);
    if(process_it != per_host_mask.end())
    {
        LOG_DEBUG("Current host rank received");
        CCL_ASSERT(process_it->first == process_idx, "Self consistency rank id check failed");
        CCL_ASSERT(process_it->second == rank_mask, "Self consistency mask check failed");
    }
    else
    {
        LOG_DEBUG("Hostname: ", host, ", updated rank: ", rank_id, ", affinity: ", rank_mask.to_string());
        per_host_mask[rank_id] = rank_mask;
    }
*/
    //TODO for indices
    auto& per_host_indices = cluster_gpu_indices[host];
    auto process_ind_it = per_host_indices.find(rank_id);
    if (process_ind_it != per_host_indices.end()) {
        LOG_DEBUG("Current host rank received");
        CCL_ASSERT(process_ind_it->first == process_idx, "Self consistency rank id check failed");
        CCL_ASSERT(process_ind_it->second == indices, "Self consistency indices check failed");
    }
    else {
        LOG_DEBUG(
            "Hostname: ", host, ", updated rank: ", rank_id, ", affinity size: ", indices.size());
        per_host_indices[rank_id] = indices;
    }
}

device_storage& process_group_context::get_device_storage() {
    CCL_ASSERT(gpu_device_storage, "Device storage must exist");
    return *gpu_device_storage;
}

/*
std::tuple<bool, std::string> process_group_context::check_device_mask_validity_across_allied_processes(ccl::process_aggregated_device_mask_t& allied_processes_mask)
{
    std::string descr;
    //temporary indices collection
    std::multiset<typename indices::value_type> expected_dupliated_indices;
    //fill duplicated devices indices across allied processes(on the same host)
    for(const auto& proc_mask : allied_processes_mask)
    {
        //user merge in c++17
        indices tmp = ccl_device_driver::get_device_indices(proc_mask.second);
        expected_dupliated_indices.insert(tmp.begin(), tmp.end());
    }
    //find duplicates
    indices duplicates;
    for(auto it = expected_dupliated_indices.begin(); it != expected_dupliated_indices.end(); ++it)
    {
        auto cnt = expected_dupliated_indices.count(*it);
        if(cnt != 1) //not unique device index across processes
        {
            duplicates.insert(*it);
        }
    }
    bool ret = true;
    if(!duplicates.empty())
    {
        ret = false;
        std::stringstream ss;
        ss << "Duplicated device ids: ";
        std::copy(duplicates.begin(), duplicates.end(), std::ostream_iterator<typename indices::value_type>(ss, ", "));
        descr = ss.str();
    }
    return { ret, descr };
}
*/

void process_group_context::dump_cluster_affinity_indices(
    const ccl::cluster_device_indices_type& indices,
    std::ostream& out) {
    out << "Cluster nodes: " << indices.size() << "\n";
    for (const auto& node_indices : indices) {
        dump_node_aggregated_indices(node_indices.first, node_indices.second, out);
        out << std::endl;
    }
}

void process_group_context::dump_node_aggregated_mask(
    const std::string& node_name,
    const ccl::process_aggregated_device_mask_t& mask,
    std::ostream& out) {
    out << "Node: " << node_name << ", processes: " << mask.size() << "\n";
    for (const auto& proc_mask : mask) {
        dump_process_mask(proc_mask.first, proc_mask.second, out);
        out << std::endl;
    }
}
void process_group_context::dump_node_aggregated_indices(
    const std::string& node_name,
    const ccl::process_device_indices_type& indices,
    std::ostream& out) {
    if (!node_name.empty()) {
        out << "Node: " << node_name << ", processes: " << indices.size() << "\n";
    }
    else {
        out << "Processes: " << indices.size() << "\n";
    }

    for (const auto& proc_idxs : indices) {
        dump_process_indices(proc_idxs.first, proc_idxs.second, out);
        out << std::endl;
    }
}

void process_group_context::dump_process_mask(size_t process_id,
                                              const ccl::device_mask_t& mask,
                                              std::ostream& out) {
    out << "Process idx: " << process_id << ", affinity: " << mask.to_string();
}

void process_group_context::dump_process_indices(size_t process_id,
                                                 const ccl::device_indices_type& indices,
                                                 std::ostream& out) {
    out << "Process idx: " << process_id << ", affinity: ";
    for (const auto& path : indices) {
        out << path << ", ";
    }
}

std::string process_group_context::to_string() const {
    auto my_processes_it = global_mask.find(my_host_name);
    CCL_ASSERT(my_processes_it == global_mask.end(), "global mask is inconsistend!");

    std::stringstream out;
    out << "My info:\nHost: " << my_host_name << ", processes: " << my_processes_it->second.size();
    process_group_context::dump_cluster_affinity_mask(global_mask, out);
    return out.str();
}

void process_group_context::dump_cluster_affinity_mask(
    const ccl::cluster_aggregated_device_mask_t& mask,
    std::ostream& out) {
    out << "Cluster nodes: " << mask.size() << "\n";
    for (const auto& node_mask : mask) {
        dump_node_aggregated_mask(node_mask.first, node_mask.second, out);
        out << std::endl;
    }
}

void process_group_context::dump_process_topologies(std::ostream& out) const {
    out << "Process threads count: " << process_device_topology.size() << std::endl;
    for (auto it = process_device_topology.begin(); it != process_device_topology.end(); ++it) {
        const auto& top = it->second;
        size_t thread = it->first;

        out << "\nProcess Thread Group: " << thread << " topology:\n" << top.to_string();
    }
}

std::vector<ccl::device_indices_type> process_group_context::get_ipc_device_indices() const {
    std::stringstream ss;
    ccl::process_device_indices_type node_mask_to_reorder = get_node_afinity_indices(get_host_id());
    if (node_mask_to_reorder.empty()) {
        ss << "process_group_context::get_ipc_device_indices failed: empty process affinities for hostname: "
           << get_host_id() << ", cluster topology:\n";
        process_group_context::dump_cluster_affinity_indices(cluster_gpu_indices, ss);
        const std::string& err = ss.str();
        LOG_ERROR("Error in ", err);
        throw std::runtime_error(err);
    }

    std::vector<ccl::device_indices_type> ipc_device_indices;
    try {
        ipc_device_indices =
            process_group_context::get_ipc_device_indices_for_id(process_idx, node_mask_to_reorder);
    }
    catch (const std::exception& ex) {
        ss << ex.what() << ", cluster topology:\n";
        process_group_context::dump_cluster_affinity_indices(cluster_gpu_indices, ss);
        const std::string& err = ss.str();
        LOG_ERROR("Error in ", err);
        throw;
    }
    return ipc_device_indices;
}

std::vector<ccl::device_indices_type> process_group_context::get_ipc_device_indices_for_id(
    size_t process_idx,
    ccl::process_device_indices_type node_indices) {
    std::stringstream ss;
    auto my_process_it = node_indices.find(process_idx);
    if (my_process_it == node_indices.end()) {
        ss << "No process id: " << process_idx << " in node affinities: ";
        process_group_context::dump_node_aggregated_indices("", node_indices, ss);
        const std::string& err = ss.str();
        LOG_ERROR(err);
        throw std::runtime_error(err);
    }

    node_indices.erase(my_process_it); //self indices erase, other are ipc

    std::vector<ccl::device_indices_type> ipc_device_indices;
    for (const auto& mask : node_indices) {
        ipc_device_indices.push_back(mask.second);
    }
    return ipc_device_indices;
}

void process_group_context::collect_cluster_colored_plain_graphs(
    const detail::colored_plain_graph_list& send_graph,
    detail::global_sorted_colored_plain_graphs& received_graphs) {
    using namespace detail::serialize;

    LOG_DEBUG("Collect cluster colored plain graphs, my process index: ",
              process_idx,
              ", graphs count: ",
              send_graph.size());

    // serialize current process graph list into bytes
    device_path_serializable::raw_data_t my_serialized_graph =
        device_path_serializer::serialize_indices(send_graph);

    size_t send_count = my_serialized_graph.size();
    std::vector<size_t> recv_counts_process_graph_sizes(ccl_communicator->size());
    {
        // collect graph lists size from cluster
        std::vector<size_t> recv_counts(ccl_communicator->size(), 1);

        LOG_DEBUG("Send graph lists size by process index: ",
                  process_idx,
                  ", serialized size: ",
                  send_count);
        ccl::stream::impl_value_t empty_stream{};
        ccl_communicator
            ->allgatherv_impl(&send_count,
                              1,
                              recv_counts_process_graph_sizes.data(),
                              recv_counts,
                              empty_stream,
                              ccl::default_allgatherv_attr,
                              {})
            .wait();
    }

    size_t global_graph_data_size = std::accumulate(
        recv_counts_process_graph_sizes.begin(), recv_counts_process_graph_sizes.end(), 0);

    // collect cluster graph lists
    device_path_serializable::raw_data_t recv_cluster_graphs;
    try {
        LOG_DEBUG(
            "Send graph list by process index: ", process_idx, ", serialized size: ", send_count);

        recv_cluster_graphs.resize(global_graph_data_size);
        ccl::stream::impl_value_t empty_stream{};
        ccl_communicator
            ->allgatherv_impl(reinterpret_cast<int8_t*>(my_serialized_graph.data()),
                              send_count,
                              reinterpret_cast<int8_t*>(recv_cluster_graphs.data()),
                              recv_counts_process_graph_sizes,
                              empty_stream,
                              ccl::default_allgatherv_attr,
                              {})
            .wait();
    }
    catch (const std::bad_alloc& ex) {
        CCL_THROW_WITH_ERROR("Memory required for global_graph_data_size size: ",
                             global_graph_data_size,
                             " bytes\nException: ",
                             ex.what());
    }
    catch (const std::exception& ex) {
        CCL_THROW_WITH_ERROR("Cannot submit global-serialized-graph requests: ", ex.what());
    }

    size_t deserialized_bytes = 0;
    size_t offset_bytes = 0;
    size_t process_num = 0;

    LOG_DEBUG("Deserialize recv_cluster_graphs");
    try {
        for (process_num = 0; process_num < ccl_communicator->size(); process_num++) {
            detail::colored_plain_graph_list graph =
                device_path_deserializer::deserialize_colored_graph_list_indices(
                    recv_cluster_graphs, deserialized_bytes, offset_bytes);
            LOG_DEBUG("Process index: ",
                      process_num,
                      ", deserialized bytes: ",
                      deserialized_bytes,
                      ", by offset: ",
                      offset_bytes);

            received_graphs.emplace(process_num, std::move(graph));
        }
    }
    catch (const std::bad_alloc& ex) {
        CCL_THROW_WITH_ERROR("Cannot deserialize recv_cluster_graphs for process num:",
                             process_num,
                             ", deserialized raw bytes: ",
                             deserialized_bytes,
                             ", processed raw bytes: ",
                             offset_bytes,
                             " \nException: ",
                             ex.what());
    }
    catch (const std::exception& ex) {
        CCL_THROW_WITH_ERROR("Cannot deserialize recv_cluster_graphs for process num:",
                             process_num,
                             ", deserialized raw bytes: ",
                             deserialized_bytes,
                             ", processed raw bytes: ",
                             offset_bytes,
                             " \nException: ",
                             ex.what());
    }

    LOG_DEBUG("Global colored_graph deserialized on process id: ", process_idx);
}

process_group_context::numa_context_base& process_group_context::get_numa_ctx() {
    return *this;
}
const process_group_context::numa_context_base& process_group_context::get_numa_ctx() const {
    return *this;
}
process_group_context::scaleup_context_base& process_group_context::get_scaleup_ctx() {
    return *this;
}
const process_group_context::scaleup_context_base& process_group_context::get_scaleup_ctx() const {
    return *this;
}
process_group_context::scaleout_context_base& process_group_context::get_scaleout_ctx() {
    return *this;
}
const process_group_context::scaleout_context_base& process_group_context::get_scaleout_ctx()
    const {
    return *this;
}
} // namespace native
