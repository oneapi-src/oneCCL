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
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "topology/topo_manager.hpp"

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include <CL/sycl/backend/level_zero.hpp>
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

topo_rank_info::topo_rank_info() : rank(ccl_comm::invalid_rank) {
    memset(uuid, 0, topo_uuid_len);
}

constexpr int topo_manager::invalid_color;
constexpr int topo_manager::max_domain_count;

void topo_manager::check_ppn(const std::set<std::string>& unique_hostnames,
                             const std::vector<std::string>& all_hostnames) {
    std::vector<int> ppn_set;
    for (auto host_name : unique_hostnames) {
        ppn_set.push_back(std::count(all_hostnames.begin(), all_hostnames.end(), host_name));
    }

    is_same_ppn = std::all_of(ppn_set.begin(), ppn_set.end(), [&ppn_set](int p) {
        return (p == ppn_set.front());
    });
}

void topo_manager::base_init(std::shared_ptr<atl_base_comm> atl_comm,
                             std::shared_ptr<ccl::device> device_ptr,
                             std::shared_ptr<ccl::context> context_ptr) {
    comm = atl_comm;

    int comm_rank = comm->get_rank();
    int comm_size = comm->get_size();

    intra_card_colors.resize(comm_size, topo_manager::invalid_color);
    inter_card_colors.resize(comm_size, topo_manager::invalid_color);
    uuids.resize(comm_size);
    rank_info_vec.resize(comm_size);
    p2p_matrix.resize(comm_size);
    for (size_t i = 0; i < p2p_matrix.size(); i++) {
        p2p_matrix[i].resize(comm_size, true);
    }

    std::vector<char> all_hostnames_raw(max_hostname_len * comm_size);
    char my_hostname[max_hostname_len] = { 0 };
    gethostname(my_hostname, max_hostname_len - 1);
    LOG_DEBUG("rank: ", comm_rank, ", size: ", comm_size, ", hostname: ", my_hostname);

    allgather(my_hostname, all_hostnames_raw.data(), max_hostname_len);

    std::set<std::string> unique_hostnames;
    std::vector<std::string> all_hostnames(comm_size);
    std::string str{};
    for (int idx = 0; idx < comm_size; idx++) {
        str = std::string(all_hostnames_raw.data() + (idx * max_hostname_len), max_hostname_len);
        unique_hostnames.insert(str);
        all_hostnames[idx] = str;
    }

    CCL_THROW_IF_NOT(!unique_hostnames.empty(), "empty unique_hostnames");

    check_ppn(unique_hostnames, all_hostnames);

    int local_idx = 0;
    for (auto host_name : unique_hostnames) {
        if (strcmp(my_hostname, host_name.c_str()) == 0) {
            host_idx = local_idx;
            break;
        }
        else {
            local_idx++;
        }
    }

    CCL_THROW_IF_NOT(host_idx != topo_manager::invalid_host_idx,
                     "invalid host index for topology building, hostname: ",
                     my_hostname);

    is_single_node = (unique_hostnames.size() == 1) ? true : false;
    is_single_card = is_single_node && (comm_size <= max_ranks_per_card);

    // exchange common rank info
    topo_rank_info rank_info{};
    rank_info.rank = comm_rank;
    rank_info.host_idx = host_idx;
    rank_info.local_proc_idx = ccl::global_data::get().executor->get_local_proc_idx();
    std::string rank_uuid = topo_manager::generate_uuid();
    std::copy(rank_uuid.begin(), rank_uuid.end(), rank_info.uuid);

    allgather(&rank_info, rank_info_vec.data(), sizeof(rank_info));

    for (size_t idx = 0; idx < rank_info_vec.size(); idx++) {
        uuids[idx] = std::string(rank_info_vec[idx].uuid);
    }

    if (ccl::global_data::env().topo_color == topo_color_mode::fixed) {
        for (int h_idx = 0; h_idx < (int)unique_hostnames.size(); h_idx++) {
            fill_fixed_colors(get_filtered_rank_info(rank_info_vec, h_idx));
        }
    }
    else if (ccl::global_data::env().topo_color == topo_color_mode::env) {
        domains = parse_topo_env();

        CCL_THROW_IF_NOT(!domains.empty(), "domains container is empty");
        LOG_INFO("domains: ", to_string(domains));

        for (int h_idx = 0; h_idx < (int)unique_hostnames.size(); h_idx++) {
            fill_env_colors(get_filtered_rank_info(rank_info_vec, h_idx));
        }
    }

    if (ccl::global_data::env().topo_color != topo_color_mode::ze)
        return;

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    if (!(device_ptr && context_ptr))
        return;

    // exchange ze specific rank info
    topo_ze_rank_info ze_rank_info{};
    std::vector<topo_ze_rank_info> ze_rank_info_vec(comm_size);

    device = sycl::get_native<ccl::utils::get_level_zero_backend()>(device_ptr.get()->get_native());
    zes_device_handle_t zes_device = (zes_device_handle_t)device;

    ze_device_properties_t dev_props = ccl::ze::default_device_props;
    ZE_CALL(zeDeviceGetProperties, (device, &dev_props));

    ze_rank_info.device_uuid = dev_props.uuid;

    zes_pci_properties_t pci_props = {};
    ZE_CALL(zesDevicePciGetProperties, (zes_device, &pci_props));
    ze_rank_info.pci_addr = pci_props.address;

    allgather(&ze_rank_info, ze_rank_info_vec.data(), sizeof(ze_rank_info));

    LOG_INFO("hostname: ",
             my_hostname,
             ", rank: ",
             comm_rank,
             ", size: ",
             comm_size,
             ", host_idx: ",
             host_idx,
             ", uuid: ",
             rank_info.uuid,
             ", device uuid: ",
             ccl::ze::to_string(ze_rank_info.device_uuid));

    // create intra card colors

    for (int h_idx = 0; h_idx < (int)unique_hostnames.size(); h_idx++) {
        fill_ze_intra_colors(get_filtered_rank_info(rank_info_vec, h_idx), ze_rank_info_vec);
    }

    // create inter card colors
    uint32_t subdev_count{};
    ZE_CALL(zeDeviceGetSubDevices, (device, &subdev_count, nullptr));
    LOG_INFO("(",
             my_hostname,
             ") rank ",
             comm_rank,
             " has ",
             subdev_count,
             " subdevices, subdeviceId=",
             ((dev_props.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)
                  ? std::to_string(dev_props.subdeviceId)
                  : "na"));

    uint32_t port_count{};
    ZE_CALL(zesDeviceEnumFabricPorts, (zes_device, &port_count, NULL));
    std::vector<zes_fabric_port_handle_t> ports(port_count);
    ZE_CALL(zesDeviceEnumFabricPorts, (zes_device, &port_count, ports.data()));

    // TODO: to be deleted, will rely on real port count
    char* all_ports_env = getenv("CCL_TOPO_ALL_PORTS");

    std::vector<topo_ze_port_info> my_ports;
    for (size_t i = 0; i < ports.size(); i++) {
        zes_fabric_port_properties_t prop;
        ZE_CALL(zesFabricPortGetProperties, (ports[i], &prop));

        zes_fabric_port_config_t config;
        ZE_CALL(zesFabricPortGetConfig, (ports[i], &config));

        if (!config.enabled) {
            LOG_INFO("(",
                     my_hostname,
                     ") rank ",
                     comm_rank,
                     " has disabled port ",
                     ccl::ze::to_string(prop.portId),
                     " skip it");
            continue;
        }

        zes_fabric_port_state_t state;
        ZE_CALL(zesFabricPortGetState, (ports[i], &state));

        if (state.status == ZES_FABRIC_PORT_STATUS_HEALTHY ||
            state.status == ZES_FABRIC_PORT_STATUS_DEGRADED) {
            // port has good status meaning it's connected

            if (all_ports_env && atoi(all_ports_env)) {
                my_ports.push_back({ prop.portId, state.remotePortId });
            }
            else {
                if (prop.onSubdevice && prop.subdeviceId == dev_props.subdeviceId)
                    my_ports.push_back({ prop.portId, state.remotePortId });
            }

            if (state.status == ZES_FABRIC_PORT_STATUS_DEGRADED) {
                LOG_INFO("(",
                         my_hostname,
                         ") rank ",
                         comm_rank,
                         " has a degraded port ",
                         ccl::ze::to_string(prop.portId));
            }
        }
    }

    // exchange port counts
    int my_port_count = (int)my_ports.size();
    std::vector<int> all_port_counts(comm_size);

    allgather(&my_port_count, all_port_counts.data(), sizeof(my_port_count));

    // exchange port info
    std::vector<int> recv_bytes(comm_size, sizeof(topo_ze_port_info) * all_port_counts[0]);
    std::vector<int> port_count_offsets(comm_size, 0);

    for (int i = 1; i < comm_size; i++) {
        recv_bytes[i] = sizeof(topo_ze_port_info) * all_port_counts[i];
        port_count_offsets[i] = port_count_offsets[i - 1] + all_port_counts[i - 1];
    }

    std::vector<topo_ze_port_info> all_ports(
        std::accumulate(all_port_counts.begin(), all_port_counts.end(), 0));

    allgatherv(my_ports.data(), all_ports.data(), recv_bytes);

    if (!comm_rank) {
        for (int rank_idx = 0; rank_idx < comm_size; rank_idx++) {
            if (all_port_counts[rank_idx])
                LOG_INFO("--- rank ", rank_info_vec[rank_idx].rank, " ---");
            for (int port_idx = port_count_offsets[rank_idx];
                 port_idx < port_count_offsets[rank_idx] + all_port_counts[rank_idx];
                 port_idx++) {
                LOG_INFO("port local ",
                         ccl::ze::to_string(all_ports[port_idx].local),
                         ", remote ",
                         ccl::ze::to_string(all_ports[port_idx].remote));
            }
        }
    }

    std::vector<std::set<int>> planes;
    for (int rank = 0; rank < comm_size; rank++) {
        int cur_plane_idx = topo_manager::invalid_plane_idx;
        for (size_t plane_idx = 0; plane_idx < planes.size(); plane_idx++) {
            if (planes[plane_idx].find(rank) != planes[plane_idx].end()) {
                cur_plane_idx = plane_idx;
                break;
            }
        }

        if (cur_plane_idx == topo_manager::invalid_plane_idx) {
            planes.push_back({ rank });
            cur_plane_idx = planes.size() - 1;
        }

        for (int my_port_idx = port_count_offsets[rank];
             my_port_idx < port_count_offsets[rank] + all_port_counts[rank];
             my_port_idx++) {
            auto remote_port = all_ports[my_port_idx].remote;
            int peer_port_idx = 0;
            while (peer_port_idx < (int)all_ports.size()) {
                if (peer_port_idx == my_port_idx) {
                    peer_port_idx += all_port_counts[rank]; // skip my ports
                }
                else {
                    auto peer_port = all_ports[peer_port_idx].local;
                    if (peer_port.fabricId == remote_port.fabricId &&
                        peer_port.attachId == remote_port.attachId &&
                        (int)peer_port.portNumber == (int)remote_port.portNumber) {
                        int peer_idx = ccl_comm::invalid_rank;
                        for (int peer_rank = 0; peer_rank < comm_size; peer_rank++) {
                            if (peer_port_idx >= port_count_offsets[peer_rank]) {
                                if (peer_rank == comm_size - 1 ||
                                    peer_port_idx < port_count_offsets[peer_rank + 1]) {
                                    peer_idx = peer_rank;
                                    break;
                                }
                            }
                        }
                        CCL_THROW_IF_NOT(peer_idx != ccl_comm::invalid_rank && peer_idx >= 0 &&
                                             peer_idx < comm_size,
                                         "peer_idx=",
                                         peer_idx,
                                         " is not valid");

                        planes[cur_plane_idx].insert(rank_info_vec[peer_idx].rank);
                        break;
                    }
                    peer_port_idx++;
                }
            }
        }
    }

    bool all_single_planes = std::all_of(planes.begin(), planes.end(), [](std::set<int>& p) {
        return (p.size() == 1);
    });

    bool all_empty_ports = std::all_of(all_port_counts.begin(), all_port_counts.end(), [](int c) {
        return (c == 0);
    });

    if (all_single_planes && all_empty_ports) {
        // use simple separation of ranks between planes
        // ranks with the same intra_card color
        // should be placed to different planes
        for (int h_idx = 0; h_idx < (int)unique_hostnames.size(); h_idx++) {
            fill_ze_inter_colors(get_filtered_rank_info(rank_info_vec, h_idx));
        }
    }
    else {
        fill_ze_inter_colors(planes);
    }

    auto devices = global_data::get().ze_data->device_handles;
    p2p_matrix = build_p2p_matrix(get_filtered_devices(devices, ze_rank_info_vec));
    LOG_DEBUG("p2p matrix: \n", to_string(p2p_matrix), "\nnumber of devices: ", devices.size());

    // update is_single_card
    bool no_invalid_colors = (std::find(intra_card_colors.begin(),
                                        intra_card_colors.end(),
                                        topo_manager::invalid_color) == intra_card_colors.end())
                                 ? true
                                 : false;
    bool all_same_colors =
        std::all_of(intra_card_colors.begin(), intra_card_colors.end(), [this](int c) {
            return (c == intra_card_colors.front());
        });
    is_single_card = (is_single_node && (comm->get_size() <= topo_manager::max_ranks_per_card) &&
                      no_invalid_colors && all_same_colors);

    if (!comm_rank) {
        for (int plane_idx = 0; plane_idx < (int)planes.size(); plane_idx++) {
            std::stringstream ss;
            std::copy(planes[plane_idx].begin(),
                      planes[plane_idx].end(),
                      std::ostream_iterator<int>(ss, " "));
            LOG_INFO("plane ", plane_idx, " contains ranks: ", ss.str());
        }
    }

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
}

void topo_manager::check_colors() {
    for (size_t domain_idx = 0; domain_idx < topo_manager::max_domain_count; domain_idx++) {
        CCL_THROW_IF_NOT(domain_idx == topo_manager::card_domain_idx ||
                             domain_idx == topo_manager::plane_domain_idx,
                         "unknown domain_idx detected");
        int max_ranks_per_subdomain = (domain_idx == topo_manager::card_domain_idx)
                                          ? topo_manager::max_ranks_per_card
                                          : topo_manager::max_ranks_per_plane;
        const std::vector<int>& colors =
            (domain_idx == topo_manager::card_domain_idx) ? intra_card_colors : inter_card_colors;
        std::string color_name = (domain_idx == topo_manager::card_domain_idx)
                                     ? topo_manager::card_domain_name
                                     : topo_manager::plane_domain_name;

        CCL_THROW_IF_NOT(
            std::find(colors.begin(), colors.end(), topo_manager::invalid_color) == colors.end(),
            color_name,
            " domain colors data is not valid");

        std::set<int> unique_colors(colors.begin(), colors.end());

        std::vector<int> counts;
        for (auto unique_color : unique_colors) {
            counts.push_back(std::count(colors.begin(), colors.end(), unique_color));
        }

        is_same_domains &=
            std::all_of(counts.begin(), counts.end(), [&counts, &max_ranks_per_subdomain](int c) {
                return (counts.front() == c && c <= max_ranks_per_subdomain);
            });
    }
}

void topo_manager::post_init() {
    for (int rank = 0; rank < comm->get_size(); rank++) {
        intra_card_colors[rank] += rank_info_vec[rank].host_idx * max_ranks_per_host;
        inter_card_colors[rank] += rank_info_vec[rank].host_idx * max_ranks_per_host;
    }

    check_colors();
}

void topo_manager::init(std::shared_ptr<atl_base_comm> atl_comm,
                        std::shared_ptr<ccl::device> device_ptr,
                        std::shared_ptr<ccl::context> context_ptr) {
    base_init(atl_comm, device_ptr, context_ptr);
    if (device_ptr) {
        post_init();
    }
}

int topo_manager::get_host_idx() const {
    return host_idx;
}

int topo_manager::get_intra_card_color(int rank) const {
    return intra_card_colors[rank];
}

int topo_manager::get_inter_card_color(int rank) const {
    return inter_card_colors[rank];
}

std::string topo_manager::get_uuid(int rank) const {
    return uuids[rank];
}

bool topo_manager::has_p2p_access() const {
    bool has_access = true;
    for (size_t i = 0; i < p2p_matrix.size(); i++) {
        for (size_t j = 0; j < p2p_matrix[i].size(); j++) {
            if (!p2p_matrix[i][j]) {
                has_access = false;
                break;
            }
        }
    }
    // LOG_DEBUG("p2p access status: ", has_access);
    return has_access;
}

bool topo_manager::has_same_ppn() const {
    return is_same_ppn;
}

bool topo_manager::has_same_domains() const {
    return is_same_domains;
}

void topo_manager::allgather(const void* send_buf, void* recv_buf, int bytes) {
    std::vector<int> recv_bytes(comm->get_size(), bytes);
    allgatherv(send_buf, recv_buf, recv_bytes);
}

void topo_manager::allgatherv(const void* send_buf,
                              void* recv_buf,
                              const std::vector<int>& recv_bytes) {
    atl_req_t req{};

    int comm_rank = comm->get_rank();
    int comm_size = comm->get_size();

    CCL_THROW_IF_NOT((int)recv_bytes.size() == comm->get_size(),
                     "unexpected recv_bytes size ",
                     recv_bytes.size(),
                     ", comm_size ",
                     comm_size);

    std::vector<int> offsets(comm_size, 0);
    for (int i = 1; i < comm_size; i++) {
        offsets[i] = offsets[i - 1] + recv_bytes[i - 1];
    }

    comm->allgatherv(0 /* ep_idx */,
                     send_buf,
                     recv_bytes[comm_rank],
                     recv_buf,
                     recv_bytes.data(),
                     offsets.data(),
                     req);
    comm->wait(0 /* ep_idx */, req);
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
bool topo_manager::is_same_pci_addr(zes_pci_address_t addr1, zes_pci_address_t addr2) {
    bool result = true;
    if (!(addr1.domain == addr2.domain && addr1.bus == addr2.bus && addr1.device == addr2.device &&
          addr1.function == addr2.function)) {
        result = false;
        LOG_DEBUG("pci addresses are not the same:"
                  " addr1: ",
                  ccl::ze::to_string(addr1),
                  " addr2: ",
                  ccl::ze::to_string(addr2));
    }

    return result;
}

bool topo_manager::is_same_dev_uuid(ze_device_uuid_t uuid1, ze_device_uuid_t uuid2) {
    bool result = true;
    std::string state = "device uuids";
    if (std::memcmp(&uuid1, &uuid2, sizeof(ze_device_uuid_t))) {
        result = false;
        state += " are not the same:";
    }
    else {
        state += " are the same:";
    }
    // LOG_DEBUG(state, " uuid1: ", ccl::ze::to_string(uuid1), ", uuid2: ", ccl::ze::to_string(uuid2));
    return result;
}

inline std::vector<ze_device_uuid_t> topo_manager::copy_dev_uuids(
    const std::vector<topo_rank_info>& info_vec,
    const std::vector<topo_ze_rank_info>& ze_rank_info_vec) {
    std::vector<ze_device_uuid_t> result;
    for (auto const& info : info_vec) {
        result.push_back(ze_rank_info_vec[info.rank].device_uuid);
    }
    return result;
}

bool topo_manager::is_sub_vector(const std::vector<ze_device_uuid_t>& vec,
                                 const std::vector<ze_device_uuid_t>& sub_vec) {
    std::vector<ze_device_uuid_t> unique_sub_vec;
    for (auto const& sub_elem : sub_vec) {
        if (std::find_if(unique_sub_vec.begin(),
                         unique_sub_vec.end(),
                         [&sub_elem](const ze_device_uuid_t& elem) {
                             return is_same_dev_uuid(sub_elem, elem);
                         }) == unique_sub_vec.end()) {
            unique_sub_vec.push_back(sub_elem);
        }
    }

    if (unique_sub_vec.size() > vec.size()) {
        LOG_DEBUG("unique sub vector size can not be greater than base vector size: unique: ",
                  unique_sub_vec.size(),
                  ", vec: ",
                  vec.size());
        return false;
    }

    for (auto const& sub_elem : sub_vec) {
        if (std::find_if(vec.begin(), vec.end(), [&sub_elem](const ze_device_uuid_t& elem) {
                return is_same_dev_uuid(sub_elem, elem);
            }) == vec.end()) {
            return false;
        }
    }
    return true;
}

std::vector<ze_device_handle_t> topo_manager::get_filtered_devices(
    const std::vector<ze_device_handle_t>& devices,
    const std::vector<topo_ze_rank_info>& ze_rank_info_vec) {
    const auto& local_info_vec = get_filtered_rank_info(rank_info_vec, host_idx);
    // get unique device uuids per host
    const auto& comm_dev_uuids = copy_dev_uuids(local_info_vec, ze_rank_info_vec);

    // check if unique device uuids are in global dev uuids
    CCL_THROW_IF_NOT(is_sub_vector(copy_dev_uuids(rank_info_vec, ze_rank_info_vec), comm_dev_uuids),
                     "comm_dev_uuids should be sub vector of global device uuids");

    // TODO: re-implement when the struct of handel device & dev_uuid will be created
    // check if unique device uuids are in node_dev_uuids
    std::vector<ze_device_uuid_t> node_dev_uuids;
    for (auto device : devices) {
        ze_device_properties_t dev_props;
        zeDeviceGetProperties(device, &dev_props);
        node_dev_uuids.push_back(dev_props.uuid);
    }

    CCL_THROW_IF_NOT(is_sub_vector(node_dev_uuids, comm_dev_uuids),
                     "comm_dev_uuids should be sub vector of node_dev_uuids");

    // get device handles which uuids from comm_dev_uuids
    std::vector<ze_device_handle_t> result;
    for (auto comm_dev_uuid : comm_dev_uuids) {
        for (auto device : devices) {
            ze_device_properties_t dev_props;
            zeDeviceGetProperties(device, &dev_props);
            if (is_same_dev_uuid(dev_props.uuid, comm_dev_uuid)) {
                result.push_back(device);
                break;
            }
        }
    }

    CCL_THROW_IF_NOT(result.size() == local_info_vec.size(),
                     "unexpected count of filtered devices: ",
                     result.size(),
                     ", expected: ",
                     local_info_vec.size());
    return result;
}

std::vector<std::vector<bool>> topo_manager::build_p2p_matrix(
    const std::vector<ze_device_handle_t>& devices) {
    size_t device_count = devices.size();
    std::vector<std::vector<bool>> matrix(device_count);

    for (uint32_t i = 0; i < device_count; i++) {
        matrix[i].resize(device_count);

        for (uint32_t j = 0; j < device_count; j++) {
            if (i == j) {
                matrix[i][j] = true;
            }
            else {
                ze_bool_t val;
                ZE_CALL(zeDeviceCanAccessPeer, (devices[i], devices[j], &val));
                matrix[i][j] = static_cast<bool>(val);
            }
        }
    }
    return matrix;
}

std::string topo_manager::to_string(const std::vector<std::vector<bool>>& matrix) {
    uint32_t device_count = matrix.size();
    std::stringstream ss;
    for (uint32_t j = 0; j < device_count; j++) {
        if (j > 9) {
            ss << "  " << j;
        }
        else {
            ss << "   " << j;
        }
    }
    ss << "\n";

    for (uint32_t i = 0; i < device_count; i++) {
        if (i > 9) {
            ss << i;
        }
        else {
            ss << " " << i;
        }
        for (uint32_t j = 0; j < device_count; j++) {
            ss << " " << matrix[i][j] << "  ";
        }
        ss << "\n";
    }
    return ss.str();
}
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

std::string topo_manager::generate_uuid() {
    std::stringstream ss;
    int pid = getpid();

    auto time = std::chrono::high_resolution_clock::now();
    uint64_t time_usec = std::chrono::duration<double, std::micro>(time.time_since_epoch()).count();

    srand(time_usec);
    int random_number = rand();

    constexpr int part_1_len = 8;
    constexpr int part_2_len = 16;

    ss << std::right << std::setfill('0') << std::setw(part_1_len)
       << std::to_string(pid).substr(0, part_1_len) << "-" << std::setw(part_1_len)
       << std::to_string(random_number).substr(0, part_1_len) << "-" << std::setw(part_2_len)
       << std::to_string(time_usec).substr(0, part_2_len);

    std::string result = ss.str();

    size_t expected_uuid_len = (topo_uuid_len - 1);
    CCL_THROW_IF_NOT(result.size() == expected_uuid_len,
                     "unexpected uuid len ",
                     result.size(),
                     ", expected ",
                     expected_uuid_len,
                     ", uuid ",
                     result);
    return result;
}

std::map<int, std::string> topo_manager::get_domain_string(const std::string& input_str,
                                                           const std::string& key) {
    std::map<int, std::string> map;
    auto str = input_str;

    size_t pos = str.find(key);
    if (pos != std::string::npos) {
        str.erase(pos, key.length() + 1); // + :
    }

    int domain_idx = topo_manager::invalid_domain_idx;
    if (key == std::string(topo_manager::card_domain_name)) {
        domain_idx = topo_manager::card_domain_idx;
    }
    else if (key == std::string(topo_manager::plane_domain_name)) {
        domain_idx = topo_manager::plane_domain_idx;
    }

    CCL_THROW_IF_NOT(
        domain_idx != topo_manager::invalid_domain_idx, "unexpected domain index: ", domain_idx);
    map.insert({ domain_idx, str });
    return map;
}

std::vector<std::string> topo_manager::get_subdomain_strings(const std::string& input_str) {
    std::vector<std::string> results;
    auto str = input_str;
    bool can_parse = true;
    do {
        auto substring = ccl::utils::get_substring_between_delims(str, "{", "}");
        results.push_back(substring);

        size_t pos = str.find(substring);
        if (pos != std::string::npos) {
            str.erase(0, str.find("}") + 1);
        }

        if (str.empty())
            can_parse = false;
    } while (can_parse);
    return results;
}

std::map<int, std::vector<std::vector<int>>> topo_manager::parse_topo_env() {
    char* env_to_parse = getenv("CCL_TOPO_COLOR");
    CCL_THROW_IF_NOT(env_to_parse, "CCL_TOPO_COLOR var is unavailable");
    std::map<int, std::vector<std::vector<int>>> domains;

    std::vector<std::string> domain_raw_strs;
    ccl::utils::str_to_array(std::string(env_to_parse), ";", domain_raw_strs);
    check_domain_count(domain_raw_strs.size());

    std::vector<std::map<int, std::string>> domain_strs;
    domain_strs.push_back(get_domain_string(domain_raw_strs[topo_manager::card_domain_idx],
                                            std::string(topo_manager::card_domain_name)));
    domain_strs.push_back(get_domain_string(domain_raw_strs[topo_manager::plane_domain_idx],
                                            std::string(topo_manager::plane_domain_name)));

    for (const auto& domain_str : domain_strs) {
        for (const auto& domain_pair : domain_str) {
            std::vector<std::vector<int>> proc_indexes;
            auto substrs = get_subdomain_strings(domain_pair.second);
            for (const auto& substr : substrs) {
                std::vector<int> procs{};
                ccl::utils::str_to_array(substr, ",", procs);
                for (const auto& proc : procs) {
                    const auto local_proc_count =
                        ccl::global_data::get().executor->get_local_proc_count();
                    CCL_THROW_IF_NOT(proc < local_proc_count,
                                     "unexpected process number: ",
                                     proc,
                                     ", it should be less than: ",
                                     local_proc_count);
                }
                proc_indexes.push_back(procs);
            }
            domains.insert({ domain_pair.first, proc_indexes });
        }
    }
    check_domain_count(domains.size());
    return domains;
}

void topo_manager::fill_env_colors(const std::vector<topo_rank_info>& info_vec) {
    for (const auto& domain : domains) {
        int color_idx = 0;
        for (size_t subdomain_idx = 0; subdomain_idx < domain.second.size(); subdomain_idx++) {
            std::vector<topo_rank_info> filtered_info_vec;
            for (size_t proc_idx = 0; proc_idx < domain.second[subdomain_idx].size(); proc_idx++) {
                for (size_t i = 0; i < info_vec.size(); i++) {
                    if (info_vec[i].local_proc_idx == domain.second[subdomain_idx][proc_idx]) {
                        filtered_info_vec.push_back(info_vec[i]);
                    }
                }
            }

            for (size_t idx = 0; idx < filtered_info_vec.size(); idx++) {
                if (domain.first == topo_manager::card_domain_idx) {
                    check_color(intra_card_colors[filtered_info_vec[idx].rank]);
                    intra_card_colors[filtered_info_vec[idx].rank] = color_idx;
                }
                else if (domain.first == topo_manager::plane_domain_idx) {
                    check_color(inter_card_colors[filtered_info_vec[idx].rank]);
                    inter_card_colors[filtered_info_vec[idx].rank] = color_idx;
                }
            }
            color_idx++;
        }
    }
}

void topo_manager::fill_fixed_colors(const std::vector<topo_rank_info>& info_vec) {
    for (size_t idx = 0; idx < info_vec.size(); idx++) {
        auto rank = info_vec[idx].rank;
        intra_card_colors[rank] = idx / 2;
        inter_card_colors[rank] = idx % 2;
    }
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
void topo_manager::fill_ze_intra_colors(const std::vector<topo_rank_info>& local_info_vec,
                                        const std::vector<topo_ze_rank_info>& ze_info_vec) {
    int card_idx = 0;
    for (size_t idx = 0; idx < local_info_vec.size(); idx++) {
        size_t card_size = 0;
        const topo_rank_info& info = local_info_vec[idx];
        int rank = info.rank;

        // check if intra's elem is already filled
        if (intra_card_colors[rank] != topo_manager::invalid_color) {
            continue;
        }
        // iterate over local info comparing pci addresses for current and peer ranks
        for (size_t peer_idx = idx; peer_idx < local_info_vec.size(); peer_idx++) {
            const topo_rank_info& peer_info = local_info_vec[peer_idx];
            int peer_rank = peer_info.rank;
            // if rank & peer rank is on the card
            if (is_same_pci_addr(ze_info_vec[rank].pci_addr, ze_info_vec[peer_rank].pci_addr)) {
                // fill intra colors
                intra_card_colors[peer_rank] = card_idx;
                card_size++;
                // if size reached card_limit
                if (card_size == max_ranks_per_card) {
                    // go to the next card
                    card_idx++;
                    // reset card size
                    card_size = 0;
                    break;
                }
            }
        }
    }
}
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

void topo_manager::fill_ze_inter_colors(const std::vector<topo_rank_info>& local_info_vec) {
    for (size_t idx = 0; idx < local_info_vec.size(); idx++) {
        const topo_rank_info& info = local_info_vec[idx];
        int rank = info.rank;

        if (inter_card_colors[rank] != topo_manager::invalid_color) {
            continue;
        }

        int plane_idx = 0;
        for (size_t peer_idx = 0; peer_idx < local_info_vec.size(); peer_idx++) {
            const topo_rank_info& peer_info = local_info_vec[peer_idx];
            int peer_rank = peer_info.rank;
            if (intra_card_colors[rank] == intra_card_colors[peer_rank]) {
                inter_card_colors[peer_rank] = plane_idx;
                plane_idx++;
            }
        }
    }
}

void topo_manager::fill_ze_inter_colors(const std::vector<std::set<int>>& planes) {
    for (int rank = 0; rank < comm->get_size(); rank++) {
        for (int plane_idx = 0; plane_idx < (int)planes.size(); plane_idx++) {
            if (planes[plane_idx].find(rank) != planes[plane_idx].end()) {
                inter_card_colors[rank] = plane_idx;
                break;
            }
        }
    }
}

inline void topo_manager::check_color(const int color) {
    CCL_THROW_IF_NOT(color == topo_manager::invalid_color,
                     "unexpected color value: ",
                     color,
                     ", expected: ",
                     topo_manager::invalid_color);
}

inline void topo_manager::check_domain_count(const size_t domain_count) {
    CCL_THROW_IF_NOT(domain_count == topo_manager::max_domain_count,
                     "unexpected domain count:",
                     domain_count,
                     ", expected:",
                     topo_manager::max_domain_count);
}

inline std::vector<topo_rank_info> topo_manager::get_filtered_rank_info(
    std::vector<topo_rank_info>& rank_info_vec,
    const int filter_idx) {
    std::vector<topo_rank_info> info_vec;
    std::copy_if(rank_info_vec.begin(),
                 rank_info_vec.end(),
                 std::back_inserter(info_vec),
                 [&filter_idx](topo_rank_info& info) {
                     return (info.host_idx == filter_idx);
                 });
    return info_vec;
}

std::string topo_manager::to_string() {
    std::stringstream ss;
    ss << "\n{\n"
       << "  comm_size: " << comm->get_size() << "\n"
       << "  is_single_node: " << is_single_node << "\n"
       << "  is_single_card: " << is_single_card << "\n"
       << "  intra_card_colors: ";
    std::copy(
        intra_card_colors.begin(), intra_card_colors.end(), std::ostream_iterator<int>(ss, " "));
    ss << "\n"
       << "  inter_card_colors: ";
    std::copy(
        inter_card_colors.begin(), inter_card_colors.end(), std::ostream_iterator<int>(ss, " "));
    ss << "\n}";

    return ss.str();
}

std::string topo_manager::to_string(const std::map<int, std::vector<std::vector<int>>>& domains) {
    std::stringstream ss;
    ss << "{\n";
    for (const auto& domain : domains) {
        for (size_t subdomain_idx = 0; subdomain_idx < domain.second.size(); subdomain_idx++) {
            for (size_t proc_idx = 0; proc_idx < domain.second[subdomain_idx].size(); proc_idx++) {
                if (subdomain_idx == 0 && proc_idx == 0) {
                    if (domain.first == topo_manager::card_domain_idx)
                        ss << "card:  ";
                    else if (domain.first == topo_manager::plane_domain_idx)
                        ss << "plane: ";
                }
                if (proc_idx == 0)
                    ss << "{ ";
                ss << domain.second[subdomain_idx][proc_idx] << " ";
                if (proc_idx == domain.second[subdomain_idx].size() - 1)
                    ss << "} ";
            }
        }
        ss << "\n";
    }
    ss << "}\n";
    return ss.str();
}

} // namespace ccl
