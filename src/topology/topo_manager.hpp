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

#include "atl/atl_base_comm.hpp"
#include "common/utils/utils.hpp"
#include "oneapi/ccl/config.h"

#ifdef CCL_ENABLE_MPI
#include <mpi.h>
#endif // CCL_ENABLE_MPI

#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <string>

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "common/ze/ze_api_wrapper.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

enum class topo_color_mode : int { fixed, ze, env };
static std::map<topo_color_mode, std::string> topo_color_names = {
    std::make_pair(topo_color_mode::fixed, "fixed"),
    std::make_pair(topo_color_mode::ze, "ze"),
    std::make_pair(topo_color_mode::env, "env")
};

static constexpr int topo_uuid_len = 35;

struct topo_rank_info {
    int rank;
    int host_idx;
    int local_proc_idx;
    char uuid[topo_uuid_len];

    topo_rank_info();
};

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
struct topo_ze_rank_info {
    ze_device_uuid_t device_uuid{};
    zes_pci_address_t pci_addr{};
};

struct topo_ze_port_info {
    zes_fabric_port_id_t local{};
    zes_fabric_port_id_t remote{};
};
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

class topo_manager {
public:
    static constexpr int invalid_color = -1;
    static constexpr int invalid_host_idx = -1;
    static constexpr int invalid_plane_idx = -1;
    static constexpr int invalid_domain_idx = -1;
    static constexpr int max_hostname_len = 256;
    static constexpr int max_ranks_per_host = 1000;
    static constexpr int max_ranks_per_card = 2;
    static constexpr int max_ranks_per_plane = 8;
    static constexpr int max_domain_count = 2;
    static constexpr int card_domain_idx = 0;
    static constexpr int plane_domain_idx = 1;
    static constexpr const char* card_domain_name = "card";
    static constexpr const char* plane_domain_name = "plane";

    topo_manager() = default;
    topo_manager(const topo_manager& other) = default;
    topo_manager& operator=(const topo_manager& other) = default;

    ~topo_manager() = default;

    void init(std::shared_ptr<atl_base_comm> atl_comm,
              std::shared_ptr<ccl::device> device_ptr,
              std::shared_ptr<ccl::context> context_ptr);

    int get_host_idx() const;
    int get_intra_card_color(int rank) const;
    int get_inter_card_color(int rank) const;
    std::string get_uuid(int rank) const;

    bool has_p2p_access() const;
    bool has_same_ppn() const;
    bool has_same_domains() const;

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    static bool is_same_pci_addr(zes_pci_address_t addr1, zes_pci_address_t addr2);
    static std::vector<std::vector<bool>> build_p2p_matrix(
        const std::vector<ze_device_handle_t>& devices);
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    static std::map<int, std::vector<std::vector<int>>> parse_topo_env();

    std::string to_string();

    bool is_single_node = false;
    bool is_single_card = false;

private:
    void base_init(std::shared_ptr<atl_base_comm> atl_comm,
                   std::shared_ptr<ccl::device> device_ptr,
                   std::shared_ptr<ccl::context> context_ptr);
    void check_colors();
    void check_ppn(const std::set<std::string>& unique_hostnames,
                   const std::vector<std::string>& all_hostnames);
    void post_init();

    void allgather(const void* send_buf, void* recv_buf, int bytes);
    void allgatherv(const void* send_buf, void* recv_buf, const std::vector<int>& recv_bytes);

    void fill_env_colors(const std::vector<topo_rank_info>& local_info_vec);
    void fill_fixed_colors(const std::vector<topo_rank_info>& info_vec);

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    void fill_ze_intra_colors(const std::vector<topo_rank_info>& local_info_vec,
                              const std::vector<topo_ze_rank_info>& ze_info_vec);
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
    void fill_ze_inter_colors(const std::vector<topo_rank_info>& local_info_vec);
    void fill_ze_inter_colors(const std::vector<std::set<int>>& planes);

    inline std::vector<topo_rank_info> get_filtered_rank_info(
        std::vector<topo_rank_info>& rank_info_vec,
        const int compare_idx);

    static inline void check_color(const int color);
    static inline void check_domain_count(const size_t domain_count);

    static std::map<int, std::string> get_domain_string(const std::string& input_str,
                                                        const std::string& key);
    static std::vector<std::string> get_subdomain_strings(const std::string& input_str);
    static std::string generate_uuid();

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    ze_device_handle_t device{};
    static std::string to_string(const std::vector<std::vector<bool>>& p2p_matrix);

    static inline std::vector<ze_device_uuid_t> copy_dev_uuids(
        const std::vector<topo_rank_info>& info_vec,
        const std::vector<topo_ze_rank_info>& ze_rank_info_vec);
    static bool is_same_dev_uuid(ze_device_uuid_t uuid1, ze_device_uuid_t uuid2);
    static bool is_sub_vector(const std::vector<ze_device_uuid_t>& vec,
                              const std::vector<ze_device_uuid_t>& sub_vec);

    std::vector<ze_device_handle_t> get_filtered_devices(
        const std::vector<ze_device_handle_t>& devices,
        const std::vector<topo_ze_rank_info>& ze_rank_info_vec);
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    std::string to_string(const std::map<int, std::vector<std::vector<int>>>& domains);

    std::shared_ptr<atl_base_comm> comm;

    int host_idx = invalid_host_idx;
    std::vector<int> intra_card_colors{};
    std::vector<int> inter_card_colors{};
    std::vector<std::string> uuids{};
    std::vector<topo_rank_info> rank_info_vec;
    std::vector<std::vector<bool>> p2p_matrix;
    std::map<int, std::vector<std::vector<int>>> domains;

    bool is_same_ppn = true;
    bool is_same_domains = true;
};

} // namespace ccl
