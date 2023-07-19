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

#include "oneapi/ccl/config.h"

#include <map>
#include <string>
#include <vector>

namespace ccl {
namespace ze {

enum class ipc_exchange_mode : int { sockets, drmfd, pidfd, none };
static std::map<ipc_exchange_mode, std::string> ipc_exchange_names = {
    std::make_pair(ipc_exchange_mode::sockets, "sockets"),
#ifdef CCL_ENABLE_DRM
    std::make_pair(ipc_exchange_mode::drmfd, "drmfd"),
#endif // CCL_ENABLE_DRM
    std::make_pair(ipc_exchange_mode::pidfd, "pidfd"),
    std::make_pair(ipc_exchange_mode::none, "none")
};

struct bdf_info {
    uint32_t domain{};
    uint32_t bus{};
    uint32_t device{};
    uint32_t function{};
};

struct device_bdf_info {
    int fd = ccl::utils::invalid_fd;
    bdf_info bdf{};
};

class fd_manager {
public:
    static constexpr int invalid_fd_idx = -1;
    static constexpr int invalid_device_idx = -1;
    static constexpr int invalid_physical_idx = -1;
    static constexpr int hexadecimal_base = 16;
    static constexpr int bdf_start_pos = 12;

    fd_manager();
    fd_manager(const fd_manager&) = delete;
    fd_manager(fd_manager&&) = delete;
    fd_manager& operator=(const fd_manager&) = delete;
    fd_manager& operator=(fd_manager&&) = delete;
    ~fd_manager();

    static int mem_handle_to_fd(int convert_from_fd, int fd);
    static int fd_to_mem_handle(int dev_fd, int handle);

    static bool is_pidfd_supported();
    static int pidfd_open(const int pid);

    // get
    std::vector<int> get_device_fds();
    std::vector<device_bdf_info> get_physical_devices();
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    static int get_physical_device_idx(std::vector<device_bdf_info> devs, ze_pci_address_ext_t pci);
#endif // ZE_PCI_PROPERTIES_EXT_NAME

private:
    void exchange_device_fds();
    std::vector<int> setup_device_fds(int local_count,
                                      int proc_idx,
                                      std::vector<bdf_info>& return_bdf);

    void close_sockets(int local_count, int proc_idx);

    std::string get_shm_filename();
    void* create_shared_memory();
    void barrier(void* mem);

    static int convert_fd_pidfd(int convert_from_fd, int handle);
    static int convert_fd_drmfd(int convert_from_fd, int handle);

    const int counter_offset = sizeof(int);
    const int size_per_proc = sizeof(pid_t);

    // init
    std::vector<int> init_device_fds();
    std::vector<bdf_info> init_device_bdfs(const size_t size);
    // fill
    std::vector<int> fill_device_fds(std::vector<std::string> dev_names);
    std::vector<device_bdf_info> fill_physical_devices();

    // find
    static int compare_bdf(const void* _a, const void* _b);
    void find_bdf(std::string dev_name, bdf_info& info);
    int find_device_idx(std::string dev_name, const size_t dev_name_idx, const size_t size);

    std::vector<bdf_info> device_bdfs;

    std::vector<int> all_socks;
    std::vector<pid_t> all_pids;

    std::vector<int> device_fds;
    std::vector<device_bdf_info> physical_devices;
};

} // namespace ze
} // namespace ccl
