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

#include "common/comm/comm.hpp"
#include "sched/entry/entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/sched.hpp"
#include "sched/ze/ze_handle_manager.hpp"

#include <poll.h>
#include <sys/un.h>
#include <ze_api.h>

class ze_handle_exchange_entry : public sched_entry {
public:
    using mem_desc_t = typename std::pair<void*, ccl::ze::ipc_mem_type>;

    static constexpr const char* class_name() noexcept {
        return "ZE_HANDLES";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    ze_handle_exchange_entry() = delete;
    explicit ze_handle_exchange_entry(ccl_sched* sched,
                                      ccl_comm* comm,
                                      const std::vector<mem_desc_t>& in_buffers,
                                      int skip_rank = -1);
    ~ze_handle_exchange_entry();

    void start() override;
    void update() override;

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "rank ",
                           rank,
                           ", comm_size ",
                           comm_size,
                           ", right_peer ",
                           right_peer_socket_name,
                           ", left_peer ",
                           left_peer_socket_name,
                           ", in_buffers size ",
                           in_buffers.size(),
                           ", handles size ",
                           handles.size(),
                           "\n");
    }

private:
    static constexpr size_t socket_max_str_len = 100;
    static constexpr int poll_expire_err_code = 0;
    static constexpr int timeout_ms = 1;
    static constexpr size_t max_pfds = 1;

    const ccl_comm* comm;

    std::vector<mem_desc_t> in_buffers;
    ccl::ze::ipc_handle_manager::mem_handle_map_t handles;

    const int rank;
    const int comm_size;
    const int skip_rank;

    int start_buf_idx{};
    int start_peer_idx{};

    std::vector<struct pollfd> poll_fds;

    int right_peer_socket{};
    int left_peer_socket{};
    int left_peer_connect_socket{};

    struct sockaddr_un right_peer_addr {
    }, left_peer_addr{};
    int right_peer_addr_len{}, left_peer_addr_len{};

    std::string right_peer_socket_name;
    std::string left_peer_socket_name;

    bool is_created{};
    bool is_connected{};
    bool is_accepted{};
    bool skip_first_send{};

    void get_fd_from_handle(const ze_ipc_mem_handle_t* handle, int* fd) noexcept;
    void get_handle_from_fd(const int* fd, ze_ipc_mem_handle_t* handle) noexcept;

    int create_server_socket(const std::string& socket_name,
                             struct sockaddr_un* socket_addr,
                             int* addr_len,
                             int comm_size);
    int create_client_socket(const std::string& left_peer_socket_name,
                             struct sockaddr_un* sockaddr_cli,
                             int* len);

    int accept_call(int connect_socket,
                    struct sockaddr_un* socket_addr,
                    int* addr_len,
                    const std::string& socket_name,
                    int& sock);
    int connect_call(int sock,
                     struct sockaddr_un* socket_addr,
                     int addr_len,
                     const std::string& socket_name);

    void sendmsg_fd(int sock, int fd, size_t mem_offset);
    void recvmsg_fd(int sock, int& fd, size_t& mem_offset);

    void sendmsg_call(int sock, int fd, size_t mem_offset);
    void recvmsg_call(int sock, int& fd, size_t& mem_offset);
    int check_msg_retval(std::string operation_name,
                         ssize_t bytes,
                         struct iovec iov,
                         struct msghdr msg,
                         size_t union_size,
                         int sock,
                         int fd);

    using mem_info_t = typename std::pair<void*, size_t>;
    mem_info_t get_mem_info(const void* ptr);

    void unlink_sockets();
    void close_sockets();
};
