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
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace net {

class connection {
public:
    using fd_t = int;

    static constexpr size_t fd_size_bytes() {
        return sizeof(fd_t);
    }

    static constexpr size_t ancillary_data_limit_fd() {
        return 10;
    }
    static constexpr size_t ancillary_data_limit_bytes() {
        return ancillary_data_limit_fd() * fd_size_bytes(); // 10 fd is limitation
    }

    connection(const connection& src) = delete;
    connection& operator=(const connection& src) = delete;

    ssize_t send_msg_with_pid_data(const std::vector<uint8_t>& data,
                                   const std::vector<size_t>& optional_pid_data_offets,
                                   int flag = 0) const;

    // For properly call resize data with *_resized before for expected sizes
    ssize_t recv_msg_with_pid_data(std::vector<uint8_t>& out_data_resized,
                                   std::vector<fd_t>& out_pids_resized,
                                   int flags = 0 /*=MSG_CMSG_CLOEXEC | MSG_WAITALL */) const;

    ssize_t send_data(const uint8_t* send_data_ptr, size_t size, int flags = 0) const;
    ssize_t recv_data(uint8_t* recv_data_ptr, size_t size, int flags = 0) const;

protected:
    explicit connection(int connected_socket);
    ~connection();

    int socket_fd{ -1 };
};

} // namespace net
