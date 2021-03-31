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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <memory>
#include <string>

namespace net {

class ipc_rx_connection;

class ipc_server {
public:
    ~ipc_server();

    void start(const std::string& path, int expected_backlog_size = 0);
    bool stop();
    bool is_ready() const noexcept;

    std::unique_ptr<ipc_rx_connection> process_connection();

private:
    int listen_fd{ -1 };
    sockaddr_un server_addr{};
    std::string server_shared_name{};
};
} // namespace net
