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
#include <stdexcept>
#include <fcntl.h>
#include <algorithm>

#include "common/log/log.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_server.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_connection.hpp"

namespace net {

ipc_server::~ipc_server() {
    stop();
    unlink(server_shared_name.c_str());
}

void ipc_server::start(const std::string& path, int expected_backlog_size) {
    LOG_INFO("Starting IPC server on addr: ", path);
    if (is_ready()) {
        throw std::runtime_error(std::string("Cannot restart ipc server with addr: ") + path);
    }

    size_t path_size_limit = sizeof(server_addr.sun_path) - 1;
    if (path.size() > path_size_limit) {
        throw std::runtime_error(std::string("Cannot start ipc server on requested addr: ") + path +
                                 " - addr size if too long: " + std::to_string(path.size()) +
                                 ", expected: " + std::to_string(path_size_limit));
    }
    path_size_limit = std::min(path_size_limit, path.size());

    LOG_TRACE("Reset previously locked handle");
    unlink(path.c_str());

    try {
        listen_fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (listen_fd == -1) {
            throw std::runtime_error(std::string("Cannot create socket, error: ") +
                                     strerror(errno));
        }

        // set non blocking
        int fileflags = fcntl(listen_fd, F_GETFL, 0);
        if (fileflags == -1) {
            throw std::runtime_error(std::string("Cannot get fcntl socket flags, error: ") +
                                     strerror(errno));
        }
        if (fcntl(listen_fd, F_SETFL, fileflags | O_NONBLOCK) == -1) {
            throw std::runtime_error(std::string("Cannot set non-blocking socket, error: ") +
                                     strerror(errno));
        }

        //allow reuse
        int enable = 1;
        if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
            throw std::runtime_error(std::string("Cannot set reuse socket, error: ") +
                                     strerror(errno));
        }

        memset(&server_addr, 0, sizeof(struct sockaddr_un));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, path.c_str(), path_size_limit);
        server_addr.sun_path[path_size_limit] = '\0';

        int ret = bind(listen_fd, (const struct sockaddr*)&server_addr, sizeof(struct sockaddr_un));
        if (ret == -1) {
            throw std::runtime_error(std::string("Cannot bind socket by addr: ") + path +
                                     ", error: " + strerror(errno));
        }

        ret = listen(listen_fd, expected_backlog_size);
        if (ret == -1) {
            throw std::runtime_error(std::string("Cannot start listen socket by addr: ") + path +
                                     ", error: " + strerror(errno));
        }

        server_shared_name = path;
    }
    catch (const std::exception& ex) {
        LOG_ERROR(ex.what());
        throw;
    }
}

bool ipc_server::stop() {
    bool ret = false;
    if (is_ready()) {
        LOG_DEBUG("Gracefully stop listener: ", listen_fd);
        shutdown(listen_fd, SHUT_RDWR);
        close(listen_fd);
        listen_fd = -1;
        ret = true;
    }
    else {
        LOG_DEBUG("Nothing to stop");
    }
    return ret;
}

bool ipc_server::is_ready() const noexcept {
    return listen_fd != -1;
}

std::unique_ptr<ipc_rx_connection> ipc_server::process_connection() {
    if (!is_ready()) {
        throw std::runtime_error(std::string(__FUNCTION__) + " - failed, ipc server is not ready");
    }

    std::unique_ptr<ipc_rx_connection> ret;

    int fd = accept(listen_fd, nullptr, nullptr);
    if (fd == -1) {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - failed, accept failed on socket: " +
                                     std::to_string(listen_fd) + ", error: " + strerror(errno));
        }
        LOG_TRACE("Nothing to accept on socket:", listen_fd);
    }
    else {
        ret.reset(new ipc_rx_connection(fd));
    }

    return ret;
}
} // namespace net
