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

#include "common/log/log.hpp"
#include "common/comm/l0/devices/communication_structs/connection.hpp"

namespace net {

connection::connection(int connected_socket) : socket_fd(connected_socket) {}

connection::~connection() {
    shutdown(socket_fd, SHUT_RDWR);
    close(socket_fd);
}

ssize_t connection::send_data(const uint8_t* send_data_ptr, size_t size, int flags) const {
    LOG_TRACE("Send data size: ", size, ", into socket: ", socket_fd);

    if (!send_data_ptr) {
        return 0;
    }

    ssize_t ret = 0;
    do {
        ret = send(socket_fd, send_data_ptr, size, flags);
    } while (ret == -1 && errno == EINTR);

    if (ret == -1 && (errno != EAGAIN || errno != EWOULDBLOCK)) {
        throw std::runtime_error(std::string("Cannot send data to socket: ") +
                                 std::to_string(socket_fd) + strerror(errno));
    }
    LOG_TRACE("Data bytes sent: ", ret, ", to socket: ", socket_fd);
    return ret;
}

ssize_t connection::recv_data(uint8_t* recv_data_ptr, size_t size, int flags) const {
    LOG_TRACE("Recv data size: ", size, ", from socket: ", socket_fd);

    if (!recv_data_ptr) {
        return 0;
    }

    ssize_t ret = 0;
    do {
        ret = recv(socket_fd, recv_data_ptr, size, flags);
    } while (ret == -1 && errno == EINTR);

    if (ret == -1 && (errno != EAGAIN || errno != EWOULDBLOCK)) {
        throw std::runtime_error(std::string("Cannot recv data from socket: ") +
                                 std::to_string(socket_fd) + strerror(errno));
    }
    LOG_TRACE("Data bytes received: ", ret, ", from socket: ", socket_fd);
    return ret;
}

ssize_t connection::send_msg_with_pid_data(const std::vector<uint8_t>& data,
                                           const std::vector<size_t>& optional_pid_data_offets,
                                           int flag) const {
    //TODO make sure limit doesn't exceed `/proc/sys/net/core/optmem_max`

    if (connection::ancillary_data_limit_bytes() < optional_pid_data_offets.size() * sizeof(fd_t)) {
        LOG_ERROR("ancillary_data_limit_bytes is to less: ",
                  connection::ancillary_data_limit_bytes(),
                  "bytes, than required: ",
                  optional_pid_data_offets.size() * sizeof(fd_t),
                  ". Recompile with large limits is required");
        abort();
    }

    // fill regular data
    struct msghdr msg = { 0 };
    struct iovec io = { .iov_base = const_cast<void*>(static_cast<const void*>(data.data())),
                        .iov_len = data.size() * sizeof(uint8_t) };

    // fill anciliary data
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_controllen =
        CMSG_SPACE(sizeof(fd_t) * optional_pid_data_offets.size()); //sizeof(u.buf);
    std::vector<uint8_t> staged_buf(msg.msg_controllen, 0);
    msg.msg_control = staged_buf.data();

    // one anciliary message for all fds
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd_t) * optional_pid_data_offets.size());
    fd_t* fdptr = (fd_t*)CMSG_DATA(cmsg);

    for (auto fd_offset_bytes_it = optional_pid_data_offets.begin();
         fd_offset_bytes_it != optional_pid_data_offets.end();
         ++fd_offset_bytes_it) {
        if (*fd_offset_bytes_it >= data.size()) {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     "Unexpected value in optional_pid_data_offets, data size: " +
                                     std::to_string(data.size()) +
                                     ", offset requested: " + std::to_string(*fd_offset_bytes_it));
        }

        const fd_t* in_fd_ptr = reinterpret_cast<const fd_t*>(data.data() + *fd_offset_bytes_it);
        memcpy(fdptr, in_fd_ptr, sizeof(fd_t));
        fdptr++;
    }

    ssize_t bytes_sent = sendmsg(socket_fd, &msg, flag);
    LOG_DEBUG("sendmsg on socket: ", socket_fd, ", result: ", bytes_sent);
    if (bytes_sent < 0) {
        throw std::runtime_error(std::string(__FUNCTION__) + " - cannot sendmsg on socket: " +
                                 std::to_string(socket_fd) + ", error: " + strerror(errno));
    }
    return bytes_sent;
}

ssize_t connection::recv_msg_with_pid_data(std::vector<uint8_t>& out_data_resized,
                                           std::vector<fd_t>& out_pids_resized,
                                           int flags) const {
    LOG_DEBUG("Prepared data size bytes: ",
              out_data_resized.size(),
              ", pid count: ",
              out_pids_resized.size(),
              ", socket: ",
              socket_fd);

    // prepare regular data
    struct iovec msg_buffer;
    msg_buffer.iov_base = out_data_resized.data();
    msg_buffer.iov_len = out_data_resized.size();

    // prepare control data
    struct msghdr msg_header = { 0 };
    msg_header.msg_iov = &msg_buffer;
    msg_header.msg_iovlen = 1;
    msg_header.msg_controllen = CMSG_SPACE(sizeof(fd_t) * out_pids_resized.size()); //sizeof(u.buf);

    std::vector<uint8_t> staged_buf(msg_header.msg_controllen, 0);
    msg_header.msg_control = staged_buf.data();

    ssize_t bytes_got = 0;
    do {
        bytes_got = recvmsg(socket_fd, &msg_header, flags);
    } while (bytes_got == -1 && errno == EINTR);

    if (bytes_got == -1) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - cannot receive data, error: " + strerror(errno));
    }

    LOG_DEBUG("Received bytes: ", bytes_got, ", from socket: ", socket_fd);

    size_t received_fd_num = 0;
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg_header);
    for (; cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg_header, cmsg)) {
        LOG_TRACE("cmsg_level: ", cmsg->cmsg_level, ", cmsg_type: ", cmsg->cmsg_type);
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            // sanity
            size_t expected_len = CMSG_LEN(out_pids_resized.size() * sizeof(fd_t));
            if (cmsg->cmsg_len < expected_len) {
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - got unexpected anciliary msg size from socket: " +
                                         std::to_string(socket_fd) +
                                         ", got: " + std::to_string(cmsg->cmsg_len) +
                                         ", but expected: " + std::to_string(expected_len));
            }

            // restore duplicated fds
            fd_t* fd_ptr = (fd_t*)CMSG_DATA(cmsg);
            for (auto& it : out_pids_resized) {
                it = *fd_ptr;
                LOG_DEBUG("got fd: ",
                          *fd_ptr,
                          ", by number: ",
                          received_fd_num,
                          ", expected count: ",
                          out_pids_resized.size());

                fd_ptr++;
                received_fd_num++;
            }
        }
    }

    LOG_DEBUG("Received fd count: ", received_fd_num);
    if (received_fd_num != out_pids_resized.size()) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - unexpected FD from socket: " + std::to_string(socket_fd) +
                                 ", received count: " + std::to_string(received_fd_num) +
                                 ", but expected: " + std::to_string(out_pids_resized.size()));
    }
    return bytes_got;
}
} // namespace net
