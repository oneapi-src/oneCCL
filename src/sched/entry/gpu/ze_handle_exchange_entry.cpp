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
#include "sched/entry/gpu/ze_handle_exchange_entry.hpp"
#include "sched/queue/queue.hpp"
#include "sched/ze_handle_manager.hpp"

#include <arpa/inet.h>
#include <CL/sycl/backend/level_zero.hpp>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static void cast_pool_to_mem_handle(ze_ipc_mem_handle_t* mem,
                                    const ze_ipc_event_pool_handle_t* pool) {
    static_assert(sizeof(ze_ipc_mem_handle_t) == sizeof(ze_ipc_event_pool_handle_t));
    memcpy(mem, pool, sizeof(*pool));
}

ze_handle_exchange_entry::ze_handle_exchange_entry(ccl_sched* sched,
                                                   ccl_comm* comm,
                                                   const std::vector<mem_desc_t>& in_buffers,
                                                   int skip_rank)
        : sched_entry(sched),
          comm(comm),
          in_buffers(in_buffers),
          rank(comm->rank()),
          comm_size(comm->size()),
          skip_rank(skip_rank) {
    LOG_DEBUG("initialization");
    CCL_THROW_IF_NOT(sched, "no sched");
    CCL_THROW_IF_NOT(!in_buffers.empty(), "in_buffers should be non empty");

    poll_fds.reserve(max_pfds);

    handles.resize(comm_size);
    for (auto& buffers : handles) {
        buffers.resize(in_buffers.size());
    }
    LOG_DEBUG("handles size: ", handles.size(), ", in_buffers size: ", in_buffers.size());

    for (size_t buf_idx = 0; buf_idx < in_buffers.size(); buf_idx++) {
        auto mem_ptr = in_buffers[buf_idx].first;
        CCL_THROW_IF_NOT(mem_ptr, "memory pointer is nullptr");
        auto mem_type = in_buffers[buf_idx].second;
        mem_info_t mem_info{};

        ze_ipc_mem_handle_t handle{};
        if (rank != skip_rank) {
            if (mem_type == ccl::ze::ipc_mem_type::memory) {
                // zeMemGetIpcHandle requires the provided pointer to be the base of an allocation.
                // We handle this the following way: for an input buffer retrieve its base pointer
                // and the offset from this base ptr. The base ptr is used for zeMemGetIpcHandle
                // and the offset is sent to the other rank. On that rank the base ptr is retrieved
                // and offsetted to get the actual input buffer ptr.
                mem_info = get_mem_info(mem_ptr);
                sched->get_memory().handle_manager.get_handle(mem_info.first, &handle);
            }
            else if (mem_type == ccl::ze::ipc_mem_type::pool) {
                ze_ipc_event_pool_handle_t pool_handle;
                sched->get_memory().handle_manager.get_handle(
                    static_cast<ze_event_pool_handle_t>(mem_ptr), &pool_handle);
                // since ze_ipc_event_pool_handle_t and ze_ipc_mem_handle_t are similar,
                // we cast ze_ipc_event_pool_handle_t to ze_ipc_mem_handle_t, but
                // maybe this is not the most correct way
                cast_pool_to_mem_handle(&handle, &pool_handle);
            }
            else {
                CCL_THROW("unknown memory type");
            }
        }

        handles[rank][buf_idx] = { handle, mem_info.second, mem_type };
        LOG_DEBUG("set IPC handle: { rank: ",
                  rank,
                  ", buf_idx: ",
                  buf_idx,
                  ", mem_type: ",
                  to_string(mem_type),
                  " }");
    }

    std::string unique_tag = std::to_string(sched->get_comm_id()) + "-" +
                             std::to_string(sched->sched_id) + "-" +
                             std::to_string(sched->get_op_id());
    right_peer_socket_name =
        "/tmp/ccl-handle-" + std::to_string((rank + 1) % comm_size) + "-" + unique_tag;
    left_peer_socket_name = "/tmp/ccl-handle-" + std::to_string(rank) + "-" + unique_tag;

    // This is a temporary workaround around to provide uniqueness of socket files created
    // in /tmp folder, otherwise this could result in issues in case of parallel runs
    // by a single/multiple users.
    // Ideally we should use process pid for this, but right now we don't have this information
    // available for all the processes, so use this env variable instead. This works with mpiexec
    // only(this is why it's the workaround rather than a complete solution)
    static const char* mpi_uuid = getenv("I_MPI_HYDRA_UUID");
    if (mpi_uuid) {
        right_peer_socket_name += std::string("-") + mpi_uuid;
        left_peer_socket_name += std::string("-") + mpi_uuid;
    }

    LOG_DEBUG("initialization complete");
}

ze_handle_exchange_entry::~ze_handle_exchange_entry() {
    close_sockets();
    unlink_sockets();
}

void ze_handle_exchange_entry::start() {
    start_buf_idx = start_peer_idx = 0;
    skip_first_send = false;
    status = ccl_sched_entry_status_started;
}

void ze_handle_exchange_entry::update() {
    if (!is_created) {
        // server
        left_peer_connect_socket = create_server_socket(
            left_peer_socket_name, &left_peer_addr, &left_peer_addr_len, comm_size);

        // client
        right_peer_socket =
            create_client_socket(right_peer_socket_name, &right_peer_addr, &right_peer_addr_len);

        is_created = true;
    }

    if (!is_connected) {
        if (connect_call(
                right_peer_socket, &right_peer_addr, right_peer_addr_len, right_peer_socket_name)) {
            return;
        }
        is_connected = true;
    }

    if (!is_accepted) {
        if (accept_call(left_peer_connect_socket,
                        &left_peer_addr,
                        &left_peer_addr_len,
                        left_peer_socket_name,
                        left_peer_socket)) {
            return;
        }

        struct pollfd poll_fd {};
        poll_fd.fd = left_peer_socket;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        poll_fds.push_back(poll_fd);

        is_accepted = true;
    }

    CCL_THROW_IF_NOT(poll_fds.size() == 1, "unexpected poll_fds size: ", poll_fds.size());

    for (size_t buf_idx = start_buf_idx; buf_idx < in_buffers.size(); buf_idx++) {
        for (int peer_idx = start_peer_idx; peer_idx < comm_size - 1; peer_idx++) {
            int peer = (comm_size + rank - 1 - peer_idx) % comm_size;

            if ((peer_idx == 0) && !skip_first_send && (rank != skip_rank)) {
                int send_fd = 0;
                // send own handle to right peer
                get_fd_from_handle(&(handles[rank][buf_idx].handle), &send_fd);
                sendmsg_call(right_peer_socket, send_fd, handles[rank][buf_idx].offset);
                skip_first_send = true;
            }

            if (peer == skip_rank)
                continue;

            int poll_ret = poll(&poll_fds[0], poll_fds.size(), timeout_ms);

            if (poll_ret == poll_expire_err_code) {
                LOG_DEBUG("poll: timeout is expired");
                return;
            }
            else if (poll_ret == POLL_ERR) {
                CCL_THROW("poll: error: ", strerror(errno), ", ret: ", poll_ret);
            }

            CCL_THROW_IF_NOT(poll_ret > 0, "unexpected poll ret: ", poll_ret);

            if (poll_fds[0].revents & POLLIN) {
                int recv_fd = 0;
                ze_ipc_mem_handle_t tmp_handle{};

                size_t mem_offset = 0;
                // recv data from left peer
                recvmsg_call(left_peer_socket, recv_fd, mem_offset);

                // invoke get_handle_from_fd to store the handle
                get_handle_from_fd(&recv_fd, &tmp_handle);

                // we don't know anything about the memory type on the other side,
                // so we take it from our list. This assumes that the lists of types (exactly types)
                // on the sending and receiving side are the same in both value and quantity
                auto mem_type = in_buffers[buf_idx].second;
                handles[peer][buf_idx] = { tmp_handle, mem_offset, mem_type };
                LOG_DEBUG("get IPC handle: { peer: ",
                          peer,
                          ", buf_idx: ",
                          buf_idx,
                          ", mem_type: ",
                          to_string(mem_type),
                          " }");

                if (peer_idx < (comm_size - 2)) {
                    // proxy data to right peer
                    sendmsg_call(right_peer_socket, recv_fd, mem_offset);
                }
                start_peer_idx++;
            }
            else if (poll_fds[0].revents & POLLERR) {
                CCL_THROW("poll: POLLERR, buf_idx: ", buf_idx, ", peer_idx ", peer_idx);
            }
            else if (poll_fds[0].revents & POLLHUP) {
                CCL_THROW("poll: POLLHUP, buf_idx: ", buf_idx, ", peer_idx ", peer_idx);
            }
            else {
                LOG_TRACE("poll: nothing to receive, buf_idx: ", buf_idx, ", peer_idx ", peer_idx);
                // nothing to receive
                // continue with the same buf_idx/peer_idx in the next update() call
                return;
            }
        }
        start_peer_idx = 0;
        skip_first_send = false;
        start_buf_idx++;
    }

    LOG_DEBUG("handles size: ", handles.size(), ", in_buffers size: ", in_buffers.size());

    sched->get_memory().handle_manager.set(handles);

    status = ccl_sched_entry_status_complete;

    LOG_DEBUG("completed: ", name());
}

int ze_handle_exchange_entry::create_server_socket(const std::string& socket_name,
                                                   struct sockaddr_un* socket_addr,
                                                   int* addr_len,
                                                   int comm_size) {
    int ret = 0;
    memset(&(*socket_addr), 0, sizeof((*socket_addr)));

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        unlink_sockets();
        CCL_THROW("cannot create a server socket: ",
                  sock,
                  ", errno: ",
                  strerror(errno),
                  ", socket_name: ",
                  socket_name);
    }

    socket_addr->sun_family = AF_UNIX;
    strncpy(socket_addr->sun_path, socket_name.c_str(), sizeof(socket_addr->sun_path) - 1);
    socket_addr->sun_path[sizeof(socket_addr->sun_path) - 1] = '\0';
    *addr_len = sizeof((*socket_addr));

    ret = fcntl(sock, F_SETFL, O_NONBLOCK);
    CCL_THROW_IF_NOT(
        !ret, "fcntl error: ", ret, ", errno: ", strerror(errno), ", socket_name: ", socket_name);

    unlink(socket_name.c_str());

    ret = bind(sock, ((struct sockaddr*)&(*socket_addr)), *addr_len);
    CCL_THROW_IF_NOT(
        !ret, "bind error: ", ret, ", errno: ", strerror(errno), ", socket_name: ", socket_name);

    ret = listen(sock, comm_size);
    CCL_THROW_IF_NOT(
        !ret, "listen error: ", ret, ", errno: ", strerror(errno), ", socket_name: ", socket_name);

    return sock;
}

int ze_handle_exchange_entry::create_client_socket(const std::string& socket_name,
                                                   struct sockaddr_un* socket_addr,
                                                   int* addr_len) {
    memset(&(*socket_addr), 0, sizeof(*(socket_addr)));

    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    CCL_THROW_IF_NOT(
        sock >= 0, "cannot create a client socket: ", sock, ", errno: ", strerror(errno));

    socket_addr->sun_family = AF_UNIX;
    strncpy(socket_addr->sun_path, socket_name.c_str(), sizeof(socket_addr->sun_path) - 1);
    socket_addr->sun_path[sizeof(socket_addr->sun_path) - 1] = '\0';
    *addr_len = sizeof((*socket_addr));

    return sock;
}

int ze_handle_exchange_entry::accept_call(int connect_socket,
                                          struct sockaddr_un* socket_addr,
                                          int* addr_len,
                                          const std::string& socket_name,
                                          int& sock) {
    sock = accept(connect_socket, ((struct sockaddr*)&(*socket_addr)), ((socklen_t*)&(*addr_len)));
    if (sock < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            LOG_TRACE("accept eagain: ", strerror(errno), ", socket_name: ", socket_name);
            return errno;
        }

        if (errno == EMFILE) {
            LOG_TRACE("accept no free fd: ", strerror(errno), ", socket_name: ", socket_name);
            return errno;
        }

        CCL_THROW(
            "accept error: ", strerror(errno), " sock: ", sock, ", socket_name: ", socket_name);
    }

    LOG_DEBUG("accept from [", comm->rank(), "] (wait) on: ", socket_name);
    return 0;
}

int ze_handle_exchange_entry::connect_call(int sock,
                                           struct sockaddr_un* socket_addr,
                                           int addr_len,
                                           const std::string& socket_name) {
    int ret = connect(sock, ((struct sockaddr*)&(*socket_addr)), addr_len);
    if (ret < 0) {
        if (errno == ECONNREFUSED || errno == ENOENT) {
            return errno;
        }
        CCL_THROW(
            "connect error: ", ret, ", errno: ", strerror(errno), ", socket_name: ", socket_name);
    }

    LOG_DEBUG("connect from: [",
              comm->rank(),
              "] to [",
              (comm->rank() - 1 + comm->size()) % comm->size(),
              "] with: ",
              socket_name);

    return 0;
}

void ze_handle_exchange_entry::sendmsg_fd(int sock, int fd, size_t mem_offset) {
    CCL_THROW_IF_NOT(fd > 0, "unexpected fd value");

    struct iovec iov {};
    iov.iov_base = &mem_offset;
    iov.iov_len = sizeof(size_t);

    char ctrl_buf[CMSG_SPACE(sizeof(fd))]{};
    struct msghdr msg {};
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(fd));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    auto cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    *reinterpret_cast<int*>(CMSG_DATA(cmsg)) = fd;

    ssize_t send_bytes = sendmsg(sock, &msg, 0);
    CCL_THROW_IF_NOT(send_bytes >= 0,
                     "sendmsg error: ",
                     send_bytes,
                     ", socket: ",
                     sock,
                     ", fd: ",
                     fd,
                     ", from: ",
                     comm->rank(),
                     ", errno: ",
                     strerror(errno));
}

void ze_handle_exchange_entry::recvmsg_fd(int sock, int& fd, size_t& mem_offset) {
    size_t buf{};
    struct iovec iov {};
    iov.iov_base = &buf;
    iov.iov_len = sizeof(size_t);

    char ctrl_buf[CMSG_SPACE(sizeof(int))]{};
    struct msghdr msg {};
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    ssize_t recv_bytes = recvmsg(sock, &msg, 0);
    CCL_THROW_IF_NOT(recv_bytes >= 0,
                     "recvmsg error: ",
                     recv_bytes,
                     ", socket: ",
                     sock,
                     ", fd: ",
                     fd,
                     ", from: ",
                     comm->rank(),
                     ", errno: ",
                     strerror(errno));

    if (msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) {
        CCL_THROW("control message is truncated");
    }

    for (auto cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) && cmsg->cmsg_level == SOL_SOCKET &&
            cmsg->cmsg_type == SCM_RIGHTS) {
            memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
            break;
        }
    }

    // we assume that the message has a strict format and size, if not this means that something
    // is wrong.
    if (msg.msg_iovlen != 1 || msg.msg_iov[0].iov_len != sizeof(size_t)) {
        CCL_THROW("received data in unexpected format");
    }

    memcpy(&mem_offset, msg.msg_iov[0].iov_base, sizeof(size_t));
}

void ze_handle_exchange_entry::sendmsg_call(int sock, int fd, size_t mem_offset) {
    sendmsg_fd(sock, fd, mem_offset);
    LOG_DEBUG("send: rank[",
              comm->rank(),
              "], send fd: ",
              fd,
              ", sock: ",
              sock,
              ", mem_offset: ",
              mem_offset);
}

void ze_handle_exchange_entry::recvmsg_call(int sock, int& fd, size_t& mem_offset) {
    recvmsg_fd(sock, fd, mem_offset);
    LOG_DEBUG(
        "recv: rank[", rank, "], got fd: ", fd, ", sock: ", sock, ", mem_offset: ", mem_offset);
}

void ze_handle_exchange_entry::get_fd_from_handle(const ze_ipc_mem_handle_t* handle,
                                                  int* fd) noexcept {
    memcpy(fd, static_cast<const void*>(handle), sizeof(*fd));
}

void ze_handle_exchange_entry::get_handle_from_fd(const int* fd,
                                                  ze_ipc_mem_handle_t* handle) noexcept {
    memcpy(handle, static_cast<const void*>(fd), sizeof(*fd));
}

ze_handle_exchange_entry::mem_info_t ze_handle_exchange_entry::get_mem_info(const void* ptr) {
    void* base_ptr{};
    size_t alloc_size{};
    sched->get_memory().handle_manager.get_address_range(ptr, &base_ptr, &alloc_size);
    return { base_ptr, static_cast<const char*>(ptr) - static_cast<char*>(base_ptr) };
}

void ze_handle_exchange_entry::unlink_sockets() {
    unlink(left_peer_socket_name.c_str());
}

void ze_handle_exchange_entry::close_sockets() {
    close(left_peer_connect_socket);
    close(left_peer_socket);
    close(right_peer_socket);
}
