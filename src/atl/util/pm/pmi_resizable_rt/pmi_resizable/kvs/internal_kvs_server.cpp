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
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <list>
#include <map>
#include <mutex>
#include <poll.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <memory>

#include "common/log/log.hpp"
#include "internal_kvs_server.hpp"

class server {
public:
    server() = default;
    kvs_status_t run(void*);
    kvs_status_t check_finalize(bool& to_finalize);
    kvs_status_t make_client_request(int& socket);
    kvs_status_t try_to_connect_new();

private:
    struct clients_info {
        int socket;
        bool in_barrier;
        clients_info(int sock, bool barrier) : socket(sock), in_barrier(barrier) {}
    };
    struct proc_info {
        std::string rank;
        size_t rank_count;
        std::string thread_id;
    };
    struct socket_info {
        int socket;
        std::string proc_id;
        proc_info process_info;
    };
    struct comm_info {
        size_t global_size = 0;
        size_t local_size = 0;
        std::list<socket_info> sockets;
        std::map<std::string, std::list<proc_info>> processes;
    };
    struct barrier_info {
        size_t global_size = 0;
        size_t local_size = 0;
        std::list<std::shared_ptr<clients_info>> clients;
    };
    enum fd_indexes { FDI_LISTENER = 0, FDI_CONTROL = 1, FDI_LAST = 2 };

    kvs_request_t request{};
    size_t count{};
    size_t client_count = 0;
    const size_t max_client_queue_size = 300;
    const size_t client_count_increase = 300;
    std::map<std::string, barrier_info> barriers;
    std::map<std::string, comm_info> communicators;
    int ret = 0;
    std::mutex server_memory_mutex;
    std::map<std::string, std::map<std::string, std::string>> requests;
    const int free_socket = -1;
    std::vector<struct pollfd> poll_fds;

    sa_family_t address_family{ AF_UNSPEC };
};

kvs_status_t server::try_to_connect_new() {
    if (poll_fds[FDI_LISTENER].revents != 0) {
        std::shared_ptr<isockaddr> addr;

        if (address_family == AF_INET) {
            addr = std::shared_ptr<isockaddr>(new sockaddr_v4());
        }
        else {
            addr = std::shared_ptr<isockaddr>(new sockaddr_v6());
        }

        int new_socket;
        socklen_t peer_addr_size = addr->size();
        if ((new_socket = accept(poll_fds[FDI_LISTENER].fd,
                                 addr->get_sock_addr_ptr(),
                                 (socklen_t*)&peer_addr_size)) < 0) {
            LOG_ERROR("server_listen_sock accept, %s", strerror(errno));
            return KVS_STATUS_FAILURE;
        }
        for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
            if (poll_fds[i].fd == free_socket) {
                poll_fds[i].fd = new_socket;
                break;
            }
        }
        client_count++;
        if (poll_fds.size() - FDI_LAST == client_count) {
            size_t old_size = poll_fds.size();
            poll_fds.resize(old_size + client_count_increase);
            for (size_t i = old_size; i < poll_fds.size(); i++) {
                poll_fds[i].fd = free_socket;
                poll_fds[i].events = POLLIN;
            }
        }
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::make_client_request(int& socket) {
    DO_RW_OP_1(
        read, socket, &request, sizeof(kvs_request_t), ret, "server: get command from client");
    if (ret == 0) {
        close(socket);
        socket = free_socket;
        client_count--;
        return KVS_STATUS_SUCCESS;
    }

    switch (request.mode) {
        case AM_PUT: {
            auto& req = requests[request.name];
            req[request.key] = request.val;
            break;
        }
        case AM_REMOVE: {
            requests[request.name].erase(request.key);
            break;
        }
        case AM_GET_VAL: {
            count = 0;
            auto it_name = requests.find(request.name);
            if (it_name != requests.end()) {
                auto it_key = it_name->second.find(request.key);
                if (it_key != it_name->second.end()) {
                    count = 1;
                    DO_RW_OP(write,
                             socket,
                             &count,
                             sizeof(size_t),
                             server_memory_mutex,
                             "server: get_value write size");

                    kvs_str_copy(request.val, it_key->second.c_str(), MAX_KVS_VAL_LENGTH);

                    DO_RW_OP(write,
                             socket,
                             &request,
                             sizeof(kvs_request_t),
                             server_memory_mutex,
                             "server: get_value write data");
                    break;
                }
            }
            DO_RW_OP(write,
                     socket,
                     &count,
                     sizeof(size_t),
                     server_memory_mutex,
                     "server: get_value write size");
            break;
        }
        case AM_GET_COUNT: {
            count = 0;
            auto it = requests.find(request.name);
            if (it != requests.end()) {
                count = it->second.size();
            }
            DO_RW_OP(
                write, socket, &count, sizeof(size_t), server_memory_mutex, "server: get_count");
            break;
        }
        case AM_GET_REPLICA: {
            char* replica_size_str = getenv(CCL_WORLD_SIZE_ENV);
            count = client_count;
            if (replica_size_str != nullptr) {
                KVS_CHECK_STATUS(safe_strtol(replica_size_str, count), "failed to convert count");
            }
            DO_RW_OP(
                write, socket, &count, sizeof(size_t), server_memory_mutex, "server: get_replica");
            break;
        }
        case AM_GET_KEYS_VALUES: {
            size_t j = 0;
            std::vector<kvs_request_t> answers;
            count = 0;
            auto it_name = requests.find(request.name);
            if (it_name != requests.end()) {
                count = it_name->second.size();
            }

            DO_RW_OP(write,
                     socket,
                     &count,
                     sizeof(size_t),
                     server_memory_mutex,
                     "server: get_keys_values write size");
            if (count == 0)
                break;

            answers.resize(count);

            for (auto it_keys : it_name->second) {
                kvs_str_copy_known_sizes(answers[j].name, request.name, MAX_KVS_NAME_LENGTH);
                kvs_str_copy(answers[j].key, it_keys.first.c_str(), MAX_KVS_KEY_LENGTH);
                kvs_str_copy(answers[j].val, it_keys.second.c_str(), MAX_KVS_VAL_LENGTH);
                j++;
            }

            DO_RW_OP(write,
                     socket,
                     answers.data(),
                     sizeof(kvs_request_t) * count,
                     server_memory_mutex,
                     "server: get_keys_values write data");

            break;
        }
        case AM_BARRIER: {
            auto& barrier_list = barriers[request.name];
            auto& clients = barrier_list.clients;
            auto client_it = std::find_if(std::begin(clients),
                                          std::end(clients),
                                          [&](std::shared_ptr<server::clients_info> v) {
                                              return v->socket == socket;
                                          });
            if (client_it == clients.end()) {
                // TODO: Look deeper to fix this error
                LOG_ERROR("Server error: Unregister Barrier request!");
                return KVS_STATUS_FAILURE;
            }
            auto client_inf = client_it->get();
            client_inf->in_barrier = true;

            if (barrier_list.global_size == barrier_list.local_size) {
                bool is_barrier_stop = true;
                for (const auto& client : clients) {
                    if (!client->in_barrier) {
                        is_barrier_stop = false;
                        break;
                    }
                }
                if (is_barrier_stop) {
                    int is_done = 1;
                    for (const auto& client : clients) {
                        client->in_barrier = false;
                        DO_RW_OP(write,
                                 client->socket,
                                 &is_done,
                                 sizeof(is_done),
                                 server_memory_mutex,
                                 "server: barrier");
                    }
                }
            }
            break;
        }
        case AM_BARRIER_REGISTER: {
            char* glob_size = request.val;
            char* local_size = strstr(glob_size, "_");
            auto& barrier = barriers[request.name];
            if (local_size == nullptr) {
                barrier.local_size++;
            }
            else {
                local_size[0] = '\0';
                local_size++;
                size_t local_size_tmp;

                KVS_CHECK_STATUS(safe_strtol(local_size, local_size_tmp),
                                 "failed to convert local_size");
                barrier.local_size += local_size_tmp;
            }
            KVS_CHECK_STATUS(safe_strtol(glob_size, barrier.global_size),
                             "failed to convert global_size");

            barrier.clients.push_back(
                std::shared_ptr<clients_info>(new clients_info(socket, false)));
            break;
        }
        case AM_SET_SIZE: {
            char* glob_size = request.val;
            KVS_CHECK_STATUS(safe_strtol(glob_size, communicators[request.key].global_size),
                             "failed to convert global_size");

            break;
        }
        case AM_INTERNAL_REGISTER: {
            auto& communicator = communicators[request.key];
            char* rank_count_str = request.val;
            char* rank = strstr(rank_count_str, "_");
            rank[0] = '\0';
            rank++;
            char* proc_id = strstr(rank, "_");
            proc_id[0] = '\0';
            proc_id++;
            char* thread_id = strstr(proc_id, "_");
            thread_id[0] = '\0';
            thread_id++;
            size_t rank_count;
            KVS_CHECK_STATUS(safe_strtol(rank_count_str, rank_count),
                             "failed to convert rank_count");
            communicators[request.key].local_size += rank_count;
            socket_info sock_info{ socket, proc_id, { rank, rank_count, thread_id } };
            communicator.processes[proc_id].push_back(sock_info.process_info);
            communicator.sockets.push_back(sock_info);
            if (communicator.local_size == communicator.global_size) {
                std::string proc_count_str = std::to_string(communicator.processes.size());
                for (auto& it : communicator.processes) {
                    it.second.sort([](proc_info a, proc_info b) {
                        return a.rank < b.rank;
                    });
                }
                for (auto& it : communicator.sockets) {
                    std::string thread_num;
                    int i = 0;
                    size_t proc_rank_count = 0;
                    auto process_info = communicator.processes[it.proc_id];
                    std::string threads_count = std::to_string(process_info.size());
                    for (auto& proc_info_it : process_info) {
                        if (it.process_info.rank == proc_info_it.rank) {
                            thread_num = std::to_string(i);
                            break;
                        }
                        i++;
                    }
                    for (auto& proc_info_it : process_info) {
                        proc_rank_count += proc_info_it.rank_count;
                    }
                    memset(request.val, 0, MAX_KVS_VAL_LENGTH);
                    /*return string: %PROC_COUNT%_%RANK_NUM%_%PROCESS_RANK_COUNT%_%THREADS_COUNT%_%THREAD_NUM% */
                    snprintf(request.val,
                             MAX_KVS_VAL_LENGTH,
                             "%s_%s_%s_%s_%s",
                             proc_count_str.c_str(),
                             it.process_info.rank.c_str(),
                             std::to_string(proc_rank_count).c_str(),
                             threads_count.c_str(),
                             thread_num.c_str());
                    DO_RW_OP(write,
                             it.socket,
                             &request,
                             sizeof(kvs_request_t),
                             server_memory_mutex,
                             "server: register write data");
                }
            }
            break;
        }
        default: {
            if (request.name[0] == '\0')
                return KVS_STATUS_SUCCESS;
            LOG_ERROR("unknown request mode - %d.\n", request.mode);
            return KVS_STATUS_FAILURE;
        }
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::check_finalize(bool& to_finalize) {
    to_finalize = false;
    if (poll_fds[FDI_CONTROL].revents != 0) {
        DO_RW_OP_1(read,
                   poll_fds[FDI_CONTROL].fd,
                   &request,
                   sizeof(kvs_request_t),
                   ret,
                   "server: get control msg from client");
        if (ret == 0) {
            close(poll_fds[FDI_CONTROL].fd);
            poll_fds[FDI_CONTROL].fd = free_socket;
        }
        if (request.mode != AM_FINALIZE) {
            LOG_ERROR("invalid access mode for local socket\n");
            return KVS_STATUS_FAILURE;
        }
        to_finalize = true;
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::run(void* args) {
    bool should_stop = false;
    int so_reuse = 1;
    poll_fds.resize(client_count_increase);
    for (auto& it : poll_fds) {
        it.fd = free_socket;
        it.events = POLLIN;
    }
    poll_fds[FDI_LISTENER].fd = ((server_args_t*)args)->sock_listener;
    address_family = ((server_args_t*)args)->args->sin_family();

#ifdef SO_REUSEPORT
    setsockopt(poll_fds[FDI_LISTENER].fd, SOL_SOCKET, SO_REUSEPORT, &so_reuse, sizeof(so_reuse));
#else
    setsockopt(poll_fds[FDI_LISTENER].fd, SOL_SOCKET, SO_REUSEADDR, &so_reuse, sizeof(so_reuse));
#endif

    if (listen(poll_fds[FDI_LISTENER].fd, max_client_queue_size) < 0) {
        LOG_ERROR("server_listen_sock listen(%s)", strerror(errno));
        return KVS_STATUS_FAILURE;
    }

    if ((poll_fds[FDI_CONTROL].fd = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_control_sock init(%s)", strerror(errno));
        return KVS_STATUS_FAILURE;
    }

    while (connect(poll_fds[FDI_CONTROL].fd,
                   ((server_args_t*)args)->args->get_sock_addr_ptr(),
                   ((server_args_t*)args)->args->size()) < 0) {
    }
    while (!should_stop || client_count > 0) {
        if (poll(poll_fds.data(), poll_fds.size(), -1) < 0) {
            if (errno != EINTR) {
                LOG_ERROR("poll(%s)", strerror(errno));
                return KVS_STATUS_FAILURE;
            }
            else {
                /* restart select */
                continue;
            }
        }

        for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
            if (poll_fds[i].fd != free_socket && poll_fds[i].revents != 0) {
                KVS_CHECK_STATUS(make_client_request(poll_fds[i].fd), "failed to make request");
            }
        }
        KVS_CHECK_STATUS(try_to_connect_new(), "failed to connect new");
        if (!should_stop) {
            KVS_CHECK_STATUS(check_finalize(should_stop), "failed to check finalize");
        }
    }

    if (poll_fds[FDI_CONTROL].fd != free_socket) {
        DO_RW_OP_1(write,
                   poll_fds[FDI_CONTROL].fd,
                   &should_stop,
                   sizeof(should_stop),
                   ret,
                   "server: send control msg to client");
    }

    close(poll_fds[FDI_CONTROL].fd);
    poll_fds[FDI_CONTROL].fd = free_socket;

    for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
        if (poll_fds[i].fd != free_socket) {
            close(poll_fds[i].fd);
            poll_fds[i].fd = free_socket;
        }
    }

    close(poll_fds[FDI_LISTENER].fd);
    poll_fds[FDI_LISTENER].fd = free_socket;
    return KVS_STATUS_SUCCESS;
}

void* kvs_server_init(void* args) {
    server s;

    if (s.run(args) != KVS_STATUS_SUCCESS) {
        LOG_ERROR("failed");
    }

    return nullptr;
}
