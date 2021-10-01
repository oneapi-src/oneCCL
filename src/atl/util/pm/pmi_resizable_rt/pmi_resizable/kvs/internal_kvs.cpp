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
#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "internal_kvs.h"
#include "internal_kvs_server.hpp"
#include "common/log/log.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/request_wrappers_k8s.hpp"

size_t internal_kvs::kvs_set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_PUT;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(request.val, kvs_val, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: put_key_value");

    return 0;
}

size_t internal_kvs::kvs_set_size(const char* kvs_name, const char* kvs_key, const char* kvs_val) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_SET_SIZE;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(request.val, kvs_val, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: set_size");

    return 0;
}

size_t internal_kvs::kvs_barrier_register(const char* kvs_name,
                                          const char* kvs_key,
                                          const char* kvs_val) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_BARRIER_REGISTER;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(request.val, kvs_val, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: barrier_register");

    return 0;
}

void internal_kvs::kvs_barrier(const char* kvs_name, const char* kvs_key, const char* kvs_val) {
    kvs_request_t request;
    int is_done;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_BARRIER;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(request.val, kvs_val, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(
        write, client_op_sock, &request, sizeof(request), client_memory_mutex, "client: barrier");

    DO_RW_OP(read,
             client_op_sock,
             &is_done,
             sizeof(is_done),
             client_memory_mutex,
             "client: barrier read data");
}

size_t internal_kvs::kvs_remove_name_key(const char* kvs_name, const char* kvs_key) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_REMOVE;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: remove_key");

    return 0;
}

size_t internal_kvs::kvs_register(const char* kvs_name, const char* kvs_key, char* kvs_val) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_INTERNAL_REGISTER;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    kvs_str_copy(request.val, kvs_val, MAX_KVS_VAL_LENGTH);
    memset(kvs_val, 0, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(
        write, client_op_sock, &request, sizeof(request), client_memory_mutex, "client: register");

    DO_RW_OP(read,
             client_op_sock,
             &request,
             sizeof(request),
             client_memory_mutex,
             "client: register read data");
    kvs_str_copy(kvs_val, request.val, MAX_KVS_VAL_LENGTH);

    return strlen(kvs_val);
}

size_t internal_kvs::kvs_get_value_by_name_key(const char* kvs_name,
                                               const char* kvs_key,
                                               char* kvs_val) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_VAL;
    size_t is_exist = 0;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    kvs_str_copy(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    memset(kvs_val, 0, MAX_KVS_VAL_LENGTH);

    DO_RW_OP(
        write, client_op_sock, &request, sizeof(request), client_memory_mutex, "client: get_value");

    DO_RW_OP(read,
             client_op_sock,
             &is_exist,
             sizeof(is_exist),
             client_memory_mutex,
             "client: get_value is_exist");

    if (is_exist) {
        DO_RW_OP(read,
                 client_op_sock,
                 &request,
                 sizeof(request),
                 client_memory_mutex,
                 "client: get_value read data");
        kvs_str_copy(kvs_val, request.val, MAX_KVS_VAL_LENGTH);
    }

    return strlen(kvs_val);
}

size_t internal_kvs::kvs_get_count_names(const char* kvs_name) {
    size_t count_names = 0;
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_COUNT;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: get_count");

    DO_RW_OP(read,
             client_op_sock,
             &count_names,
             sizeof(size_t),
             client_memory_mutex,
             "client: get_count read data");

    return count_names;
}

size_t internal_kvs::kvs_get_keys_values_by_name(const char* kvs_name,
                                                 char*** kvs_keys,
                                                 char*** kvs_values) {
    size_t count = 0;
    size_t i;
    kvs_request_t request;
    kvs_request_t* answers;

    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_KEYS_VALUES;
    kvs_str_copy(request.name, kvs_name, MAX_KVS_NAME_LENGTH);

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: get_keys_values");

    DO_RW_OP(read,
             client_op_sock,
             &count,
             sizeof(size_t),
             client_memory_mutex,
             "client: get_keys_values read size");

    if (count == 0)
        return count;

    answers = (kvs_request_t*)calloc(count, sizeof(kvs_request_t));
    DO_RW_OP(read,
             client_op_sock,
             answers,
             sizeof(kvs_request_t) * count,
             client_memory_mutex,
             "client: get_keys_values read data");
    if (kvs_keys != nullptr) {
        if (*kvs_keys != nullptr)
            free(*kvs_keys);

        *kvs_keys = (char**)calloc(count, sizeof(char*));
        if ((*kvs_keys) == nullptr) {
            LOG_ERROR("Memory allocation failed");
            exit(1);
        }
        for (i = 0; i < count; i++) {
            (*kvs_keys)[i] = (char*)calloc(MAX_KVS_KEY_LENGTH, sizeof(char));
            kvs_str_copy((*kvs_keys)[i], answers[i].key, MAX_KVS_KEY_LENGTH);
        }
    }
    if (kvs_values != nullptr) {
        if (*kvs_values != nullptr)
            free(*kvs_values);

        *kvs_values = (char**)calloc(count, sizeof(char*));
        if ((*kvs_values) == nullptr) {
            LOG_ERROR("Memory allocation failed");
            exit(1);
        }
        for (i = 0; i < count; i++) {
            (*kvs_values)[i] = (char*)calloc(MAX_KVS_VAL_LENGTH, sizeof(char));
            kvs_str_copy((*kvs_values)[i], answers[i].val, MAX_KVS_VAL_LENGTH);
        }
    }

    free(answers);

    return count;
}

size_t internal_kvs::kvs_get_replica_size(void) {
    size_t replica_size = 0;
    if (ip_getting_mode == IGT_K8S) {
        replica_size = request_k8s_get_replica_size();
    }
    else {
        kvs_request_t request;
        memset(&request, 0, sizeof(kvs_request_t));
        request.mode = AM_GET_REPLICA;

        DO_RW_OP(write,
                 client_op_sock,
                 &request,
                 sizeof(kvs_request_t),
                 client_memory_mutex,
                 "client: get_replica");

        DO_RW_OP(read,
                 client_op_sock,
                 &replica_size,
                 sizeof(size_t),
                 client_memory_mutex,
                 "client: get_replica read size");
    }
    return replica_size;
}

size_t internal_kvs::init_main_server_by_k8s() {
    char port_str[MAX_KVS_VAL_LENGTH];
    request_k8s_kvs_init();

    SET_STR(port_str, INT_STR_SIZE, "%d", local_server_address->get_sin_port());

    request_k8s_kvs_get_master(local_host_ip, main_host_ip, port_str);

    main_port = safe_strtol(port_str, nullptr, 10);
    main_server_address->set_sin_port(main_port);
    main_server_address->set_sin_addr(main_host_ip);
    return 0;
}

size_t internal_kvs::init_main_server_by_env() {
    char* port = nullptr;

    const char* tmp_host_ip = (!server_address.empty()) ? server_address.c_str()
                                                        : std::getenv(CCL_KVS_IP_PORT_ENV.c_str());

    if (tmp_host_ip == nullptr) {
        LOG_ERROR("specify ", CCL_KVS_IP_PORT_ENV);
        return 1;
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, tmp_host_ip, CCL_IP_LEN);
    if ((port = strstr(main_host_ip, "_")) == nullptr) {
        if ((port = strstr(main_host_ip, ":")) == nullptr) {
            LOG_ERROR("set ", CCL_KVS_IP_PORT_ENV, " in format <ip>_<port>\n");
            return 1;
        }
    }
    port[0] = '\0';
    port++;

    main_port = safe_strtol(port, nullptr, 10);
    main_server_address->set_sin_port(main_port);
    main_server_address->set_sin_addr(main_host_ip);
    return 0;
}

size_t internal_kvs::init_main_server_by_string(const char* main_addr) {
    char* port = nullptr;
    local_server_address->set_sin_addr(local_host_ip);

    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("init_main_server_by_string: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    size_t sin_port = local_server_address->get_sin_port();
    while (bind(server_listen_sock,
                local_server_address->get_sock_addr_ptr(),
                local_server_address->size()) < 0) {
        sin_port++;
        local_server_address->set_sin_port(sin_port);
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, main_addr, CCL_IP_LEN);

    if ((port = strstr(main_host_ip, "_")) == nullptr) {
        if ((port = strstr(main_host_ip, ":")) == nullptr) {
            LOG_ERROR(
                "init_main_server_by_string: set ", CCL_KVS_IP_PORT_ENV, " in format <ip>_<port>");
            return 1;
        }
    }
    port[0] = '\0';
    port++;

    main_port = safe_strtol(port, nullptr, 10);
    main_server_address->set_sin_port(main_port);
    main_server_address->set_sin_addr(main_host_ip);
    return 0;
}

int internal_kvs::fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    bool is_supported_iface = false;
    if (getifaddrs(&ifaddr) < 0) {
        LOG_ERROR("fill_local_host_ip: can not get host IP");
        return -1;
    }

    const char iface_name[] = "lo";
    char* iface_name_env = std::getenv(CCL_KVS_IFACE_ENV.c_str());
    local_host_ips.clear();
    local_host_ipv6s.clear();
    local_host_ipv4s.clear();

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;
        if (iface_name_env) {
            is_supported_iface = strstr(ifa->ifa_name, iface_name_env);
        }
        else {
            is_supported_iface = strstr(ifa->ifa_name, iface_name) == NULL;
        }
        if (is_supported_iface) {
            family = ifa->ifa_addr->sa_family;
            if (family == AF_INET || family == AF_INET6) {
                memset(local_ip, 0, CCL_IP_LEN);
                int res = getnameinfo(
                    ifa->ifa_addr,
                    (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6),
                    local_ip,
                    CCL_IP_LEN,
                    NULL,
                    0,
                    NI_NUMERICHOST);
                if (res != 0) {
                    std::string s("fill_local_host_ip: getnameinfo error > ");
                    s.append(gai_strerror(res));
                    LOG_ERROR(s.c_str());
                    return -1;
                }

                local_host_ips.push_back(local_ip);
                if (family == AF_INET6) {
                    char* scope_id_ptr = nullptr;
                    if ((scope_id_ptr = strchr(local_ip, SCOPE_ID_DELIM))) {
                        uint32_t scope_id = ((struct sockaddr_in6*)(ifa->ifa_addr))->sin6_scope_id;
                        sprintf(scope_id_ptr + 1, "%u", scope_id);
                    }
                    local_host_ipv6s.push_back(local_ip);
                }
                else {
                    local_host_ipv4s.push_back(local_ip);
                }
            }
        }
    }
    if (local_host_ips.empty()) {
        LOG_ERROR("fill_local_host_ip: can't find interface ",
                  iface_name_env ? iface_name_env : "",
                  " to get host IP");
        return -1;
    }

    memset(local_host_ip, 0, CCL_IP_LEN);

    char* kvs_prefer_ipv6 = std::getenv(CCL_KVS_PREFER_IPV6_ENV.c_str());
    size_t is_kvs_prefer_ipv6 = kvs_prefer_ipv6 ? safe_strtol(kvs_prefer_ipv6, nullptr, 10) : 0;

    if (is_kvs_prefer_ipv6) {
        if (!local_host_ipv6s.empty()) {
            address_family = AF_INET6;
        }
        else {
            LOG_WARN("ipv6 addresses are not found, fallback to ipv4");
            address_family = AF_INET;
        }
    }
    else {
        address_family = (!local_host_ipv4s.empty()) ? AF_INET : AF_INET6;
    }

    if (address_family == AF_INET) {
        main_server_address = std::shared_ptr<isockaddr>(new sockaddr_v4());
        local_server_address = std::shared_ptr<isockaddr>(new sockaddr_v4());
        kvs_str_copy(local_host_ip, local_host_ipv4s.front().c_str(), CCL_IP_LEN);
    }
    else {
        main_server_address = std::shared_ptr<isockaddr>(new sockaddr_v6());
        local_server_address = std::shared_ptr<isockaddr>(new sockaddr_v6());
        kvs_str_copy(local_host_ip, local_host_ipv6s.front().c_str(), CCL_IP_LEN);
    }
    LOG_DEBUG("use ", address_family == AF_INET ? "ipv4" : "ipv6", ": ", local_host_ip);

    freeifaddrs(ifaddr);
    return 0;
}

size_t internal_kvs::kvs_main_server_address_reserve(char* main_address) {
    if (!server_address.empty())
        return 0;

    if (fill_local_host_ip() < 0) {
        LOG_ERROR("reserve_main_address: failed to get local host IP");
        exit(EXIT_FAILURE);
    }

    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("reserve_main_address: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    main_server_address->set_sin_addr(local_host_ip);
    local_server_address->set_sin_addr(local_host_ip);
    size_t sin_port = main_server_address->get_sin_port();

    while (bind(server_listen_sock,
                main_server_address->get_sock_addr_ptr(),
                main_server_address->size()) < 0) {
        sin_port++;
        main_server_address->set_sin_port(sin_port);
    }
    local_server_address->set_sin_port(main_server_address->get_sin_port());

    memset(main_address, '\0', CCL_IP_LEN);
    snprintf(main_address, CCL_IP_LEN, "%s", local_host_ip);
    snprintf(main_address + strlen(local_host_ip),
             INT_STR_SIZE + 1,
             "_%d",
             main_server_address->get_sin_port());

    return 0;
}

size_t internal_kvs::init_main_server_address(const char* main_addr) {
    char* ip_getting_type = std::getenv(CCL_KVS_IP_EXCHANGE_ENV.c_str());

    if (local_host_ips.empty()) {
        if (fill_local_host_ip() < 0) {
            LOG_ERROR("init_main_server_address: failed to get local host ip");
            exit(EXIT_FAILURE);
        }
    }

    if (ip_getting_type) {
        if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_ENV.c_str())) {
            ip_getting_mode = IGT_ENV;
        }
        else if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_K8S.c_str())) {
            ip_getting_mode = IGT_K8S;
        }
        else {
            LOG_ERROR("unknown ", CCL_KVS_IP_EXCHANGE_ENV, ": ", ip_getting_type);
            return 1;
        }
    }

    if (server_address.empty()) {
        if (main_addr != NULL) {
            ip_getting_mode = IGT_ENV;
            if (server_listen_sock == 0)
                init_main_server_by_string(main_addr);
            return 0;
        }
    }
    else {
        ip_getting_mode = IGT_ENV;
    }

    local_server_address->set_sin_addr(local_host_ip);

    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        ;
        LOG_ERROR("init_main_server_address: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    switch (ip_getting_mode) {
        case IGT_K8S: {
            size_t sin_port = local_server_address->get_sin_port();
            while (bind(server_listen_sock,
                        local_server_address->get_sock_addr_ptr(),
                        local_server_address->size()) < 0) {
                sin_port++;
                local_server_address->set_sin_port(sin_port);
            }

            local_port = local_server_address->get_sin_port();
            return init_main_server_by_k8s();
        }
        case IGT_ENV: {
            int res = init_main_server_by_env();
            int is_master_node = 0;

            if (res)
                return res;

            if (strstr(local_host_ip, main_host_ip)) {
                is_master_node = 1;
            }
            else {
                auto main_node_ip =
                    std::find(local_host_ips.begin(), local_host_ips.end(), main_host_ip);
                if (main_node_ip != local_host_ips.end()) {
                    is_master_node = 1;
                    memset(local_host_ip, 0, CCL_IP_LEN);
                    kvs_str_copy_known_sizes(local_host_ip, main_host_ip, CCL_IP_LEN);
                    local_server_address->set_sin_addr(local_host_ip);
                }
            }
            if (is_master_node) {
                if (bind(server_listen_sock,
                         main_server_address->get_sock_addr_ptr(),
                         main_server_address->size()) < 0) {
                    LOG_INFO("port [", main_server_address->get_sin_port(), "] is busy");
                    local_port = local_server_address->get_sin_port();
                    while (bind(server_listen_sock,
                                local_server_address->get_sock_addr_ptr(),
                                local_server_address->size()) < 0) {
                        local_port++;
                        local_server_address->set_sin_port(local_port);
                    }
                }
                else {
                    local_port = main_server_address->get_sin_port();
                }
            }
            else {
                local_port = local_server_address->get_sin_port();
                while (bind(server_listen_sock,
                            local_server_address->get_sock_addr_ptr(),
                            local_server_address->size()) < 0) {
                    local_port++;
                    local_server_address->set_sin_port(local_port);
                }
            }

            return res;
        }
        default: {
            LOG_ERROR("unknown ", CCL_KVS_IP_EXCHANGE_ENV);
            return 1;
        }
    }
}

size_t internal_kvs::kvs_init(const char* main_addr) {
    int err;
    socklen_t len = 0;
    std::shared_ptr<isockaddr> addr;

    time_t start_time;
    time_t connection_time = 0;

    if (init_main_server_address(main_addr)) {
        LOG_ERROR("kvs_init: init main server address error");
        close(client_op_sock);
        close(server_control_sock);
        client_op_sock = 0;
        server_control_sock = 0;
        return 1;
    }

    if (address_family == AF_INET) {
        addr = std::shared_ptr<isockaddr>(new sockaddr_v4());
        addr->set_sin_addr("127.0.0.1");
    }
    else {
        addr = std::shared_ptr<isockaddr>(new sockaddr_v6());
        addr->set_sin_addr("::1");
    }

    if ((client_op_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("kvs_init: client_op_sock init");
        return 1;
    }

    if ((server_control_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("kvs_init: server_control_sock init");
        return 1;
    }

    size_t sin_port = addr->get_sin_port();
    while (bind(server_control_sock, addr->get_sock_addr_ptr(), addr->size()) < 0) {
        sin_port++;
        addr->set_sin_port(sin_port);
    }

    if (listen(server_control_sock, 1) < 0) {
        LOG_ERROR("kvs_init: server_control_sock listen");
        exit(EXIT_FAILURE);
    }

    getsockname(server_control_sock, addr->get_sock_addr_ptr(), &len);
    server_args args;
    args.args = addr;
    args.sock_listener = server_listen_sock;
    err = pthread_create(&kvs_thread, nullptr, kvs_server_init, &args);
    if (err) {
        LOG_ERROR("kvs_init: failed to create kvs server thread, pthread_create returns ", err);
        return 1;
    }

    if ((client_control_sock = accept(server_control_sock, nullptr, nullptr)) < 0) {
        LOG_ERROR("kvs_init: server_control_sock accept");
        exit(EXIT_FAILURE);
    }

    /* Wait connection to master */
    start_time = time(nullptr);
    do {
        err = connect(
            client_op_sock, main_server_address->get_sock_addr_ptr(), main_server_address->size());
        connection_time = time(nullptr) - start_time;
    } while ((err < 0) && (connection_time < CONNECTION_TIMEOUT));

    if (connection_time >= CONNECTION_TIMEOUT) {
        LOG_ERROR("kvs_init: connection error: timeout limit (",
                  connection_time,
                  " > ",
                  CONNECTION_TIMEOUT);
        exit(EXIT_FAILURE);
    }

    if (strstr(main_host_ip, local_host_ip) && local_port == main_port) {
        is_master = 1;
    }
    is_inited = true;

    return 0;
}

size_t internal_kvs::kvs_finalize(void) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));

    close(client_op_sock);
    client_op_sock = 0;
    if (kvs_thread != 0) {
        void* exit_code;
        bool is_stop;
        int err;
        request.mode = AM_FINALIZE;
        DO_RW_OP(write,
                 client_control_sock,
                 &request,
                 sizeof(request),
                 client_memory_mutex,
                 "client: finalize start");
        DO_RW_OP(read,
                 client_control_sock,
                 &is_stop,
                 sizeof(is_stop),
                 client_memory_mutex,
                 "client: finalize complete");

        err = pthread_join(kvs_thread, &exit_code);
        if (err) {
            LOG_ERROR("kvs_finalize: failed to stop kvs server thread, pthread_join returns ", err);
        }

        kvs_thread = 0;

        close(client_control_sock);
        close(server_control_sock);

        client_control_sock = 0;
        server_control_sock = 0;
    }

    if (ip_getting_mode == IGT_K8S)
        request_k8s_kvs_finalize(is_master);
    is_inited = false;

    return 0;
}

internal_kvs::~internal_kvs() {
    if (is_inited)
        kvs_finalize();
}

void sockaddr_v4::set_sin_addr(const char* src) {
    int ret = inet_pton(addr.sin_family, src, &(addr.sin_addr));
    if (ret <= 0) {
        if (ret == 0) {
            LOG_ERROR(
                "inet_pton error - invalid network address, af: ", addr.sin_family, ", src: ", src);
        }
        else if (ret < 0) {
            LOG_ERROR("inet_pton error - af: ",
                      addr.sin_family,
                      ", src: ",
                      src,
                      ", error: ",
                      strerror(errno));
        }
        exit(1);
    }
}

void sockaddr_v6::set_sin_addr(const char* src) {
    char src_copy[internal_kvs::CCL_IP_LEN] = { 0 };
    kvs_str_copy(src_copy, src, internal_kvs::CCL_IP_LEN);

    char* scope_id_ptr = nullptr;
    if ((scope_id_ptr = strchr(src_copy, internal_kvs::SCOPE_ID_DELIM))) {
        addr.sin6_scope_id = safe_strtol(scope_id_ptr + 1, nullptr, 10);
        *scope_id_ptr = '\0';
    }

    int ret = inet_pton(addr.sin6_family, src_copy, &(addr.sin6_addr));
    if (ret <= 0) {
        if (ret == 0) {
            LOG_ERROR("inet_pton error - invalid network address, af: ",
                      addr.sin6_family,
                      ", src_copy: ",
                      src_copy);
        }
        else if (ret < 0) {
            LOG_ERROR("inet_pton error - af: ",
                      addr.sin6_family,
                      ", src_copy: ",
                      src_copy,
                      ", error: ",
                      strerror(errno));
        }
        exit(1);
    }

    LOG_DEBUG("addr: ", src_copy, ", scope_id: ", addr.sin6_scope_id);
}
