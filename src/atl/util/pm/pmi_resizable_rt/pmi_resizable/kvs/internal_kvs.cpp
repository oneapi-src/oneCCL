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

    SET_STR(port_str, INT_STR_SIZE, "%d", local_server_address.sin_port);

    request_k8s_kvs_get_master(local_host_ip, main_host_ip, port_str);

    main_port = safe_strtol(port_str, nullptr, 10);
    main_server_address.sin_port = main_port;
    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        LOG_ERROR("invalid address/ address not supported: ", main_host_ip);
        return 1;
    }
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
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        LOG_ERROR("ivalid address / address not supported: ", main_host_ip);
        return 1;
    }
    return 0;
}

size_t internal_kvs::init_main_server_by_string(const char* main_addr) {
    char* port = nullptr;
    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = default_start_port;

    main_server_address.sin_family = AF_INET;

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("init_main_server_by_string: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    while (bind(server_listen_sock,
                (const struct sockaddr*)&local_server_address,
                sizeof(local_server_address)) < 0) {
        local_server_address.sin_port++;
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
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        LOG_ERROR("init_main_server_by_string: invalid address / address not supported: ",
                  main_host_ip);
        LOG_ERROR("init_main_server_by_string: inet_pton");
        return 1;
    }
    return 0;
}

int internal_kvs::fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    if (getifaddrs(&ifaddr) < 0) {
        LOG_ERROR("fill_local_host_ip: can not get host IP");
        return -1;
    }

    const char iface_name[] = "lo";
    local_host_ips.clear();

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;
        if (strstr(ifa->ifa_name, iface_name) == NULL) {
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
            }
        }
    }
    if (local_host_ips.empty()) {
        LOG_ERROR("fill_local_host_ip: can't find interface to get host IP");
        return -1;
    }

    memset(local_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(local_host_ip, local_host_ips.front().c_str(), CCL_IP_LEN);

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

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("reserve_main_address: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    main_server_address.sin_family = AF_INET;
    main_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    main_server_address.sin_port = default_start_port;
    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);

    while (bind(server_listen_sock,
                (const struct sockaddr*)&main_server_address,
                sizeof(main_server_address)) < 0) {
        main_server_address.sin_port++;
    }
    local_server_address.sin_port = main_server_address.sin_port;

    memset(main_address, '\0', CCL_IP_LEN);
    snprintf(main_address, CCL_IP_LEN, "%s", local_host_ip);
    snprintf(main_address + strlen(local_host_ip),
             INT_STR_SIZE + 1,
             "_%d",
             main_server_address.sin_port);

    return 0;
}

size_t internal_kvs::init_main_server_address(const char* main_addr) {
    char* ip_getting_type = std::getenv(CCL_KVS_IP_EXCHANGE_ENV.c_str());

    memset(local_host_ip, 0, CCL_IP_LEN);
    if (fill_local_host_ip() < 0) {
        LOG_ERROR("init_main_server_address: failed to get local host IP");
        exit(EXIT_FAILURE);
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

    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = default_start_port;

    main_server_address.sin_family = AF_INET;

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        ;
        LOG_ERROR("init_main_server_address: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    switch (ip_getting_mode) {
        case IGT_K8S: {
            while (bind(server_listen_sock,
                        (const struct sockaddr*)&local_server_address,
                        sizeof(local_server_address)) < 0) {
                local_server_address.sin_port++;
            }

            local_port = local_server_address.sin_port;
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
                    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
                }
            }
            if (is_master_node) {
                if (bind(server_listen_sock,
                         (const struct sockaddr*)&main_server_address,
                         sizeof(main_server_address)) < 0) {
                    printf("port [%d] is busy\n", main_server_address.sin_port);
                    while (bind(server_listen_sock,
                                (const struct sockaddr*)&local_server_address,
                                sizeof(local_server_address)) < 0) {
                        local_server_address.sin_port++;
                    }
                    local_port = local_server_address.sin_port;
                }
                else {
                    local_port = main_server_address.sin_port;
                }
            }
            else {
                while (bind(server_listen_sock,
                            (const struct sockaddr*)&local_server_address,
                            sizeof(local_server_address)) < 0) {
                    local_server_address.sin_port++;
                }
                local_port = local_server_address.sin_port;
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
    struct sockaddr_in addr;
    time_t start_time;
    time_t connection_time = 0;
    memset(&addr, 0, sizeof(struct sockaddr_in));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = default_start_port;

    if ((client_op_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("kvs_init: client_op_sock init");
        return 1;
    }

    if ((server_control_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("kvs_init: server_control_sock init");
        return 1;
    }

    if (init_main_server_address(main_addr)) {
        LOG_ERROR("kvs_init: init main server address error");
        close(client_op_sock);
        close(server_control_sock);
        client_op_sock = 0;
        server_control_sock = 0;
        return 1;
    }

    while (bind(server_control_sock, (const struct sockaddr*)&addr, sizeof(addr)) < 0) {
        addr.sin_port++;
    }

    if (listen(server_control_sock, 1) < 0) {
        LOG_ERROR("kvs_init: server_control_sock listen");
        exit(EXIT_FAILURE);
    }

    getsockname(server_control_sock, (struct sockaddr*)&addr, &len);
    server_args args;
    args.args = &addr;
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
            client_op_sock, (struct sockaddr*)&main_server_address, sizeof(main_server_address));
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
