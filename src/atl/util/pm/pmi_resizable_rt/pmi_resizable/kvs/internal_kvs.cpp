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
#include <netinet/in.h>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "util/pm/pmi_resizable_rt/pmi_resizable/helper.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "internal_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/request_wrappers_k8s.hpp"

#define CCL_KVS_IP_PORT_ENV         "CCL_KVS_IP_PORT"
#define CCL_KVS_IP_EXCHANGE_ENV     "CCL_KVS_IP_EXCHANGE"
#define CCL_KVS_IP_EXCHANGE_VAL_ENV "env"
#define CCL_KVS_IP_EXCHANGE_VAL_K8S "k8s"

#define MAX_CLIENT_COUNT   300
#define CONNECTION_TIMEOUT 120

static pthread_t kvs_thread = 0;

static char main_host_ip[CCL_IP_LEN];
char local_host_ip[CCL_IP_LEN];

static size_t main_port;
static size_t local_port;
static size_t is_master = 0;
static std::mutex client_memory_mutex;
static std::mutex server_memory_mutex;

static struct sockaddr_in main_server_address;
static struct sockaddr_in local_server_address;

static int
    client_op_sock; /* used on client side to send commands and to recv result to/from server */
static int
    server_listen_sock; /* used on server side to handle new incoming connect requests from clients */

static int client_control_sock; /* used on client side to control local kvs server */
static int server_control_sock; /* used on server side to be controlled by local client */

typedef enum ip_getting_type {
    IGT_K8S = 0,
    IGT_ENV = 1,
} ip_getting_type_t;

static ip_getting_type_t ip_getting_mode = IGT_K8S;

typedef enum kvs_access_mode {
    AM_CONNECT = -1,
    //    AM_DISCONNECT = 1,
    AM_PUT = 2,
    AM_REMOVE = 3,
    AM_GET_COUNT = 4,
    AM_GET_VAL = 5,
    AM_GET_KEYS_VALUES = 6,
    AM_GET_REPLICA = 7,
    AM_FINALIZE = 8,
} kvs_access_mode_t;

typedef struct kvs_request {
    kvs_access_mode_t mode;
    char name[MAX_KVS_NAME_LENGTH];
    char key[MAX_KVS_KEY_LENGTH];
    char val[MAX_KVS_VAL_LENGTH];
} kvs_request_t;

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
    if (kvs_keys != NULL) {
        if (*kvs_keys != NULL)
            free(*kvs_keys);

        *kvs_keys = (char**)calloc(count, sizeof(char*));
        if ((*kvs_keys) == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (i = 0; i < count; i++) {
            (*kvs_keys)[i] = (char*)calloc(MAX_KVS_KEY_LENGTH, sizeof(char));
            kvs_str_copy((*kvs_keys)[i], answers[i].key, MAX_KVS_KEY_LENGTH);
        }
    }
    if (kvs_values != NULL) {
        if (*kvs_values != NULL)
            free(*kvs_values);

        *kvs_values = (char**)calloc(count, sizeof(char*));
        if ((*kvs_values) == NULL) {
            printf("Memory allocation failed\n");
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

void* kvs_server_init(void* args) {
    struct sockaddr_in addr;
    int server_control_sock;
    kvs_request_t request;
    size_t count;
    size_t client_count = 0;
    int should_stop = 0;
    fd_set read_fds;
    int i, client_op_sockets[MAX_CLIENT_COUNT], max_sd, sd;
    int so_reuse = 1;
    int ret = 0;

#ifdef SO_REUSEPORT
    setsockopt(server_listen_sock, SOL_SOCKET, SO_REUSEPORT, &so_reuse, sizeof(so_reuse));
#else
    setsockopt(server_listen_sock, SOL_SOCKET, SO_REUSEADDR, &so_reuse, sizeof(so_reuse));
#endif

    for (i = 0; i < MAX_CLIENT_COUNT; i++) {
        client_op_sockets[i] = 0;
    }

    if ((server_control_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("server: server_control_sock init");
        exit(EXIT_FAILURE);
    }

    while (connect(server_control_sock, (struct sockaddr*)args, sizeof(addr)) < 0) {
    }

    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;

    if (listen(server_listen_sock, MAX_CLIENT_COUNT) < 0) {
        perror("server: server_listen_sock listen");
        exit(EXIT_FAILURE);
    }

    while (!should_stop || client_count > 1) {
        FD_ZERO(&read_fds);
        FD_SET(server_listen_sock, &read_fds);
        FD_SET(server_control_sock, &read_fds);
        max_sd = server_listen_sock;

        for (i = 0; i < MAX_CLIENT_COUNT; i++) {
            sd = client_op_sockets[i];

            if (sd > 0)
                FD_SET(sd, &read_fds);

            if (sd > max_sd)
                max_sd = sd;
        }

        if (server_control_sock > max_sd)
            max_sd = server_control_sock;

        if (select(max_sd + 1, &read_fds, NULL, NULL, NULL) < 0) {
            if (errno != EINTR) {
                perror("server: select");
                exit(EXIT_FAILURE);
            }
            else {
                /* restart select */
                continue;
            }
        }

        if (FD_ISSET(server_control_sock, &read_fds)) {
            DO_RW_OP_1(read,
                       server_control_sock,
                       &request,
                       sizeof(kvs_request_t),
                       ret,
                       "server: get control msg from client");
            if (ret == 0) {
                close(server_control_sock);
                server_control_sock = 0;
            }
            if (request.mode != AM_FINALIZE) {
                printf("server: invalid access mode for local socket\n");
                exit(EXIT_FAILURE);
            }
            should_stop = 1;
        }

        for (i = 0; i < MAX_CLIENT_COUNT; i++) {
            sd = client_op_sockets[i];
            if (sd == 0)
                continue;

            if (FD_ISSET(sd, &read_fds)) {
                DO_RW_OP_1(read,
                           sd,
                           &request,
                           sizeof(kvs_request_t),
                           ret,
                           "server: get command from client");
                if (ret == 0) {
                    close(sd);
                    client_op_sockets[i] = 0;
                    client_count--;
                    continue;
                }

                switch (request.mode) {
                    case AM_CONNECT: {
                        client_count++;
                        break;
                    }
                    case AM_PUT: {
                        put_key(request.name, request.key, request.val, ST_SERVER);
                        break;
                    }
                    case AM_REMOVE: {
                        remove_val(request.name, request.key, ST_SERVER);
                        break;
                    }
                    case AM_GET_VAL: {
                        count = get_val(request.name, request.key, request.val, ST_SERVER);
                        DO_RW_OP(write,
                                 client_op_sockets[i],
                                 &count,
                                 sizeof(size_t),
                                 server_memory_mutex,
                                 "server: get_value write size");
                        if (count != 0)
                            DO_RW_OP(write,
                                     client_op_sockets[i],
                                     &request,
                                     sizeof(kvs_request_t),
                                     server_memory_mutex,
                                     "server: get_value write data");
                        break;
                    }
                    case AM_GET_COUNT: {
                        count = get_count(request.name, ST_SERVER);
                        DO_RW_OP(write,
                                 client_op_sockets[i],
                                 &count,
                                 sizeof(size_t),
                                 server_memory_mutex,
                                 "server: get_count");
                        break;
                    }
                    case AM_GET_REPLICA: {
                        char* replica_size_str = getenv(CCL_WORLD_SIZE_ENV);
                        count = (replica_size_str != NULL) ? strtol(replica_size_str, NULL, 10)
                                                           : client_count;
                        DO_RW_OP(write,
                                 client_op_sockets[i],
                                 &count,
                                 sizeof(size_t),
                                 server_memory_mutex,
                                 "server: get_replica");
                        break;
                    }
                    case AM_GET_KEYS_VALUES: {
                        char** kvs_keys = NULL;
                        char** kvs_values = NULL;
                        size_t j;
                        kvs_request_t* answers = NULL;

                        count = get_keys_values(request.name, &kvs_keys, &kvs_values, ST_SERVER);

                        DO_RW_OP(write,
                                 client_op_sockets[i],
                                 &count,
                                 sizeof(size_t),
                                 server_memory_mutex,
                                 "server: get_keys_values write size");
                        if (count == 0)
                            break;

                        answers = (kvs_request_t*)calloc(count, sizeof(kvs_request_t));
                        if (answers == NULL) {
                            printf("Memory allocation failed\n");
                            break;
                        }
                        for (j = 0; j < count; j++) {
                            kvs_str_copy(answers[j].name, request.name, MAX_KVS_NAME_LENGTH);
                            kvs_str_copy(answers[j].key, kvs_keys[j], MAX_KVS_KEY_LENGTH);
                            kvs_str_copy(answers[j].val, kvs_values[j], MAX_KVS_VAL_LENGTH);
                        }

                        DO_RW_OP(write,
                                 client_op_sockets[i],
                                 answers,
                                 sizeof(kvs_request_t) * count,
                                 server_memory_mutex,
                                 "server: get_keys_values write data");

                        free(answers);
                        for (j = 0; j < count; j++) {
                            free(kvs_keys[j]);
                            free(kvs_values[j]);
                        }
                        free(kvs_keys);
                        free(kvs_values);
                        break;
                    }
                    default: {
                        if (request.name[0] == '\0')
                            continue;
                        printf("server: unknown request mode - %d.\n", request.mode);
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

        if (FD_ISSET(server_listen_sock, &read_fds)) {
            int new_socket;
            socklen_t peer_addr_size = sizeof(addr);
            if ((new_socket = accept(
                     server_listen_sock, (struct sockaddr*)&addr, (socklen_t*)&peer_addr_size)) <
                0) {
                perror("server: server_listen_sock accept");
                exit(EXIT_FAILURE);
            }
            for (i = 0; i < MAX_CLIENT_COUNT; i++) {
                if (client_op_sockets[i] == 0) {
                    client_op_sockets[i] = new_socket;
                    break;
                }
            }
            if (i >= MAX_CLIENT_COUNT) {
                printf("server: no free sockets\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    kvs_keeper_clear(ST_SERVER);

    if (server_control_sock) {
        DO_RW_OP_1(write,
                   server_control_sock,
                   &should_stop,
                   sizeof(should_stop),
                   ret,
                   "server: send control msg to client");
    }

    close(server_control_sock);
    server_control_sock = 0;

    for (i = 0; i < MAX_CLIENT_COUNT; i++) {
        if (client_op_sockets[i] != 0) {
            close(client_op_sockets[i]);
            client_op_sockets[i] = 0;
        }
    }

    close(server_listen_sock);
    server_listen_sock = 0;

    return NULL;
}

size_t init_main_server_by_k8s(void) {
    char port_str[MAX_KVS_VAL_LENGTH];
    request_k8s_kvs_init();

    SET_STR(port_str, INT_STR_SIZE, "%d", local_server_address.sin_port);

    request_k8s_kvs_get_master(local_host_ip, main_host_ip, port_str);

    main_port = strtol(port_str, NULL, 10);
    main_server_address.sin_port = main_port;
    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        printf("invalid address/ address not supported: %s\n", main_host_ip);
        return 1;
    }
    return 0;
}

size_t init_main_server_by_env(void) {
    char* tmp_host_ip;
    char* port = NULL;

    tmp_host_ip = getenv(CCL_KVS_IP_PORT_ENV);

    if (tmp_host_ip == NULL) {
        printf("specify %s\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, tmp_host_ip, CCL_IP_LEN);
    if ((port = strstr(main_host_ip, "_")) == NULL) {
        printf("set %s in format <ip>_<port>\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }
    port[0] = '\0';
    port++;

    main_port = strtol(port, NULL, 10);
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        printf("ivalid address / address not supported: %s\n", main_host_ip);
        return 1;
    }
    return 0;
}

size_t init_main_server_by_string(const char* main_addr) {
    char* port = NULL;
    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = 1;

    main_server_address.sin_family = AF_INET;

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("init_main_server_by_string: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    while (bind(server_listen_sock,
                (const struct sockaddr*)&local_server_address,
                sizeof(local_server_address)) < 0) {
        local_server_address.sin_port++;
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, main_addr, CCL_IP_LEN);

    if ((port = strstr(main_host_ip, "_")) == NULL) {
        printf("init_main_server_by_string: set %s in format <ip>_<port>\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }
    port[0] = '\0';
    port++;

    main_port = strtol(port, NULL, 10);
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0) {
        printf("init_main_server_by_string: invalid address / address not supported: %s\n",
               main_host_ip);
        perror("init_main_server_by_string: inet_pton");
        return 1;
    }
    return 0;
}

size_t internal_kvs::kvs_main_server_address_reserve(char* main_address) {
    FILE* fp;
    char* additional_local_host_ips;
    if ((fp = popen(GET_IP_CMD, READ_ONLY)) == NULL) {
        perror("reserve_main_address: can not get host IP");
        exit(EXIT_FAILURE);
    }
    CHECK_FGETS(fgets(local_host_ip, CCL_IP_LEN, fp), local_host_ip);
    pclose(fp);

    while (local_host_ip[strlen(local_host_ip) - 1] == '\n' ||
           local_host_ip[strlen(local_host_ip) - 1] == ' ')
        local_host_ip[strlen(local_host_ip) - 1] = NULL_CHAR;
    if ((additional_local_host_ips = strstr(local_host_ip, " ")) != NULL)
        additional_local_host_ips[0] = NULL_CHAR;

    if (strlen(local_host_ip) >= CCL_IP_LEN - INT_STR_SIZE - 1) {
        printf("reserve_main_address: local host IP is too long: %zu, expected: %d\n",
               strlen(local_host_ip),
               CCL_IP_LEN - INT_STR_SIZE - 1);
        exit(EXIT_FAILURE);
    }

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("reserve_main_address: server_listen_sock init");
        exit(EXIT_FAILURE);
    }

    main_server_address.sin_family = AF_INET;
    main_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    main_server_address.sin_port = 1;
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

size_t init_main_server_address(const char* main_addr) {
    char* ip_getting_type = getenv(CCL_KVS_IP_EXCHANGE_ENV);
    FILE* fp;
    char* additional_local_host_ips;

    if ((fp = popen(GET_IP_CMD, READ_ONLY)) == NULL) {
        perror("init_main_server_address: can not get host IP");
        exit(EXIT_FAILURE);
    }

    memset(local_host_ip, 0, CCL_IP_LEN);
    CHECK_FGETS(fgets(local_host_ip, CCL_IP_LEN, fp), local_host_ip);
    pclose(fp);

    while (local_host_ip[strlen(local_host_ip) - 1] == '\n' ||
           local_host_ip[strlen(local_host_ip) - 1] == ' ')
        local_host_ip[strlen(local_host_ip) - 1] = NULL_CHAR;

    if ((additional_local_host_ips = strstr(local_host_ip, " ")) != NULL) {
        additional_local_host_ips[0] = NULL_CHAR;
        additional_local_host_ips++;
    }

    if (ip_getting_type) {
        if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_ENV)) {
            ip_getting_mode = IGT_ENV;
        }
        else if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_K8S)) {
            ip_getting_mode = IGT_K8S;
        }
        else {
            printf("unknown %s: %s\n", CCL_KVS_IP_EXCHANGE_ENV, ip_getting_type);
            return 1;
        }
    }

    if (main_addr != NULL) {
        ip_getting_mode = IGT_ENV;
        if (server_listen_sock == 0)
            init_main_server_by_string(main_addr);
        return 0;
    }

    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = 1;

    main_server_address.sin_family = AF_INET;

    if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        ;
        perror("init_main_server_address: server_listen_sock init");
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
                if (additional_local_host_ips && strstr(additional_local_host_ips, main_host_ip)) {
                    is_master_node = 1;
                    memset(local_host_ip, 0, CCL_IP_LEN);
                    strncpy(local_host_ip, main_host_ip, CCL_IP_LEN);
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
            printf("unknown %s\n", CCL_KVS_IP_EXCHANGE_ENV);
            return 1;
        }
    }
}

size_t internal_kvs::kvs_init(const char* main_addr) {
    int err;
    socklen_t len = 0;
    struct sockaddr_in addr;
    kvs_request_t request;
    time_t start_time;
    time_t connection_time = 0;
    memset(&request, 0, sizeof(kvs_request_t));
    memset(&addr, 0, sizeof(struct sockaddr_in));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = 1;

    if ((client_op_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("kvs_init: client_op_sock init");
        return 1;
    }

    if ((server_control_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("kvs_init: server_control_sock init");
        return 1;
    }

    if (init_main_server_address(main_addr)) {
        printf("kvs_init: init main server address error\n");
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
        perror("kvs_init: server_control_sock listen");
        exit(EXIT_FAILURE);
    }

    getsockname(server_control_sock, (struct sockaddr*)&addr, &len);
    err = pthread_create(&kvs_thread, NULL, kvs_server_init, &addr);
    if (err) {
        printf("kvs_init: failed to create kvs server thread, pthread_create returns %d\n", err);
        return 1;
    }

    if ((client_control_sock = accept(server_control_sock, NULL, NULL)) < 0) {
        perror("kvs_init: server_control_sock accept");
        exit(EXIT_FAILURE);
    }

    /* Wait connection to master */
    start_time = time(NULL);
    do {
        err = connect(
            client_op_sock, (struct sockaddr*)&main_server_address, sizeof(main_server_address));
        connection_time = time(NULL) - start_time;
    } while ((err < 0) && (connection_time < CONNECTION_TIMEOUT));

    if (connection_time >= CONNECTION_TIMEOUT) {
        printf("kvs_init: connection error: timeout limit (%ld > %d)\n",
               connection_time,
               CONNECTION_TIMEOUT);
        exit(EXIT_FAILURE);
    }

    request.mode = AM_CONNECT;

    DO_RW_OP(write,
             client_op_sock,
             &request,
             sizeof(kvs_request_t),
             client_memory_mutex,
             "client: connect");

    if (strstr(main_host_ip, local_host_ip) && local_port == main_port) {
        is_master = 1;
    }
    is_inited = true;

    return 0;
}

size_t internal_kvs::kvs_finalize(void) {
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));

    if (kvs_thread != 0) {
        void* exit_code;
        int err;
        request.mode = AM_FINALIZE;

        DO_RW_OP(write,
                 client_control_sock,
                 &request,
                 sizeof(kvs_request_t),
                 client_memory_mutex,
                 "client: finalize start");

        DO_RW_OP(read,
                 client_control_sock,
                 &err,
                 sizeof(int),
                 client_memory_mutex,
                 "client: finalize complete");

        err = pthread_join(kvs_thread, &exit_code);
        if (err) {
            printf("kvs_finalize: failed to stop kvs server thread, pthread_join returns %d\n",
                   err);
        }

        kvs_thread = 0;

        close(client_control_sock);
        close(server_control_sock);

        client_control_sock = 0;
        server_control_sock = 0;
    }
    close(client_op_sock);
    client_op_sock = 0;

    if (ip_getting_mode == IGT_K8S)
        request_k8s_kvs_finalize(is_master);
    is_inited = false;

    return 0;
}
internal_kvs::~internal_kvs() {
    if (is_inited)
        kvs_finalize();
}
