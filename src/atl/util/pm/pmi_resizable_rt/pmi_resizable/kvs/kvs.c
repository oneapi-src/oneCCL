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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include "kvs.h"
#include "def.h"
#include "kvs_keeper.h"
#include "request_wrappers_k8s.h"

#define CCL_KVS_IP_PORT_ENV "CCL_KVS_IP_PORT"
#define CCL_KVS_IP_EXCHANGE_ENV "CCL_KVS_IP_EXCHANGE"
#define CCL_KVS_IP_EXCHANGE_VAL_ENV "env"
#define CCL_KVS_IP_EXCHANGE_VAL_K8S "k8s"

#define MAX_CLIENT_COUNT 300

static pthread_t thread = 0;
static char main_host_ip[CCL_IP_LEN];
char local_host_ip[CCL_IP_LEN];
static int sock_listener = 0;
static size_t main_port;
static size_t local_port;
static size_t is_master = 0;

typedef enum ip_getting_type {
    IGT_K8S = 0,
    IGT_ENV = 1,
} ip_getting_type_t;

static ip_getting_type_t ip_getting_mode = IGT_K8S;

typedef enum kvs_access_mode {
    AM_CONNECT = 0,
    AM_DISCONNECT = 1,
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


static struct sockaddr_in main_server_address;
static struct sockaddr_in local_server_address;
static int sock_sender, local_sock, accepted_local_sock;

size_t kvs_set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val)
{
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_PUT;
    STR_COPY(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    STR_COPY(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    STR_COPY(request.val, kvs_val, MAX_KVS_VAL_LENGTH);

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    return 0;
}

size_t kvs_remove_name_key(const char* kvs_name, const char* kvs_key)
{
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_REMOVE;
    STR_COPY(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    STR_COPY(request.key, kvs_key, MAX_KVS_KEY_LENGTH);

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    return 0;
}

size_t kvs_get_value_by_name_key(const char* kvs_name, const char* kvs_key, char* kvs_val)
{
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_VAL;
    size_t is_exist = 0;
    STR_COPY(request.name, kvs_name, MAX_KVS_NAME_LENGTH);
    STR_COPY(request.key, kvs_key, MAX_KVS_KEY_LENGTH);
    memset(kvs_val, 0, MAX_KVS_VAL_LENGTH);

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    CHECK_RW_OP(read(sock_sender, &is_exist, sizeof(size_t)), sizeof(size_t));
    if (is_exist)
    {
        CHECK_RW_OP(read(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));
        STR_COPY(kvs_val, request.val, MAX_KVS_VAL_LENGTH);
    }

    return strlen(kvs_val);
}

size_t kvs_get_count_names(const char* kvs_name)
{
    size_t count_names = 0;
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_COUNT;
    STR_COPY(request.name, kvs_name, MAX_KVS_NAME_LENGTH);

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    CHECK_RW_OP(read(sock_sender, &count_names, sizeof(size_t)), sizeof(size_t));

    return count_names;
}

size_t kvs_get_keys_values_by_name(const char* kvs_name, char*** kvs_keys, char*** kvs_values)
{
    size_t count;
    size_t i;
    kvs_request_t request;
    kvs_request_t* answers;

    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_GET_KEYS_VALUES;
    STR_COPY(request.name, kvs_name, MAX_KVS_NAME_LENGTH);

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    CHECK_RW_OP(read(sock_sender, &count, sizeof(size_t)), sizeof(size_t));

    if (count == 0)
        return count;

    answers = (kvs_request_t*)calloc(count, sizeof(kvs_request_t));

    CHECK_RW_OP(read(sock_sender, answers, sizeof(kvs_request_t) * count),
                sizeof(kvs_request_t) * count);
    if (kvs_keys != NULL)
    {
        if (*kvs_keys != NULL)
            free(*kvs_keys);

        *kvs_keys = (char**)calloc(count, sizeof(char*));
        for (i = 0; i < count; i++)
        {
            (*kvs_keys)[i] = (char*)calloc(MAX_KVS_KEY_LENGTH, sizeof(char));
            STR_COPY((*kvs_keys)[i], answers[i].key, MAX_KVS_KEY_LENGTH);
        }
    }
    if (kvs_values != NULL)
    {
        if (*kvs_values != NULL)
            free(*kvs_values);

        *kvs_values = (char**)calloc(count, sizeof(char*));
        for (i = 0; i < count; i++)
        {
            (*kvs_values)[i] = (char*)calloc(MAX_KVS_VAL_LENGTH, sizeof(char));
            STR_COPY((*kvs_values)[i], answers[i].val, MAX_KVS_VAL_LENGTH);
        }
    }

    free(answers);

    return count;
}

size_t kvs_get_replica_size(void)
{
    size_t replica_size = 0;
    if (ip_getting_mode == IGT_K8S)
    {
        replica_size = request_k8s_get_replica_size();
    }
    else
    {
        kvs_request_t request;
        memset(&request, 0, sizeof(kvs_request_t));
        request.mode = AM_GET_REPLICA;

        CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

        CHECK_RW_OP(read(sock_sender, &replica_size, sizeof(size_t)), sizeof(size_t));
    }
    return replica_size;
}

void* kvs_server_init(void* args)
{
    struct sockaddr_in addr;
    int local_sock;
    kvs_request_t request;
    size_t count;
    size_t clients_count = 0;
    int is_stop = 0;
    fd_set read_fds;
    int i, client_socket[MAX_CLIENT_COUNT], max_sd, sd;

    for (i = 0; i < MAX_CLIENT_COUNT; i++)
    {
        client_socket[i] = 0;
    }

    if ((local_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("Server: socket init failed - %s\n", strerror(errno));
        exit(1);
    }


    while (connect(local_sock, (struct sockaddr*) args, sizeof(addr)) < 0)
    {}

    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;

    if (listen(sock_listener, MAX_CLIENT_COUNT) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    while (!is_stop)
    {
        FD_ZERO(&read_fds);
        FD_SET(sock_listener, &read_fds);
        FD_SET(local_sock, &read_fds);
        max_sd = sock_listener;
        for (i = 0; i < MAX_CLIENT_COUNT; i++)
        {
            sd = client_socket[i];

            if (sd > 0)
                FD_SET(sd, &read_fds);

            if (sd > max_sd)
                max_sd = sd;
        }
        if (local_sock > max_sd)
            max_sd = local_sock;
        if ((select(max_sd + 1, &read_fds, NULL, NULL, NULL) < 0) && (errno != EINTR))
        {
            perror("select");
            exit(EXIT_FAILURE);
        }

        if (FD_ISSET(sock_listener, &read_fds))
        {
            int new_socket;
            if ((new_socket = accept(sock_listener,
                                     (struct sockaddr*) &addr, (socklen_t*) &addr)) < 0)
            {
                perror("accept");
                exit(EXIT_FAILURE);
            }
            for (i = 0; i < MAX_CLIENT_COUNT; i++)
            {
                if (client_socket[i] == 0)
                {
                    client_socket[i] = new_socket;
                    break;
                }
            }
            if (i >= MAX_CLIENT_COUNT)
            {
                printf("server: Not enough free sockets\n");
                exit(1);
            }
            continue;
        }
        if (FD_ISSET(local_sock, &read_fds))
        {
            if ((read(local_sock, &request, sizeof(kvs_request_t))) == 0)
            {
                close(local_sock);
                local_sock = 0;
            }
            if (request.mode != AM_FINALIZE)
            {
                printf("server: Wrong access mode for local socket.\n");
                exit(1);
            }
        }
        else
        {
            for (i = 0; i < MAX_CLIENT_COUNT; i++)
            {
                sd = client_socket[i];

                if (FD_ISSET(sd, &read_fds))
                {
                    if ((read(sd, &request, sizeof(kvs_request_t))) == 0)
                    {
                        close(sd);
                        client_socket[i] = 0;
                    }
                    break;
                }
            }
            if (i >= MAX_CLIENT_COUNT || client_socket[i] == 0)
            {
                continue;
            }
        }

        switch (request.mode)
        {
            case AM_CONNECT:
            {
                clients_count++;
                break;
            }
            case AM_DISCONNECT:
            {
                clients_count--;
                break;
            }
            case AM_PUT:
            {
                put_key(request.name, request.key, request.val, ST_SERVER);
                break;
            }
            case AM_REMOVE:
            {
                remove_val(request.name, request.key, ST_SERVER);
                break;
            }
            case AM_GET_VAL:
            {
                count = get_val(request.name, request.key, request.val, ST_SERVER);
                CHECK_RW_OP(write(client_socket[i], &count, sizeof(size_t)), sizeof(size_t));
                if (count != 0)
                    CHECK_RW_OP(write(client_socket[i], &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));
                break;
            }
            case AM_GET_COUNT:
            {
                count = get_count(request.name, ST_SERVER);
                CHECK_RW_OP(write(client_socket[i], &count, sizeof(size_t)), sizeof(size_t));
                break;
            }
            case AM_GET_REPLICA:
            {
                char* replica_size_str = getenv(CCL_WORLD_SIZE_ENV);
                count = (replica_size_str != NULL) ? strtol(replica_size_str, NULL, 10) : clients_count;
                CHECK_RW_OP(write(client_socket[i], &count, sizeof(size_t)), sizeof(size_t));
                break;
            }
            case AM_GET_KEYS_VALUES:
            {
                char** kvs_keys = NULL;
                char** kvs_values = NULL;
                size_t j;
                kvs_request_t* answers = NULL;

                count = get_keys_values(request.name, &kvs_keys, &kvs_values, ST_SERVER);

                CHECK_RW_OP(write(client_socket[i], &count, sizeof(size_t)), sizeof(size_t));
                if (count == 0)
                    break;

                answers = (kvs_request_t*)calloc(count, sizeof(kvs_request_t));
                for (j = 0; j < count; j++)
                {
                    STR_COPY(answers[j].name, request.name, MAX_KVS_NAME_LENGTH);
                    STR_COPY(answers[j].key, kvs_keys[j], MAX_KVS_KEY_LENGTH);
                    STR_COPY(answers[j].val, kvs_values[j], MAX_KVS_VAL_LENGTH);
                }

                CHECK_RW_OP(write(client_socket[i], answers, sizeof(kvs_request_t) * count),
                            sizeof(kvs_request_t) * count);

                free(answers);
                for (j = 0; j < count; j++)
                {
                    free(kvs_keys[j]);
                    free(kvs_values[j]);
                }
                free(kvs_keys);
                free(kvs_values);
                break;
            }
            case AM_FINALIZE:
            {
                is_stop = 1;
                kvs_keeper_clear(ST_SERVER);
                break;
            }
            default:
                exit(1);
        }
    }

    CHECK_RW_OP(write(local_sock, &is_stop, sizeof(int)), sizeof(int));
    close(local_sock);
    for (i = 0; i < MAX_CLIENT_COUNT; i++)
    {
        if (client_socket[i] != 0)
            close(client_socket[i]);
    }
    close(sock_listener);
    return NULL;
}

size_t init_main_server_by_k8s(void)
{
    char port_str[MAX_KVS_VAL_LENGTH];
    request_k8s_kvs_init();

    SET_STR(port_str, INT_STR_SIZE, "%d", local_server_address.sin_port);

    request_k8s_kvs_get_master(local_host_ip, main_host_ip, port_str);

    main_port = strtol(port_str, NULL, 10);
    main_server_address.sin_port = main_port;
    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0)
    {
        printf("\nInvalid address/ Address not supported: %s\n", main_host_ip);
        return 1;
    }
    return 0;
}

size_t init_main_server_by_env(void)
{
    char* tmp_host_ip;
    char* port = NULL;

    tmp_host_ip = getenv(CCL_KVS_IP_PORT_ENV);

    if (tmp_host_ip == NULL)
    {
        printf("You must set %s\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }

    if ((port = strstr(tmp_host_ip, "_")) == NULL)
    {
        printf("You must set %s like IP_PORT\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }
    port[0] = '\0';
    port++;

    memset(main_host_ip, 0, CCL_IP_LEN);
    STR_COPY(main_host_ip, tmp_host_ip, CCL_IP_LEN);

    main_port = strtol(port, NULL, 10);
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0)
    {
        printf("\nInvalid address/ Address not supported: %s\n", main_host_ip);
        return 1;
    }
    return 0;
}

size_t init_main_server_by_string(const char* main_addr)
{
    char* port = NULL;
    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = 1;

    main_server_address.sin_family = AF_INET;

    if ((sock_listener = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("Server: socket init failed - %s\n", strerror(errno));
        exit(1);
    }

    while (bind(sock_listener, (const struct sockaddr*) &local_server_address, sizeof(local_server_address)) < 0)
    {
        local_server_address.sin_port++;
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    STR_COPY(main_host_ip, main_addr, CCL_IP_LEN);

    if ((port = strstr(main_host_ip, "_")) == NULL)
    {
        printf("You must set %s like IP_PORT\n", CCL_KVS_IP_PORT_ENV);
        return 1;
    }
    port[0] = '\0';
    port++;

    main_port = strtol(port, NULL, 10);
    main_server_address.sin_port = main_port;

    if (inet_pton(AF_INET, main_host_ip, &(main_server_address.sin_addr)) <= 0)
    {
        printf("\nInvalid address/ Address not supported: %s(%s)\n", main_host_ip, strerror(errno));
        return 1;
    }
    return 0;
}

size_t main_server_address_reserve(char* main_address)
{
    FILE* fp;
    if ((fp = popen(CHECKER_IP, READ_ONLY)) == NULL)
    {
        printf("Can't get host IP - %s\n", strerror(errno));
        exit(1);
    }
    CHECK_FGETS(fgets(local_host_ip, CCL_IP_LEN, fp), local_host_ip);
    pclose(fp);
    while (local_host_ip[strlen(local_host_ip) - 1] == '\n' ||
        local_host_ip[strlen(local_host_ip) - 1] == ' ')
        local_host_ip[strlen(local_host_ip) - 1] = '\0';

    if ((sock_listener = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("Server: socket init failed - %s\n", strerror(errno));
        exit(1);
    }
    main_server_address.sin_family = AF_INET;
    main_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    main_server_address.sin_port = 1;
    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);

    while (bind(sock_listener, (const struct sockaddr*) &main_server_address,
                sizeof(main_server_address)) < 0)
    {
        main_server_address.sin_port++;
    }
    local_server_address.sin_port = main_server_address.sin_port;

    memset(main_address, '\0', CCL_IP_LEN);
    snprintf(main_address, CCL_IP_LEN, "%s_%d", local_host_ip, main_server_address.sin_port);

    return 0;
}

size_t init_main_server_address(const char* main_addr)
{
    char* ip_getting_type = getenv(CCL_KVS_IP_EXCHANGE_ENV);
    FILE* fp;
    char* point_to_space;

    if ((fp = popen(CHECKER_IP, READ_ONLY)) == NULL)
    {
        printf("Can't get host IP\n");
        exit(1);
    }
    CHECK_FGETS(fgets(local_host_ip, CCL_IP_LEN, fp), local_host_ip);
    pclose(fp);

    while (local_host_ip[strlen(local_host_ip) - 1] == '\n' ||
           local_host_ip[strlen(local_host_ip) - 1] == ' ')
        local_host_ip[strlen(local_host_ip) - 1] = '\0';
    if ((point_to_space = strstr(local_host_ip, " ")) != NULL)
        point_to_space[0] = NULL_CHAR;

    if (ip_getting_type)
    {
        if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_ENV))
        {
            ip_getting_mode = IGT_ENV;
        }
        else if (strstr(ip_getting_type, CCL_KVS_IP_EXCHANGE_VAL_K8S))
        {
            ip_getting_mode = IGT_K8S;
        }
        else
        {
            printf("Unknown %s: %s\n", CCL_KVS_IP_EXCHANGE_ENV, ip_getting_type);
            return 1;
        }
    }

    if (main_addr != NULL)
    {
        ip_getting_mode = IGT_ENV;
        if (sock_listener == 0)
            init_main_server_by_string(main_addr);
        return 0;
    }

    local_server_address.sin_family = AF_INET;
    local_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
    local_server_address.sin_port = 1;

    main_server_address.sin_family = AF_INET;

    if ((sock_listener = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("Server: socket init failed - %s\n", strerror(errno));
        exit(1);
    }

    switch (ip_getting_mode)
    {
        case IGT_K8S:
        {
            while (bind(sock_listener, (const struct sockaddr*) &local_server_address, sizeof(local_server_address)) < 0)
            {
                local_server_address.sin_port++;
            }

            local_port = local_server_address.sin_port;
            return init_main_server_by_k8s();
        }
        case IGT_ENV:
        {
            int res = init_main_server_by_env();

            if (res)
                return res;

            if (strstr(local_host_ip, main_host_ip))
            {
                if (bind(sock_listener, (const struct sockaddr*) &main_server_address,
                         sizeof(main_server_address)) < 0)
                {
                    printf("PORT %d busy\n", main_server_address.sin_port);
                    while (bind(sock_listener, (const struct sockaddr*) &local_server_address,
                                sizeof(local_server_address)) < 0)
                    {
                        local_server_address.sin_port++;
                    }
                    local_port = local_server_address.sin_port;
                }
                else
                {
                    local_port = main_server_address.sin_port;
                }
            }
            else
            {
                while (bind(sock_listener, (const struct sockaddr*) &local_server_address,
                            sizeof(local_server_address)) < 0)
                {
                    local_server_address.sin_port++;
                }
                local_port = local_server_address.sin_port;
            }

            return res;
        }
        default:
        {
            printf("Unknown %s\n", CCL_KVS_IP_EXCHANGE_ENV);
            return 1;
        }
    }
}

size_t kvs_init(const char* main_addr)
{
    int err;
    socklen_t len = 0;
    struct sockaddr_in addr;
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    memset(&addr, 0, sizeof(struct sockaddr_in));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = 1;

    if ((sock_sender = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return 1;
    }
    if ((local_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error: %s\n", strerror(errno));
        return 1;
    }
    if (init_main_server_address(main_addr))
    {
        printf("Init main server address error\n");
        close(sock_sender);
        close(local_sock);
        return 1;
    }
    while (bind(local_sock, (const struct sockaddr*) &addr, sizeof(addr)) < 0)
    {
        addr.sin_port++;
    }

    if (listen(local_sock, 1) < 0)
    {
        printf("listener error: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    getsockname(local_sock, (struct sockaddr*) &addr, &len);
    err = pthread_create(&thread, NULL, kvs_server_init, &addr);
    if (err)
    {
        printf("error while creating listener thread, pthread_create returns %d\n", err);
        return 1;
    }

    if ((accepted_local_sock  = accept(local_sock, NULL, NULL)) < 0)
    {
        printf("Client: accept error: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    while (connect(sock_sender, (struct sockaddr*) &main_server_address, sizeof(main_server_address)) < 0)
    {}

    request.mode = AM_CONNECT;

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

    if (strstr(main_host_ip, local_host_ip) &&
        local_port == main_port)
    {
        is_master = 1;
    }

    return 0;
}

size_t kvs_finalize(void)
{
    kvs_request_t request;
    memset(&request, 0, sizeof(kvs_request_t));
    request.mode = AM_DISCONNECT;

    CHECK_RW_OP(write(sock_sender, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));
    if (thread != 0)
    {
        void* exit_code;
        int err;
        request.mode = AM_FINALIZE;

        CHECK_RW_OP(write(accepted_local_sock, &request, sizeof(kvs_request_t)), sizeof(kvs_request_t));

        CHECK_RW_OP(read(accepted_local_sock, &err, sizeof(int)), sizeof(int));

        err = pthread_cancel(thread);

        if (err)
        {
            printf("error while canceling progress listener, pthread_cancel returns %d\n", err);
        }

        err = pthread_join(thread, &exit_code);
        if (err)
        {
            printf("error while joining progress listener, pthread_join returns %d\n", err);
        }
        close(accepted_local_sock);
        close(local_sock);
    }
    close(sock_sender);

    if (ip_getting_mode == IGT_K8S)
        request_k8s_kvs_finalize(is_master);

    return 0;
}
