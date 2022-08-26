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
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "def.h"
#include "pmi_listener.hpp"

#define KVS_LISTENER "CCL_LISTENER"

#define LISTENER_TIMEOUT 5

static int sock_sender;
static size_t num_listeners;
static int sock_listener = -1;
static struct sockaddr_in* server_addresses = NULL;
static struct sockaddr_in addr;
static int num_changes = 0;

void pmi_listener::set_applied_count(int count) {
    num_changes -= count;
}

kvs_status_t pmi_listener::collect_sock_addr(std::shared_ptr<helper> h) {
    FILE* fp;
    size_t i, j;
    kvs_status_t res = KVS_STATUS_SUCCESS;
    size_t glob_num_listeners;
    std::vector<std::string> sock_addr_str(1);
    std::vector<std::string> hosts_names_str(1);
    char my_ip[MAX_KVS_VAL_LENGTH];
    char* point_to_space;

    if ((fp = popen(GET_IP_CMD, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get host IP");
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(my_ip, MAX_KVS_VAL_LENGTH, fp), my_ip);
    pclose(fp);
    while (my_ip[strlen(my_ip) - 1] == '\n' || my_ip[strlen(my_ip) - 1] == ' ')
        my_ip[strlen(my_ip) - 1] = '\0';
    if ((point_to_space = strstr(my_ip, " ")) != NULL)
        point_to_space[0] = NULL_CHAR;

    KVS_CHECK_STATUS(h->get_keys_values_by_name(
                         KVS_LISTENER, hosts_names_str, sock_addr_str, glob_num_listeners),
                     "failed to get sock info");
    num_listeners = glob_num_listeners;

    for (i = 0; i < num_listeners; i++) {
        if (strstr(hosts_names_str[i].c_str(), my_hostname)) {
            num_listeners--;
            break;
        }
    }

    if (num_listeners == 0) {
        res = KVS_STATUS_SUCCESS;
        goto exit;
    }

    if ((sock_sender = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        LOG_ERROR("Socket creation error");
        res = KVS_STATUS_FAILURE;
        goto exit;
    }

    if (server_addresses != NULL) {
        free(server_addresses);
    }

    server_addresses = (struct sockaddr_in*)malloc((num_listeners) * sizeof(struct sockaddr_in));
    if (server_addresses == NULL) {
        LOG_ERROR("nmemory allocation failed");
        res = KVS_STATUS_FAILURE;
        goto exit;
    }

    /*get listener addresses*/
    for (i = 0, j = 0; i < num_listeners; i++, j++) {
        char* point_to_port = strstr(const_cast<char*>(sock_addr_str[j].c_str()), "_");
        if (point_to_port == NULL) {
            LOG_ERROR("Wrong address_port record: %s", sock_addr_str[j]);
            res = KVS_STATUS_FAILURE;
            goto exit;
        }
        point_to_port[0] = NULL_CHAR;
        point_to_port++;
        if (strstr(const_cast<char*>(hosts_names_str[j].c_str()), my_hostname)) {
            i--;
            continue;
        }

        if (safe_strtol(point_to_port, server_addresses[i].sin_port) != KVS_STATUS_SUCCESS) {
            LOG_ERROR("failed to convert sin_port");
            res = KVS_STATUS_FAILURE;
            goto exit;
        }
        server_addresses[i].sin_family = AF_INET;

        if (inet_pton(AF_INET, sock_addr_str[j].c_str(), &(server_addresses[i].sin_addr)) <= 0) {
            LOG_ERROR("Invalid address/ Address not supported: %s", sock_addr_str[j].c_str());
            res = KVS_STATUS_FAILURE;
            goto exit;
        }
    }
exit:
    return res;
}

kvs_status_t pmi_listener::clean_listener(std::shared_ptr<helper> h) {
    KVS_CHECK_STATUS(h->remove_name_key(KVS_LISTENER, my_hostname), "failed to remove host info");
    close(sock_listener);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_listener::send_notification(int sig, std::shared_ptr<helper> h) {
    size_t i;
    char message[INT_STR_SIZE];

    KVS_CHECK_STATUS(collect_sock_addr(h), "failed to collect sock info");

    SET_STR(message, INT_STR_SIZE, "%s", "Update!");
    for (i = 0; i < num_listeners; ++i) {
        sendto(sock_sender,
               message,
               INT_STR_SIZE,
               MSG_DONTWAIT,
               (const struct sockaddr*)&(server_addresses[i]),
               sizeof(server_addresses[i]));
    }
    if (sig) {
        KVS_CHECK_STATUS(clean_listener(h), "failed to clean listener");
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_listener::run_listener(std::shared_ptr<helper> h) {
    socklen_t len = 0;
    char recv_buf[INT_STR_SIZE];
    memset(recv_buf, 0, INT_STR_SIZE);

    if (sock_listener == -1) {
        FILE* fp;
        char addr_for_kvs[REQUEST_POSTFIX_SIZE];
        int addr_len = sizeof(addr);
        char my_ip[MAX_KVS_VAL_LENGTH];
        char* point_to_space;
        struct timeval timeout;
        timeout.tv_sec = LISTENER_TIMEOUT;
        timeout.tv_usec = 0;

        if ((fp = popen(GET_IP_CMD, READ_ONLY)) == NULL) {
            printf("Can't get host IP\n");
            exit(1);
        }
        CHECK_FGETS(fgets(my_ip, MAX_KVS_VAL_LENGTH, fp), my_ip);
        pclose(fp);
        while (my_ip[strlen(my_ip) - 1] == '\n' || my_ip[strlen(my_ip) - 1] == ' ')
            my_ip[strlen(my_ip) - 1] = '\0';
        if ((point_to_space = strstr(my_ip, " ")) != NULL)
            point_to_space[0] = NULL_CHAR;
        if ((sock_listener = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            LOG_ERROR("socket error(%s)", strerror(errno));
            return KVS_STATUS_FAILURE;
        }

        memset(&addr, 0, sizeof(addr));

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = 0;

        if (bind(sock_listener, (const struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LOG_ERROR("bind error(%s)", strerror(errno));
            return KVS_STATUS_FAILURE;
        }

        getsockname(sock_listener, (struct sockaddr*)&addr, (socklen_t*)&addr_len);

        SET_STR(
            addr_for_kvs, REQUEST_POSTFIX_SIZE, KVS_NAME_TEMPLATE_I, my_ip, (size_t)addr.sin_port);
        KVS_CHECK_STATUS(h->set_value(KVS_LISTENER, my_hostname, addr_for_kvs),
                         "failed to set addr info");
        if (setsockopt(sock_listener, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
            perror("Error");
        }
        num_changes = 0;
    }

    while (num_changes <= 0) {
        int ret = recvfrom(sock_listener,
                           (char*)recv_buf,
                           INT_STR_SIZE,
                           MSG_WAITALL,
                           (struct sockaddr*)&addr,
                           &len);
        if (ret == -1) {
            if (errno == EAGAIN) {
                return KVS_STATUS_SUCCESS;
            }
            if (errno != EINTR) {
                LOG_ERROR("listner: accept error: %s\n", strerror(errno));
                return KVS_STATUS_FAILURE;
            }
        }
        num_changes++;
    }

    return KVS_STATUS_SUCCESS;
}
