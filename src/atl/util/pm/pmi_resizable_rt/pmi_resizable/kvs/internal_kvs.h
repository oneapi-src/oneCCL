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

#include <cstddef>
#include <list>
#include <mutex>
#include <netinet/ip.h>

#include "ikvs_wrapper.h"

class internal_kvs final : public ikvs_wrapper {
public:
    size_t kvs_set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val) override;

    size_t kvs_remove_name_key(const char* kvs_name, const char* kvs_key) override;

    size_t kvs_get_value_by_name_key(const char* kvs_name,
                                     const char* kvs_key,
                                     char* kvs_val) override;

    size_t kvs_register(const char* kvs_name, const char* kvs_key, char* kvs_val);

    size_t kvs_set_size(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    size_t kvs_barrier_register(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    void kvs_barrier(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    size_t kvs_init(const char* main_addr) override;

    size_t kvs_main_server_address_reserve(char* main_addr) override;

    size_t kvs_get_count_names(const char* kvs_name) override;

    size_t kvs_finalize() override;

    size_t kvs_get_keys_values_by_name(const char* kvs_name,
                                       char*** kvs_keys,
                                       char*** kvs_values) override;

    size_t kvs_get_replica_size() override;

    ~internal_kvs() override;

    void set_server_address(const std::string& server_addr) {
        server_address = server_addr;
    }

private:
    static const int CCL_IP_LEN = 128;
    size_t init_main_server_by_string(const char* main_addr);
    size_t init_main_server_by_env();
    size_t init_main_server_by_k8s();
    size_t init_main_server_address(const char* main_addr);
    int fill_local_host_ip();
    bool is_inited{ false };

    pthread_t kvs_thread = 0;

    char main_host_ip[CCL_IP_LEN];
    std::list<std::string> local_host_ips;
    char local_host_ip[CCL_IP_LEN];

    size_t main_port;
    size_t local_port;
    size_t is_master = 0;
    std::mutex client_memory_mutex;

    struct sockaddr_in main_server_address;
    struct sockaddr_in local_server_address;

    int client_op_sock; /* used on client side to send commands and to recv result to/from server */

    int client_control_sock; /* used on client side to control local kvs server */
    int server_control_sock; /* used on server side to be controlled by local client */

    typedef enum ip_getting_type {
        IGT_K8S = 0,
        IGT_ENV = 1,
    } ip_getting_type_t;

    ip_getting_type_t ip_getting_mode = IGT_K8S;

    const std::string CCL_KVS_IP_PORT_ENV = "CCL_KVS_IP_PORT";
    const std::string CCL_KVS_IP_EXCHANGE_ENV = "CCL_KVS_IP_EXCHANGE";
    const std::string CCL_KVS_IP_EXCHANGE_VAL_ENV = "env";
    const std::string CCL_KVS_IP_EXCHANGE_VAL_K8S = "k8s";

    const int CONNECTION_TIMEOUT = 120;
    int server_listen_sock; /* used on server side to handle new incoming connect requests from clients */
    std::string server_address{};
    const size_t default_start_port = 4096;
};
