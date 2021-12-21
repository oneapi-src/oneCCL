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
#include <memory>

#include "ikvs_wrapper.h"

class isockaddr {
public:
    virtual in_port_t get_sin_port() = 0;
    virtual void set_sin_port(in_port_t) = 0;
    virtual const void* get_sin_addr_ptr() = 0;
    virtual kvs_status_t set_sin_addr(const char*) = 0;
    virtual struct sockaddr* get_sock_addr_ptr() = 0;
    virtual sa_family_t sin_family() = 0;
    virtual size_t size() = 0;
    virtual ~isockaddr() = default;

protected:
    const size_t default_start_port = 4096;
};

class internal_kvs final : public ikvs_wrapper {
public:
    kvs_status_t kvs_set_value(const char* kvs_name,
                               const char* kvs_key,
                               const char* kvs_val) override;

    kvs_status_t kvs_remove_name_key(const char* kvs_name, const char* kvs_key) override;

    kvs_status_t kvs_get_value_by_name_key(const char* kvs_name,
                                           const char* kvs_key,
                                           char* kvs_val) override;

    kvs_status_t kvs_register(const char* kvs_name, const char* kvs_key, char* kvs_val);

    kvs_status_t kvs_set_size(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    kvs_status_t kvs_barrier_register(const char* kvs_name,
                                      const char* kvs_key,
                                      const char* kvs_val);

    kvs_status_t kvs_barrier(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    kvs_status_t kvs_init(const char* main_addr) override;

    kvs_status_t kvs_main_server_address_reserve(char* main_addr) override;

    kvs_status_t kvs_get_count_names(const char* kvs_name, int& count_names) override;

    kvs_status_t kvs_finalize() override;

    kvs_status_t kvs_get_keys_values_by_name(const char* kvs_name,
                                             char*** kvs_keys,
                                             char*** kvs_values,
                                             size_t& count) override;

    kvs_status_t kvs_get_replica_size(size_t& replica_size) override;

    ~internal_kvs() override;

    void set_server_address(const std::string& server_addr) {
        server_address = server_addr;
    }

    static const int CCL_IP_LEN = 128;
    static const char SCOPE_ID_DELIM = '%';

private:
    kvs_status_t init_main_server_by_string(const char* main_addr);
    kvs_status_t init_main_server_by_env();
    kvs_status_t init_main_server_by_k8s();
    kvs_status_t init_main_server_address(const char* main_addr);
    kvs_status_t fill_local_host_ip();
    bool is_inited{ false };

    pthread_t kvs_thread = 0;

    char main_host_ip[CCL_IP_LEN];
    std::list<std::string> local_host_ips;
    std::list<std::string> local_host_ipv4s;
    std::list<std::string> local_host_ipv6s;
    char local_host_ip[CCL_IP_LEN];

    size_t main_port;
    size_t local_port;
    size_t is_master = 0;
    std::mutex client_memory_mutex;

    std::shared_ptr<isockaddr> main_server_address;
    std::shared_ptr<isockaddr> local_server_address;

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
    const std::string CCL_KVS_PREFER_IPV6_ENV = "CCL_KVS_PREFER_IPV6";
    const std::string CCL_KVS_IFACE_ENV = "CCL_KVS_IFACE";

    const std::string CCL_KVS_IP_EXCHANGE_VAL_ENV = "env";
    const std::string CCL_KVS_IP_EXCHANGE_VAL_K8S = "k8s";

    const int CONNECTION_TIMEOUT = 120;

    int server_listen_sock; /* used on server side to handle new incoming connect requests from clients */
    std::string server_address{};

    sa_family_t address_family{ AF_UNSPEC };
};

class sockaddr_v4 : public isockaddr {
public:
    sockaddr_v4() {
        memset(&addr, 0, sizeof(sockaddr_in));
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_family = AF_INET;
        addr.sin_port = default_start_port;
    }
    in_port_t get_sin_port() override {
        return addr.sin_port;
    }
    void set_sin_port(in_port_t sin_port) override {
        addr.sin_port = sin_port;
    }
    struct sockaddr* get_sock_addr_ptr() override {
        return (struct sockaddr*)&addr;
    }
    const void* get_sin_addr_ptr() override {
        return &(addr.sin_addr);
    }
    kvs_status_t set_sin_addr(const char* src) override;
    sa_family_t sin_family() override {
        return addr.sin_family;
    }
    size_t size() override {
        return sizeof(addr);
    }
    ~sockaddr_v4() override = default;

private:
    struct sockaddr_in addr;
};
class sockaddr_v6 : public isockaddr {
public:
    sockaddr_v6() {
        memset(&addr, 0, sizeof(sockaddr_in6));
        addr.sin6_addr = IN6ADDR_ANY_INIT;
        addr.sin6_family = AF_INET6;
        addr.sin6_port = default_start_port;
    }
    in_port_t get_sin_port() override {
        return addr.sin6_port;
    }
    void set_sin_port(in_port_t sin_port) override {
        addr.sin6_port = sin_port;
    }
    const void* get_sin_addr_ptr() override {
        return &(addr.sin6_addr);
    }
    kvs_status_t set_sin_addr(const char* src) override;
    struct sockaddr* get_sock_addr_ptr() override {
        return (struct sockaddr*)&addr;
    }
    sa_family_t sin_family() override {
        return addr.sin6_family;
    }
    size_t size() override {
        return sizeof(addr);
    }
    ~sockaddr_v6() override = default;

private:
    struct sockaddr_in6 addr;
};
