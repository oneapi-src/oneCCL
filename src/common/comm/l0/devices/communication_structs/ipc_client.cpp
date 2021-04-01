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
#include <fcntl.h>
#include <algorithm>

#include "common/log/log.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_client.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_connection.hpp"

namespace net {

ipc_client::~ipc_client() {
    stop_all();
}

std::shared_ptr<ipc_tx_connection> ipc_client::create_connection(const std::string& addr) {
    LOG_DEBUG("Create or find existing connection to: ", addr);
    auto it = connections.find(addr);
    if (it != connections.end()) {
        LOG_DEBUG("Get existing conenction");
        return it->second;
    }

    std::shared_ptr<ipc_tx_connection> tx_conn;
    try {
        tx_conn.reset(new ipc_tx_connection(addr));
    }
    catch (const std::exception& ex) {
        LOG_ERROR(
            "Cannot create TX connection to other IPC server on: ", addr, ", error: ", ex.what());
        throw;
    }

    connections.emplace(addr, tx_conn);

    LOG_DEBUG("Connections created, total tx connections: ", connections.size());
    return tx_conn;
}

bool ipc_client::stop_all() {
    LOG_DEBUG("Stop connections: ", connections.size());
    for (auto& conn_pair : connections) {
        LOG_DEBUG("schedule stop connection to: ", conn_pair.first);
        conn_pair.second.reset();
    }
    return true;
}
} // namespace net
