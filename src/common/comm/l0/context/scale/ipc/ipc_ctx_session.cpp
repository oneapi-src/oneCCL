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
#include <sstream>
#include "common/comm/l0/context/scale/ipc/ipc_ctx_session.hpp"
#include "common/comm/l0/context/scale/ipc/ipc_ctx_utils.hpp"
#include "common/log/log.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

#include "common/comm/l0/devices/ccl_ipc_gpu_comm.hpp"

#include "common/comm/l0/devices/communication_structs/ipc_client.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_connection.hpp"

namespace native {

session::session(origin_ipc_memory_container&& ipc_src_memory_handles,
                 size_t source_ipc_device_rank)
        : source_device_rank(source_ipc_device_rank),
          source_ipc_memory_storage(std::move(ipc_src_memory_handles)) {
    LOG_DEBUG("Got IPC handles: ", source_ipc_memory_storage.size());
    for (const auto& h : source_ipc_memory_storage) {
        LOG_DEBUG("handle: ", native::to_string(h.get()));
    }

    //to match recevied messag
    send_tag = session_table::get_unique_tag();
    finished.store(false);
}

session::~session() {}

void session::start(net::ipc_client* client, const std::string& addr) {
    if (!client) {
        LOG_ERROR("Empty client for addr: ", addr);
        abort();
    }

    if (finished.load()) {
        //No need ask handles again
        LOG_DEBUG("session: ", reinterpret_cast<void*>(this), ", finished already");
        return;
    }

    //create connection and send data
    std::shared_ptr<net::ipc_tx_connection> tx_connection = client->create_connection(addr);

    // send tag and device rank additionally
    std::array<uint8_t, sizeof(source_device_rank) + sizeof(send_tag)> payload{};
    *reinterpret_cast<size_t*>(payload.data()) = source_device_rank;
    *reinterpret_cast<size_t*>(payload.data() + sizeof(source_device_rank)) = send_tag;
    source_ipc_raw_data = tx_connection->send_ipc_memory_ext(
        source_ipc_memory_storage, payload.data(), payload.size());

    //TODO RING only: peer-to-peer
    data_to_recover.raw_data.resize(source_ipc_raw_data.size());
    LOG_DEBUG("Rank: ",
              source_device_rank,
              ", prepared IPC handles exhange bytes for receive:",
              data_to_recover.raw_data.size());
}

bool session::process(const ccl_ipc_gpu_comm* indexed_ipc_dst_devices,
                      const net::ipc_rx_connection* incoming_connection) {
    size_t existing_recovered_ipc_size = data_to_recover.ipc_memory_storage.size();
    LOG_DEBUG("session: ",
              reinterpret_cast<void*>(this),
              ", recovered ipc storage size: ",
              existing_recovered_ipc_size);
    if (existing_recovered_ipc_size and finished.load()) {
        return true;
    }

    //wait data
    size_t received_rank = 0;
    size_t received_tag = 0;
    size_t handles_data_offset = sizeof(received_rank) + sizeof(received_tag);
    std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>> handles =
        incoming_connection->receive_ipc_memory_ext(data_to_recover.raw_data, handles_data_offset);

    //TODO get tag
    received_rank = *reinterpret_cast<size_t*>(data_to_recover.raw_data.data());
    received_tag =
        *reinterpret_cast<size_t*>(data_to_recover.raw_data.data() + sizeof(source_device_rank));
    (void)received_tag;

    std::shared_ptr<ccl_context> ctx;

    //restore handles
    size_t num_handles = 0;
    for (auto& recv_ip_handle : handles) {
        std::shared_ptr<ccl_device> owner_device = recv_ip_handle->get_owner().lock();
        LOG_DEBUG("Found IPC owner comm device: ",
                  indexed_ipc_dst_devices->to_string(),
                  ",\nIPC handle:\n",
                  native::to_string(recv_ip_handle->get()));

        try {
            // restore IPC memory object from comm device
            auto restored = owner_device->get_ipc_memory(std::move(recv_ip_handle), ctx);
            data_to_recover.ipc_memory_storage[indexed_ipc_dst_devices].push_back(
                std::move(restored));
            LOG_DEBUG("IPC handle by index: ", num_handles, " restored");
        }
        catch (const std::exception& ex) {
            LOG_ERROR("Cannot recover IPC handle by index: ", num_handles, ", error:\n", ex.what());
            throw;
        }
        num_handles++;
    }

    // handles received
    finished.store(true);
    return true;
}

std::string session::to_string() const {
    std::stringstream ss;
    ss << "tag: " << send_tag << ", src_dev_rank: " << source_device_rank
       << ", src_raw_size: " << source_ipc_raw_data.size()
       << ", handles cnt: " << source_ipc_memory_storage.size()
       << ", data_recover: " << data_to_recover.ipc_memory_storage.size()
       << ", is finished: " << finished.load();

    return ss.str();
}

size_t session::get_send_tag() const {
    return send_tag;
}

void session_table::start_session(std::shared_ptr<session> sess,
                                  net::ipc_client* client,
                                  const std::string& peer_addr) {
    sess->start(client, peer_addr);
}

size_t session_table::get_unique_tag() {
    static std::atomic<size_t> tag_counter{ 1 };
    return tag_counter.fetch_add(1);
}

std::string session_table::to_string() const {
    std::stringstream ss;
    ss << "sessions count: " << sessions.size() << std::endl;
    for (const auto& val : sessions) {
        ss << "[" << val.first << ", " << reinterpret_cast<void*>(val.second.get()) << "]\n"
           << val.second->to_string() << std::endl;
    }
    return ss.str();
}
} // namespace native
