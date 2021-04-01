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

#include "common/log/log.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_connection.hpp"

#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"

namespace net {

// Rx receive IPC data from IPC_SOURCE_DEVICE
ipc_rx_connection::ipc_rx_connection(int socket) : connection(socket) {
    //disable WRITE
    shutdown(socket_fd, SHUT_WR);
}

std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>>
ipc_rx_connection::receive_ipc_memory(std::vector<uint8_t>& out_data_resized,
                                      size_t& out_received_rank) const {
    std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>> ret =
        receive_ipc_memory_ext(out_data_resized, sizeof(size_t));

    out_received_rank = *(reinterpret_cast<const size_t*>(out_data_resized.data()));
    LOG_DEBUG("Received IPC handles count: ", ret.size(), ", from rank: ", out_received_rank);
    return ret;
}

// For properly call resize data with *_resized before for expected sizes
std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>>
ipc_rx_connection::receive_ipc_memory_ext(std::vector<uint8_t>& out_data_resized,
                                          size_t out_data_offset_bytes) const {
    LOG_DEBUG("Try to receive ip memory for expected bytes count: ", out_data_resized.size());

    constexpr size_t handle_size =
        native::ccl_device::device_ipc_memory_handle::get_size_for_serialize();

    size_t handles_count = (out_data_resized.size() - out_data_offset_bytes) / handle_size;
    size_t bytes_rest = (out_data_resized.size() - out_data_offset_bytes) % handle_size;

    std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>> ret;
    LOG_DEBUG("Expected receive bytes: ",
              out_data_resized.size(),
              ", handles count: ",
              handles_count,
              ", bytes in rest: ",
              bytes_rest);
    try {
        if (bytes_rest != 0) {
            throw std::runtime_error(
                std::string("Unexpected bytes to receive: ") +
                std::to_string(out_data_resized.size()) +
                ", handles count: " + std::to_string(handles_count) +
                ", bytes in rest should be zero, got: " + std::to_string(bytes_rest));
        }

        // receive data
        std::vector<connection::fd_t> out_pids_resized(handles_count, 0);
        ret = receive_raw_ipc_memory_ext(out_data_resized, out_pids_resized, out_data_offset_bytes);

        LOG_DEBUG("Received IPC handles: ", ret.size(), ", with offset: ", out_data_offset_bytes);
        size_t num_handles = 0;
        for (auto& handle : ret) {
            // override duplicated fd by SCM_RIGHTS
            connection::fd_t* pid_ptr = reinterpret_cast<connection::fd_t*>(handle->get_ptr());
            connection::fd_t new_pid = out_pids_resized[num_handles];

            LOG_DEBUG("Override FD: ",
                      *pid_ptr,
                      ", with new FD: ",
                      new_pid,
                      ", for IPC handle num: ",
                      num_handles);
            *pid_ptr = new_pid;
            num_handles++;
        }
    }
    catch (const std::exception& ex) {
        LOG_ERROR("Cannot receive IPC handles, error:\n", ex.what());
        throw;
    }
    return ret;
}

std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>>
ipc_rx_connection::receive_raw_ipc_memory(std::vector<uint8_t>& out_data_resized,
                                          std::vector<connection::fd_t>& out_pids_resized,
                                          size_t& received_rank) const {
    std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>> ret =
        receive_raw_ipc_memory_ext(out_data_resized, out_pids_resized, sizeof(size_t));

    received_rank = *(reinterpret_cast<const size_t*>(out_data_resized.data()));
    LOG_DEBUG("Deserialized IPC handles count: ", ret.size(), ", from rank: ", received_rank);
    return ret;
}

std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>>
ipc_rx_connection::receive_raw_ipc_memory_ext(std::vector<uint8_t>& out_data_resized,
                                              std::vector<connection::fd_t>& out_pids_resized,
                                              size_t out_data_offset_bytes) const {
    LOG_DEBUG("Try to receive ip memory for expected bytes count: ",
              out_data_resized.size(),
              ", fd count: ",
              out_pids_resized.size(),
              ", with offset",
              out_data_offset_bytes);
    if (out_data_resized.size() < out_data_offset_bytes) {
        LOG_ERROR("not enough bytes in out_data_resized: ",
                  out_data_resized.size(),
                  ", for given offset: ",
                  out_data_offset_bytes);
        abort();
    }

    ssize_t read_bytes = 0;
    try {
        read_bytes = recv_msg_with_pid_data(
            out_data_resized, out_pids_resized, MSG_CMSG_CLOEXEC | MSG_WAITALL);
        LOG_DEBUG("Read bytes count: ", read_bytes);

        if (static_cast<size_t>(read_bytes) < out_data_resized.size()) {
            throw std::runtime_error(std::string("Too many bytes received: ") +
                                     std::to_string(read_bytes));
        }
    }
    catch (const std::exception& ex) {
        LOG_ERROR("Cannot receive IPC handles, error: ", ex.what());
        throw;
    }

    //get ipc handles
    size_t recv_data_size = out_data_resized.size();
    const uint8_t* recv_data_start = out_data_resized.data();

    recv_data_start += out_data_offset_bytes;
    recv_data_size -= out_data_offset_bytes;

    std::vector<std::shared_ptr<native::ccl_device::device_ipc_memory_handle>> ret;
    ret.reserve(out_pids_resized.size());

    size_t num_handles = 0;
    size_t expected_handles = out_pids_resized.size();
    std::shared_ptr<native::ccl_device_platform> ipc_platform;
    std::shared_ptr<native::ccl_context> ctx;
    LOG_DEBUG("Deserialize IPC handles count: ", expected_handles);
    while (num_handles < expected_handles and recv_data_size > 0) {
        LOG_DEBUG(
            "Start restore handle num: ", num_handles, ", expected count: ", expected_handles);
        try {
            // deserialize handle
            auto recv_ip_handle = native::ccl_device::device_ipc_memory_handle::deserialize<
                native::ccl_device::device_ipc_memory_handle>(
                &recv_data_start, recv_data_size, ctx, ipc_platform);
            // remember ipc handle
            ret.push_back(std::move(recv_ip_handle));
        }
        catch (const std::exception& ex) {
            LOG_ERROR(
                "Cannot deserialize IPC handle by index: ", num_handles, ", error:\n", ex.what());
            throw;
        }
        num_handles++;
    }

    LOG_DEBUG("Deserialized IPC handles count: ", ret.size());
    return ret;
}

// Tx receive IPC data from IPC_SOURCE_DEVICE
ipc_tx_connection::ipc_tx_connection(const std::string& addr) : connection(-1) {
    try {
        socket_fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (socket_fd == -1) {
            throw std::runtime_error(std::string("Cannot create client socket, error: ") +
                                     strerror(errno));
        }

        memset(&peer_addr, 0, sizeof(struct sockaddr_un));
        peer_addr.sun_family = AF_UNIX;
        strncpy(peer_addr.sun_path, addr.c_str(), sizeof(peer_addr.sun_path) - 1);
        peer_addr.sun_path[sizeof(peer_addr.sun_path) - 1] = '\0';

        // make connect
        int ret = -1;
        while ((ret = connect(
                    socket_fd, (const struct sockaddr*)&peer_addr, sizeof(struct sockaddr_un))) ==
               -1)
            if (errno != EINTR && errno != EINPROGRESS) {
                throw std::runtime_error(std::string("Cannot connect socket: ") +
                                         std::to_string(socket_fd) + " to peer: " + addr +
                                         ", error: " + strerror(errno));
            }
    }
    catch (const std::exception& ex) {
        LOG_ERROR(ex.what());
        throw;
    }

    LOG_DEBUG("Socket connected: ", socket_fd, " to peer: ", addr);

    //disable READ
    shutdown(socket_fd, SHUT_RD);
}

std::vector<uint8_t> ipc_tx_connection::send_ipc_memory(
    const std::vector<native::ccl_device::device_ipc_memory_handle>& handles,
    size_t send_rank) const {
    return send_ipc_memory_ext(handles, reinterpret_cast<uint8_t*>(&send_rank), sizeof(send_rank));
}

std::vector<uint8_t> ipc_tx_connection::send_ipc_memory_ext(
    const std::vector<native::ccl_device::device_ipc_memory_handle>& handles,
    const uint8_t* payload,
    size_t payload_size) const {
    LOG_DEBUG("Send IPC handles: ", handles.size(), ", payload size: ", payload_size);
    for (const auto& h : handles) {
        LOG_DEBUG("handle: ", native::to_string(h.get()));
    }

    std::vector<uint8_t> out_raw_data;
    size_t out_raw_data_initial_offset_bytes = payload_size;

    constexpr size_t handle_size =
        native::ccl_device::device_ipc_memory_handle::get_size_for_serialize();

    size_t bytes_to_send = handle_size * handles.size() + out_raw_data_initial_offset_bytes;
    out_raw_data.resize(bytes_to_send);

    // fill send_buf & pid buf
    std::vector<size_t> pids_offset_bytes;
    pids_offset_bytes.reserve(handles.size());

    size_t serialize_offset = out_raw_data_initial_offset_bytes;
    for (const auto& ipc_handle : handles) {
        serialize_offset += ipc_handle.serialize(out_raw_data, serialize_offset);
        pids_offset_bytes.push_back(serialize_offset -
                                    sizeof(native::ccl_device::device_ipc_memory_handle::handle_t));

        LOG_DEBUG("Serialized bytes: ",
                  serialize_offset,
                  ", with pid offset by: ",
                  pids_offset_bytes.back());
    }

    memcpy(reinterpret_cast<uint8_t*>(out_raw_data.data()),
           payload,
           out_raw_data_initial_offset_bytes);

    CCL_ASSERT(serialize_offset == bytes_to_send,
               "Expected data to send and actually serialized are differ");

    ssize_t send_bytes = 0;
    try {
        send_bytes = connection::send_msg_with_pid_data(out_raw_data, pids_offset_bytes);
    }
    catch (const std::exception& ex) {
        LOG_ERROR("Cannot send IPC handles, error: ", ex.what());
        throw;
    }

    LOG_DEBUG("Handles serialized count: ",
              handles.size(),
              ", data bytes: ",
              serialize_offset,
              ", sent bytes: ",
              send_bytes);
    return out_raw_data;
}
} // namespace net
