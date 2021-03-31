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
#include <atomic>
#include <map>
#include <memory>
#include "common/comm/l0/context/scaling_ctx/ipc_ctx_utils.hpp"
#include "common/comm/l0/context/scaling_ctx/ipc_session_key.hpp"
#include "common/comm/l0/modules/supported_modules.hpp"

namespace ccl {
class host_communicator;
}

namespace net {
class ipc_client;
class ipc_rx_connection;
} // namespace net

namespace native {
class ccl_ipc_gpu_comm;

/* Low levels session
 * contains raw data for net operations
 */
class session {
public:
    using raw_data_t = std::vector<uint8_t>;
    using origin_ipc_memory_container = std::vector<ccl_device::device_ipc_memory_handle>;

    session(origin_ipc_memory_container&& ipc_src_momory_handles, size_t source_ipc_device_rank);
    virtual ~session();

    session(const session& src) = delete;
    session& operator=(const session& src) = delete;

    struct recovered_handles_storage {
        using restored_ipc_memory_container = std::vector<ccl_device::device_ipc_memory>;

        raw_data_t raw_data;

        std::map<const ccl_ipc_gpu_comm*, restored_ipc_memory_container> ipc_memory_storage;
    };

    void start(net::ipc_client* client, const std::string& addr);

    bool process(const ccl_ipc_gpu_comm* indexed_ipc_dst_devices,
                 const net::ipc_rx_connection* incoming_connection);

    size_t get_send_tag() const;
    std::string to_string() const;

    virtual void visit(
        const ccl_ipc_gpu_comm* source,
        native::supported_device_modules<ipc_dst_device_coll_module>& ipc_modules) = 0;

protected:
    size_t source_device_rank{};
    raw_data_t source_ipc_raw_data;
    origin_ipc_memory_container source_ipc_memory_storage;
    recovered_handles_storage data_to_recover;

    size_t send_tag{};
    std::atomic<bool> finished;
};

using shared_session_ptr = std::shared_ptr<session>;

/* High level session
 * Contains collective communication data
 */
template <ccl_coll_type coll_type, class kernel_params, ccl::device_topology_type class_id>
struct typed_ipc_session : public session {
    typed_ipc_session(origin_ipc_memory_container&& ipc_src_memory_handles,
                      size_t source_ipc_device_rank)
            : session(std::move(ipc_src_memory_handles), source_ipc_device_rank) {}

    void visit(const ccl_ipc_gpu_comm* source,
               native::supported_device_modules<ipc_dst_device_coll_module>& ipc_modules) override {
        //get appropriate module
        using module_t =
            ipc_dst_device_coll_module<coll_type, ccl::group_split_type::cluster, class_id>;
        std::shared_ptr<module_t>& module_ptr = std::get<::utils::enum_to_underlying(class_id)>(
            std::get<::utils::enum_to_underlying(ccl::group_split_type::cluster)>(
                std::get<coll_type>(ipc_modules)));
        assert(module_ptr);

        // get appropriate kernel
        auto& kernel = module_ptr->template get_class<typename module_t::main_class>()
                           .template get<kernel_params>();
        using kernel_t = typename std::decay<decltype(kernel)>::type;

        // get recovered ipc handles
        auto data_it = data_to_recover.ipc_memory_storage.find(source);
        if (data_it == data_to_recover.ipc_memory_storage.end()) {
            abort();
        }

        // bind data
        const recovered_handles_storage::restored_ipc_memory_container& ipc_handles =
            data_it->second;
        typename kernel_t::tmp_recv_buf_arg_type tmp_recv_buf =
            reinterpret_cast<typename kernel_t::tmp_recv_buf_arg_type>(
                ipc_handles.at(0).get().pointer);
        kernel.template set_arg<typename kernel_t::tmp_recv_buf_arg>(tmp_recv_buf);

        typename kernel_t::income_data_flag_arg_type inc =
            reinterpret_cast<typename kernel_t::income_data_flag_arg_type>(
                ipc_handles.at(1).get().pointer);
        kernel.template set_arg<typename kernel_t::income_data_flag_arg>(inc);

        typename kernel_t::ready_to_recv_flag_arg_type ready =
            reinterpret_cast<typename kernel_t::ready_to_recv_flag_arg_type>(
                ipc_handles.at(2).get().pointer);
        kernel.template set_arg<typename kernel_t::ready_to_recv_flag_arg>(ready);
    }
};

// session owner
// TODO not thread-safe
struct session_table {
    using session_key_t = ipc_session_key;

    template <ccl::device_topology_type class_id, class ipc_invoke_params_type>
    std::shared_ptr<session> create_session(const session_key_t& key,
                                            net::ipc_client* client,
                                            const std::string& peer_addr,
                                            ipc_invoke_params_type&& params,
                                            size_t source_device_rank) {
        using specific_session = typed_ipc_session<ipc_invoke_params_type::get_coll_type(),
                                                   typename ipc_invoke_params_type::kernel_params_t,
                                                   class_id>;
        auto sess =
            std::make_shared<specific_session>(std::move(params.handles), source_device_rank);
        sessions.emplace(key, sess);

        start_session(sess, client, peer_addr);
        return sess;
    }

    std::string to_string() const;
    std::map<session_key_t, shared_session_ptr> sessions{};

    static size_t get_unique_tag();

private:
    void start_session(std::shared_ptr<session> sess,
                       net::ipc_client* client,
                       const std::string& peer_addr);
};

using shared_session_table_ptr = std::shared_ptr<session_table>;
} // namespace native
