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

#include <initializer_list>
#include <iterator>
#include "ccl_types.hpp"
#include "ccl.hpp"
#include "common/datatype/datatype.hpp"
#include "comp/comp.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "sched/entry/coll/direct/base_coll_entry.hpp"

#include "common/comm/l0/context/device_storage.hpp"
namespace native
{

template<class from_entry>
class l0_allgather_handles_entry : public base_coll_entry
{
public:
    using dependent_entry = from_entry;
    using gpu_comm = typename dependent_entry::gpu_comm;
    using processing_type = typename dependent_entry::processing_type;

    friend class ccl_gpu_comm;

    static constexpr const char* class_name() noexcept
    {
        return "L0_ALLGATHER_HANDLES";
    }

    static constexpr ccl_coll_type type() noexcept
    {
        return ccl_coll_allgatherv;//TODO
    }

    static constexpr ccl_coll_type dependent_type() noexcept
    {
        return dependent_entry::type();
    }

    static constexpr ccl::device_topology_type dependent_topology()
    {
        return dependent_entry::get_topology();
    }

    l0_allgather_handles_entry() = delete;

    l0_allgather_handles_entry(ccl_sched* sched,
                              std::shared_ptr<gpu_comm> comm,
                              std::shared_ptr<ccl::communicator> ccl_comm,
                              device_storage& global_device_storage,
                              std::vector<ccl_device::device_ipc_memory_handle>&& send_data) :
        base_coll_entry(sched),
        comm_addr(comm->template get_comm_data<dependent_topology()>()),
        ccl_communicator(ccl_comm),
        node_device_storage(global_device_storage),
        send_handles(std::move(send_data))
    {
        LOG_DEBUG(class_name()," entry req ", &req, ", rank: ", comm_addr.to_string());
    }

    void start() override
    {
        size_t comm_size = ccl_communicator->size();
        LOG_INFO(class_name()," entry req ", &req, ", rank: ", comm_addr.to_string());

        // serialize data for native allgather algo
        plain_send_data.clear();
        constexpr size_t handle_size = ccl_device::device_ipc_memory_handle::get_size_for_serialize();
        size_t send_bytes = handle_size * send_handles.size() + sizeof(size_t);
        plain_send_data.resize(send_bytes);

        // fill send_buf
        size_t serialize_offset = 0;
        *(reinterpret_cast<size_t*>(plain_send_data.data())) = comm_addr.rank;
        serialize_offset += sizeof(size_t);
        for(auto& ipc_handle : send_handles)
        {
            serialize_offset += ipc_handle.serialize(plain_send_data, serialize_offset);
        }

        CCL_ASSERT(serialize_offset == send_bytes, "Expected data to send and actually serialized are differ");

        //prepare recv_buf
        plain_recv_data.resize(send_bytes * (comm_size)); //all others and me
        recv_bytes.resize(comm_size);
        std::fill(recv_bytes.begin(), recv_bytes.end(), send_bytes);

        int step = 0;
        offsets.resize(comm_size);
        std::generate(offsets.begin(), offsets.end(), [&step, send_bytes]
        {
            int prev = step;
            step += send_bytes;
            return prev;
        });

        LOG_INFO(class_name(), " entry req ", &req,
                 ", send_bytes ", send_bytes,
                 ", waiting recv_bytes: ", plain_recv_data.size());

        request = ccl_communicator->allgatherv((char*)plain_send_data.data(), send_bytes,
                                               (char*)plain_recv_data.data(), recv_bytes.data());
        status = ccl_sched_entry_status_started;

        //TODO prepare foreign_device_ipc_mem_storage handles array
    }

    void update() override
    {
        if (request->test())
        {
            LOG_DEBUG(class_name(), " entry req ", &req, ", rank: ", comm_addr.to_string(), "gathering completed");

            //TODO
            /*
            std::stringstream ss;
            std::copy(plain_recv_data.begin(), plain_recv_data.end(), std::ostream_iterator<int>(ss, ","));
            LOG_INFO(class_name(), " recevied: ", ss.str());
            */
            //get ipc handles
            size_t recv_data_size = plain_recv_data.size();
            const uint8_t* recv_data_start = plain_recv_data.data();

            //TODO make preallocation for ipc_memory_container in start or in cnstructor!!!
            while(recv_data_size > 0)
            {
                size_t received_rank_idx = *(reinterpret_cast<const size_t*>(recv_data_start));
                recv_data_start += sizeof(size_t);
                recv_data_size -= sizeof(size_t);
                LOG_DEBUG("Received IPC rank: ", received_rank_idx, ", on rank: ", comm_addr.to_string());

                //TODO
                size_t num_handles = 0;
                while (num_handles < send_handles.size())
                {
                    /* TODO - do not deserilize  own rank IPC handles, just skip all
                     * Current deserizliation just for testing
                     */
                    auto recv_ip_handle = ccl_device::device_ipc_memory_handle::deserialize<ccl_device::device_ipc_memory_handle>(&recv_data_start,
                                                                                        recv_data_size,
                                                                                        get_platform());


                    std::shared_ptr<ccl_ipc_gpu_comm> ipc_mem_owner;
                    {
                        auto acc = node_device_storage.get_node_storage();
                        auto& device_cont = acc.get();
                        auto& ipc_device_cont = ccl_tuple_get<indexed_device_container<ccl_ipc_gpu_comm>>(device_cont);

                        auto ipc_device_cont_it = ipc_device_cont.find(received_rank_idx);
                        if(ipc_device_cont_it == ipc_device_cont.end())
                        {
                            if(received_rank_idx != comm_addr.rank)
                            {
                                LOG_ERROR("No device owner for ipc handle detected, ipc handle: ", native::to_string(recv_ip_handle->get()),
                                    ", suggested device handle: ", *recv_ip_handle->get_owner().lock(),
                                    ", suggested rank: ", received_rank_idx,
                                    ". Please check your configuration setup");

                                status = ccl_sched_entry_status_failed;
                                abort();
                                return;
                            }
                            LOG_INFO("Find own gpu device, skip");
                            num_handles++;
                            continue;
                        }

                        ipc_mem_owner = ipc_device_cont_it->second;
                    }

                    LOG_DEBUG("Find gpu device: ", ipc_mem_owner->to_string(), ", IPC handle: ", native::to_string(recv_ip_handle->get()));

                    // create IPC memory object & remember in shared storage
                    foreign_device_ipc_mem_storage[ipc_mem_owner].push_back(ipc_mem_owner->get_device().get_ipc_memory(std::move(recv_ip_handle)));

                    num_handles++;
                }
            }

            LOG_INFO("All handles deserialized. Start ipc kernel arguments binding", ", rank: ", comm_addr.to_string());
            for(auto& dev_handle_pair : foreign_device_ipc_mem_storage)
            {
                auto &ipc_device = dev_handle_pair.first;
                ipc_memory_container& handles = dev_handle_pair.second;

                LOG_INFO("Bind kernel arguments: ", handles.size(), ", rank: ", comm_addr.to_string());
                CCL_ASSERT(handles.size() == send_handles.size(), "Received unexpected memory handles count");

                //Bind


                using kernel_ipc_typed = typename dependent_entry::kernel_ipc_typed;
                kernel_ipc_typed& unreach_rank_main_func = ipc_device->get_gpu_kernel<dependent_type(), dependent_topology(), processing_type>();

                typename kernel_ipc_typed::tmp_recv_buf_arg_type tmp_recv_buf = reinterpret_cast<typename kernel_ipc_typed::tmp_recv_buf_arg_type>(handles.at(0).get().pointer);
                unreach_rank_main_func.template set_arg<typename kernel_ipc_typed::tmp_recv_buf_arg>(tmp_recv_buf);

                typename kernel_ipc_typed::income_data_flag_arg_type inc = reinterpret_cast<typename kernel_ipc_typed::income_data_flag_arg_type>(handles.at(1).get().pointer);
                unreach_rank_main_func.template set_arg<typename kernel_ipc_typed::income_data_flag_arg>(inc);

                typename kernel_ipc_typed::ready_to_recv_flag_arg_type ready = reinterpret_cast<typename kernel_ipc_typed::ready_to_recv_flag_arg_type>(handles.at(2).get().pointer);
                unreach_rank_main_func.template set_arg<typename kernel_ipc_typed::ready_to_recv_flag_arg>(ready);
            }

            status = ccl_sched_entry_status_complete;
        }
        else
        {
            LOG_TRACE(class_name(), " entry req ", &req, ", rank: ", comm_addr.to_string(), " is not ready yet");
        }
    }

    const char* name() const override
    {
        return class_name();
    }

protected:

    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                            class_name(),
                            ", dt ", global_data.dtypes->name(dtype),
                            ", cnt ", cnt,
                            ", comm_id ", sched->coll_param.comm->id(),
                            ", req ",&req,
                            "\n");
    }

private:
    topology_addr<dependent_topology()> comm_addr;
    std::shared_ptr<ccl::communicator> ccl_communicator;
    device_storage& node_device_storage;

    std::vector<ccl_device::device_ipc_memory_handle> send_handles;
    std::vector<uint8_t> plain_send_data;
    std::vector<uint8_t> plain_recv_data;
    std::vector<size_t> recv_bytes;
    std::vector<int> offsets;

    std::vector<ccl_device::device_ipc_memory> recv_ipc_handles;

    using ipc_memory_container = std::vector<ccl_device::device_ipc_memory>;
    std::map<std::shared_ptr<ccl_ipc_gpu_comm>, ipc_memory_container> foreign_device_ipc_mem_storage;
    size_t cnt;
    ccl_datatype dtype;

    ccl::communicator::coll_request_t request;
    atl_req_t req{};


};
}
