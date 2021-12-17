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
#include <unordered_map>
#include "atl/atl_base_comm.hpp"
#include "coll/algorithms/allreduce/allreduce_2d.hpp"
#include "common/comm/communicator_traits.hpp"
#include "common/comm/comm_interface.hpp"
#include "common/comm/comm_id_storage.hpp"
#include "common/comm/atl_tag.hpp"
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/tree.hpp"
#include "common/utils/utils.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/event.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"
#include "types_generator_defines.hpp"
#include "unordered_coll/unordered_coll.hpp"

// index = local_rank, value = global_rank
using ccl_rank2rank_map = std::vector<int>;

class ikvs_wrapper;

inline ccl_stream* get_stream_ptr(const ccl::stream::impl_value_t& stream) {
    if (stream.get() && stream->is_sycl_device_stream())
        return stream.get();
    else
        return nullptr;
}

using ccl_rank2rank_map = std::vector<int>;

class ccl_comm;
namespace ccl {
namespace v1 {
class kvs_interface;
}
} // namespace ccl

// The main purpose of the internal part is to hold shareable parts of ccl_comm which don't need to
// be copied/reset on ccl_comm's copy.
class alignas(CACHELINE_SIZE) ccl_comm_internal {
public:
    static void ccl_comm_reset_thread_barrier();
    ccl_comm_internal() = delete;
    ccl_comm_internal(const ccl_comm_internal& other) = delete;
    ccl_comm_internal& operator=(const ccl_comm_internal& other) = delete;

    ccl_comm_internal(int rank, int size, std::shared_ptr<atl_base_comm> atl);

    ccl_comm_internal(int rank,
                      int size,
                      ccl_rank2rank_map&& ranks,
                      std::shared_ptr<atl_base_comm> atl);

    //TODO non-implemented
    //1) cluster_devices_count (devices 1000) -> (processes 10)
    //2) blocking until all thread -> calls ccl_comm
    //3) return 'thread_count'

    // ccl_comm( {0,1,2,3...}, 1000, kvs )
    // from 20 processes from ranks 0,1,2,3. Each rank contains 10 threads
    // communicator: size in {20} and ranks in {0..19}
    // communicator: return threads count in process {10}
    // communicator: return devices counts per thread in process
    ccl_comm_internal(const std::vector<int>& local_ranks,
                      int comm_size,
                      std::shared_ptr<ccl::kvs_interface> kvs_instance);

    ~ccl_comm_internal() = default;

    int rank() const noexcept {
        return m_rank;
    }

    int size() const noexcept {
        return m_size;
    }

    int pof2() const noexcept {
        return m_pof2;
    }

    const ccl_double_tree& dtree() const {
        return m_dtree;
    }

    void reset(int rank, int size) {
        m_rank = rank;
        m_size = size;
        m_pof2 = ccl_pof2(m_size);
    }

    const ccl_rank2rank_map& get_local2global_map() {
        return m_local2global_map;
    }

    /**
     * Maximum available number of active communicators
     */
    static constexpr ccl_sched_id_t max_comm_count = std::numeric_limits<ccl_comm_id_t>::max();
    /**
     * Maximum value of schedule id in scope of the current communicator
     */
    static constexpr ccl_sched_id_t max_sched_count = std::numeric_limits<ccl_sched_id_t>::max();

    std::shared_ptr<atl_base_comm> atl;
    std::unique_ptr<ccl_unordered_coll_manager> unordered_coll_manager;
    std::unique_ptr<ccl_allreduce_2d_builder> allreduce_2d_builder;

private:
    int m_rank;
    int m_size;
    int m_pof2;

    ccl_rank2rank_map m_local2global_map{};
    ccl_double_tree m_dtree;
};

class alignas(CACHELINE_SIZE) ccl_comm : public ccl::communicator_interface {
public:
    using traits = ccl::host_communicator_traits;

    // traits
    bool is_host() const noexcept override {
        return traits::is_host();
    }

    bool is_cpu() const noexcept override {
        return traits::is_cpu();
    }

    bool is_gpu() const noexcept override {
        return traits::is_gpu();
    }

    bool is_accelerator() const noexcept override {
        return traits::is_accelerator();
    }

    bool is_ready() const override {
        return true;
    }

    ccl::device_index_type get_device_path() const override;
    ccl::communicator_interface::device_t get_device() const override;
    ccl::communicator_interface::context_t get_context() const override;

    const ccl::comm_split_attr& get_comm_split_attr() const override {
        return comm_attr;
    }

    ccl::group_split_type get_topology_type() const override {
        CCL_THROW(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
        return ccl::group_split_type::undetermined;
    }

    ccl::device_topology_type get_topology_class() const override {
        CCL_THROW(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
        return ccl::device_topology_type::undetermined;
    }

    ccl::communicator_interface_ptr split(const ccl::comm_split_attr& attr) override;

    // collectives operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& op_stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override;
    ccl::event barrier_impl(const ccl::stream::impl_value_t& op_stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif // CCL_ENABLE_SYCL

    COMM_IMPL_DECLARATION;
    COMM_IMPL_CLASS_DECLARATION
    COMM_IMPL_SPARSE_DECLARATION;
    COMM_IMPL_SPARSE_CLASS_DECLARATION

    ccl_comm();
    ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs);
    ccl_comm(int size, int rank, ccl::shared_ptr_class<ikvs_wrapper> kvs);
    ccl_comm(ccl::unified_device_type&& device,
             ccl::unified_context_type&& context,
             std::shared_ptr<atl_base_comm> atl);
    ccl_comm(std::shared_ptr<atl_base_comm> atl);

public:
    ccl_comm(int rank,
             int size,
             ccl_comm_id_storage::comm_id&& id,
             std::shared_ptr<atl_base_comm> atl,
             bool share_resources = false,
             bool is_sub_communicator = false);

    ccl_comm(int rank,
             int size,
             ccl_comm_id_storage::comm_id&& id,
             ccl_rank2rank_map&& ranks,
             std::shared_ptr<atl_base_comm> atl,
             bool share_resources = false,
             bool is_sub_communicator = false);

    //TODO non-implemented
    //1) cluster_devices_count (devices 1000) -> (processes 10)
    //2) blocking until all thread -> calls ccl_comm
    //3) return 'thread_count'

    // ccl_comm( {0,1,2,3...}, 1000, kvs )
    // from 20 processes from ranks 0,1,2,3. Each rank contains 10 threads
    // communicator: size in {20} and ranks in {0..19}
    // communicator: return threads count in process {10}
    // communicator: return devices counts per thread in process
    ccl_comm(const std::vector<int>& local_ranks,
             int comm_size,
             std::shared_ptr<ccl::kvs_interface> kvs_instance,
             ccl_comm_id_storage::comm_id&& id,
             bool share_resources = false,
             bool is_sub_communicator = false);

private:
    // This is copy-constructor alike which basically means to copy-construct from src
    // but replace m_id with id's value.
    // We can't have a simple copy constructor here due to comm_id type limitation
    ccl_comm(const ccl_comm& src, ccl_comm_id_storage::comm_id&& id);

public:
    ccl_comm(ccl_comm& src) = delete;
    ccl_comm(ccl_comm&& src) = default;
    ccl_comm& operator=(ccl_comm& src) = delete;
    ccl_comm& operator=(ccl_comm&& src) = default;
    ~ccl_comm() = default;
    std::shared_ptr<atl_base_comm> get_atl_comm() const;
    std::shared_ptr<ccl_comm> get_r2r_comm();
    std::shared_ptr<ccl_comm> get_node_comm();
    std::shared_ptr<ccl_comm> get_even_comm();
    std::shared_ptr<ccl_comm> get_pair_comm();

    // troubleshooting
    std::string to_string() const;
    std::string to_string_ext() const;

    static constexpr int invalid_rank = -1;

    /**
     * Returns the number of @c rank in the global communicator
     * @param rank a rank which is part of the current communicator
     * @return number of @c rank in the global communicator
     */
    int get_global_rank(int rank, bool only_global = false) const;
    int get_rank_from_global(int global_rank) const;

    int rank() const override {
        return comm_rank;
    }

    int size() const override {
        return comm_size;
    }

    int pof2() const noexcept {
        return comm_impl->pof2();
    }

    ccl_comm_id_t id() const noexcept {
        return comm_id->value();
    }

    const ccl_double_tree& dtree() const {
        return comm_impl->dtree();
    }

    std::unique_ptr<ccl_unordered_coll_manager>& get_unordered_coll_manager() {
        return comm_impl->unordered_coll_manager;
    }
    std::unique_ptr<ccl_allreduce_2d_builder>& get_allreduce_2d_builder() {
        return comm_impl->allreduce_2d_builder;
    }

    ccl_comm* create_with_color(int color,
                                ccl_comm_id_storage* comm_ids,
                                bool share_resources) const;

    std::shared_ptr<ccl_comm> clone_with_new_id(ccl_comm_id_storage::comm_id&& id);

    ccl_sched_id_t get_sched_id(bool use_internal_space) {
        ccl_sched_id_t& next_sched_id =
            (use_internal_space) ? next_sched_id_internal : next_sched_id_external;

        ccl_sched_id_t first_sched_id = (use_internal_space)
                                            ? static_cast<ccl_sched_id_t>(0)
                                            : ccl_comm_internal::max_sched_count / 2;

        ccl_sched_id_t max_sched_id = (use_internal_space) ? ccl_comm_internal::max_sched_count / 2
                                                           : ccl_comm_internal::max_sched_count;

        ccl_sched_id_t id = next_sched_id;

        ++next_sched_id;

        if (next_sched_id == max_sched_id) {
            /* wrap the sched numbers around to the start */
            next_sched_id = first_sched_id;
        }

        LOG_DEBUG("sched_id ", id, ", comm_id ", this->id(), ", next sched_id ", next_sched_id);

        return id;
    }

    /**
     * Maximum available number of active communicators
     */
    static constexpr ccl_sched_id_t max_comm_count = ccl_comm_internal::max_comm_count;
    /**
     * Maximum value of schedule id in scope of the current communicator
     */
    static constexpr ccl_sched_id_t max_sched_count = ccl_comm_internal::max_sched_count;

    void allocate_resources();

private:
    // This is an internal part of the communicator, we store there only the fileds should be shared
    // across ccl_comm copies/clones. Everything else must go to ccl_comm.
    std::shared_ptr<ccl_comm_internal> comm_impl;

    ccl::unified_device_type device;
    ccl::unified_context_type context;

    // TODO: double check if these can be moved to comm_impl as shared fields
    std::shared_ptr<ccl_comm> r2r_comm;
    std::shared_ptr<ccl_comm> node_comm;
    std::shared_ptr<ccl_comm> even_comm;
    std::shared_ptr<ccl_comm> pair_comm;
    ccl::comm_split_attr comm_attr;

    // these fields are duplicate with the ones in ccl_comm_internal, but having them here
    // allows to get them without going through the shared_ptr inderection.
    int comm_rank;
    int comm_size;

    // comm_id is not default constructible but ccl_comm is, so use unique_ptr here
    std::unique_ptr<ccl_comm_id_storage::comm_id> comm_id;
    ccl_sched_id_t next_sched_id_internal;
    ccl_sched_id_t next_sched_id_external;

    ccl_comm* get_impl() {
        return this;
    }

    void create_sub_comms(std::shared_ptr<atl_base_comm> atl);
}; // class ccl_comm
