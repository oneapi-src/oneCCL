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
#include "comm/comm_interface.hpp"
#include "comm/atl_tag.hpp"
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#include "types_generator_defines.hpp"
#include "topology/topo_manager.hpp"
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

// comm-specific environment
// based on global environment
// and adjusted according to comm parameters
class ccl_comm_env {
public:
    ccl_comm_env(std::shared_ptr<ccl::device> device);
    ccl_comm_env(const ccl_comm_env& other) = delete;
    ccl_comm_env& operator=(const ccl_comm_env& other) = delete;
    ~ccl_comm_env() = default;

    std::string to_string() const;

#ifdef CCL_ENABLE_SYCL
    int get_enable_topo_algo() const {
        return enable_topo_algo;
    }

    ccl::ze::copy_engine_mode get_ze_copy_engine() const {
        return ze_copy_engine;
    }
#endif // CCL_ENABLE_SYCL

private:
    std::shared_ptr<ccl::device> device;

#ifdef CCL_ENABLE_SYCL
    int enable_topo_algo;
    ccl::ze::copy_engine_mode ze_copy_engine;
    ccl::ze::h2d_copy_engine_mode ze_h2d_copy_engine;

#endif // CCL_ENABLE_SYCL
};

// the main purpose of internal comm is to hold
// shareable parts of ccl_comm which don't need to
// be copied/reset on ccl_comm's copy
class alignas(CACHELINE_SIZE) ccl_internal_comm {
public:
    ccl_internal_comm() = delete;
    ccl_internal_comm(const ccl_internal_comm& other) = delete;
    ccl_internal_comm& operator=(const ccl_internal_comm& other) = delete;
    ccl_internal_comm(int comm_id, int rank, int size, std::shared_ptr<atl_base_comm> comm);
    ~ccl_internal_comm() = default;

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

    void reset(int rank, int size);

    std::shared_ptr<atl_base_comm> atl_comm;
    std::unique_ptr<ccl_unordered_coll_manager> unordered_coll_manager;

private:
    int m_rank;
    int m_size;
    int m_pof2;

    ccl_double_tree m_dtree;
};

class alignas(CACHELINE_SIZE) ccl_comm : public ccl::comm_interface {
public:
    static constexpr int invalid_rank = -1;

    void init(int comm_id,
              std::shared_ptr<atl_base_comm> atl_comm,
              bool share_resources = false,
              bool is_sub_communicator = false);
    ccl_comm(int comm_id,
             std::shared_ptr<atl_base_comm> atl_comm,
             bool share_resources,
             bool is_sub_communicator);
    ccl_comm(std::shared_ptr<atl_base_comm> atl_comm,
             bool share_resources = false,
             bool is_sub_communicator = false);
    ccl_comm();

    ccl_comm(ccl_comm& src) = delete;
    ccl_comm(ccl_comm&& src) = default;
    ccl_comm& operator=(ccl_comm& src) = delete;
    ccl_comm& operator=(ccl_comm&& src) = default;
    ~ccl_comm() = default;

    void set_parent_comm(ccl_comm* comm) {
        parent_comm = comm;
    }

    ccl_comm* get_parent_comm() {
        return parent_comm;
    }

    static ccl_comm* create(device_t device,
                            context_t context,
                            int size,
                            int rank,
                            ccl::shared_ptr_class<ccl::kvs_interface> kvs);
    static ccl_comm* create(int size, int rank, ccl::shared_ptr_class<ccl::kvs_interface> kvs);
    static ccl_comm* create(int size, ccl::shared_ptr_class<ccl::kvs_interface> kvs);

private:
    ccl_comm(device_t device, context_t context, std::shared_ptr<atl_base_comm> atl_comm);
    ccl_comm(int size, int rank, ccl::shared_ptr_class<ikvs_wrapper> kvs);
    ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs);

    // copy-constructor with explicit comm_id
    ccl_comm(const ccl_comm& src, int comm_id);

    void create_topo_subcomms();

    ccl_comm* get_impl() {
        return this;
    }

    static std::shared_ptr<ikvs_wrapper> get_kvs_wrapper(std::shared_ptr<ccl::kvs_interface> kvs);

public:
    ccl_comm* create_subcomm(int color, int key = 0) const;

    std::shared_ptr<ccl_comm> clone_with_new_id(int comm_id);

    void allocate_resources();

    ccl::comm_interface_ptr split(const ccl::comm_split_attr& attr) override;

    std::string to_string() const;
    std::string to_string_ext() const;

    /**
     * Returns the number of @c rank in the global communicator
     * @param rank a rank which is part of the current communicator
     * @return number of @c rank in the global communicator
     */
    int get_global_rank(int rank) const;

    int get_rank_from_global(int global_rank) const;
    bool try_get_rank_from_global(int global_rank) const;
    ccl_sched_id_t get_sched_id(bool use_internal_space, bool is_pt2pt);

    device_ptr_t get_device() const override {
        return device_ptr;
    }

    context_ptr_t get_context() const override {
        return context_ptr;
    }

    std::shared_ptr<atl_base_comm> get_atl_comm() const {
        return comm_impl->atl_comm;
    }

    int get_comm_id() const {
        return comm_impl->atl_comm->get_comm_id();
    }

    std::shared_ptr<ccl_comm> get_r2r_comm() const {
        return r2r_comm;
    }

    std::shared_ptr<ccl_comm> get_node_comm() const {
        return node_comm;
    }

    std::shared_ptr<ccl_comm> get_even_comm() const {
        return even_comm;
    }

    std::shared_ptr<ccl_comm> get_pair_comm() const {
        return pair_comm;
    }

    const ccl_rank2rank_map& get_local2global_map() const {
        return local2global_map;
    }

    const ccl::topo_manager& get_topo_manager() const {
        if (parent_comm) {
            return parent_comm->get_topo_manager();
        }
        else {
            return topo_manager;
        }
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::shared_ptr<ccl::ze::fd_manager> get_fd_manager() const {
        if (parent_comm) {
            return parent_comm->get_fd_manager();
        }
        else {
            return fd_manager;
        }
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    std::shared_ptr<ccl_comm_env> get_env() const {
        return env;
    }

    std::unique_ptr<ccl_unordered_coll_manager>& get_unordered_coll_manager() const {
        return comm_impl->unordered_coll_manager;
    }

    int rank() const override {
        return comm_rank;
    }

    int size() const override {
        return comm_size;
    }

    int pof2() const noexcept {
        return comm_impl->pof2();
    }

    int id() const noexcept {
        return get_comm_id();
    }

    const ccl_double_tree& dtree() const {
        return comm_impl->dtree();
    }

    // collectives operation declarations
    ccl::event barrier(const ccl::stream::impl_value_t& stream,
                       const ccl::barrier_attr& attr,
                       const ccl::vector_class<ccl::event>& deps = {}) override;
    ccl::event barrier_impl(const ccl::stream::impl_value_t& stream,
                            const ccl::barrier_attr& attr,
                            const ccl::vector_class<ccl::event>& deps = {});

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif // CCL_ENABLE_SYCL

    COMM_IMPL_DECLARATION;
    COMM_IMPL_CLASS_DECLARATION

private:
    // this is an internal part of the communicator
    // we store there only the fields which should be shared
    // across ccl_comm copies/clones
    // everything else must go to ccl_comm
    std::shared_ptr<ccl_internal_comm> comm_impl;

    ccl_comm* parent_comm = nullptr;
    // ccl::device/context hasn't got a default c-tor
    // that's why we use shared_ptr<ccl::device/context>
    device_ptr_t device_ptr;
    context_ptr_t context_ptr;

    // TODO: double check if these can be moved to comm_impl as shared fields
    std::shared_ptr<ccl_comm> r2r_comm;
    std::shared_ptr<ccl_comm> node_comm;
    std::shared_ptr<ccl_comm> even_comm;
    std::shared_ptr<ccl_comm> pair_comm;

    // these fields are duplicate with the ones in ccl_internal_comm
    // but having them here allows to get them without going
    // through the shared_ptr indirection
    int comm_rank;
    int comm_size;

    ccl_rank2rank_map local2global_map{};
    ccl::topo_manager topo_manager;
    std::shared_ptr<ccl_comm_env> env;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::shared_ptr<ccl::ze::fd_manager> fd_manager;
    void init_ipc_exchange_mode(std::shared_ptr<ccl_comm> comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    ccl_sched_id_t next_sched_id_internal;
    ccl_sched_id_t next_sched_id_external;

}; // class ccl_comm
