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
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

// sync_ptrs is used for counting in local kernel_barrier
static size_t *sync_ptrs = nullptr;
constexpr int sync_ptrs_count = 2;

// three tmp buffers - 1: work_buf, 2: tmp_send_buf, 3: tmp_recv_buf
constexpr int tmp_bufs_count = 3;
// tmp_bufs are used as work buf and to copy input/output
static std::array<void *, tmp_bufs_count> tmp_bufs;
// ipc exchanged pointers to remote tmp buffers
static std::array<void *, MAX_NODE_RANKS> remote_tmp_bufs[tmp_bufs_count];
static std::array<void *, MAX_GPUS> remote_even_tmp_bufs[tmp_bufs_count];
static std::array<void *, MAX_TILES> remote_pair_tmp_bufs[tmp_bufs_count];

size_t tmp_buf_size_per_rank = 0;

std::pair<ccl_sched *, ze_handle_exchange_entry *> do_ipc_exchange(ccl_comm *comm,
                                                                   ccl_stream *stream,
                                                                   std::vector<void *> ptrs) {
    ccl_comm *node_comm = comm->get_node_comm().get();
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    for (auto ptr : ptrs) {
        in_buffers.emplace_back(ptr, ccl::ze::ipc_mem_type::memory);
    }

    ccl_coll_param param{};
    param.comm = comm;
    param.stream = stream;
    ccl_coll_attr attr{};
    ccl_sched *sched = ccl_sched::create(param, attr);
    ccl::utils::pt2pt_handle_exchange_info info = {};
    int skip_rank = ccl_comm::invalid_rank;

    ze_handle_exchange_entry *exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
    // start the entry
    exchange_entry->start();
    while (!exchange_entry->is_completed()) {
        exchange_entry->update(); //    128us
    }
    return { sched, exchange_entry };
}

static sycl::queue create_sycl_queue(sycl::queue q, int ordinal, int index) {
    // TODO: should we use the parameter q or a new queue?
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    // Create Command Queue
    ze_command_queue_desc_t Qdescriptor = {};
    Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Qdescriptor.ordinal = ordinal;
    Qdescriptor.index = index;
    Qdescriptor.flags = 0;
    Qdescriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    //ze_command_queue_handle_t ze_cmd_queue = nullptr;
    //ze_result_t result = zeCommandQueueCreate(ze_ctx, ze_dev, &Qdescriptor, &ze_cmd_queue);

    ze_command_list_handle_t ze_imm_cmd_list = nullptr;
    ze_result_t result =
        zeCommandListCreateImmediate(ze_ctx, ze_dev, &Qdescriptor, &ze_imm_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueCreate failed\n";
        return q;
    }

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{
        ze_dev
    };
    sycl::device InteropDevice =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
        ze_ctx,
        std::vector<sycl::device>(1, InteropDevice),
        sycl::ext::oneapi::level_zero::ownership::keep
    };
    sycl::context InteropContext =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

    //sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCQ{
    //  ze_cmd_queue, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCL{
        ze_imm_cmd_list, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep
    };

    //return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCQ, InteropContext);
    return sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCL,
                                                                  InteropContext);
}

// TODO: find the number of link copy engines using L0 APIs
static constexpr int num_lce = 7;
// main copy engine
static sycl::queue q_me;
// linke copy engine
static sycl::queue q_le[num_lce];

static void create_copy_engine_queues(sycl::queue q) {
    for (int i = 0; i < num_lce; i++) {
        q_le[i] = create_sycl_queue(q, 2, i);
    }
    q_me = create_sycl_queue(q, 1, 0);
}

static void comm_barrier(const std::shared_ptr<ccl_comm> comm) {
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        ccl::impl_dispatch disp;
        comm->barrier(disp(ccl::default_stream), ccl::default_barrier_attr).wait();
    }
    else {
        // based on atl invocation from allreduce_scaleout_sycl
        // call ccl::wrapper for MPI/OFI.
        int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl_comm->barrier(ep_idx, req));

        ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
        if (!req.is_completed) {
            // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
        }
        else {
            // The operation was probably blocking, since it finished really quickly
        }
    }
}

void coll_init(ccl_comm *comm, ccl_stream *global_stream) {
    static bool is_initial_invocation = true;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    ccl_barrier_data bd = node_comm->barrier_data();

    // if communicator is used for first time then do ipc exchage
    // to get remote ptrs used for barrier counting and remote tmp bufs
    if (!bd.is_set()) {
        std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
        std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
        std::vector<std::shared_ptr<ccl_comm>> sub_comms{ node_comm, even_comm, pair_comm };

        sycl::queue q = global_stream->get_native_stream();
        sycl::queue q_worker(q.get_device());

        // alloc sync pointers to be used for global comm_barrier across ranks
        constexpr int num_slots = ccl_barrier_data::slots;
        const size_t ptr_count = num_slots * sub_comms.size();
        size_t *ptrs = sycl::malloc_device<size_t>(ptr_count, q);
        q.memset(ptrs, 0, ptr_count * sizeof(size_t)).wait();

        std::vector<void *> ipc_ptrs{ ptrs, ptrs + num_slots, ptrs + 2 * num_slots };

        // do one time initializations
        if (is_initial_invocation) {
            is_initial_invocation = false;

            create_copy_engine_queues(q);

            // allocate sync_ptrs for local kernel barrier
            sync_ptrs = sycl::malloc_device<size_t>(sync_ptrs_count, q);
            q.memset(sync_ptrs, 0, sync_ptrs_count * sizeof(size_t)).wait();

            //set up temp buf to be used for large collectives
            const size_t tmp_buf_size = ccl::global_data::env().sycl_tmp_buf_size / tmp_bufs_count;
            const size_t tmp_buf_size_per_rank_orig =
                tmp_buf_size / ccl::global_data::get().get_local_proc_count();

            // adjust tmp_buf_size_per_rank to align in all ranks
            const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
            tmp_buf_size_per_rank = (tmp_buf_size_per_rank_orig / align_bytes) * align_bytes;

            char *tmp_buf = sycl::aligned_alloc_device<char>(
                CCL_REG_MSG_ALIGNMENT, tmp_buf_size * tmp_bufs_count, q);

            for (int i = 0; i < tmp_bufs_count; i++) {
                tmp_bufs[i] = tmp_buf + i * tmp_buf_size;
            }
        }

        // set up temp buf to be used for small collectives
        const int small_buf_ipc_idx = ipc_ptrs.size();
        char *tmp_buf = sycl::aligned_alloc_device<char>(
            CCL_REG_MSG_ALIGNMENT, ccl_tmp_bufs::buf_size * ccl_tmp_bufs::buf_count, q);
        for (int i = 0; i < ccl_tmp_bufs::buf_count; i++) {
            void *tmp_buf_ptr = tmp_buf + i * ccl_tmp_bufs::buf_size;
            node_comm->set_tmp_buf(tmp_buf_ptr, i);
            ipc_ptrs.push_back(tmp_buf_ptr);
        }

        // add tmp buf pointers of large buffers
        const int large_buf_ipc_idx = ipc_ptrs.size();
        ipc_ptrs.insert(std::end(ipc_ptrs), std::begin(tmp_bufs), std::end(tmp_bufs));

        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ipc_ptrs);

        // add comm_barrier sync pointers to each communicator
        for (size_t i = 0; i < sub_comms.size(); i++) {
            auto remote_ptrs = get_ipc_ptrs<size_t, MAX_NODE_RANKS>(
                sub_comms[i], i, ipc_ptrs[i], sched, q_worker, 1);
            sub_comms[i]->set_barrier_ptrs(remote_ptrs);
        }
        // get ipc pointers for small tmp buffers and add them to node_comm
        for (size_t i = 0, j = small_buf_ipc_idx; i < ccl_tmp_bufs::buf_count; i++, j++) {
            auto remote_ptrs =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm, j, ipc_ptrs[j], sched, q_worker, 1);
            node_comm->set_remote_tmp_bufs(remote_ptrs, i);
        }
        // get ipc pointers for large tmp_buffers
        for (size_t i = 0, j = large_buf_ipc_idx; i < tmp_bufs.size(); i++, j++) {
            remote_tmp_bufs[i] =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm, j, ipc_ptrs[j], sched, q_worker, 1);
            remote_even_tmp_bufs[i] =
                get_ipc_ptrs<void, MAX_GPUS>(even_comm, j, ipc_ptrs[j], sched, q_worker, 1);
            remote_pair_tmp_bufs[i] =
                get_ipc_ptrs<void, MAX_TILES>(pair_comm, j, ipc_ptrs[j], sched, q_worker, 1);
        }

        q_worker.wait();

        delete exchange_entry;
        delete sched;

        auto evt = q.submit([=](sycl::handler &h) {
            h.host_task([node_comm]() {
                comm_barrier(node_comm);
            });
        });
        evt.wait();
    }
}

size_t *get_sync_ptr(bool is_next) {
    assert(sync_ptrs != nullptr);
    static size_t count = 0;
    size_t index = (is_next ? ++count : count) % sync_ptrs_count;
    return sync_ptrs + index;
}

void *get_tmp_buf(int index) {
    return tmp_bufs[index];
}

std::array<void *, MAX_NODE_RANKS> get_remote_node_tmp_buf(int index) {
    return remote_tmp_bufs[index];
}

std::array<void *, MAX_GPUS> get_remote_even_tmp_buf(int index) {
    return remote_even_tmp_bufs[index];
}

std::array<void *, MAX_TILES> get_remote_pair_tmp_buf(int index) {
    return remote_pair_tmp_bufs[index];
}

size_t get_tmp_buf_size_per_rank() {
    return tmp_buf_size_per_rank;
}

std::vector<sycl::event> get_sycl_events(const ccl::vector_class<ccl::event> &deps) {
    std::vector<sycl::event> ret;
    for (auto &dep : deps) {
        ret.push_back(dep.get_native());
    }
    return ret;
}

// invoke the global communication barrier kernel
sycl::event invoke_barrier(const std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           const std::vector<sycl::event> &dep_events,
                           bool use_cpu) {
    sycl::event e;
    if (use_cpu) {
        e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.host_task([comm]() {
                comm_barrier(comm);
            });
        });
    }
    else {
        ccl_barrier_data barrier_data = comm->barrier_inc();
        e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.parallel_for(
                sycl::nd_range<1>(MAX_NODE_RANKS, MAX_NODE_RANKS),
                [=](sycl::nd_item<1> it)
                    [[intel::reqd_sub_group_size(16)]] { comm_barrier(barrier_data, it); });
        });
    }
    return e;
}

int get_num_lce() {
    return num_lce;
}

// get main copy engine
sycl::queue get_mce_queue(sycl::queue q) {
    return q_me;
}

// get link copy engine
sycl::queue get_lce_queue(sycl::queue q, int index) {
    return q_le[index];
}

void copy_data(const int dsize,
               const int N,
               std::array<void *, MAX_GPUS> dst,
               std::array<void *, MAX_GPUS> src,
               const size_t count,
               sycl::queue q,
               std::vector<sycl::event> deps,
               std::vector<sycl::event> &out) {
    for (int i = 0; i < N; i++) {
        sycl::event e = q.submit([=](sycl::handler &h) {
            h.depends_on(deps);
            h.memcpy(dst[i], src[i], dsize * count);
        });
        out.push_back(e);
    }
}
