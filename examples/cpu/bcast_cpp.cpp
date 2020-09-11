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
#include "base.hpp"
#include "mpi.h"

void run_collective(const char* cmd_name,
                    std::vector<float>& buf,
                    ccl::communicator& comm,
                    ccl::broadcast_attr& coll_attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float received;

    if (comm.rank() == COLL_ROOT) {
        for (size_t idx = 0; idx < buf.size(); idx++) {
            buf[idx] = static_cast<float>(idx);
        }
    }
    comm.barrier();

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        auto req = comm.broadcast(buf.data(), buf.size(), COLL_ROOT, coll_attr);
        req->wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < buf.size(); idx++) {
        received = buf[idx];
        if (received != idx) {
            fprintf(stderr,
                    "idx %zu, expected %4.4f, got %4.4f\n",
                    idx,
                    static_cast<float>(idx),
                    received);

            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    comm.barrier();

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

int main() {
    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto& env = ccl::environment::instance();
    (void)env;

    /* create CCL internal KVS */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = env.create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = env.create_kvs(main_addr);
    }

    auto comm = env.create_communicator(size, rank, kvs);
    auto coll_attr = ccl::environment::instance().create_operation_attr<ccl::broadcast_attr>();

    MSG_LOOP(comm, std::vector<float> buf(msg_count);
             coll_attr.set<ccl::operation_attr_id::to_cache>(0);
             run_collective("warmup_bcast", buf, comm, coll_attr);
             coll_attr.set<ccl::operation_attr_id::to_cache>(1);
             run_collective("persistent_bcast", buf, comm, coll_attr);
             coll_attr.set<ccl::operation_attr_id::to_cache>(0);
             run_collective("regular_bcast", buf, comm, coll_attr););

    MPI_Finalize();
    return 0;
}
