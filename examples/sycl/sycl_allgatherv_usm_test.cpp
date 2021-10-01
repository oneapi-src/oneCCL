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
#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    const size_t count = 10 * 1024 * 1024;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    queue q;
    if (!create_sycl_queue(argc, argv, rank, q)) {
        return -1;
    }

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::shared;
    if (argc > 2) {
        usm_alloc_type = usm_alloc_type_from_string(argv[2]);
    }

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* create buffers */
    auto send_buf = allocator.allocate(count, usm_alloc_type);
    auto recv_buf = allocator.allocate(count * size, usm_alloc_type);

    buffer<int> expected_buf(count * size);
    buffer<int> check_buf(count * size);
    vector<size_t> recv_counts(size, count);

    /* open buffers and modify them on the device side */
    auto e = q.submit([&](auto &h) {
        accessor expected_buf_acc(expected_buf, h, write_only);
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = rank + 1;
            recv_buf[id] = -1;
            for (int i = 0; i < size; i++) {
                expected_buf_acc[i * count + id] = i + 1;
            }
        });
    });

    /* do not wait completion of kernel and provide it as dependency for operation */
    vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));

    /* invoke allagtherv */
    auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
    ccl::allgatherv(send_buf, count, recv_buf, recv_counts, comm, stream, attr, deps).wait();

    /* open recv_buf and check its correctness on the device side */
    q.submit([&](auto &h) {
        accessor expected_buf_acc(expected_buf, h, read_only);
        accessor check_buf_acc(check_buf, h, write_only);
        h.parallel_for(size * count, [=](auto id) {
            if (recv_buf[id] != expected_buf_acc[id]) {
                check_buf_acc[id] = -1;
            }
        });
    });

    if (!handle_exception(q))
        return -1;

    /* print out the result of the test on the host side */
    {
        host_accessor check_buf_acc(check_buf, read_only);
        size_t i;
        for (i = 0; i < size * count; i++) {
            if (check_buf_acc[i] == -1) {
                cout << "FAILED\n";
                break;
            }
        }
        if (i == size * count) {
            cout << "PASSED\n";
        }
    }

    return 0;
}
