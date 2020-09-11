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

int main(int argc, char **argv) {
    int i = 0;
    int j = 0;
    int size = 0;
    int rank = 0;
    size_t sendbuf_count = 0;
    size_t recvbuf_count = 0;
    ccl_stream_type_t stream_type;

    cl::sycl::queue q;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (create_sycl_queue(argc, argv, q, stream_type) != 0) {
        MPI_Finalize();
        return -1;
    }

    /* create CCL internal KVS */
    auto &env = ccl::environment::instance();
    (void)env;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::environment::instance().create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::environment::instance().create_kvs(main_addr);
    }

    /* create SYCL communicator */
    auto ctx = q.get_context();
    auto communcators = ccl::environment::instance().create_device_communicators(
        size,
        ccl::vector_class<ccl::pair_class<ccl::rank_t, cl::sycl::device>>{
            { rank, q.get_device() } },
        ctx,
        kvs);
    auto &comm = *communcators.begin();

    /* create SYCL stream */
    auto stream = ccl::environment::instance().create_stream(q);

    sendbuf_count = COUNT + rank;
    recvbuf_count = COUNT * size + ((size - 1) * size) / 2;
    cl::sycl::buffer<int, 1> sendbuf(sendbuf_count);
    cl::sycl::buffer<int, 1> expected_buf(recvbuf_count);
    cl::sycl::buffer<int, 1> recvbuf(recvbuf_count);

    std::vector<size_t> recv_counts(size, 0);
    std::iota(recv_counts.begin(), recv_counts.end(), COUNT);

    {
        /* open buffers and initialize them on the CPU side */
        auto host_acc_sbuf = sendbuf.get_access<mode::write>();
        auto host_acc_rbuf = recvbuf.get_access<mode::write>();
        auto expected_acc_buf = expected_buf.get_access<mode::write>();

        for (i = 0; i < sendbuf_count; i++) {
            host_acc_sbuf[i] = rank;
        }
        for (i = 0; i < recvbuf_count; i++) {
            host_acc_rbuf[i] = -1;
        }
        size_t idx = 0;
        for (i = 0; i < size; i++) {
            for (j = 0; j < COUNT + i; j++) {
                expected_acc_buf[idx + j] = i + 1;
            }
            idx += COUNT + i;
        }
    }

    /* open sendbuf and modify it on the target device side */
    /* we try in-place updates so we do the modification in the appropriate place in recvbuf */
    size_t idx_rbuf = 0;
    for (int i = 0; i < rank; i++)
        idx_rbuf += recv_counts[i];

    q.submit([&](handler &cgh) {
        auto dev_acc_sbuf = sendbuf.get_access<mode::read>(cgh);
        auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
        cgh.parallel_for<class allgatherv_test_sbuf_modify>(
            range<1>{ sendbuf_count }, [=](item<1> id) {
                dev_acc_rbuf[idx_rbuf + id[0]] = dev_acc_sbuf[id] + 1;
            });
    });

    handle_exception(q);
    /* invoke ccl_allgatherv on the CPU side */
    /* request in-place operation by providing recvbuf as input */
    auto attr = ccl::environment::instance().create_operation_attr<ccl::allgatherv_attr>();
    comm.allgatherv(recvbuf, sendbuf_count, recvbuf, recv_counts, stream, attr)->wait();

    /* open recvbuf and check its correctness on the target device side */
    q.submit([&](handler &cgh) {
        auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
        auto expected_acc_buf_dev = expected_buf.get_access<mode::read>(cgh);
        cgh.parallel_for<class allgatherv_test_rbuf_check>(
            range<1>{ recvbuf_count }, [=](item<1> id) {
                if (dev_acc_rbuf[id] != expected_acc_buf_dev[id]) {
                    dev_acc_rbuf[id] = -1;
                }
            });
    });

    handle_exception(q);

    /* print out the result of the test on the CPU side */
    if (rank == COLL_ROOT) {
        auto host_acc_rbuf_new = recvbuf.get_access<mode::read>();
        for (i = 0; i < recvbuf_count; i++) {
            if (host_acc_rbuf_new[i] == -1) {
                cout << "FAILED";
                break;
            }
        }
        if (i == recvbuf_count) {
            cout << "PASSED" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
