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
#include <numeric>
#include <vector>
#include <iostream>

#include "oneapi/ccl.hpp"
#include "gtest/gtest.h"
#include "mpi.h"

class alltoallv_test : public ::testing::Test {
protected:
    void SetUp() override {
        ccl::init();

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    void TearDown() override {
        // Don't do finalize if the case has failed, this
        // could lead to a deadlock due to inconsistent state.
        if (HasFatalFailure()) {
            return;
        }

        int is_finalized = 0;
        MPI_Finalized(&is_finalized);

        if (!is_finalized)
            MPI_Finalize();
    }

    int size;
    int rank;
};

// there are 3 ranks, rank 0 is able to send and receive data to/from others(its send and receive total count > 0)
// rank 1 only sends data but not receives it(its recv_count == 0 for all ranks), and rank 2 only receives data but
// not sends it.
// also rank 1 sets its recv_buf to nullptr(it's not used anyway due to 0 recv count), the same is done on rank 2 for send buf
// in the testcase we simply run alltoallv with these parameters and after that check that both rank 0 and rank 2 received
// the correct data.
// TODO: once we add more tests, move some common parts out of this test
TEST_F(alltoallv_test, alltoallv_empty_recv_count) {
    const size_t count = 1000;

    int i = 0;

    ASSERT_EQ(size, 3) << "Test expects 3 ranks";

    sycl::queue q;
    ASSERT_TRUE(q.get_device().is_gpu())
        << "Test expects gpu device, please use SYCL_DEVICE_FILTER accordingly";

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, /*kvs*/ {});

    /* create stream */
    auto stream = ccl::create_stream(q);

    // TODO: find a proper way to choose between shared and device pointers(i.e. env variable)
    /* create buffers */
    auto send_buf = sycl::malloc_device<int>(count * size, q);
    auto recv_buf = sycl::malloc_device<int>(count * size, q);

    // we have 2 ranks in total: rank 1 doesn't receive anything, rank 2 - doesn't send anything
    int empty_recv_rank = 1;
    int empty_send_rank = 2;

    std::vector<size_t> send_counts(size, count);
    std::vector<size_t> recv_counts(size, count);

    // update counts so the corresponding rank doesn't receive anything and others doesn't send anything to it
    send_counts[empty_recv_rank] = 0;
    if (rank == empty_recv_rank) {
        std::fill(recv_counts.begin(), recv_counts.end(), 0);
    }

    recv_counts[empty_send_rank] = 0;
    if (rank == empty_send_rank) {
        std::fill(send_counts.begin(), send_counts.end(), 0);
    }
    q.memset(recv_buf, 0, count * size).wait();

    std::vector<sycl::event> events;
    size_t offset = 0;
    for (int i = 0; i < send_counts.size(); ++i) {
        auto e = q.submit([&](auto& h) {
            h.parallel_for(send_counts[i], [=](auto id) {
                send_buf[id + offset] = i + 1;
            });
        });
        offset += send_counts[i];
        events.push_back(e);
    }

    // do not wait completion of kernel and provide it as dependency for operation
    std::vector<ccl::event> deps;
    for (auto e : events) {
        deps.push_back(ccl::create_event(e));
    }

    // invoke alltoall
    auto attr = ccl::create_operation_attr<ccl::alltoallv_attr>();
    int* invalid_ptr = (int*)0x00ffff;
    // pass an invalid pointer to make sure it's correctly handled and not dereferenced due to 0 count
    if (rank == empty_recv_rank) {
        recv_buf = invalid_ptr;
    }
    else if (rank == empty_send_rank) {
        send_buf = invalid_ptr;
    }

    ccl::alltoallv(send_buf, send_counts, recv_buf, recv_counts, comm, stream, attr, deps).wait();

    // if our rank is the one that didn't receive anything, than just exit and don't do any checking
    if (rank == empty_recv_rank)
        return;

    size_t total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    sycl::buffer<int> check_buf(count * size);
    q.submit([&](auto& h) {
         sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
         h.parallel_for(total_recv, [=, rnk = rank](auto id) {
             // we expect that size - 1 chunks are properly filled with data and the last one is
             // unchanged as we have one rank that doesn't send anything
             if (recv_buf[id] != rnk + 1) {
                 check_buf_acc[id] = -1;
             }
             else {
                 check_buf_acc[id] = 0;
             }
         });
     }).wait_and_throw();

    /* print out the result of the test on the host side */
    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        for (i = 0; i < total_recv; i++) {
            ASSERT_NE(check_buf_acc[i], -1) << "Check failed for receive buffer";
        }
    }

    return;
}
