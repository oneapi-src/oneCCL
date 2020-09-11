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
    int size = 0;
    int rank = 0;
    ccl_stream_type_t stream_type;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cl::sycl::queue q;
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

    cl::sycl::buffer<int, 1> sendbuf(COUNT * size);
    cl::sycl::buffer<int, 1> recvbuf(COUNT * size);

    {
        /* open buffers and initialize them on the CPU side */
        auto host_acc_sbuf = sendbuf.get_access<mode::write>();
        auto host_acc_rbuf = recvbuf.get_access<mode::write>();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < COUNT; j++) {
                host_acc_sbuf[(i * COUNT) + j] = i;
                host_acc_rbuf[(i * COUNT) + j] = -1;
            }
        }
    }

    /* open sendbuf and modify it on the target device side */
    q.submit([&](handler &cgh) {
        auto dev_acc_sbuf = sendbuf.get_access<mode::write>(cgh);
        cgh.parallel_for<class alltoall_test_sbuf_modify>(range<1>{ COUNT * size },
                                                          [=](item<1> id) {
                                                              dev_acc_sbuf[id] += 1;
                                                          });
    });

    handle_exception(q);

    /* invoke ccl_alltoall on the CPU side */
    auto attr = ccl::environment::instance().create_operation_attr<ccl::alltoall_attr>();
    comm.alltoall(sendbuf, recvbuf, COUNT, stream, attr)->wait();

    /* open recvbuf and check its correctness on the target device side */
    q.submit([&](handler &cgh) {
        auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
        cgh.parallel_for<class alltoall_test_rbuf_check>(range<1>{ COUNT * size }, [=](item<1> id) {
            if (dev_acc_rbuf[id] != rank + 1) {
                dev_acc_rbuf[id] = -1;
            }
        });
    });

    handle_exception(q);

    /* print out the result of the test on the CPU side */
    if (rank == COLL_ROOT) {
        auto host_acc_rbuf_new = recvbuf.get_access<mode::read>();
        for (i = 0; i < COUNT * size; i++) {
            if (host_acc_rbuf_new[i] == -1) {
                cout << "FAILED" << std::endl;
                break;
            }
        }
        if (i == COUNT * size) {
            cout << "PASSED" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
