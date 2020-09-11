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

int main(int argc, char** argv) {
    int i = 0;
    size_t size = 0;
    size_t rank = 0;
    ccl_stream_type_t stream_type;

    cl::sycl::queue q;
    cl::sycl::buffer<int, 1> buf(COUNT);

    auto comm = ccl::environment::instance().create_communicator();

    rank = comm->rank();
    size = comm->size();

    if (create_sycl_queue(argc, argv, q, stream_type) != 0) {
        return -1;
    }
    /* create SYCL stream */
    auto stream = ccl::environment::instance().create_stream(q);

    {
        /* open buf and initialize it on the CPU side */
        auto host_acc_sbuf = buf.get_access<mode::write>();
        for (i = 0; i < COUNT; i++) {
            if (rank == COLL_ROOT)
                host_acc_sbuf[i] = rank;
            else
                host_acc_sbuf[i] = 0;
        }
    }

    /* open buf and modify it on the target device side */
    q.submit([&](handler& cgh) {
        auto dev_acc_sbuf = buf.get_access<mode::write>(cgh);
        cgh.parallel_for<class bcast_test_sbuf_modify>(range<1>{ COUNT }, [=](item<1> id) {
            dev_acc_sbuf[id] += 1;
        });
    });

    handle_exception(q);

    /* invoke ccl_bcast on the CPU side */
    comm->bcast(buf,
                COUNT,
                COLL_ROOT,
                nullptr, /* attr */
                stream)
        ->wait();

    /* open buf and check its correctness on the target device side */
    q.submit([&](handler& cgh) {
        auto dev_acc_rbuf = buf.get_access<mode::write>(cgh);
        cgh.parallel_for<class bcast_test_rbuf_check>(range<1>{ COUNT }, [=](item<1> id) {
            if (dev_acc_rbuf[id] != COLL_ROOT + 1) {
                dev_acc_rbuf[id] = -1;
            }
        });
    });

    handle_exception(q);

    /* print out the result of the test on the CPU side */
    if (rank == COLL_ROOT) {
        auto host_acc_rbuf_new = buf.get_access<mode::read>();
        for (i = 0; i < COUNT; i++) {
            if (host_acc_rbuf_new[i] == -1) {
                cout << "FAILED" << std::endl;
                break;
            }
        }
        if (i == COUNT) {
            cout << "PASSED" << std::endl;
        }
    }

    return 0;
}
