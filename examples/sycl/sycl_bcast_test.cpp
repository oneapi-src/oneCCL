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
#include "ccl.h"
#include "sycl_base.hpp"

int main(int argc, char **argv)
{
    int i = 0;
    size_t size = 0;
    size_t rank = 0;

    cl::sycl::queue q;
    cl::sycl::buffer<int, 1> buf(COUNT);

    ccl_request_t request;
    ccl_stream_t stream;

    ccl_init();
    ccl_get_comm_rank(NULL, &rank);
    ccl_get_comm_size(NULL, &size);
    
    if (create_sycl_queue(argc, argv, q) != 0) {
        return -1;
    }
    /* create SYCL stream */
    ccl_stream_create(ccl_stream_sycl, &q, &stream);

    /* open buf and initialize it on the CPU side */
    auto host_acc_buf = buf.get_access<mode::write>();

    for (i = 0; i < COUNT; i++) {
        host_acc_buf[i] = rank;
    }

    /* open buf and modify it on the target device side */
    q.submit([&](cl::sycl::handler& cgh) {
        auto dev_acc_buf = buf.get_access<mode::write>(cgh);
        cgh.parallel_for<class allreduce_test_sbuf_modify>(range<1>{COUNT}, [=](item<1> id) {
            dev_acc_buf[id] += 1;
        });
    });

    /* invoke ccl_bcast on the CPU side */
    ccl_bcast(&buf,
              COUNT,
              ccl_dtype_int,
              COLL_ROOT,
              NULL, /* attr */
              NULL, /* comm */
              stream,
              &request);

    ccl_wait(request);

    /* open buf and check its correctness on the target device side */
    q.submit([&](handler& cgh) {
        auto dev_acc_buf = buf.get_access<mode::write>(cgh);
        cgh.parallel_for<class bcast_test_rbuf_check>(range<1>{COUNT}, [=](item<1> id) {
            if (dev_acc_buf[id] != COLL_ROOT + 1) {
                dev_acc_buf[id] = -1;
            }
        });
    });

    /* print out the result of the test on the CPU side */
    if (rank == COLL_ROOT) {
        auto host_acc_buf_new = buf.get_access<mode::read>();
        for (i = 0; i < COUNT; i++) {
            if (host_acc_buf_new[i] == -1) {
                cout << "FAILED" << std::endl;
                break;
            }
        }
        if (i == COUNT) {
            cout << "PASSED" << std::endl;
        }
    }

    ccl_stream_free(stream);

    ccl_finalize();

    return 0;
}
