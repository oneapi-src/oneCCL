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

int main(int argc, char **argv)
{
    int i = 0;
    size_t size = 0;
    size_t rank = 0;

    cl::sycl::queue q;
    cl::sycl::buffer<int, 1> sendbuf(COUNT);
    cl::sycl::buffer<int, 1> recvbuf(COUNT);

    auto comm = ccl::environment::instance().create_communicator();

    rank = comm->rank();
    size = comm->size();

    if (create_sycl_queue(argc, argv, q) != 0) {
        return -1;
    }
    /* create SYCL stream */
    auto stream = ccl::environment::instance().create_stream(ccl::stream_type::sycl, &q);

    /* open buffers and initialize them on the CPU side */
    auto host_acc_sbuf = sendbuf.get_access<mode::write>();
    auto host_acc_rbuf = recvbuf.get_access<mode::write>();
    for (i = 0; i < COUNT; i++) {
        host_acc_sbuf[i] = rank;
        host_acc_rbuf[i] = -1;
    }

    /* open sendbuf and modify it on the target device side */
    q.submit([&](handler& cgh){
       auto dev_acc_sbuf = sendbuf.get_access<mode::write>(cgh);
       cgh.parallel_for<class allreduce_test_sbuf_modify>(range<1>{COUNT}, [=](item<1> id) {
           dev_acc_sbuf[id] += 1;
       });
    });

    /* invoke ccl_allreduce on the CPU side */
    comm->allreduce(sendbuf,
                   recvbuf,
                   COUNT,
                   ccl::reduction::sum,
                   nullptr, /* attr */
                   stream)->wait();

    /* open recvbuf and check its correctness on the target device side */
    q.submit([&](handler& cgh){
       auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
       cgh.parallel_for<class allreduce_test_rbuf_check>(range<1>{COUNT}, [=](item<1> id) {
           if (dev_acc_rbuf[id] != size*(size+1)/2) {
               dev_acc_rbuf[id] = -1;
           }
       });
    });

    /* print out the result of the test on the CPU side */
    if (rank == COLL_ROOT){
        auto host_acc_rbuf_new = recvbuf.get_access<mode::read>();
        for (i = 0; i < COUNT; i++) {
            if (host_acc_rbuf_new[i] == -1) {
                cout<<"FAILED"<< std::endl;
                break;
            }
        }
        if (i == COUNT) {
            cout<<"PASSED"<< std::endl;
        }
    }

    return 0;
}
