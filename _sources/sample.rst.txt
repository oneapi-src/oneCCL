==================
Sample Application
==================

The sample code below shows how to use |product_short| API to perform allreduce communication for SYCL* USM buffers: 

::

    #include "sycl_base.hpp"

    using namespace std;
    using namespace sycl;

    int main(int argc, char *argv[]) {

        const size_t count = 10 * 1024 * 1024;

        int i = 0;
        int size = 0;
        int rank = 0;

        ccl::init();

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        queue q;
        if (!create_sycl_queue(argc, argv, q)) {
            MPI_Finalize();
            return -1;
        }

        /* allocate USM buffers */
        auto send_buf = aligned_alloc_shared<int>(64, count, q);
        auto recv_buf = aligned_alloc_shared<int>(64, count, q);

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

        {
            /* open buffers and initialize them on the host side */
            for (i = 0; i < count; i++) {
                send_buf[i] = rank;
                recv_buf[i] = -1;
            }
        }

        /* open send_buf and modify it on the device side */
        q.submit([&](auto &h) {
            h.parallel_for(count, [=](auto id) {
                send_buf[id] += 1;
            });
        });

        if (!handle_exception(q))
            return -1;

        /* invoke allreduce */
        ccl::allreduce(send_buf, recv_buf, count, ccl::reduction::sum, comm, stream).wait();

        /* open recv_buf and check its correctness on the device side */
        buffer<int> check_buf(count);
        q.submit([&](auto &h) {
            accessor check_buf_acc(check_buf, h, write_only);
            h.parallel_for(count, [=](auto id) {
                if (recv_buf[id] != size * (size + 1) / 2) {
                    check_buf_acc[id] = -1;
                }
            });
        });

        if (!handle_exception(q))
            return -1;

        /* print out the result of the test on the host side */
        {
            host_accessor check_buf_acc(check_buf, read_only);
            for (i = 0; i < count; i++) {
                if (check_buf_acc[i] == -1) {
                    cout << "FAILED\n";
                    break;
                }
            }
            if (i == count) {
                cout << "PASSED\n";
            }
        }

        free(send_buf, q);
        free(recv_buf, q);

        MPI_Finalize();

        return 0;
    }




Build details
*************

#. |product_short| should be built with SYCL* support.

#. Set up the library environment (see :doc:`prerequisites`).

#. Use ``clang++`` compiler to build the sample:

    ::

        clang++ -I${CCL_ROOT}/include -L${CCL_ROOT}/lib/ -lsycl -lccl -o sample sample.cpp


Run the sample
**************

Intel\ |reg|\  MPI Library is required for running the sample. Make sure that MPI environment is set up.

To run the sample, use the following command:

::

    mpiexec <parameters> ./sample

where ``<parameters>`` represents optional mpiexec parameters such as node count, processes per node, hosts, and so on.
