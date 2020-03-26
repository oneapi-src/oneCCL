GPU support
===========

You can choose between CPU and GPU backends by specifying ``ccl_stream_type`` value during the ccl stream object creation:

- For GPU backend, specify ``ccl_stream_sycl`` as the first argument. 
- For collective operations, which operate on SYCL* stream, C version of |product_short| API expects communication buffers to be ``sycl::buffer*`` objects casted to ``void*``.

The example below demonstrates these concepts.

Example
-------

Consider a simple ``allreduce`` example for GPU. 

#. Create a GPU ccl stream object:

    .. tabs::

        .. group-tab:: |c_api|

            ::

                ccl_stream_create(ccl_stream_sycl, &q, &stream);

        .. group-tab:: |cpp_api|

            ::

                ccl::stream_t stream = ccl::environment::instance().create_stream(cc::stream_type::sycl, &q);

    ``q`` is an object of type ``sycl::queue``.

#. To illustrate the ``ccl_allreduce`` execution, initialize ``sendbuf`` (in real scenario it is provided by application):

    ::

        auto host_acc_sbuf = sendbuf.get_access<mode::write>();
        for (i = 0; i < COUNT; i++) {
            host_acc_sbuf[i] = rank;
        }

#. For demostration purposes only, modify the ``sendbuf`` on the GPU side:

    ::

        q.submit([&](cl::sycl::handler& cgh) {
            auto dev_acc_sbuf = sendbuf.get_access<mode::write>(cgh);
            cgh.parallel_for<class allreduce_test_sbuf_modify>(range<1>{COUNT}, [=](item<1> id) {
                dev_acc_sbuf[id] += 1;
            });
        });

    ``ccl_allreduce`` invocation performs reduction of values from all processes and then distributes the result to all processes.
    In this case, the result is an array with the size equal to the number of processes (:math:`\text{#processes}`),
    where all elements are equal to the sum of arithmetical progression:

    .. math::
        \text{#processes} \cdot (\text{#processes} - 1) / 2

    .. tabs::

        .. group-tab:: |c_api|

            ::

                ccl_allreduce(&sendbuf,
                              &recvbuf,
                              COUNT,
                              ccl_dtype_int,
                              ccl_reduction_sum,
                              NULL, /* attr */
                              NULL, /* comm */
                              stream,
                              &request);
                ccl_wait(request);

        .. group-tab:: |cpp_api|

            ::

                comm.allreduce(sendbuf,
                               recvbuf,
                               COUNT,
                               ccl::reduction::sum,
                               nullptr, /* attr */
                               stream)->wait();

#. Check the correctness of ``ccl_allreduce`` on the GPU:

    ::

        q.submit([&](handler& cgh) {
            auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
            cgh.parallel_for<class allreduce_test_rbuf_check>(range<1>{COUNT}, [=](item<1> id) {
                if (dev_acc_rbuf[id] != size*(size+1)/2) {
                dev_acc_rbuf[id] = -1;
                }
            });
        });

    ::

        if (rank == COLL_ROOT) {
            auto host_acc_rbuf_new = recvbuf.get_access<mode::read>();
            for (i = 0; i < COUNT; i++) {
                if (host_acc_rbuf_new[i] == -1) {
                    cout << "FAILED" << endl;
                    break;
                }
            }
            if (i == COUNT) {
                cout<<"PASSED"<<endl;
            }
        }

.. note::

    When using C version of |product_short| API, it is required to explicitly free the created GPU ccl stream object:

    ::

        ccl_stream_free(stream);

    For C++ version of |product_short| API this is performed implicitly.
