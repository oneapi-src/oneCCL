Generic Workflow
=================

Below is a generic flow for using C++ API of oneCCL:

#. Initialize the library:

        ::

                ccl::environment::instance();

   Alternatively, you can create communicator objects:

        ::

                ccl::communicator_t comm = ccl::environment::instance().create_communicator();

#. Execute collective operation of choice on this communicator:

        ::

                auto request = comm.allreduce(...);
                request->wait();
