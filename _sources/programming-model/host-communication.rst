.. _`Communicator`: https://spec.oneapi.com/versions/latest/elements/oneCCL/source/spec/main_objects.html#communicator

==================
Host Communication
==================

The communication operations between processes are provided by `Communicator`_.

The example below demonstrates the main concepts of communication on host memory buffers.

.. rubric:: Example

Consider a simple oneCCL ``allreduce`` example for CPU.

1. Create a communicator object with user-supplied size, rank, and key-value store:

   .. code:: cpp

      auto ccl_context = ccl::create_context();
      auto ccl_device = ccl::create_device();

      auto comms = ccl::create_communicators(
         size,
         vector_class<pair_class<size_t, device>>{ { rank, ccl_device } },
         ccl_context,
         kvs);

   Or for convenience use non-vector form without device and context parameters.

   .. code:: cpp

      auto comm = ccl::create_communicator(size, rank, kvs);

2. Initialize ``send_buf`` (in real scenario it is supplied by the user):

   .. code:: cpp

      const size_t elem_count = <N>;

      /* initialize send_buf */
      for (idx = 0; idx < elem_count; idx++) {
         send_buf[idx] = rank + 1;
      }

3. ``allreduce`` invocation performs the reduction of values from all the processes and then distributes the result to all the processes. In this case, the result is an array with ``elem_count`` elements, where all elements are equal to the sum of arithmetical progression:

   .. math::
      p \cdot (p + 1) / 2

   .. code:: cpp

      ccl::allreduce(send_buf,
                     recv_buf,
                     elem_count,
                     reduction::sum,
                     comm).wait();

4. Check the correctness of ``allreduce`` operation:

   .. code:: cpp

      auto comm_size = comm.size();
      auto expected = comm_size * (comm_size + 1) / 2;

      for (idx = 0; idx < elem_count; idx++) {
         if (recv_buf[idx] != expected) {
               std::count << "unexpected value at index " << idx << std::endl;
               break;
         }
      }
