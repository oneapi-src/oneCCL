==================
Sample Application
==================

The sample code below shows how to use |product_short| API to perform allreduce communication for SYCL* memory: buffer objects and USM.

.. tabs::

   .. tab:: SYCL Buffers

      .. literalinclude:: ../../../../examples/sycl/sycl_allreduce_test.cpp 
         :language: cpp

   .. tab:: SYCL USM

      .. literalinclude:: ../../../../examples/sycl/sycl_allreduce_usm_test.cpp
         :language: cpp


Build details
*************

#. :ref:`Build <enable_sycl>` |product_short| with ``SYCL`` support (only DPC++ is supported).

#. :ref:`Set up <prerequisites>` the library environment.

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
