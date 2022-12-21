==================
Sample Application
==================

The sample code below shows how to use |product_short| API to perform allreduce communication for SYCL USM memory.

.. literalinclude:: sample.cpp
   :language: cpp


Build details
*************

#. :ref:`Build <enable_sycl>` |product_short| with ``SYCL`` support (only Intel\ |reg|\  oneAPI DPC++/C++ Compiler is supported).

#. :ref:`Set up <prerequisites>` the library environment.

#. Use the C++ driver with the -fsycl option to build the sample:

   ::

      icpx -o sample sample.cpp -lccl -lmpi


Run the sample
**************

Intel\ |reg|\  MPI Library is required for running the sample. Make sure that MPI environment is set up.

To run the sample, use the following command:

::

    mpiexec <parameters> ./sample

where ``<parameters>`` represents optional mpiexec parameters such as node count, processes per node, hosts, and so on.
