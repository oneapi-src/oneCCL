.. _mpi: https://software.intel.com/content/www/us/en/develop/tools/mpi-library.html

=======================================================
|product_full|
=======================================================
   
|product_full| (|product_short|) provides an efficient implementation of communication patterns used in deep learning. 

|product_short| features include:

- Built on top of lower-level communication middleware – |mpi|_ and `libfabrics <https://github.com/ofiwg/libfabric>`_.
- Optimized to drive scalability of communication patterns by allowing to easily trade-off compute for communication performance.
- Enables a set of DL-specific optimizations, such as prioritization, persistent operations, or out-of-order execution.
- Works across various interconnects: Intel(R) Omni-Path Architecture, InfiniBand*, and Ethernet.
- Provides common API sufficient to support communication workflows within Deep Learning frameworks (such as PyTorch*, Horovod*).

|product_short| package comprises the |product_short| Software Development Kit (SDK) and the Intel(R) MPI Library Runtime components.

Contents:
=========

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   prerequisites.rst
   installation.rst
   sample.rst

.. toctree::
   :maxdepth: 1
   :caption: Programming Model

   specification.rst
   host_communication.rst
   device_communication.rst

.. toctree::
   :maxdepth: 1
   :caption: General Configuration

   operation_execution.rst
   transport_selection.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Advanced Configuration

   collective_algorithms_selection.rst
   operation_caching.rst
   operation_prioritization.rst
   operation_fusion.rst
   unordered_collectives.rst
   sparse_collectives.rst

.. toctree::
   :maxdepth: 1
   :caption: Reference Materials

   env_variables.rst
   api/library_root.rst

.. toctree::
   :hidden: 
   :caption: Notices and Disclaimers

   legal.rst   
   optimization_notice.rst   