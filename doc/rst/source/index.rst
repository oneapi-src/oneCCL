.. _mpi: https://software.intel.com/content/www/us/en/develop/tools/mpi-library.html

=======================================================
|product_full|
=======================================================
   
|product_full| (|product_short|) provides an efficient implementation of communication patterns used in deep learning. 

|product_short| features include:

- Built on top of lower-level communication middleware – |mpi|_ and `libfabrics <https://github.com/ofiwg/libfabric>`_.
- Optimized to drive scalability of communication patterns by allowing to easily trade off compute for communication performance.
- Enables a set of DL-specific optimizations, such as prioritization, persistent operations, or out-of-order execution.
- Works across various interconnects: Intel(R) Omni-Path Architecture, InfiniBand*, and Ethernet.
- Provides common API sufficient to support communication workflows within Deep Learning / distributed frameworks (such as PyTorch*, Horovod*).

|product_short| package comprises the |product_short| Software Development Kit (SDK) and the Intel(R) MPI Library Runtime components.


.. toctree::
   :maxdepth: 2
   :caption: Get Started

   introduction/release-notes.rst
   introduction/installation.rst
   introduction/sample.rst
   introduction/cmake-configuration.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   programming-model.rst
   general-configuration.rst
   advanced-configuration.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Reference

   api.rst
   env-variables.rst

.. toctree::
   :hidden: 
   :caption: Notices and Disclaimers

   legal.rst