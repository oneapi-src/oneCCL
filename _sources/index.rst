.. _mpi: https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html

=======================================================
|product_full|
=======================================================
   
|product_full| (|product_short|) provides an efficient implementation of communication patterns used in deep learning. 

|product_short| features include:

- Built on top of lower-level communication middleware – |mpi|_ and `libfabrics <https://github.com/ofiwg/libfabric>`_.
- Optimized to drive scalability of communication patterns by allowing to easily trade off compute for communication performance.
- Works across various interconnects: InfiniBand*, Cornelis Networks*, and Ethernet.
- Provides common API sufficient to support communication workflows within Deep Learning / distributed frameworks (such as `PyTorch* <https://github.com/pytorch/pytorch>`_, `Horovod* <https://github.com/horovod/horovod>`_).

|product_short| package comprises the |product_short| Software Development Kit (SDK) and the |mpi| Runtime components.


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
