.. _mpi: https://software.intel.com/content/www/us/en/develop/tools/mpi-library.html

===================
Transport selection
===================

|product_short| supports two transports for inter-node communication: |mpi|_ and `Libfabric* <https://github.com/ofiwg/libfabric>`_.

The transport selection is controlled by :ref:`CCL_ATL_TRANSPORT`.

In case of MPI over Libfabric implementation (for example, Intel\ |reg|\  MPI Library 2019) or in case of direct Libfabric transport, 
the selection of specific Libfabric provider is controlled by the ``FI_PROVIDER`` environment variable.
