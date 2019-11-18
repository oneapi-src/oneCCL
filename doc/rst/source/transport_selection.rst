Transport selection
===================

oneCCL supports two transports for inter-node communication: MPI and libfabrics.

The transport selection is controlled by :ref:`CCL_ATL_TRANSPORT`.

In case of MPI over libfaric implementation (e.g. Intel\ |reg|\  MPI Library 2019) or in case of direct libfabric transport, 
the selection of specific libfaric provider is controlled by ``FI_PROVIDER`` environment variable.
