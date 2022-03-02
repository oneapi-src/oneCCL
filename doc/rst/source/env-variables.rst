=====================
Environment Variables
=====================

Collective algorithms selection
###############################

CCL_<coll_name>
***************
**Syntax**

To set a specific algorithm for the whole message size range:

::

  CCL_<coll_name>=<algo_name>

To set a specific algorithm for a specific message size range:

::

  CCL_<coll_name>="<algo_name_1>[:<size_range_1>][;<algo_name_2>:<size_range_2>][;...]"

Where:

- ``<coll_name>`` is selected from a list of available collective operations (`Available collectives`_).
- ``<algo_name>`` is selected from a list of available algorithms for a specific collective operation (`Available algorithms`_).
- ``<size_range>`` is described by the left and the right size borders in a format ``<left>-<right>``. 
  Size is specified in bytes. Use reserved word ``max`` to specify the maximum message size.

|product_short| internally fills algorithm selection table with sensible defaults. User input complements the selection table. 
To see the actual table values set ``CCL_LOG_LEVEL=info``.

.. rubric:: Example

:: 

  CCL_ALLREDUCE="recursive_doubling:0-8192;rabenseifner:8193-1048576;ring:1048577-max"

Available collectives
*********************

Available collective operations (``<coll_name>``):

-   ``ALLGATHERV``
-   ``ALLREDUCE``
-   ``ALLTOALL``
-   ``ALLTOALLV``
-   ``BARRIER``
-   ``BCAST``
-   ``REDUCE``
-   ``REDUCE_SCATTER``


Available algorithms
********************

Available algorithms for each collective operation (``<algo_name>``):

``ALLGATHERV`` algorithms
+++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left
   
   * - ``direct``
     - Based on ``MPI_Iallgatherv``
   * - ``naive``
     - Send to all, receive from all
   * - ``flat``
     - Alltoall-based algorithm
   * - ``multi_bcast``
     - Series of broadcast operations with different root ranks


``ALLREDUCE`` algorithms
++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Iallreduce``
   * - ``rabenseifner``
     - Rabenseifner’s algorithm
   * - ``nreduce``
     - May be beneficial for imbalanced workloads
   * - ``ring`` 
     - reduce_scatter + allgather ring.
       Use ``CCL_RS_CHUNK_COUNT`` and ``CCL_RS_MIN_CHUNK_SIZE``
       to control pipelining on reduce_scatter phase.
   * - ``double_tree``
     - Double-tree algorithm
   * - ``recursive_doubling``
     - Recursive doubling algorithm
   * - ``2d``
     - Two-dimensional algorithm (reduce_scatter + allreduce + allgather)


``ALLTOALL`` algorithms
++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ialltoall``
   * - ``naive``
     - Send to all, receive from all


``ALLTOALLV`` algorithms
++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ialltoallv``
   * - ``naive``
     - Send to all, receive from all


``BARRIER`` algorithms
++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left
   
   * - ``direct``
     - Based on ``MPI_Ibarrier``
   * - ``ring``
     - Ring-based algorithm


``BCAST`` algorithms
++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ibcast``
   * - ``ring`` 
     - Ring
   * - ``double_tree``
     - Double-tree algorithm
   * - ``naive``
     - Send to all from root rank


``REDUCE`` algorithms
+++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ireduce``
   * - ``rabenseifner``
     - Rabenseifner’s algorithm
   * - ``tree``
     - Tree algorithm
   * - ``double_tree``
     - Double-tree algorithm


``REDUCE_SCATTER`` algorithms
+++++++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ireduce_scatter_block``
   * - ``ring`` 
     - Use ``CCL_RS_CHUNK_COUNT`` and ``CCL_RS_MIN_CHUNK_SIZE``
       to control pipelining.


CCL_RS_CHUNK_COUNT
++++++++++++++++++
**Syntax**

:: 

  CCL_RS_CHUNK_COUNT=<value>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value> 
     - Description
   * - ``COUNT``
     - Maximum number of chunks.

**Description**

Set this environment variable to specify maximum number of chunks for reduce_scatter phase in ring allreduce.


CCL_RS_MIN_CHUNK_SIZE
+++++++++++++++++++++
**Syntax**

:: 

  CCL_RS_MIN_CHUNK_SIZE=<value>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value> 
     - Description
   * - ``SIZE``
     - Minimum number of bytes in chunk.

**Description**

Set this environment variable to specify minimum number of bytes in chunk for reduce_scatter phase in ring allreduce. Affects actual value of ``CCL_RS_CHUNK_COUNT``.



Workers
#######


The group of environment variables to control worker threads.


CCL_WORKER_COUNT
****************
**Syntax**

:: 

  CCL_WORKER_COUNT=<value>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value> 
     - Description
   * - ``N``
     - The number of worker threads for |product_short| rank (``1`` if not specified).

**Description**

Set this environment variable to specify the number of |product_short| worker threads.


CCL_WORKER_AFFINITY
*******************
**Syntax**

:: 

  CCL_WORKER_AFFINITY=<cpulist>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <cpulist>
     - Description
   * - ``auto``
     - Workers are automatically pinned to last cores of pin domain.
       Pin domain depends from process launcher.
       If ``mpirun`` from |product_short| package is used then pin domain is MPI process pin domain.
       Otherwise, pin domain is all cores on the node.
   * - ``<cpulist>``
     - A comma-separated list of core numbers and/or ranges of core numbers for all local workers, one number per worker.
       The i-th local worker is pinned to the i-th core in the list.
       For example ``<a>,<b>-<c>`` defines list of cores contaning core with number ``<a>``
       and range of cores with numbers from ``<b>`` to ``<c>``.
       The number should not exceed the number of cores available on the system.

**Description**

Set this environment variable to specify cpu affinity for |product_short| worker threads.


CCL_WORKER_MEM_AFFINITY
***********************
**Syntax**

:: 

  CCL_WORKER_MEM_AFFINITY=<nodelist>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <nodelist>
     - Description
   * - ``auto``
     - Workers are automatically pinned to NUMA nodes that correspond to CPU affinity of workers.
   * - ``<nodelist>``
     - A comma-separated list of NUMA node numbers for all local workers, one number per worker.
       The i-th local worker is pinned to the i-th NUMA node in the list.
       The number should not exceed the number of NUMA nodes available on the system.

**Description**

Set this environment variable to specify memory affinity for |product_short| worker threads.


ATL
###


The group of environment variables to control ATL (abstract transport layer).


CCL_ATL_TRANSPORT
*****************
**Syntax**

:: 

  CCL_ATL_TRANSPORT=<value>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value> 
     - Description
   * - ``mpi``
     - MPI transport (**default**).
   * - ``ofi``
     - OFI (libfabric\*) transport.

**Description**

Set this environment variable to select the transport for inter-process communications.


CCL_ATL_HMEM
************
**Syntax**

::

  CCL_ATL_HMEM=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``1``
     - Enable heterogeneous memory support on the transport layer.
   * - ``0``
     - Disable heterogeneous memory support on the transport layer (**default**).

**Description**

Set this environment variable to enable handling of HMEM/GPU buffers by the transport layer.
The actual HMEM support depends on the limitations on the transport level and system configuration.


Multi-NIC
#########


``CCL_MNIC``, ``CCL_MNIC_NAME`` and ``CCL_MNIC_COUNT`` define filters to select multiple NICs.
|product_short| workers will be pinned on selected NICs in a round-robin way.


CCL_MNIC
********
**Syntax**

::

  CCL_MNIC=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``global``
     - Select all NICs available on the node.
   * - ``local``
     - Select all NICs local for the NUMA node that corresponds to process pinning.
   * - ``none``
     - Disable special NIC selection, use a single default NIC (**default**).

**Description**

Set this environment variable to control multi-NIC selection by NIC locality.


CCL_MNIC_NAME
*************
**Syntax**

::

  CCL_MNIC_NAME=<namelist>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <namelist>
     - Description
   * - ``<namelist>``
     - A comma-separated list of NIC full names or prefixes to filter NICs.
       Use the ``^`` symbol to exclude NICs starting with the specified prefixes. For example,
       if you provide a list ``mlx5_0,mlx5_1,^mlx5_2``, NICs with the names ``mlx5_0`` and ``mlx5_1``
       will be selected, while ``mlx5_2`` will be excluded from the selection.

**Description**

Set this environment variable to control multi-NIC selection by NIC names.


CCL_MNIC_COUNT
**************
**Syntax**

::

  CCL_MNIC_COUNT=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``N``
     - The maximum number of NICs that should be selected for |product_short| workers.
       If not specified then equal to the number of |product_short| workers.

**Description**

Set this environment variable to specify the maximum number of NICs to be selected.
The actual number of NICs selected may be smaller due to limitations on transport level or system configuration.


Low-precision datatypes
#######################


The group of environment variables to control processing of low-precision datatypes.


CCL_BF16
********
**Syntax**

::

  CCL_BF16=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``avx512f``
     - Select implementation based on ``AVX512F`` instructions.
   * - ``avx512bf``
     - Select implementation based on ``AVX512_BF16`` instructions.

**Description**

Set this environment variable to select implementation for BF16 <-> FP32 conversion on reduction phase of collective operation.
Default value depends on instruction set support on specific CPU. ``AVX512_BF16``-based implementation has precedence over ``AVX512F``-based one.


CCL_FP16
********
**Syntax**

::

  CCL_FP16=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``f16c``
     - Select implementation based on ``F16C`` instructions.
   * - ``avx512f``
     - Select implementation based on ``AVX512F`` instructions.

**Description**

Set this environment variable to select implementation for FP16 <-> FP32 conversion on reduction phase of collective operation.
Default value depends on instruction set support on specific CPU. ``AVX512F``-based implementation has precedence over ``F16C``-based one.


CCL_LOG_LEVEL
#############
**Syntax**

:: 

  CCL_LOG_LEVEL=<value>

**Arguments**

.. list-table:: 
   :header-rows: 1
   :align: left
   
   * - <value> 
   * - ``error``
   * - ``warn`` (**default**)
   * - ``info``
   * - ``debug``
   * - ``trace``

**Description**

Set this environment variable to control logging level.


CCL_ITT_LEVEL
#############
**Syntax**

::

  CCL_ITT_LEVEL=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``1``
     - Enable support for ITT profiling.
   * - ``0``
     - Disable support for ITT profiling (**default**).

**Description**

Set this environment variable to specify ``Intel(R) Instrumentation and Tracing Technology`` (ITT) profiling level.
Once the environment variable is enabled (value > 0), it is possible to collect and display profiling
data for |product_short| using tools such as ``Intel(R) VTune(TM) Amplifier``.


Fusion
######


The group of environment variables to control fusion of collective operations.


CCL_FUSION
**********

**Syntax**

::

  CCL_FUSION=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``1``
     - Enable fusion of collective operations
   * - ``0``
     - Disable fusion of collective operations (**default**)

**Description**

Set this environment variable to control fusion of collective operations.
The real fusion depends on additional settings described below.


CCL_FUSION_BYTES_THRESHOLD
**************************
**Syntax**

::

  CCL_FUSION_BYTES_THRESHOLD=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``SIZE``
     - Bytes threshold for a collective operation. If the size of a communication buffer in bytes is less than or equal
       to ``SIZE``, then |product_short| fuses this operation with the other ones.

**Description**

Set this environment variable to specify the threshold of the number of bytes for a collective operation to be fused.


CCL_FUSION_COUNT_THRESHOLD
**************************
**Syntax**

::

  CCL_FUSION_COUNT_THRESHOLD=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``COUNT``
     - The threshold for the number of collective operations.
       |product_short| can fuse together no more than ``COUNT`` operations at a time.

**Description**

Set this environment variable to specify count threshold for a collective operation to be fused.


CCL_FUSION_CYCLE_MS
*******************
**Syntax**

::

  CCL_FUSION_CYCLE_MS=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``MS``
     - The frequency of checking for collectives operations to be fused, in milliseconds:

       - Small ``MS`` value can improve latency.
       - Large ``MS`` value can help to fuse larger number of operations at a time.

**Description**

Set this environment variable to specify the frequency of checking for collectives operations to be fused.


CCL_PRIORITY
############
**Syntax**

::

  CCL_PRIORITY=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``direct``
     - You have to explicitly specify priority using ``priority``.
   * - ``lifo``
     - Priority is implicitly increased on each collective call. You do not have to specify priority.
   * - ``none``
     - Disable prioritization (**default**).

**Description**

Set this environment variable to control priority mode of collective operations.


CCL_MAX_SHORT_SIZE
##################
**Syntax**

::

  CCL_MAX_SHORT_SIZE=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``SIZE``
     - Bytes threshold for a collective operation (``0`` if not specified). If the size of a communication buffer in bytes is less than or equal to ``SIZE``, then |product_short| does not split operation between workers. Applicable for ``allreduce``, ``reduce`` and ``broadcast``.

**Description**

Set this environment variable to specify the threshold of the number of bytes for a collective operation to be split.


CCL_SYCL_OUTPUT_EVENT
#####################
**Syntax**

::

  CCL_SYCL_OUTPUT_EVENT=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``1``
     - Enable support for SYCL output event.
   * - ``0``
     - Disable support for SYCL output event (**default**).

**Description**

Set this environment variable to control support for SYCL output event.
Once the support is enabled, you can retrieve SYCL output event from |product_short| event using ``get_native()`` method.
|product_short| event must be associated with |product_short| communication operation.


CCL_ZE_LIBRARY_PATH
###################
**Syntax**

::

  CCL_ZE_LIBRARY_PATH=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``PATH/NAME``
     - Specify the name and full path to the ``Level-Zero`` library for dynamic loading by |product_short|.

**Description**

Set this environment variable to specify the name and full path to ``Level-Zero`` library. The path should be absolute and validated. Set this variable if ``Level-Zero`` is not located in the default path. By default |product_short| uses ``libze_loader.so`` name for dynamic loading.
