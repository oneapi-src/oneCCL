=====================
Environment Variables
=====================

Collective Algorithms Selection
###############################
oneCCL supports collective operations for the host (CPU) memory buffers and device (GPU) memory buffers. Below you can see how to select the collective algorithm depending on the type of buffer being utilized. 

Device (GPU) Memory Buffers
***************************
Collectives that use GPU buffers are implemented using two phases:

* Scaleup phase. Communication between ranks/processes in the same node.
* Scaleout phase. Communication between ranks/processes on different nodes.

SCALEUP
+++++++
Use the following environment variables to select the scaleup algorithm:

:: 
  
  CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL 

**Syntax**

:: 
  
  CCL_REDUCE_SCATTER_MONOLITHIC_KERNEL=<value>

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value>
     - Description
   * - ``1``
     - Uses compute kernels to transfer data across GPUs for the ``ALLREDUCE``, ``REDUCE``, and ``REDUCE_SCATTER`` collectives. 
   * - ``0``
     - Uses copy engines to transfer data across GPUs for the ``ALLREDUCE``, ``REDUCE``, and ``REDUCE_SCATTER`` collectives. The default value.


**Description**

Set this environment variable to enable compute kernels for the ``ALLREDUCE``, ``REDUCE``, and ``REDUCE_SCATTER`` collectives using device (GPU) buffers. 



CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL
+++++++++++++++++++++++++++++++++++++++++

**Syntax**

::

  CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=<value> 


**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value>
     - Description
   * - ``1``
     - Uses compute kernels to transfer data across GPUs for the ``ALLGATHERV`` collective. The default value.
   * - ``0``
     - Uses copy engines to transfer data across GPUs for the ``ALLGATHERV`` collective. 
  
**Description**

Set this environment variable to enable compute kernels that pipeline data transfers across tiles in the same GPU with data transfers across different GPUs,  for the ALLGATHERV collective using device (GPU) buffers. 



CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL 
+++++++++++++++++++++++++++++++++++++++++++++

**Syntax**

::

  CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=<value> 


**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value>
     - Description
   * - ``1``
     - Uses compute kernels for the ``ALLREDUCE``, ``REDUCE``, and ``REDUCE_SCATTER`` collectives. The default value. 
   * - ``0``
     - Uses copy engines to transfer data across GPUs for the ``ALLREDUCE``, ``REDUCE``, and ``REDUCE_SCATTER collectives``. 
  
**Description**

Set this environment variable to enable compute kernels that pipeline data transfers across tiles in the same GPU with data transfers across different GPUs, for the ALLREDUCE, REDUCE, and REDUCE_SCATTER collectives using the device (GPU) buffers. 
 

CCL_ALLTOALLV_MONOLITHIC_KERNEL 
+++++++++++++++++++++++++++++++

**Syntax**

::

  CCL_ALLTOALLV_MONOLITHIC_KERNEL=<value> 

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left
   
   * - <value>
     - Description
   * - ``1``
     - Uses compute kernels to transfer data across GPUs for the ``ALLTOALL`` and ``ALLTOALLV`` collectives. The default value.
   * - ``0``
     - Uses copy engines to transfer data across GPUs for the ``ALLTOALL`` and ``ALLTOALLV`` collectives.
  
**Description**

Set this environment variable to enable compute kernels for the ``ALLTOALL`` and ``ALLTOALLV`` collectives using device (GPU) buffers
``CCL_<coll_name>_SCALEOUT``. 

CCL_SKIP_SCHEDULER  
++++++++++++++++++

**Syntax**

::

  CCL_SKIP_SCHEDULER=<value> 

**Arguments**

.. list-table:: 
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``1``
     - Enable SYCL kernels.
   * - ``0``
     - Disable SYCL kernels. Default value.
  
**Description**

Setting this environment variable to 1 enables SYCL kernel-based implementation for allgatherv, allreduce, and reduce_scatter.
This is a new optimization that optimizes all message sizes and supports some data types (int32, fp32, fp16, and bf16), sum operation, and single node. 
If the support is unavailable with SYCL kernels, oneCCL falls back to other implementations. Therefore, you can safely set up this environment variable. 

SCALEOUT
++++++++

The following environment variables can be used to select the scaleout algorithm used. 

**Syntax**

To set a specific algorithm for scaleout for the device (GPU) buffers for the whole message size range:

::
   
   CCL_<coll_name>_SCALEOUT=<algo_name>

To set a specific algorithm for scaleout for the device (GPU) buffers for a specific message size range:

::

  CCL_<coll_name>_SCALEOUT="<algo_name_1>[:<size_range_1>][;<algo_name_2>:<size_range_2>][;...]"


Where:

* ``<coll_name>`` is selected from a list of the available collective operations (`Available collectives`_).
* ``<algo_name>`` is selected from a list of the available algorithms for the specific collective operation (`Available collectives`_).
* ``<size_range>`` is described by the left and the right size borders in the ``<left>-<right>`` format. The size is specified in bytes. To specify the maximum message size, use reserved word max. 

oneCCL internally fills the algorithm selection table with sensible defaults. Your input complements the selection table. 
To see the actual table values, set ``CCL_LOG_LEVEL=info``.

.. rubric:: Example

:: 

  CCL_ALLREDUCE_SCALEOUT="recursive_doubling:0-8192;rabenseifner:8193-1048576;ring:1048577-max"

Available Collectives
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
   * - ``ring``
     - Ring-based algorithm


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
     - Two-dimensional algorithm (reduce_scatter + allreduce + allgather). Only available for the host (CPU) buffers.


``ALLTOALL`` algorithms
++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ialltoall``
   * - ``naive``
     - Send to all, receive from all
   * - ``scatter``
     - Scatter-based algorithm


``ALLTOALLV`` algorithms
++++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left

   * - ``direct``
     - Based on ``MPI_Ialltoallv``
   * - ``naive``
     - Send to all, receive from all
   * - ``scatter``
     - Scatter-based algorithm


``BARRIER`` algorithms
++++++++++++++++++++++

.. list-table:: 
   :widths: 25 50
   :align: left
   
   * - ``direct``
     - Based on ``MPI_Ibarrier``
   * - ``ring``
     - Ring-based algorithm

.. note:: The ``BARRIER``` algorithm does not support the ``CCL_BARRIER_SCALEOUT`` environment variable. To change the algorithm for ``BARRIER``, use the ``CCL_BARRIER`` environment variable.


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

.. note:: The ``BCAST`` algorithm does not yet support the ``CCL_BCAST_SCALEOUT`` environment variable. To change the algorithm for ``BCAST``, use the ``CCL_BCAST`` environment variable.


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

.. note:: The ``REDUCE_SCATTER`` algorithm does not yet support the ``CCL_REDUCE_SCATTER_SCALEOUT`` environment variable. To change the algorithm for ``REDUCE_SCATTER``, use the ``CCL_REDUCE_SCATTER`` environment variable.

Host (CPU) Memory Buffers
*************************

CCL_<coll_name>
+++++++++++++++

**Syntax**

To set a specific algorithm for the host (CPU) buffers for the whole message size range:

::

  CCL_<coll_name>=<algo_name>

To set a specific algorithm for the host (CPU) buffers for a specific message size range:

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
       The core number should not exceed the number of cores available on the system. The length of the list should be equal to the number of workers.

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

CCL_ATL_SHM
***********

**Syntax**
:: 

  CCL_ATL_SHM=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``0``
     - Disables the OFI shared memory provider. The default value.
   * - ``1``
     - Enables the OFI shared memory provider. 

**Description**

Set this environment variable to enable the OFI shared memory provider to communicate between ranks in the same node of the host (CPU) buffers.
This capability requires OFI as the transport (``CCL_ATL_TRANSPORT=ofi``). 

The OFI/SHM provider has support to utilize the `Intel(R) Data Streaming Accelerator* (DSA) <https://01.org/blogs/2019/introducing-intel-data-streaming-accelerator>`_. 
To run it with DSA*, you need:
* Linux* OS kernel support for the DSA* shared work queues
* Libfabric* 1.17 or later

To enable DSA, set the following environment variables:   

.. code::

    FI_SHM_DISABLE_CMA=1  
    FI_SHM_USE_DSA_SAR=1  

Refer to Libfabric* Programmer's Manual for the additional details about DSA* support in the SHM provider: https://ofiwg.github.io/libfabric/main/man/fi_shm.7.html. 

CCL_PROCESS_LAUNCHER
********************

**Syntax**
:: 

  CCL_PROCESS_LAUNCHER=<value>
  
**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``hydra``
     - Uses the MPI hydra job launcher. The default value.
   * - ``torch``
     - Uses a torch job launcher. 
   * - ``pmix``
     - Is used with the PALS job launcher that uses the pmix API. The ``mpiexec`` command should be similar to:
       
       ::
         
         CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi mpiexec -np 2 -ppn 2 --pmi=pmix ...
   * - ``none``
     - No job launcher is used. You should specify the values for ``CCL_LOCAL_SIZE and CCL_LOCAL_RANK``.  


**Description**

Set this environment variable to specify the job launcher.


CCL_LOCAL_SIZE
**************

**Syntax**
:: 

  CCL_LOCAL_SIZE=<value>
  
  
**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``SIZE``
     - A total number of ranks on the local host.

**Description**

Set this environment variable to specify a total number of ranks on a local host. 

CCL_LOCAL_RANK
**************

**Syntax**
:: 

  CCL_LOCAL_RANK=<value>
  
**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``RANK``
     - Rank number of the current process on the local host. 
     
 
**Description**
 
Set this environment variable to specify the rank number of the current process in the local host.
  
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

Set this environment variable to specify Intel\ |reg|\  Instrumentation and Tracing Technology (ITT) profiling level.
Once the environment variable is enabled (value > 0), it is possible to collect and display profiling
data for |product_short| using tools such as Intel\ |reg|\  VTune\ |tm|\  Profiler.


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
     - Enable support for SYCL output event (**default**).
   * - ``0``
     - Disable support for SYCL output event.

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


Point-To-Point Operations
*************************

CCL_RECV 
#########

**Syntax**

::

  CCL_RECV=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``direct``
     - Based on the MPI*/OFI* transport layer.
   * - ``topo``
     - Uses XeLinks across GPUs in a multi-GPU node. Default for GPU buffers.  
   * - ``offload``
     - Based on the MPI*/OFI* transport layer and GPU RDMA when supported by the hardware.



CCL_SEND 
#########

**Syntax**

::

  CCL_SEND=<value>

**Arguments**

.. list-table::
   :widths: 25 50
   :header-rows: 1
   :align: left

   * - <value>
     - Description
   * - ``direct``
     - Based on the MPI*/OFI* transport layer.
   * - ``topo``
     - Uses XeLinks across GPUs in a multi-GPU node. Default for GPU buffers.  
   * - ``offload``
     - Based on the MPI*/OFI* transport layer and GPU RDMA when supported by the hardware.
