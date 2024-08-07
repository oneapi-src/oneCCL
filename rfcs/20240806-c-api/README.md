# C API Design Document (RFC)


## Introduction

The oneCCL communication library’s current APIs is defined in the [oneAPI
specification][ccl-spec]. However, other APIs used by similar collective
communication libraries differ from those used by oneCCL. For example, see
[NCCL][nccl-spec] from Nvidia, [RCCL][rccl-spec] from AMD, and hccl from
Habana. This RFC asks for feedback about aligning the oneCCL APIs to be closer
to other vendor libraries, since this facilitates integration with frameworks
and upstreaming to the open source. 

One difference between oneCCL and other vendors communication libraries is that
all other communication libraries have a C API, while oneCCL has a C++ API.
This is because oneCCL was designed to integrate with SYCL, which is based on
C++. One of the goals of oneCCL is to support different hardware and vendors,
such as Intel Data Center GPU Max Series, Intel Core and Intel Xeon family,
Intel Gaudi, Nvidia or AMD GPUs, among others. 

[ccl-spec]: https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneCCL/source/index.html
[hccl-spec]: https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/C_API.html
[nccl-spec]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html
[rccl-spec]: https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/api-library.html#api-library

## Proposal

The proposal is to define a C-like API that aligns with current APIs in other
communication libraries, while introducing a few changes, as described next:

1. Most APIs are C-based like other communication libraries. C++ data
   structures are hidden behind handles returned to the user, such as
   `ccl::stream` and `ccl::comm`. 

2. The API is extended with two C++ API functions to support `sycl::queue`:

   - `onecclResult_t  onecclCreateStream(sycl::queue, &oneccl_stream)`
   - `onecclResult_t onecclReleaseStream(oneccl_stream)`

   Once the sycl::queue is registered, it is hidden behind the ccl stream
   handle

3. Add functions to allow users to explicitly control the lifetime of objects,
   instead of relying on the C++ destructors

   - `onecclResult_t onecclCommFinalize(comm)`
   - `onecclResult_t onecclCommDestroy(comm)`

4. Drop support for out-of-order SYCL queue and SYCL buffers. The current
   oneCCL library support out of order SYCL queues, but this feature is not
   used by the users of the library. In general, the collective operations are
   submitted to an in-order queue. When out-of order behavior is required,
   commands are submitted to a different in-order queue, and the two queues are
   synchronized.
   
5. Drop support for SYCL buffers. Only [Unified Shared Memory][usm-example] is
   supported.

[usm-example]: https://www.intel.com/content/www/us/en/developer/articles/code-sample/dpcpp-usm-code-sample.html

### APIs

The tables below contain the NCCL API, the corresponding new proposed oneCCL
API, and the current oneCCL API. 

#### APIs related with communicator creation.

| NCCL              | oneCCL (proposed C)          | oneCCL (current, C++)   |
|-------------------|------------------------------|-------------------------|
|`cudaError_t`      |`onecclResult_t cudaSetDevice(device)(1)`| N/A          |
|`ncclResult_t ncclGetUniqueId (id)`|	`onecclResult_t onecclGetUniqueId (id)`| `ccl::create_main_kvs(); ccl::create_kvs(main_addr);`|
|`ncclResult_t ncclCommInitRank(comm, size, id, rank)`|`onecclResult_t onecclCommInitRank(comm, size, id, rank)`|`comm cl::create_communicator(size, rank, device, context, kvs) comms ccl:create_communicators(size, rank, device, context, kvs)`|
|`ncclResult_t ncclCommInitRankConfig(comm, size, id, rank, attr)`|`onecclResult_t onecclCommInitRankConfig(comm, size, id, rank, attr)`|`comm ccl:create_communicator(size, rank, device, context, kvs, attr)`|
|`ncclResult_t ncclCommInitAll (comms, ndev, dev_list)`|`onecclResult_t onecclCommInitAll(comms,ndev,dev_list)`| Not currently available.Working on adding support.|
|`ncclCommSplit`    |	Not implemented	              | Not implemented        |
|`nccltResult ncclCommFinalize(comm)`|`onecclResult_t onecclCommFinalize(comm)`| N/A |
|`ncclResult_t ncclCommDestroy(comm)`|`onecclResult_t onecclCommDestroy(comm)`|	Destructor |

Notice that cudaSetDevice(device) is a CUDA call, not a NCCL call. If an
equivalent call is available in SYCL (or calling language), the proposed
onecclSetDevice(device) will not be needed.     

#### APIs related with Collective Communication operations

| NCCL              | oneCCL (proposed C)          | oneCCL (current, C++)   |
|-------------------|------------------------------|-------------------------|
|`ncclResult_t ncclAllgather (sendbuff,recvbuff,count, datatype, op, comm, stream)`|`onecclResult_t onecclAllgather(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream)`|`ccl::event communicator::allgather (2) (sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream, deps)`|
|`ncclResult_t ncclAllreduce(sendbuff,recvbuff, count, datatype, op, comm, stream)`|`onecclResult_t onecclAllreduce(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream)`|`ccl::event 
communicator::allreduce(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream, deps)`|
|`ncclResult_t ncclBroadcast(sendbuff,recvbuff,count, datatype, op, comm, stream)`|`onecclResult_t onecclBroadcast(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream)`|`ccl::event communicator::broadcast (3) (sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream, deps)`|
|`ncclResult_t ncclReduce(sendbuff,recvbuff,count, datatype, op, comm, stream)`|`onecclResult_t onecclReduce(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream)`|`ccl::event communicator::reduce(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream, deps)`|
|`ncclResult_t ncclReduceScatter(sendbuff,recvbuff, count, datatype, op, comm, stream)`|`onecclResult_t onecclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream)`|`ccl::event communicator::reduce_scatter(sendbuff, recvbuff, count, datatype, op, comm, oneccl_stream, deps)`|
| N/A	               |`onecclAlltoall onecclAlltoallv` We could deprecate|`communicator::alltoall communicator::alltoallv`|
| N/A                |`onecclBarrier` We could deprecate and use Allreduce with 1 Byte|`ccl::event communicator::barrier`|

- Currently oneCCL contains Allgatherv, but this will be deprecated in the
  future
- The current API is slightly different, but the next oneCCL release will align
  the Broadcast with the one shown here

#### Group APIs

| NCCL              | oneCCL (proposed C)          | oneCCL (current, C++)   |
|-------------------|------------------------------|-------------------------|
|`ncclResult_t ncclGroupStart()`|`onecclResult_t onecclGroupStart()`| N/A    |
|`ncclResult_t ncclGroupEnd()`  |`onecclResult_t onecclGroupEnd()`  |	N/A    |

#### Point to Point APIs

| NCCL              | oneCCL (proposed C)          | oneCCL (current, C++)   |
|-------------------|------------------------------|-------------------------|
|`ncclResult_t ncclSend(sendbuf, count, datatype, peer, comm, stream)`|`onecclResult_t onecclSend(sendbuf, count, datatype, peer, comm, oneccl_stream)`|`ccl::event communicator::send(sendbuf, count,datatype, peer, comm, oneccl_stream)`|
|`ncclResult_t ncclRecv(…)`|`onecclResult_t onecclRecv(…)`|`communicator::recv`|

#### Other APIs

| NCCL              | oneCCL (proposed C)          | oneCCL (current, C++)   |
|-------------------|------------------------------|-------------------------|
|`ncclResult_t ncclCommCount(comm, size)`|`onecclResult_t onecclCommCount(comm, size)`|`size communicator::size()`|
|`ncclResult_t ncclCommCuDevice(comm, device)`|`onecclResult_t onecclCommGetDevice(comm, device)`|`device communicator::get_device()`|
|`ncclResult_t ncclCommUserRank(comm, rank)`|`onecclResult_t onecclCommUserRank(comm, rank)`|`rank communicator::rank()`|
|`ncclResult_t ncclGetVersion(version)`|`onecclResult_t onecclGetVersion(version)`|`version ccl:get_library_version()`|
|`ncclCommAbort`    |	Not implemented              | N/A                     |
|`ncclCommGetAsyncError`|	Not implemented	         | N/A                     |
|`ncclGetLastError` |	Not implemented              | N/A                     |
|`ncclGetErrorString`| Not implemented             | N/A                     |
