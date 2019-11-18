Collective Opertations
======================

|product_full| introduces the following list of communication primitives:

- `Allgatherv`_
- `Allreduce`_
- `Reduce`_
- `Alltoall`_
- `Barrier`_
- `Broadcast`_

These operations are collective, meaning that all participants of oneCCL communicator should make a call.

Allgatherv
**********

Allgatherv is a collective communication operation that collects data from all processes within oneCCL communicator. 
Each participant gets the same result data. Different participants can contribute segments of different sizes.

**C version of oneCCL API:**

::

    ccl_status_t ccl_allgatherv(
        const void* send_buf,
        size_t send_count,
        void* recv_buf,
        const size_t* recv_counts,
        ccl_datatype_t dtype,
        const ccl_coll_attr_t* attr,
        ccl_comm_t comm,
        ccl_stream_t stream,
        ccl_request_t* req);

**C++ version of oneCCL API:**

::

    template<class buffer_type>
    coll_request_t communicator::allgatherv(const buffer_type* send_buf, 
                                size_t send_count,
                                buffer_type* recv_buf,
                                const size_t* recv_counts,
                                const ccl::coll_attr* attr,
                                const ccl::stream_t& stream);

send_buf
    the buffer with ``count`` elements of type ``buffer_type`` that stores local data to be sent
recv_buf [out]
    the buffer to store received data, must have the same dimension as ``send_buf``
count
    the number of elements of type ``buffer_type`` to be sent by a participant of oneCCL communicator
recv_counts
    the number of elements of type ``buffer_type`` to be received from each participant of oneCCL communicator
dtype
    datatype of the elements (for C++ API gets inferred from the buffer type)
attr
    optional attributes that customize operation
comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)


.. _allreduce:

Allreduce
*********

Global reduction operations such as sum, max, min, or user-defined functions, where the result is returned to all members of oneCCL communicator.

**C version of oneCCL API:**

::

    ccl_status_t ccl_allreduce(
        const void* send_buf,
        void* recv_buf,
        size_t count,
        ccl_datatype_t dtype,
        ccl_reduction_t reduction,
        const ccl_coll_attr_t* attr,
        ccl_comm_t comm,
        ccl_stream_t stream,
        ccl_request_t* req);

**C++ version of oneCCL API:**

::

    template<class buffer_type>
    coll_request_t communicator::allreduce(const buffer_type* send_buf,
                            buffer_type* recv_buf,
                            size_t count,
                            ccl::reduction reduction,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream);

send_buf
    the buffer with ``count`` elements of ``buffer_type`` that stores local data to be reduced
recv_buf [out]
    the buffer to store reduced result, must have the same dimension as ``send_buf``
count
    the number of elements of ``buffer_type`` in ``send_buf``
dtype
    datatype of the elements (for C++ API gets inferred from the buffer type)
reduction
    type of reduction operation to be applied
attr
    optional attributes that customize operation
comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

Reduce
******

Global reduction operations such as sum, max, min, or user-defined functions, where the result is returned to a single member of oneCCL communicator (root).


**C version of oneCCL API:**

::

    ccl_status_t ccl_reduce(
        const void* send_buf,
        void* recv_buf,
        size_t count,
        ccl_datatype_t dtype,
        ccl_reduction_t reduction,
        size_t root,
        const ccl_coll_attr_t* attr,
        ccl_comm_t comm,
        ccl_stream_t stream,
        ccl_request_t* req);

**C++ version of oneCCL API:**

::

    template<class buffer_type>
    coll_request_t communicator::reduce(const buffer_type* send_buf,
                            buffer_type* recv_buf,
                            size_t count,
                            ccl::reduction reduction,
                            size_t root,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream);

send_buf
    the buffer with ``count`` elements of ``buffer_type`` that stores local data to be reduced
recv_buf [out]
    the buffer to store reduced result, must have the same dimension as ``send_buf``
count
    the number of elements of ``buffer_type`` in ``send_buf``
dtype
    datatype of the elements (for C++ API gets inferred from the buffer type)
reduction
    type of reduction operation to be applied
root
    the rank of the process that gets the result of reduction
attr
    optional attributes that customize operation
comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

The following reduction operations are supported for `Allreduce`_ and `Reduce`_ primitives:

ccl_reduction_sum
    elementwise summation
ccl_reduction_prod
    elementwise multiplication
ccl_reduction_min
    elementwise min
ccl_reduction_max
    elementwise max
ccl_reduction_custom:
    class of user-defined operations

Alltoall
********

oneCCL Alltoall is an extension of oneCCL Allgather for cases when each process
sends different data to each receiver. The :math:`j`-th block sent from the process :math:`i` is received
by the process :math:`j` and is placed in the :math:`i`-th block of ``recvbuf``.
For each pair of processes the amount of sent data must be equal to 
the amount of received data.

**C version of oneCCL API:**

::

    ccl_status_t ccl_alltoall(
                    const void* send_buf,
                    void* recv_buf,
                    size_t count,
                    ccl_datatype_t dtype,
                    const ccl_coll_attr_t* attr,
                    ccl_comm_t comm,
                    ccl_stream_t stream,
                    ccl_request_t* req);

**C++ version of oneCCL API:**

::

    template<class buffer_type>
    coll_request_t communicator::alltoall(const buffer_type* send_buf,
                                        buffer_type* recv_buf,
                                        size_t count,
                                        const ccl::coll_attr* attr,
                                        const ccl::stream_t& stream);


send_buf
    the buffer with ``count`` elements of ``buffer_type`` that stores local data to be sent
recv_buf [out]
    the buffer to store received data, must have the same dimension as ``send_buf``
count
    the number of elements of type ``buffer_type`` to be sent to or received from each participant of oneCCL communicator
dtype
    datatype of the elements (for C++ API gets inferred from the buffer type)
attr
    optional attributes that customize operation
comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

Barrier
*******

Blocking barrier synchronization across all members of oneCCL communicator.

**C version of oneCCL API:**

::

    ccl_status_t ccl_barrier(ccl_comm_t comm,
                            ccl_stream_t stream);

**C++ version of oneCCL API:**

::

    void communicator::barrier(const ccl::stream_t& stream);

comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation

Broadcast
*********

Collective communication operation that broadcasts data from one participant of oneCCL communicator (denoted as root) to all other participants.

**C version of oneCCL API:**

::

    ccl_status_t ccl_bcast(
        void* buf,
        size_t count,
        ccl_datatype_t dtype,
        size_t root,
        const ccl_coll_attr_t* attr,
        ccl_comm_t comm,
        ccl_stream_t stream,
        ccl_request_t* req);

**C++ version of oneCCL API:**

::

    template<class buffer_type>
    col_request_t communicator::bcast(buffer_type* buf, size_t count,
                            size_t root,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream);

buf
    serves as send buffer for root and as receive buffer for other participants
count
    the number of elements of type ``buffer_type`` in ``send_buf``
dtype
    datatype of the elements (for C++ API gets inferred from the buffer type)
root
    the rank of the process that broadcasts the data
attr
    optional attributes that customize the operation
comm
    oneCCL communicator for the operation
stream
    oneCCL stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)


