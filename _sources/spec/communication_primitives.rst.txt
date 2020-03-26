Collective Opertations
======================

|product_full| introduces the following list of communication primitives:

- `Allgatherv`_
- `Allreduce`_
- `Alltoall`_
- `Alltoallv`_
- `Barrier`_
- `Broadcast`_
- `Reduce`_

These operations are collective, meaning that all participants of a |product_short| communicator should make a call.

Allgatherv
**********

Allgatherv is a collective communication operation that collects data from all processes within a |product_short| communicator. 
Each participant gets the same result data. Different participants can contribute segments of different sizes.

.. tabs::

    .. group-tab:: |c_api|

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

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            coll_request_t communicator::allgatherv(
                                        const buffer_type* send_buf, 
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
    the number of elements of type ``buffer_type`` to be sent by a participant of a |product_short| communicator
recv_counts
    the number of elements of type ``buffer_type`` to be received from each participant of a |product_short| communicator
dtype
    datatype of the elements (for C++ API it is inferred from the buffer type)
attr
    optional attributes that customize operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)


.. _allreduce:

Allreduce
*********

Allreduce includes global reduction operations such as sum, max, min, or user-defined functions, where the result is returned to all members of |product_short| communicator.

.. tabs::

    .. group-tab:: |c_api|

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

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            coll_request_t communicator::allreduce(
                                    const buffer_type* send_buf,
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
    datatype of the elements (for C++ API it is inferred from the buffer type)
reduction
    type of reduction operation to be applied
attr
    optional attributes that customize operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)


Alltoall
********

Alltoall is a collective operation in which all processes send the same amount of data to each other and receive the same amount of data from each other. 
The :math:`j`-th block sent from the :math:`i`-th process is received by the :math:`j`-th process and is placed in the :math:`i`-th block of ``recvbuf``.

.. tabs::

    .. group-tab:: |c_api|

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

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            coll_request_t communicator::alltoall(
                                                const buffer_type* send_buf,
                                                buffer_type* recv_buf,
                                                size_t count,
                                                const ccl::coll_attr* attr,
                                                const ccl::stream_t& stream);


send_buf
    the buffer with ``count`` elements of ``buffer_type`` that stores local data to be sent
recv_buf [out]
    the buffer to store received data, must have the same dimension as ``send_buf``
count
    the number of elements of type ``buffer_type`` to be sent to or received from each participant of |product_short| communicator
dtype
    datatype of the elements (for C++ API it is inferred from the buffer type)
attr
    optional attributes that customize operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

Alltoallv
*********

Alltoallv is a generalized version of `Alltoall`_. 
Alltoallv adds flexibility by allowing a varying amount of data from each process.

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t CCL_API ccl_alltoallv(
                                const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl_datatype_t dtype,
                                const ccl_coll_attr_t* attr,
                                ccl_comm_t comm,
                                ccl_stream_t stream,
                                ccl_request_t* req);

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            coll_request_t communicator::alltoallv(
                                    const buffer_type* send_buf,
                                    const size_t* send_counts,
                                    buffer_type* recv_buf,
                                    const size_t* recv_counts,
                                    const ccl::coll_attr* attr,
                                    const ccl::stream_t& stream);

send_buf
    the buffer with elements of ``buffer_type`` that stores local data to be sent to all participants
send_counts
    the number of elements of type ``buffer_type`` to be sent to each participant
recv_buf [out]
    the buffer to store received data from all participants
recv_counts
    the number of elements of type ``buffer_type`` to be received from each participant
dtype
    datatype of the elements (for C++ API it is inferred from the buffer type)
attr
    optional attributes that customize operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

Barrier
*******

Blocking barrier synchronization across all members of |product_short| communicator.

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t ccl_barrier(ccl_comm_t comm,
                                     ccl_stream_t stream);

    .. group-tab:: |cpp_api|

        ::

            void communicator::barrier(const ccl::stream_t& stream);

comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation

Broadcast
*********

Collective communication operation that broadcasts data from one participant of |product_short| communicator (denoted as root) to all other participants.

.. tabs::

    .. group-tab:: |c_api|

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

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            col_request_t communicator::bcast(
                                    buffer_type* buf,
                                    size_t count,
                                    size_t root,
                                    const ccl::coll_attr* attr,
                                    const ccl::stream_t& stream);

buf
    serves as send buffer for root and as receive buffer for other participants
count
    the number of elements of type ``buffer_type`` in ``send_buf``
dtype
    datatype of the elements (for C++ API it is inferred from the buffer type)
root
    the rank of the process that broadcasts the data
attr
    optional attributes that customize the operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
req
    object that can be used to track the progress of the operation (returned value for C++ API)

Reduce
******

Reduce includes global reduction operations such as sum, max, min, or user-defined functions, where the result is returned to a single member of |product_short| communicator (root).

.. tabs::

    .. group-tab:: |c_api|

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

    .. group-tab:: |cpp_api|

        ::

            template<class buffer_type>
            coll_request_t communicator::reduce(
                                    const buffer_type* send_buf,
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
    datatype of the elements (for C++ API it is inferred from the buffer type)
reduction
    type of reduction operation to be applied
root
    the rank of the process that gets the result of reduction
attr
    optional attributes that customize operation
comm
    |product_short| communicator for the operation
stream
    |product_short| stream associated with the operation
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
