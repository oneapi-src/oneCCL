============================
Sparse collective operations
============================

Language models typically feature huge embedding tables within their topology. 
This makes straight-forward gradient computation followed by ``allreduce`` for the whole set of weights not feasible in practice
due to both performance and memory footprint reasons. 
Thus, gradients for such layers are usually computed for a smaller sub-tensor on each iteration, and communication pattern,
which is required to average the gradients across processes, does not map well to allreduce API. 

To address these scenarios, frameworks usually utilize the ``allgather`` primitive, which may be suboptimal if there is a lot of intersections between sub-tensors from different processes.

Latest research paves the way to handling such communication in a more optimal manner, but each of these approaches has its own application area. 
The ultimate goal of |product_short| is to provide a common API for sparse collective operations that would simplify framework design by allowing under-the-hood implementation of different approaches.

|product_short| can work with sparse tensors represented by two tensors: one for indices and one for values.


Sparse allreduce is a collective communication operation that makes global reduction operation on sparse buffers from all ranks of communicator and distributes result back to all ranks. Sparse buffers are defined by separate index and value buffers.

.. code:: cpp

    ccl::event sparse_allreduce(const void* send_ind_buf,
                                size_t send_ind_count,
                                const void* send_val_buf,
                                size_t send_val_count,
                                void* recv_ind_buf,
                                size_t recv_ind_count,
                                void* recv_val_buf,
                                size_t recv_val_count,
                                ccl::datatype ind_dtype,
                                ccl::datatype val_dtype,
                                ccl::reduction rtype,
                                const ccl::communicator& comm,
                                const ccl::stream& stream,
                                const ccl::sparse_allreduce_attr& attr = ccl::default_sparse_allreduce_attr,
                                const ccl::vector_class<ccl::event>& deps = {});

send_ind_buf
    the buffer of indices with ``send_ind_count`` elements of type ``ind_dtype``
send_ind_count
    the number of elements of type ``ind_type`` in ``send_ind_buf``
send_val_buf
    the buffer of values with ``send_val_count`` elements of type ``val_dtype``
send_val_count
    the number of elements of type ``val_type`` in  ``send_val_buf``
recv_ind_buf [out]
    the buffer to store reduced indices, unused
recv_ind_count [out]
    the number of elements in ``recv_ind_buf``, unused
recv_val_buf [out]
     the buffer to store reduced values, unused
recv_val_count [out]
    the number of elements in ``recv_val_buf``, unused
ind_dtype
    the datatype of elements in ``send_ind_buf`` and ``recv_ind_buf``
val_dtype
    the the datatype of elements in ``send_val_buf`` and ``recv_val_buf``
rtype
    the type of the reduction operation to be applied
comm
    the communicator that defines a group of ranks for the operation
stream
    an optional stream associated with the operation
attr
    optional attributes to customize the operation
deps
    an optional vector of the events that the operation should depend on
return ``event``
    an object to track the progress of the operation


For ``sparse_allreduce``, a completion callback or an allocation callback is required.

Use the following fields in operation attribute:

- ``completion_fn`` - a completion callback function pointer
- ``alloc_fn`` - an allocation callback function pointer
- ``fn_ctx``- an user context pointer of type ``void*``

Completion callback should follow the signature:

.. code:: cpp

        typedef void (*completion_fn)
        (
            const void*,   // idx_buf
            size_t,        // idx_count
            ccl::datatype, // idx_dtype
            const void*,   // val_buf
            size_t,        // val_count
            ccl::datatype, // val_dtype
            const void*    // user_context
        );

Note that ``idx_buf`` and ``val_buf`` are temporary buffers.
Thus, the data from these buffers should be copied. Use ``user_context`` for this purpose.


Allocation callback should follow the signature:

.. code:: cpp

        typedef void (*alloc_fn)
        (
            size_t,        // idx_count
            ccl::datatype, // idx_dtype
            size_t,        // val_count
            ccl::datatype, // val_dtype
            const void*,   // user_context
            void**,        // out_idx_buf
            void**         // out_val_buf
        );


For more details, refer to `this example <https://github.com/oneapi-src/oneCCL/blob/master/examples/cpu/sparse_allreduce.cpp>`_.


.. note::
    WARNING: ``ccl::sparse_allreduce`` is experimental and subject to change.
