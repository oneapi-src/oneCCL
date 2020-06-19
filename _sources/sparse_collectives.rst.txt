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

The ``sparse_allreduce`` function has the following parameters:

-	``send_ind_buf`` - a buffer of indices with ``send_ind_count`` elements of ``index_dtype``
-	``send_int_count`` - the number of ``send_ind_buf`` elements of type ``index_type``
-	``send_val_buf`` - a buffer of values with ``send_val_count`` elements of ``value_dtype``
-	``send_val_count`` - the number of ``send_val_buf`` elements of type ``value_type``
-	``recv_ind_buf`` - a buffer to store reduced indices (ignored for now) 
-	``recv_ind_count`` - the number of reduced indices (ignored for now)
-	``recv_val_buf``` - a buffer to store reduced values (ignored for now)
-	``recv_val_count`` - the number of reduced values (ignored for now)
-	``index_dtype`` - index type of elements in ``send_ind_buf`` and ``recv_ind_buf`` buffers
-	``value_dtype`` - data type of elements in ``send_val_buf`` and ``recv_val_buf`` buffers
-	``reduction`` - the type of reduction operation to be applied
-	``attributes`` - attributes that customize operation
-	returns ``ccl::request`` object to track the progress of the operation

For ``sparse_allreduce``, a completion callback is required to get the results.
Use the following :ref:`Collective Call Attributes` fields:

-	``sparse_allreduce_completion_fn`` - a completion callback function pointer (must not be set to ``NULL``)
-	``sparse_allreduce_completion_ctx``- a user context pointer of type ``void*``

Here is an example of a function definition for ``sparse_allreduce`` completion callback:

::

  ccl_status_t sparse_allreduce_completion_fn(
      const void* indices_buf, size_t indices_count, ccl_datatype_t indices_datatype,
      const void* values_buf, size_t values_count, ccl_datatype_t values_datatype,
      const ccl_fn_context_t* fn_ctx, const void* user_ctx)
  {
      /* 
        Note that indices_buf and values_buf are temporary buffers.
        Thus, the data from these buffers should be copied. Use user_ctx for
        this purpose. 
      */
      return ccl_status_success;
  }

For more details, refer to `this example <https://github.com/oneapi-src/oneCCL/blob/master/examples/cpu/sparse_test_algo.hpp>`_