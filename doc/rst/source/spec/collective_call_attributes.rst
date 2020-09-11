Collective Call Attributes
*********************************

::

    /* Extendable list of collective attributes */
    typedef struct
    {
        /**
        * Callbacks into application code
        * for pre-/post-processing data
        * and custom reduction operation
        */
        ccl_prologue_fn_t prologue_fn;
        ccl_epilogue_fn_t epilogue_fn;
        ccl_reduction_fn_t reduction_fn;

        /* Priority for collective operation */
        size_t priority;

        /* Blocking/non-blocking */
        int synchronous;

        /* Persistent/non-persistent */
        int to_cache;

        /* Treat buffer as vector/regular - applicable for allgatherv only */
        int vector_buf;

        /**
        * Id of the operation. If specified, new communicator is created and collective
        * operations with the same @b match_id are executed in the same order.
        */
        const char* match_id;

        /* Sparse allreduce specific */
        ccl_sparse_allreduce_completion_fn_t sparse_allreduce_completion_fn;
        ccl::sparse_allreduce_alloc_fn_t sparse_allreduce_alloc_fn;
        const void* sparse_allreduce_fn_ctx;
        ccl::sparse_coalesce_mode sparse_coalesce_mode;

    } ccl_coll_attr_t;

``ccl_coll_attr_t`` (``ccl::coll_attr`` in C++ version of API) is an extendable structure that serves as a modificator of communication primitive behaviour. 
It can be optionally passed into any collective operation exposed by |product_short|.
