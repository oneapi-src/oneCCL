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
        /**
        * Id of the operation. If specified, new communicator will be created and collective
        * operations with the same @b match_id will be executed in the same order.
        */
        const char* match_id;
    } ccl_coll_attr_t;

``ccl_coll_attr_t`` (``ccl::coll_attr`` in C++ version of API) is extendable structure which serves as a modificator of communication primitive behaviour. 
It can be optionally passed into any collective operation exposed by oneCCL.
