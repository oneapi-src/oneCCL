Error Handling
==============

Error handling in oneCCL is implemented differently in C and C++ versions of API. 
C version of API uses error codes that are returned by every exposed function, while C++ API uses exceptions.

**C version of oneCCL API:**

::

    typedef enum
    {
        ccl_status_success               = 0,
        ccl_status_out_of_resource       = 1,
        ccl_status_invalid_arguments     = 2,
        ccl_status_unimplemented         = 3,
        ccl_status_runtime_error         = 4,
        ccl_status_blocked_due_to_resize = 5,

        ccl_status_last_value
    } ccl_status_t;

**C++ version of oneCCL API:**

::

    class ccl_error : public std::runtime_error