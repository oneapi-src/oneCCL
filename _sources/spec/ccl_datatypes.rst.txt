Data Types
==========

|product_short| specification defines the following data types that can be used for collective communication operations:

.. tabs::

    .. group-tab:: |c_api|

        ::

            typedef enum
            {
                ccl_dtype_char   = 0,
                ccl_dtype_int    = 1,
                ccl_dtype_bfp16  = 2,
                ccl_dtype_float  = 3,
                ccl_dtype_double = 4,
                ccl_dtype_int64  = 5,
                ccl_dtype_uint64 = 6,
            } ccl _datatype_t;

    .. group-tab:: |cpp_api|

        ::

            enum class data_type
            {
                dt_char = ccl_dtype_char,
                dt_int = ccl_dtype_int,
                dt_bfp16 = ccl_dtype_bfp16,
                dt_float = ccl_dtype_float,
                dt_double = ccl_dtype_double,
                dt_int64 = ccl_dtype_int64,
                dt_uint64 = ccl_dtype_uint64,
            };

ccl_dtype_char
    Corresponds to *char* in C language
ccl_dtype_int
    Corresponds to *signed int* in C language
ccl_dtype_bfp16
    BFloat16 datatype
ccl_dtype_float
    Corresponds to *float* in C language
ccl_dtype_double
    Corresponds to *double* in C language
ccl_dtype_int64
    Corresponds to *int64_t* in C language
ccl_dtype_uint64
    Corresponds to *uint64_t* in C language
