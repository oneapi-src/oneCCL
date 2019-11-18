oneCCL Concepts
===============

|product_full| introduces the following list of concepts:

- `oneCCL Environment`_
- `oneCCL Stream`_
- `oneCCL Communicator`_

oneCCL Environment
******************

oneCCL Enviroment is a singleton object which is used as an entry point into oneCCL library and which is defined only for C++ version of API. 
oneCCL Environment exposes a number of helper methods to manage other CCL objects, such as streams, communicators, etc.

oneCCL Stream
*************

**C version of oneCCL API:**

::

    typedef void* ccl_stream_t;

**C++ version of oneCCL API:**

::

    class stream;
    using stream_t = std::unique_ptr<ccl::stream>;

CCL Stream encapsulates execution context for communication primitives declared by oneCCL specification. It is an opaque handle that is managed by oneCCL API:

**C version of oneCCL API:**

::

    ccl_status_t ccl_stream_create(ccl_stream_type_t type,
                                        void* native_stream,
                                        ccl_stream_t* ccl_stream);
    ccl_status_t ccl_stream_free(ccl_stream_t stream);

**C++ version of oneCCL API:**

::

    class environment
    {
    public:
    ...
        /**
        * Creates a new ccl stream of @c type with @c native stream
        * @param type the @c ccl::stream_type and may be @c cpu or @c sycl (if configured)
        * @param native_stream the existing handle of stream
        */
        stream_t create_stream(ccl::stream_type type = ccl::stream_type::cpu, void* native_stream = nullptr) const;
    }

When you create oneCCL stream object using the API described above, you need to specify the stream type and pass the pointer to the underlying command queue object. 
For example, for oneAPI device you should pass ``ccl::stream_type::sycl`` and ``cl::sycl::queue`` objects.

oneCCL Communicator
*******************

**C version of oneCCL API:**

::

    typedef void* ccl_comm_t;

**C++ version of oneCCL API:**

::

    class communicator;
    using communicator_t = std::unique_ptr<ccl::communicator>;

oneCCL Communicator defines participants of collective communication operations. It is an opaque handle that is managed by oneCCL API:

**C version of oneCCL API:**

::

    ccl_status_t ccl_comm_create(ccl_comm_t* comm,
                                        const ccl_comm_attr_t* attr);
    ccl_status_t ccl_comm_free(ccl_comm_t comm);

**C++ version of oneCCL API:**

::

    class environment
    {
    public:
    ...
        /**
        * Creates a new communicator according to @c attr parameters
        * or creates a copy of global communicator, if @c attr is @c nullptr(default)
        * @param attr
        */
        communicator_t create_communicator(const ccl::comm_attr* attr = nullptr) const;
    }

When you create oneCCL Communicator, you can optionally specify attributes that control the runtime behaviour of oneCCL implementation.

oneCCL Communicator Attributes
------------------------------

::

    typedef struct
    {
        /**
        * Used to split global communicator into parts. Ranks with identical color
        * will form a new communicator.
        */
        int color;
    } ccl_comm_attr_t;

``ccl_comm_attr_t`` (``ccl::comm_attr`` in C++ version of API) is extendable structure that serves as a modificator of communicator behaviour. 
