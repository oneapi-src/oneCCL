oneCCL Concepts
===============

|product_full| introduces the following list of concepts:

- `oneCCL Environment`_
- `oneCCL Stream`_
- `oneCCL Communicator`_

oneCCL Environment
******************

|product_short| Environment is a singleton object that is used as an entry point into |product_short|. It is defined only for C++ version of API. 
|product_short| Environment exposes a number of helper methods to manage other CCL objects, such as streams or communicators.

oneCCL Stream
*************

.. tabs::

    .. group-tab:: |c_api|

        ::

            typedef void* ccl_stream_t;

    .. group-tab:: |cpp_api|

        ::

            class stream;
            using stream_t = std::unique_ptr<ccl::stream>;

CCL Stream encapsulates execution context for communication primitives declared by |product_short| specification. It is an opaque handle that is managed by |product_short| API:

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t ccl_stream_create(ccl_stream_type_t type,
                                           void* native_stream,
                                           ccl_stream_t* ccl_stream);
            ccl_status_t ccl_stream_free(ccl_stream_t stream);

    .. group-tab:: |cpp_api|  

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

When you create a |product_short| stream object using the API described above, you need to specify the stream type and pass the pointer to the underlying command queue object. 
For example, for oneAPI device you should pass ``ccl::stream_type::sycl`` and ``cl::sycl::queue`` objects.

oneCCL Communicator
*******************

.. tabs::

    .. group-tab:: |c_api|

        ::

            typedef void* ccl_comm_t;

    .. group-tab:: |cpp_api|

        ::

            class communicator;
            using communicator_t = std::unique_ptr<ccl::communicator>;

|product_short| Communicator defines participants of collective communication operations. It is an opaque handle that is managed by |product_short| API:

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t ccl_comm_create(ccl_comm_t* comm,
                                         const ccl_comm_attr_t* attr);
            ccl_status_t ccl_comm_free(ccl_comm_t comm);

    .. group-tab:: |cpp_api|

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

When you create a |product_short| Communicator, you can optionally specify attributes that control the runtime behaviour of |product_short| implementation.

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

``ccl_comm_attr_t`` (``ccl::comm_attr`` in C++ version of API) is an extendable structure that serves as a modificator of communicator behaviour. 
