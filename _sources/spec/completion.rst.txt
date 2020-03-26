Track Communication Progress
===============================

You can track the progress for any of the collective operations provided by |product_short| using the `Test`_ or `Wait`_ function for the `Request`_ object. 

Request
*******

Each collective communication operation of |product_short| returns a request that can be used
to query completion of this operation or to block the execution while the operation is in progress.
|product_short| request is an opaque handle that is managed by corresponding APIs.

.. tabs::

    .. group-tab:: |c_api|

        ::

            typedef void* ccl_request_t;

    .. group-tab:: |cpp_api|

        ::

            /**
            * A request interface that allows the user to track collective operation progress
            */
            class request
            {
            public:
                /**
                * Blocking wait for collective operation completion
                */
                virtual void wait() = 0;

                /**
                * Non-blocking check for collective operation completion
                * @retval true if the operations has been completed
                * @retval false if the operations has not been completed
                */
                virtual bool test() = 0;

                virtual ~request() = default;
            };



Test
****

Non-blocking operation that returns the completion status.

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t ccl_test(ccl_request_t req, int* is_completed);

        req
            requests the handle for communication operation being tracked
        is_completed
            indicates the status: 
            
            - 0 - operation is in progress 
            - otherwise, the operation is completed

    .. group-tab:: |cpp_api|

        ::

            bool request::test();

        Returnes the value that indicates the status: 

        - 0 - operation is in progress 
        - otherwise, the operation is completed

Wait
****

Operation that blocks the execution until communication operation is completed.

.. tabs::

    .. group-tab:: |c_api|

        ::

            ccl_status_t ccl_wait(ccl_request_t req);

        req
            requests the handle for communication operation being tracked

    .. group-tab:: |cpp_api|

        ::

            void request::wait();
