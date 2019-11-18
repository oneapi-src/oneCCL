Track Communication Progress
===============================

The progress for any of the collective operations provided by oneCCL can be tracked using `Test`_ or `Wait`_ function for `Request`_ object. 

Request
*******

Each collective communication operation of oneCCL returns a request that can be used
to query completion of this operation or to block the execution while the operation is in progress.
oneCCL request is an opaque handle that is managed by corresponding APIs.

**C version of oneCCL API:**

::

    typedef void* ccl_request_t;

**C++ version of oneCCL API:**

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

**C version of oneCCL API:**

::

    ccl_status_t ccl_test(ccl_request_t req, int* is_completed);

req
    requests the handle for communication operation being tracked
is_completed
    indicates the status: 
    
      - 0 - operation is in progress 
      - otherwise, the operation is completed

**C++ version of oneCCL API:**

::

    bool request::test();

Returnes the value that indicates the status: 

- 0 - operation is in progress 
- otherwise, the operation is completed.

Wait
****

Operation that blocks the execution until communication operation is completed.

**C version of oneCCL API:**

::

    ccl_status_t ccl_wait(ccl_request_t req);

req
    requests the handle for communication operation being tracked

**C++ version of oneCCL API:**

::

    void request::wait();
