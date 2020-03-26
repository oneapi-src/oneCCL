.. highlight:: bash

Fault tolerance / elasticity
############################

Main instructions
+++++++++++++++++

Before launchung ranks, you can specify :ref:`CCL_WORLD_SIZE` = N, where N is the number of ranks to start.
If k8s with k8s manager support is used, then N is equal to ``replicasize`` by default.

You can specify your own function that decides what |product_short| should do on the "world" resize event: 

- wait
- use current "world" information 
- finalize

::

  typedef enum ccl_resize_action
  {
    // wait additional changes for number of ranks
    ccl_ra_wait     = 0,

    // run with current number of ranks
    ccl_ra_use      = 1,

    // finalize work
    ccl_ra_finalize = 2,

  } ccl_resize_action_t;

  typedef ccl_resize_action_t(*ccl_on_resize_fn_t)(size_t comm_size);

  ccl_set_resize_callback(ccl_on_resize_fn_t callback);

In case the number of ranks is changed, this function is called on |product_short| level. 
Application level (e.g. framework) should return the action that |product_short| should perform.

Setting this function to ``NULL`` (default value) means that |product_short| will work with exactly
:ref:`CCL_WORLD_SIZE` or ``replicasize`` ranks without fault tolerant / elasticity.


Examples
++++++++

Without k8s manager
*******************

To run ranks in k8s without k8s manager, for example, set of pods:

-   :ref:`CCL_PM_TYPE` = resizable
-   :ref:`CCL_K8S_API_ADDR` = k8s server address and port (in a format of IP:PORT)
-   Set same label :ref:`CCL_JOB_NAME` = job_name on each pod
-   Run your example

Using k8s manager
*****************

To run ranks in k8s use statefulset / deployment as a manager:

-   :ref:`CCL_PM_TYPE` = resizable
-   :ref:`CCL_K8S_API_ADDR` = k8s server address
-   :ref:`CCL_K8S_MANAGER_TYPE` = k8s
-   Run your example

Without mpirun
**************

To run ranks without ``mpirun``:

-   :ref:`CCL_PM_TYPE` = resizable
-   :ref:`CCL_KVS_IP_EXCHANGE` = env
-   :ref:`CCL_KVS_IP_PORT` = ip_port of one of your nodes where you run the example
-   Run your example
