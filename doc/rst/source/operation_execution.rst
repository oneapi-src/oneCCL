.. highlight:: bash

=====================================
Execution of Communication Operations
=====================================

Communication operations are executed by CCL worker threads (workers). The number of workers is controlled by the :ref:`CCL_WORKER_COUNT` environment variable.

Workers affinity is controlled by :ref:`CCL_WORKER_AFFINITY`.

By setting workers affinity you can specify which CPU cores are used to host CCL workers. The general rule of thumb is to use different CPU cores for compute (e.g. by specifying ``KMP_AFFINITY``) and for communication.

There are two ways to set workers affinity: explicit and automatic.

Explicit setup
##############

To set affinity explicitly, pass ID of the cores to be bound to to  the ``CCL_WORKER_AFFINITY`` environment variable. 

Example
+++++++

In the example below, |product_short| creates 4 threads and pins them to cores with numbers 3, 4, 5, and 6, respectively:
::

   export CCL_WORKER_COUNT=4
   export CCL_WORKER_AFFINITY=3,4,5,6

Automatic setup
###############

.. note:: Automatic pinning only works if application is launched using ``mpirun`` provided by the |product_short| distribution package.

To set affinity automatically, set ``CCL_WORKER_AFFINITY`` to ``auto``. 

Example
+++++++

In the example below, |product_short| creates four threads and pins them to the last four cores available for the process launched:
::

   export CCL_WORKER_COUNT=4
   export CCL_WORKER_AFFINITY=auto

.. note:: The exact IDs of CPU cores depend on the parameters passed to ``mpirun``.
