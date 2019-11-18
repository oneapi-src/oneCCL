Prioritization of collective operations
****************************************

oneCCL supports prioritization of collective operations that control the order in which individual collective operations are executed. 
This allows to postpone execution of non-urgent operations to complete urgent operations earlier, which may be beneficial for many use cases.

The collective prioritization is controlled by priority value. Note that the priority must be a non-negative number with a higher number standing for a higher priority.

There are few prioritization modes:

-   None - default mode when all collective operations have the same priority.
-	Direct - priority is explicitly specified by users using ``coll_attr.priority``.
-	LIFO (Last In, First Out) - priority is implicitly increased on each collective call. In this case user doesn't have to specify priority.

The prioritization mode is controlled by :ref:`CCL_PRIORITY`.
