Fusion of collective operations
*******************************

In some cases, it may be beneficial to postpone execution of collective operations and execute them all together as a single operation in a batch mode. 
This can reduce operation setup overhead and improve interconnect saturation. 

|product_short| provides several knobs to enable and control such optimization:

- The fusion is enabled by :ref:`CCL_FUSION`.
- The advanced configuration is controlled by:

  * :ref:`CCL_FUSION_BYTES_THRESHOLD`
  * :ref:`CCL_FUSION_COUNT_THRESHOLD`
  * :ref:`CCL_FUSION_CYCLE_MS`

For now, this functionality is supported for :ref:`allreduce <allreduce>` operations only.
