Fusion of collective operations
*******************************

In some cases it may be benefial to postpone execution of collective operations and execute them all together as a single operation in a batch mode. 
This can amortize operation setup overhead and improve interconnect saturation. 

oneCCL provides several knobs to enable and control such optimization:

- The fusion is enabled by :ref:`CCL_FUSION`.
- The advanced configuration is controlled by:

  * :ref:`CCL_FUSION_BYTES_THRESHOLD`
  * :ref:`CCL_FUSION_COUNT_THRESHOLD`
  * :ref:`CCL_FUSION_CYCLE_MS`

For now this functionality is supported for :ref:`allreaduce <allreduce>` operations only.
