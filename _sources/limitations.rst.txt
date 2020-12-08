===========
Limitations
===========

The list of scenarious not yet supported by oneCCL:

- Creation of multiple ranks within single process
- Handling of dependencies as operation parameter (for example, ``deps`` vector in ``ccl::allreduce(..., deps)``)
- Float16 datatype support
