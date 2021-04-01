===========
Limitations
===========

The list of scenarios not yet supported by oneCCL:

- Creation of multiple ranks within single process
- Handling of dependencies as operation parameter (for example, ``deps`` vector in ``ccl::allreduce(..., deps)``)
