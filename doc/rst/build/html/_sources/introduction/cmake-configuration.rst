Use oneCCL package from CMake
=============================

``oneCCLConfig.cmake`` and ``oneCCLConfigVersion.cmake`` are included into oneCCL distribution.

With these files, you can integrate oneCCL into a user project with the
`find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ command.
Successful invocation of ``find_package(oneCCL <options>)`` creates imported target ``oneCCL``
that can be passed to the
`target_link_libraries <https://cmake.org/cmake/help/latest/command/target_link_libraries.html>`_ command.

For example:

.. code-block:: cmake
   
   project(Foo)
   add_executable(foo foo.cpp)

   # Search for oneCCL
   find_package(oneCCL REQUIRED)

   # Connect oneCCL to foo
   target_link_libraries(foo oneCCL)

oneCCLConfig files generation
*****************************

To generate oneCCLConfig files for oneCCL package,
use the provided ``cmake/scripts/config_generation.cmake`` file:

.. code-block:: bash

   cmake [-DOUTPUT_DIR=<output_dir>] -P cmake/script/config_generation.cmake
