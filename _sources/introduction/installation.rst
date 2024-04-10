.. |sys_req| replace:: |product_full| System Requirements
.. _sys_req: https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-collective-communication-library-system-requirements.html
.. |tgz_file| replace:: tar.gz file
.. _tgz_file: https://github.com/oneapi-src/oneCCL/releases
.. highlight:: bash

==================
Installation Guide
==================

This page explains how to install and configure the |product_full| (|product_short|).
|product_short| supports different installation scenarios using command line interface.

System Requirements
*******************

Visit |sys_req|_ to learn about hardware and software requirements for |product_short|.

Installation using Command Line Interface
*****************************************

To install |product_short| using command line interface (CLI), follow these steps:

#. Go to the ``ccl`` folder:

   ::

      cd ccl

#. Create a new folder:

   ::

      mkdir build

#. Go to the folder created:

   ::

      cd build

#. Launch CMake:

   ::

      cmake ..

#. Install the product:

   ::

      make -j install

In order to have a clear build, create a new ``build`` directory and invoke ``cmake`` within the directory.

Custom Installation
^^^^^^^^^^^^^^^^^^^

You can customize CLI-based installation (for example, specify directory, compiler, and build type):

* To specify **installation directory**, modify the ``cmake`` command:

  ::

    cmake .. -DCMAKE_INSTALL_PREFIX=</path/to/installation/directory>

  If no ``-DCMAKE_INSTALL_PREFIX`` is specified, |product_short| is installed into the ``_install`` subdirectory of the current build directory. For example, ``ccl/build/_install``.

* To specify **compiler**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_C_COMPILER=<c_compiler> -DCMAKE_CXX_COMPILER=<cxx_compiler>

.. _enable_sycl:

*  To enable ``SYCL`` devices communication support, specify ``SYCL`` compiler (only Intel\ |reg|\  oneAPI DPC++/C++ Compiler is supported):

  ::

     cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp

* To specify the **build type**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release]

* To enable ``make`` verbose output to see all parameters used by ``make`` during compilation and linkage, modify the ``make`` command as follows:

  ::

     make -j VERBOSE=1 install

Find More
*********
- `oneCCL Get Started Guide <https://www.intel.com/content/www/us/en/docs/oneccl/get-started-guide/current/overview.html>`_
- `oneCCL GitHub Source Code Repository <https://github.com/oneapi-src/oneCCL>`_
- `oneCCL Documentation <https://oneapi-src.github.io/oneCCL/??#>`_
