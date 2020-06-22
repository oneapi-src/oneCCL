.. |sys_req| replace:: |product_full| System Requirements
.. _sys_req: https://software.intel.com/en-us/articles/oneapi-collective-communication-library-system-requirements
.. highlight:: bash

Installation
=============

This page explains how to install and configure the |product_full| (|product_short|).

|product_short| supports different installation scenarios:

* `Installation using command line interface`_
* `Installation using tar.gz`_
* `Installation using RPM`_

.. note:: Visit |sys_req|_ to learn about hardware and software requirements for |product_short|.

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

* To speciify **installation directory**, modify the ``cmake`` command:

  ::

    cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory

  If no ``-DCMAKE_INSTALL_PREFIX`` is specified, |product_short| is installed into the ``_install`` subdirectory of the current build directory.
  For example, ``ccl/build/_install``.

* To specify **compiler**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=your_cxx_compiler

  If ``CMAKE_CXX_COMPILER`` requires ``SYCL`` cross-platform abstraction level it should be specified in ``-DCOMPUTE_RUNTIME`` ( ``compute++`` and ``dpcpp`` supported only):

  ::

     cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=compute++ -DCOMPUTE_RUNTIME=computecpp
     cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=dpcpp -DCOMPUTE_RUNTIME=dpcpp

  OpenCL search location path hint can be specified by using standart environment ``OPENCLROOT`` additionally:

  ::

     OPENCLROOT=your_opencl_location cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=compute++ -DCOMPUTE_RUNTIME=computecpp


* To specify the **build type**, modify the ``cmake`` command:

  ::

     cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release|RelWithDebInfo|MinSizeRel]

* To enable ``make`` verbose output to see all parameters used by ``make`` during compilation and linkage, modify the ``make`` command as follows:

  ::

     make -j VERBOSE=1

* To archive installed files:

  ::

     make -j install

* To build with Address Sanitizer, modify the ``cmake`` command as follow:

  ::

     cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_ASAN=true

  Make sure that ``libasan.so`` exists.

  .. note::

     Address sanitizer only works in the debug build.

Binary releases are available on our release page.

Installation using tar.gz
*************************

To install |product_short| using the tar.gz file in a user mode, execute the following commands:

.. prompt:: bash

   tar zxf l_ccl-devel-64-<version>.<update>.<package#>.tgz
   cd l_ccl_<version>.<update>.<package#>
   ./install.sh

There is no uninstall script. To uninstall |product_short|, delete the whole installation directory.

Installation using RPM
**********************

You can get |product_short| through the RPM Package Manager. To install the library in a root mode using RPM, follow these steps:

#. Log in as root.

#. Install the following package:

  .. prompt:: bash

     rpm -i intel-ccl-devel-64-<version>.<update>-<package#>.x86_64.rpm

     where ``<version>.<update>-<package#>`` is a string. For example, ``2017.0-009``.

To uninstall |product_short| using the RPM Package Manager, execute this command:

  .. prompt:: bash

     rpm -e intel-ccl-devel-64-<version>.<update>-<package#>.x86_64
