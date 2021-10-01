.. _`here`: https://github.com/ofiwg/libfabric/releases/tag/v1.13.1
.. _`documentation`: https://one-api.gitlab-pages.devtools.intel.com/level_zero/core/PROG.html#affinity-mask

=====================================
Enabling OFI/verbs dmabuf support
=====================================

|product_short| provides experimental support for device memory transfers using Linux dmabuf,
which is exposed through OFI API for verbs provider.


Requirements
############

- Linux kernel version >= 5.12
- RDMA core version >= 34.0
- level-zero-devel package


Limitations
###########

- Only first tile should be used from each GPU card.
  For example, if GPU with 2 tiles is used then set ZE_AFFINITY_MASK=0.0.
  More information about GPU selection can be found in level-zero `documentation`_.


Build instructions
##################

OFI
***

::

    git clone --single-branch --branch v1.13.1 https://github.com/ofiwg/libfabric.git
    cd libfabric
    ./autogen.sh
    ./configure --prefix=<ofi_install_dir> --enable-verbs=<rdma_core_install_dir> --enable-ze-dlopen=yes
    make -j install

.. note::
    You may also get OFI release package directly from `here`_.
    No need to run autogen.sh if using the release package.

|product_short|
***************

::

    cmake -DCMAKE_INSTALL_PREFIX=<ccl_install_dir> -DLIBFABRIC_DIR=<ofi_install_dir> -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp -DCOMPUTE_BACKEND=dpcpp_level_zero -DENABLE_OFI_HMEM=1 ..
    make -j install


Run instructions
################

Run allreduce test with ring algorithm and SYCL USM device buffers.

::

    source <ccl_install_dir>/env/setvars.sh
    export LD_LIBRARY_PATH=<ofi_install_path>/lib:${LD_LIBRARY_PATH}
    CCL_ATL_TRANSPORT=ofi CCL_ATL_HMEM=1 CCL_ALLREDUCE=ring FI_PROVIDER=verbs mpiexec -n 2 <ccl_install_dir>/examples/sycl/sycl_allreduce_usm_test gpu device
