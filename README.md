# Intel(R) oneAPI Collective Communications Library (oneCCL)

## Prerequisites

- Ubuntu* 18
- GNU*: C, C++ 4.8.5 or higher.

## Documentation

- [oneCCL Documentation (Beta)](https://intel.github.io/oneccl/)
- [Release notes](https://software.intel.com/en-us/articles/oneapi-collective-communication-library-ccl-release-notes) 

## Installation
### General installation scenario

```
cd oneccl
mkdir build
cd build
cmake ..
make -j install
```

If a "clear" build is needed, then one should create a new build directory and invoke `cmake` within it.

### Specify installation directory
Modify `cmake` command as follows:

```
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory
```

If no `-DCMAKE_INSTALL_PREFIX` is specified then oneCCL will be installed into `_install` subdirectory of the current
build directory, e.g. `ccl/build/_install`.

### Specify compiler
Modify `cmake` command as follows:

```
cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=your_cxx_compiler
```

### Specify build type
Modify `cmake` command as follows:

```
cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release|RelWithDebInfo|MinSizeRel]
```

### Enable `make` verbose output
Modify `make` command as follows to see all parameters used by `make` during compilation
and linkage:

```
make -j VERBOSE=1
```

### Build with Address Sanitizer
Modify `cmake` command as follows:
```
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_ASAN=true
```
---
**Note:** Address Sanitizer only works in Debug build.

---

Make sure that libasan.so exists.

## Usage

### Launching Example Application
Use the command:
```
$ source <install_dir>/env/setvars.sh
$ cd <install_dir>/examples
$ mpirun -n 2 ./common/benchmark
```
### Setting workers affinity
There are two ways to set workers threads affinity: explicit and automatic.

#### Setting affinity explicitly
1. Set the environment variable *CCL_WORKER_COUNT* with the desired number of worker threads.
2. Set the environment variable *CCL_WORKER_AFFINITY* with the IDs of the cores to be bound to.

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=3,4,5,6
```
With the variables above oneCCL will create 4 threads and pin them to the cores under the numbers 3, 4, 5 and 6 accordingly

#### Setting affinity automatically
---
**NOTE:** Automatic pinning only works if application has been launched using *mpirun* provided by oneCCL distribution package.

---
1. Set the environment variable *CCL_WORKER_COUNT* with the desired number of worker threads.
2. Set the environment variable *CCL_WORKER_AFFINITY* with the value *auto*.

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=auto
```
With the variables above oneCCL will create 4 threads and pin them to the last 4 cores available for the launched process.

The exact IDs of CPU cores depend on parameters passed to *mpirun*. 

## Migration from MLSL to oneCCL ##

Intel® MLSL is no longer supported, no new releases are available. Please switch to the new API introduced in [Intel® oneAPI Collective Communications Library (oneCCL)](http://github.com/intel/oneccl)
There are some examples that can help you get started with oneCCL, simply try to perform the following:

```
$ cd ./mlsl_to_oneccl
$ . ${MLSL_ROOT}/intel64/bin/mlslvars.sh
$ . ${CCL_ROOT}/env/vars.sh
$ make run -f Makefile
```

If you used MLSL before, here is an example that demonstrates the key differences between libraries' APIs.

```diff
#include <iostream>
#include <stdio.h>
- #include "mlsl.hpp"
+ #include "ccl.hpp"

- using namespace MLSL;
+ using namespace ccl;

#define COUNT 128
 
int main(int argc, char** argv)
{
    int i, size, rank;
 
    auto sendbuf = new float[COUNT];
    auto recvbuf = new float[COUNT];
 
-    Environment::GetEnv().Init(&argc, &argv);
-    rank = Environment::GetEnv().GetProcessIdx();
-    size = Environment::GetEnv().GetProcessCount();     
-    auto dist = Environment::GetEnv().CreateDistribution(size, 1);
+    auto stream = environment::instance().create_stream();
+    auto comm = environment::instance().create_communicator();
+    rank = comm->rank();
+    size = comm->size();
 
    /* initialize sendbuf */
    for (i = 0; i < COUNT; i++)
        sendbuf[i] = rank;
 
    /* invoke allreduce */
-    auto req = dist->AllReduce(sendbuf, recvbuf, COUNT,                      
-                               DT_FLOAT, RT_SUM, GT_GLOBAL);
-    Environment::GetEnv().Wait(req);
+    comm->allreduce(sendbuf, recvbuf, COUNT,
+                    reduction::sum,
+                    nullptr /* coll_attr */,
+                    stream)->wait(); 
    /* check correctness of recvbuf */
    float expected = (size - 1) * ((float)size / 2);
    for (i = 0; i < COUNT; i++)
    {
        if (recvbuf[i] != expected)
        {
            std::cout << "idx " << i
                      << ": got " << recvbuf[i]
                      << " but expected " << expected
                      << std::endl;
            break;
        }
    }
 
    if (i == COUNT && rank == 0)
        std::cout << "PASSED" << std::endl;
 
-    Environment::GetEnv().DeleteDistribution(dist);
-    Environment::GetEnv().Finalize();
 
    delete[] sendbuf;
    delete[] recvbuf;
 
    return 0;
}
```

## FAQ

### When do I need a clean build? When should I remove my favorite build directory?

In most cases, there is no need to remove the current build directory. Just run `make` to 
compile and link changed files. Only if you see some suspicious build errors after a significant 
change in your code (e.g. after rebase or change of branch), you may want to clean the build directory.


