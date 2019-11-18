# oneccl
Intel(R) oneAPI Collective Communications Library
## Prerequisites

Ubuntu* 18
GNU*: C, C++ 4.8.5 or higher.

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
Modify `cmake` command as follow:

```
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory
```

If no `-DCMAKE_INSTALL_PREFIX` is specified then ccl will be installed into `_install` subdirectory of the current
build directory, e.g. `ccl/build/_install`

### Specify compiler
Modify `cmake` command as follow:

```
cmake .. -DCMAKE_C_COMPILER=your_c_compiler -DCMAKE_CXX_COMPILER=your_cxx_compiler
```

### Specify build type
Modify `cmake` command as follow:

```
cmake .. -DCMAKE_BUILD_TYPE=[Debug|Release|RelWithDebInfo|MinSizeRel]
```

### Enable `make` verbose output
Modify `make` command as follow to see all parameters used by `make` during compilation
and linkage:

```
make -j VERBOSE=1
```

### Build with address sanitizer
Modify `cmake` command as follow:
```
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_ASAN=true
```
*Note:* address sanitizer only works in Debug build

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
There are two ways to set workers threads affinity - explicit and automatic

#### Setting affinity explicitly
1. Set environment variable *CCL_WORKER_COUNT* with desired number of workers threads
2. Set environment variable *CCL_WORKER_AFFINITY* with IDs of cores to be bound to

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=3,4,5,6
```
With variables above CCL will create 4 threads and pin them to cores with numbers 3,4,5 and 6 accordingly

#### Setting affinity automatically
*NOTE:* automatic pinning only works if application has been launched using *mpirun* provided by CCL distribution package.

1. Set environment variable *CCL_WORKER_COUNT* with desired number of workers threads
2. Set environment variable *CCL_WORKER_AFFINITY* with value *auto*

Example:
```
export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY=auto
```
With variables above CCL will create 4 threads and pin them to the last 4 cores available for the launched process.

The exact IDs of CPU cores depend on parameters passed to *mpirun* 

## FAQ

### When do I need a clean build? When should I remove my favorite build directory?

In the most cases there is no need in removal of the current build directory. Just run `make` to 
compile and link changed files. Only if one sees some suspicious build errors after significant 
change in the code (e.g. after rebase or change of branch) then it is a hint to clean build directory.

Release notes available by link: https://software.intel.com/en-us/articles/oneapi-collective-communication-library-ccl-release-notes

