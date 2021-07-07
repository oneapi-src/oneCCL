/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#ifndef HWLOC_WRAPPER_H
#define HWLOC_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hwloc.h"
#include <sys/syscall.h>

#define GETTID() syscall(SYS_gettid)

#define HWLOC_ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, \
                    "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n", \
                    GETTID(), \
                    __FILE__, \
                    __FUNCTION__, \
                    __LINE__, \
                    #cond, \
                    ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while (0)

typedef enum { HWLOC_SUCCESS, HWLOC_FAILURE, HWLOC_UNSUPPORTED } hwloc_status_t;

inline const char* hwloc_status_to_str(hwloc_status_t status) {
    switch (status) {
        case HWLOC_SUCCESS: return "SUCCESS";
        case HWLOC_FAILURE: return "FAILURE";
        case HWLOC_UNSUPPORTED: return "UNSUPPORTED";
        default: return "UNKNOWN";
    }
}

typedef struct {
    hwloc_topology_t topology;
    hwloc_cpuset_t bindset;
    int initialized;
} hwloc_info_t;

hwloc_status_t hwloc_init();
hwloc_status_t hwloc_finalize();
int hwloc_is_initialized();

/*
 * return true if pci device is close to this process
 */
int hwloc_is_dev_close_by_pci(int domain, int bus, int dev, int func);

#ifdef __cplusplus
}
#endif

#endif /* HWLOC_WRAPPER_H */
