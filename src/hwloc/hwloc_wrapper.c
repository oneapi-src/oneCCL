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
#include "hwloc_wrapper.h"

static hwloc_info_t hwloc_info = { .initialized = 0 };

hwloc_status_t hwloc_init() {
    hwloc_status_t ret = HWLOC_SUCCESS;

    hwloc_info.initialized = 0;
    hwloc_info.bindset = hwloc_bitmap_alloc();

    if (hwloc_topology_init(&hwloc_info.topology) < 0) {
        printf("hwloc_topology_init failed (%s)\n", strerror(errno));
        goto err;
    }

    hwloc_topology_set_io_types_filter(hwloc_info.topology, HWLOC_TYPE_FILTER_KEEP_ALL);

    if (hwloc_topology_load(hwloc_info.topology) < 0) {
        printf("hwloc_topology_load failed (%s)\n", strerror(errno));
        goto err;
    }

    if (hwloc_get_proc_cpubind(
            hwloc_info.topology, getpid(), hwloc_info.bindset, HWLOC_CPUBIND_PROCESS) < 0) {
        printf("hwloc_get_proc_cpubind failed (%s)\n", strerror(errno));
        goto err;
    }

    hwloc_info.initialized = 1;

    return ret;

err:
    return HWLOC_FAILURE;
}

hwloc_status_t hwloc_finalize() {
    hwloc_status_t ret = HWLOC_SUCCESS;

    hwloc_topology_destroy(hwloc_info.topology);
    hwloc_bitmap_free(hwloc_info.bindset);
    hwloc_info.initialized = 0;

    return ret;
}

int hwloc_is_initialized() {
    return hwloc_info.initialized;
}

static hwloc_obj_t hwloc_get_first_non_io_obj_by_pci(int domain, int bus, int dev, int func) {
    hwloc_obj_t io_device = hwloc_get_pcidev_by_busid(hwloc_info.topology, domain, bus, dev, func);
    HWLOC_ASSERT(io_device,
                 "failed to get PCI device with domain %d, bus %d, dev %d, func %d",
                 domain,
                 bus,
                 dev,
                 func);
    hwloc_obj_t first_non_io = hwloc_get_non_io_ancestor_obj(hwloc_info.topology, io_device);
    HWLOC_ASSERT(first_non_io, "failed to get ancestor of PCI device");
    return first_non_io;
}

int hwloc_is_dev_close_by_pci(int domain, int bus, int dev, int func) {
    int is_close = 0;

    if (!hwloc_is_initialized())
        return is_close;

    hwloc_obj_t first_non_io = hwloc_get_first_non_io_obj_by_pci(domain, bus, dev, func);

    /* determine if PCI device is "close" to process by checking if process's affinity is included
     * in PCI device's affinity or if PCI device's affinity is included in process's affinity */
    is_close = (hwloc_bitmap_isincluded(hwloc_info.bindset, first_non_io->cpuset) ||
                hwloc_bitmap_isincluded(first_non_io->cpuset, hwloc_info.bindset));

    return is_close;
}
