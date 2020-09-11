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
#include <stdio.h>

#include "base.h"

int main() {
    test_init();

    ccl_version_t version;
    ccl_status_t ret = ccl_get_version(&version);
    if (ret != ccl_status_success) {
        perror("Cannot get CCL version\n");
        return -1;
    }

    printf("CCL library info:\nVersion:\nmajor: %d\nminor: %d\nupdate: %d\n"
           "\nProduct: %s\nBuild date: %s\nFull: %s\n",
           version.major,
           version.minor,
           version.update,
           version.product_status,
           version.build_date,
           version.full);

    printf("\nCCL API info:\nVersion:\nmajor: %d\nminor: %d\nupdate: %d\n"
           "\nProduct: %s\nBuild date: %s\nFull: %s\n",
           CCL_MAJOR_VERSION,
           CCL_MINOR_VERSION,
           CCL_UPDATE_VERSION,
           CCL_PRODUCT_STATUS,
           CCL_PRODUCT_BUILD_DATE,
           CCL_PRODUCT_FULL);

    if (CCL_MAJOR_VERSION == version.major && CCL_MINOR_VERSION == version.minor) {
        printf("API & library versions compatible\n");
    }
    else {
        perror("API & library versions are not compatible!\n");
        return -1;
    }

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");
    return 0;
}
