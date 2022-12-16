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
#pragma once

#include "oneapi/ccl/config.h"

#include <string>
#include <vector>

#ifdef CCL_ENABLE_PMIX
#include <pmix.h>
#endif // CCL_ENABLE_PMIX

namespace ccl {

#ifdef CCL_ENABLE_PMIX
typedef struct pmix_lib_ops {
    decltype(::PMIx_Init) *PMIx_Init;
    decltype(::PMIx_Error_string) *PMIx_Error_string;
    decltype(::PMIx_Get) *PMIx_Get;
    decltype(::PMIx_Finalize) *PMIx_Finalize;
    decltype(::PMIx_Value_destruct) *PMIx_Value_destruct;
} pmix_lib_ops_t;

static std::vector<std::string> pmix_fn_names = { "PMIx_Init",
                                                  "PMIx_Error_string",
                                                  "PMIx_Get",
                                                  "PMIx_Finalize",
                                                  "PMIx_Value_destruct" };

extern ccl::pmix_lib_ops_t pmix_lib_ops;

#define PMIx_Init           ccl::pmix_lib_ops.PMIx_Init
#define PMIx_Error_string   ccl::pmix_lib_ops.PMIx_Error_string
#define PMIx_Get            ccl::pmix_lib_ops.PMIx_Get
#define PMIx_Finalize       ccl::pmix_lib_ops.PMIx_Finalize
#define PMIx_Value_destruct ccl::pmix_lib_ops.PMIx_Value_destruct

bool get_pmix_local_coord(int *local_proc_idx, int *local_proc_count);
#endif // CCL_ENABLE_PMIX

void pmix_api_init();
void pmix_api_fini();

} //namespace ccl
