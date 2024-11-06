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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "comm/comm.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "common/global/global.hpp"
#include "coll/selection/selection.hpp"

ccl_selector_param create_ccl_selector_param(ccl_coll_type ctype,
                                             size_t count,
                                             ccl::datatype dtype,
                                             ccl_comm* comm,
                                             ccl_stream* stream,
                                             void* buf,
                                             ccl::reduction reduction = ccl::reduction::custom,
                                             bool is_vector_buf = false,
                                             bool is_sycl_buf = false,
                                             int peer_rank = CCL_INVALID_PEER_RANK_IDX,
                                             ccl_coll_algo hint_algo = {},
                                             bool is_scaleout = false);

bool can_use_sycl_kernels(const ccl_selector_param& param);
