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
#include "include/oneapi/ccl/types.hpp"
#include "coll/algorithm/algorithms_enum.hpp"

namespace native {

template <class... T>
struct communiaction_data_holder {
    template <class U, ccl_coll_type type>
    struct data_for_algo_t {
        U data;
    };

    template <class Data, ccl_coll_type... types>
    using data_storage_t = std::tuple<data_for_algo_t<Data, types>...>;
};
} // namespace native
CCL_COLL_TYPE_LIST
