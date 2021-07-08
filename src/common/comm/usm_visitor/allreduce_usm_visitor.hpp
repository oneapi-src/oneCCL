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

#include "oneapi/ccl.hpp"
#include "oneapi/ccl/type_traits.hpp"

template <class communicator_impl>
struct allreduce_usm_visitor {
    using self_t = communicator_impl;

    self_t* get_self() {
        return static_cast<self_t*>(this);
    }

    const self_t* get_self() const {
        return static_cast<const self_t*>(const_cast<allreduce_usm_visitor*>(this)->get_self());
    }

    template <class... Args>
    bool visit(ccl::event& req,
               ccl::datatype dtype,
               const void* send_buf,
               void* recv_buf,
               size_t count,
               Args&&... args) {
        bool processed = false;
        LOG_TRACE("comm: ",
                  /*get_self()->to_string(),*/
                  " - starting to find visitor for datatype: ",
                  ccl::to_string(dtype),
                  " , handle: ",
                  utils::enum_to_underlying(dtype));
        req = get_self()->template allreduce_impl<uint8_t>((const uint8_t*)(const void*)send_buf,
                                                           (uint8_t*)(void*)recv_buf,
                                                           count,
                                                           std::forward<Args>(args)...);

        return processed;
    }
};
