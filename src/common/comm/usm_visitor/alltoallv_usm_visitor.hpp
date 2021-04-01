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
struct alltoallv_usm_visitor {
    using self_t = communicator_impl;

    self_t* get_self() {
        return static_cast<self_t*>(this);
    }

    const self_t* get_self() const {
        return static_cast<const self_t*>(const_cast<alltoallv_usm_visitor*>(this)->get_self());
    }

    template <class... Args>
    bool visit(ccl::event& req,
               ccl::datatype dtype,
               const void* send_buf,
               const ccl::vector_class<size_t>& send_count,
               void* recv_buf,
               const ccl::vector_class<size_t>& recv_counts,
               Args&&... args) {
        bool processed = false;
        LOG_TRACE("comm: ",
                  /*get_self()->to_string(),*/
                  " - starting to find visitor for datatype: ",
                  ccl::to_string(dtype),
                  " , handle: ",
                  utils::enum_to_underlying(dtype));

        switch (dtype) {
            case ccl::datatype::int8: {
                using type = int8_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint8: {
                using type = uint8_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::int16: {
                using type = int16_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint16: {
                using type = uint16_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::int32: {
                using type = int32_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint32: {
                using type = uint32_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::int64: {
                using type = int64_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint64: {
                using type = uint64_t;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::float16: {
                using type = ccl::float16;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::float32: {
                using type = float;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::float64: {
                using type = double;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::bfloat16: {
                using type = ccl::bfloat16;
                req = get_self()->template alltoallv_impl<type>(static_cast<const type*>(send_buf),
                                                                send_count,
                                                                static_cast<type*>(recv_buf),
                                                                recv_counts,
                                                                std::forward<Args>(args)...);
                processed = true;
                break;
            }
            default: {
                CCL_THROW("unknown datatype ", dtype);
                LOG_DEBUG("comm: ",
                          /*get_self()->to_string(),*/
                          " - no found visitor for datatype: ",
                          ccl::to_string(dtype),
                          " , handle: ",
                          utils::enum_to_underlying(dtype),
                          ", use RAW types");
                break;
            }
        }
        return processed;
    }
};
