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
struct broadcast_usm_visitor {
    using self_t = communicator_impl;

    self_t* get_self() {
        return static_cast<self_t*>(this);
    }

    const self_t* get_self() const {
        return static_cast<const self_t*>(const_cast<broadcast_usm_visitor*>(this)->get_self());
    }

    template <class... Args>
    bool visit(ccl::event& req, ccl::datatype dtype, void* buf, size_t count, Args&&... args) {
        bool processed = false;
        LOG_TRACE("comm: ",
                  get_self()->to_string(),
                  " - starting to find visitor for datatype: ",
                  ccl::to_string(dtype),
                  " , handle: ",
                  utils::enum_to_underlying(dtype));

        CCL_THROW("unexpected path");

        switch (dtype) {
            case ccl::datatype::int8: {
                using type = char;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint8: {
                throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                     " - USM convertation of: " + ccl::to_string(dtype) +
                                     " is not supported for such configuration");
                break;
            }
            case ccl::datatype::int16: {
                throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                     " - USM convertation of: " + ccl::to_string(dtype) +
                                     " is not supported for such configuration");
                break;
            }
            case ccl::datatype::uint16: {
                throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                     " - USM convertation of: " + ccl::to_string(dtype) +
                                     " is not supported for such configuration");
                break;
            }
            case ccl::datatype::int32: {
                using type = int32_t;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint32: {
                throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                     " - USM convertation of: " + ccl::to_string(dtype) +
                                     " is not supported for such configuration");
                break;
            }
            case ccl::datatype::int64: {
                using type = int64_t;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::uint64: {
                using type = uint64_t;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::float16: {
                throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                     " - USM convertation of: " + ccl::to_string(dtype) +
                                     " is not supported for such configuration");
                break;
            }
            case ccl::datatype::float32: {
                using type = float;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::float64: {
                using type = double;
                req = get_self()->template broadcast_impl<type>(
                    static_cast<type*>(buf), count, std::forward<Args>(args)...);
                processed = true;
                break;
            }
            case ccl::datatype::bfloat16: {
                throw ccl::exception(
                    std::string(__PRETTY_FUNCTION__) +
                    " - USM convertationf loat16  is not supported for such configuration");
                break;
            }
            default: {
                LOG_DEBUG("comm: ",
                          get_self()->to_string(),
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
