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

#include "common/utils/tuple.hpp"

template<class Func, cl::sycl::access::mode access_mode>
struct sycl_buffer_visitor
{
    sycl_buffer_visitor(const ccl_datatype& dtype, size_t cnt, size_t offset, const ccl_buffer& buf, Func f) :
        requested_dtype(dtype),
        requested_cnt(cnt),
        requested_offset(offset),
        requested_buf(buf),
        callback(f)
    {}


    template<size_t index, class specific_sycl_buffer>
    void invoke()
    {
        if (index == requested_dtype.idx())
        {
            LOG_DEBUG("visitor matched index: ", index,
                      ", ccl: ", ccl::global_data::get().dtypes->name(requested_dtype),
                      ", in: ", __PRETTY_FUNCTION__);

            size_t bytes = requested_cnt * requested_dtype.size();
            auto out_buf_acc =
                static_cast<specific_sycl_buffer*>(requested_buf.get_ptr(bytes))->template get_access<access_mode>();
            CCL_ASSERT(requested_cnt <= out_buf_acc.get_count());
            void* out_pointer = out_buf_acc.get_pointer();
            LOG_DEBUG("requested_cnt: ", requested_cnt,
                      ", requested_dtype.size(): ", requested_dtype.size(),
                      ", requested_offset: ", requested_offset,
                      ", bytes: ", bytes,
                      ", out_buf_acc.get_count(): ", out_buf_acc.get_count());
            callback((char*)out_pointer + requested_offset, bytes);
        }
        else
        {
            LOG_TRACE("visitor skipped index: ", index,
                      ", ccl: ", ccl::global_data::get().dtypes->name(requested_dtype),
                      ", in: ", __PRETTY_FUNCTION__);
        }

    }
    const ccl_datatype& requested_dtype;
    size_t requested_cnt;
    size_t requested_offset;
    const ccl_buffer& requested_buf;
    Func callback;
};



template<cl::sycl::access::mode access_mode, class Func>
sycl_buffer_visitor<Func, access_mode> make_visitor(const ccl_datatype& dtype, size_t cnt, size_t offset, const ccl_buffer& buf, Func f)
{
    return sycl_buffer_visitor<Func, access_mode>(dtype, cnt, offset, buf, f);
}
