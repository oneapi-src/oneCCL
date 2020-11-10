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

#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "common/utils/buffer.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/tuple.hpp"
#include "oneapi/ccl/native_device_api/interop_utils.hpp"

enum class sycl_copy_direction { d2h, h2d };

std::string to_string(sycl_copy_direction val);

#ifdef CCL_ENABLE_SYCL

template <sycl_copy_direction direction>
struct sycl_copier {
    sycl_copier(ccl_buffer in_buf,
                ccl_buffer out_buf,
                size_t count,
                const ccl_datatype& dtype,
                size_t in_buf_offset)
            : in_buf(in_buf),
              out_buf(out_buf),
              count(count),
              dtype(dtype),
              in_buf_offset(in_buf_offset) {}

    bool is_completed() {
        return (e.get_info<sycl::info::event::command_execution_status>() ==
                sycl::info::event_command_status::complete)
                   ? true
                   : false;
    }

    void set_queue(sycl::queue external_q) {
        q = external_q;
    }

    template <size_t index, class specific_sycl_buffer>
    void invoke() {
        if (index == (int)(dtype.idx())) {
            LOG_DEBUG("visitor matched index: ",
                      index,
                      ", ccl: ",
                      ccl::global_data::get().dtypes->name(dtype),
                      ", in: ",
                      __PRETTY_FUNCTION__);

            size_t bytes = count * dtype.size();

            void* in_buf_ptr = in_buf.get_ptr(bytes);
            void* out_buf_ptr = out_buf.get_ptr(bytes);

            void* void_device_ptr =
                (direction == sycl_copy_direction::h2d) ? out_buf_ptr : in_buf_ptr;

            /*
              don't print this pointer through CCL logger
              as in case of char/int8_t it will be interpreted as string
              and logger will try access device memory
              use void_device_ptr instead
            */
            typename specific_sycl_buffer::value_type* device_ptr =
                static_cast<typename specific_sycl_buffer::value_type*>(void_device_ptr);

            auto device_ptr_type = cl::sycl::get_pointer_type(device_ptr, q.get_context());

            CCL_THROW_IF_NOT((device_ptr_type == cl::sycl::usm::alloc::device ||
                              device_ptr_type == cl::sycl::usm::alloc::unknown),
                             "unexpected USM type ",
                             native::detail::usm_to_string(device_ptr_type),
                             " for device_ptr ",
                             device_ptr);

            specific_sycl_buffer* device_buf_ptr = nullptr;

            if (device_ptr_type == cl::sycl::usm::alloc::device) {
                /* do nothing, provided device USM pointer can be used as is in copy kernel */
            }
            else {
                /* cast pointer into SYCL buffer */
                device_buf_ptr = static_cast<specific_sycl_buffer*>(void_device_ptr);
            }

            LOG_DEBUG("count: ",
                      count,
                      ", in_buf_offset: ",
                      in_buf_offset,
                      ", dtype_size: ",
                      dtype.size(),
                      ", bytes: ",
                      bytes,
                      ", direction: ",
                      to_string(direction),
                      ", in_buf_ptr: ",
                      in_buf_ptr,
                      ", out_buf_ptr: ",
                      out_buf_ptr,
                      ", device_ptr: ",
                      void_device_ptr,
                      ", is_device_usm: ",
                      (device_buf_ptr) ? "no" : "yes",
                      ", device_ptr usm_type: ",
                      native::detail::usm_to_string(device_ptr_type));

            size_t offset = in_buf_offset;

            if (device_buf_ptr) {
                specific_sycl_buffer host_buf(
                    static_cast<typename specific_sycl_buffer::value_type*>(
                        (direction == sycl_copy_direction::h2d) ? in_buf_ptr : out_buf_ptr),
                    count,
                    cl::sycl::property::buffer::use_host_ptr{});

                e = q.submit([&](cl::sycl::handler& h) {
                    auto& src_buf =
                        (direction == sycl_copy_direction::h2d) ? host_buf : *device_buf_ptr;
                    auto& dst_buf =
                        (direction == sycl_copy_direction::h2d) ? *device_buf_ptr : host_buf;
                    auto src_buf_acc =
                        src_buf.template get_access<cl::sycl::access::mode::read>(h, count, offset);
                    auto dst_buf_acc =
                        dst_buf.template get_access<cl::sycl::access::mode::write>(h);
                    h.copy(src_buf_acc, dst_buf_acc);
                });
            }
            else {
                e = q.memcpy(
                    out_buf_ptr,
                    static_cast<typename specific_sycl_buffer::value_type*>(in_buf_ptr) + offset,
                    count * dtype.size());

                /* TODO: remove explicit wait */
                e.wait();
            }
        }
        else {
            LOG_TRACE("visitor skipped index: ",
                      index,
                      ", ccl: ",
                      ccl::global_data::get().dtypes->name(dtype),
                      ", in: ",
                      __PRETTY_FUNCTION__);
        }
    }

    ccl_buffer in_buf;
    ccl_buffer out_buf;
    size_t count;
    const ccl_datatype& dtype;
    cl::sycl::queue q;
    size_t in_buf_offset;
    sycl::event e;
};

#endif /* CCL_ENABLE_SYCL */
