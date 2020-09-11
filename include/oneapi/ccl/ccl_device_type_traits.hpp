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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl_type_traits.hpp'"
#endif

#ifdef MULTI_GPU_SUPPORT
#include <ze_api.h>
#endif

namespace ccl {

template <class type>
struct api_type_info {
    static constexpr bool is_supported() {
        return false;
    }
    static constexpr bool is_class() {
        return false;
    }
};

#define SUPPORTED_KERNEL_NATIVE_DATA_TYPES char, int, float, ccl::bfp16, double, int64_t, uint64_t

#define API_CLASS_TYPE_INFO(api_type) \
    template <> \
    struct api_type_info<api_type> { \
        static constexpr bool is_supported() { \
            return true; \
        } \
        static constexpr bool is_class() { \
            return std::is_class<api_type>::value; \
        } \
    };

template <class native_stream>
constexpr bool is_stream_supported() {
    return api_type_info</*typename std::remove_pointer<typename std::remove_cv<*/
                         native_stream /*>::type>::type*/>::is_supported();
}

template <class native_event>
constexpr bool is_event_supported() {
    return api_type_info</*typename std::remove_pointer<typename std::remove_cv<*/
                         native_event /*>::type>::type*/>::is_supported();
}

#ifdef CCL_ENABLE_SYCL
API_CLASS_TYPE_INFO(cl::sycl::device)

API_CLASS_TYPE_INFO(cl::sycl::queue);
API_CLASS_TYPE_INFO(cl_command_queue);

API_CLASS_TYPE_INFO(cl::sycl::context)
API_CLASS_TYPE_INFO(cl_context);

API_CLASS_TYPE_INFO(cl::sycl::event);
API_CLASS_TYPE_INFO(cl_event)

template <>
struct generic_device_type<CCL_ENABLE_SYCL_TRUE> {
    using handle_t = cl::sycl::device;
    using impl_t = handle_t;
    using ccl_native_t = impl_t;

    generic_device_type(device_index_type id,
                        cl::sycl::info::device_type = cl::sycl::info::device_type::gpu);
    generic_device_type(const cl::sycl::device& device);
    device_index_type get_id() const;
    ccl_native_t& get() noexcept;

    cl::sycl::device device;
};

template <>
struct generic_device_context_type<CCL_ENABLE_SYCL_TRUE> {
    using handle_t = cl_context;
    using impl_t = cl::sycl::context;
    using ccl_native_t = impl_t;

    generic_device_context_type(ccl_native_t ctx);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t context;
};

template <>
struct generic_platform_type<CCL_ENABLE_SYCL_TRUE> {
    using handle_t = cl::sycl::platform;
    using impl_t = handle_t;
    using ccl_native_t = impl_t;

    generic_platform_type(ccl_native_t& pl);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t platform;
};

template <>
struct generic_stream_type<CCL_ENABLE_SYCL_TRUE> {
    using handle_t = cl_command_queue;
    using impl_t = cl::sycl::queue;
    using ccl_native_t = impl_t;

    generic_stream_type(handle_t q);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t queue;
};

template <>
struct generic_event_type<CCL_ENABLE_SYCL_TRUE> {
    using handle_t = cl_event;
    using impl_t = cl::sycl::event;
    using ccl_native_t = impl_t;

    generic_event_type(handle_t e);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;
    ccl_native_t event;
};

#else

#ifdef MULTI_GPU_SUPPORT
}
namespace native {
class ccl_device;
class ccl_context;
class ccl_device_platform;

template <class handle_type, class resource_owner>
class cl_base;

using ccl_device_event = cl_base<ze_event_handle_t, ccl_device>;
using ccl_device_queue = cl_base<ze_command_queue_handle_t, ccl_device>;
} // namespace native

namespace ccl {
API_CLASS_TYPE_INFO(std::shared_ptr<native::ccl_device_event>);
API_CLASS_TYPE_INFO(std::shared_ptr<native::ccl_device_queue>);
//API_CLASS_TYPE_INFO(ze_command_queue_handle_t);
API_CLASS_TYPE_INFO(ze_event_handle_t);

template <>
struct generic_device_type<CCL_ENABLE_SYCL_FALSE> {
    using handle_t = device_index_type;
    using impl_t = native::ccl_device;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_device_type(device_index_type id);
    device_index_type get_id() const noexcept;
    ccl_native_t get() noexcept;

    handle_t device;
};

#ifndef ze_context_handle_t
#define ze_context_handle_t void*
#endif

template <>
struct generic_device_context_type<CCL_ENABLE_SYCL_FALSE> {
    using handle_t = ze_context_handle_t;
    using impl_t = native::ccl_context;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_device_context_type(handle_t ctx);
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t context;
};

template <>
struct generic_platform_type<CCL_ENABLE_SYCL_FALSE> {
    using handle_t = native::ccl_device_platform;
    using impl_t = handle_t;
    using ccl_native_t = std::shared_ptr<impl_t>;

    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;
};

template <>
struct generic_stream_type<CCL_ENABLE_SYCL_FALSE> {
    using handle_t = ze_command_queue_handle_t;
    using impl_t = handle_t;
    using ccl_native_t = std::shared_ptr<native::ccl_device_queue>;

    generic_stream_type(handle_t q);
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t queue;
};

template <>
struct generic_event_type<CCL_ENABLE_SYCL_FALSE> {
    using handle_t = ze_event_handle_t;
    using impl_t = handle_t;
    using ccl_native_t = std::shared_ptr<native::ccl_device_event>;

    generic_event_type(handle_t e);
    ccl_native_t get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t event;
};
#else /* MULTI_GPU_SUPPORT */
// no sycl no multu gpu ...
#undef CCL_ENABLE_SYCL_V
#define CCL_ENABLE_SYCL_V -1
#endif
#endif /* else for  CCL_ENABLE_SYCL */

using unified_device_type = generic_device_type<CCL_ENABLE_SYCL_V>;
using unified_device_context_type = generic_device_context_type<CCL_ENABLE_SYCL_V>;
using unified_platform_type = generic_platform_type<CCL_ENABLE_SYCL_V>;
using unified_stream_type = generic_stream_type<CCL_ENABLE_SYCL_V>;
using unified_event_type = generic_event_type<CCL_ENABLE_SYCL_V>;

//TMP - matching device index into native device object
template <class... Args>
unified_device_type create_from_index(Args&&... args) {
    return unified_device_type(std::forward<Args>(args)...);
}

} // namespace ccl
