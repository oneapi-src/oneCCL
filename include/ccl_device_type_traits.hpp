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
namespace ccl
{

template<class type>
struct api_type_info
{
    static constexpr bool is_supported() { return false; }
    static constexpr bool is_class()     { return false; }
};

#define SUPPORTED_KERNEL_NATIVE_DATA_TYPES      char, int, float, ccl::bfp16, double, int64_t, uint64_t

#define API_CLASS_TYPE_INFO(api_type)                                                                           \
    template<>                                                                                                  \
    struct api_type_info<api_type>                                                                              \
    {                                                                                                           \
        static constexpr bool is_supported() { return true; }                                                   \
        static constexpr bool is_class()     { return std::is_class<api_type>::value; }                         \
    };

    template <class native_stream>
    constexpr bool is_stream_supported()
    {
        return api_type_info</*typename std::remove_pointer<typename std::remove_cv<*/native_stream/*>::type>::type*/>::is_supported();
    }

#ifdef CCL_ENABLE_SYCL
    API_CLASS_TYPE_INFO(cl::sycl::queue);
#endif

#ifdef MULTI_GPU_SUPPORT
    API_CLASS_TYPE_INFO(ze_command_queue_handle_t);

template<>
struct ccl_device_attributes_traits<ccl_device_preferred_topology_class>
{
    using type = device_topology_class;
};

template<>
struct ccl_device_attributes_traits<ccl_device_preferred_group>
{
    using type = device_topology_group;
};


#ifdef CCL_ENABLE_SYCL
template<>
struct generic_device_type<CCL_ENABLE_SYCL_TRUE>
{
    using device_t = cl::sycl::device;
    using native_t = device_t;
    using native_pointer_t = native_t*;
    using native_reference_t = native_t&;

    generic_device_type(device_index_type id, cl::sycl::info::device_type = cl::sycl::info::device_type::gpu);
    generic_device_type(const cl::sycl::device &device);
    device_index_type get_id() const noexcept;
    native_reference_t get() noexcept;

    cl::sycl::device device;
};

template<>
struct generic_device_context_type<CCL_ENABLE_SYCL_TRUE>
{
    using context_t = cl::sycl::context;
    using native_t = context_t;
    using native_pointer_t = native_t*;
    using native_reference_t = native_t&;
    using native_const_reference_t = const native_t&;

    native_reference_t get() noexcept;
    native_const_reference_t get() const noexcept;
};
#else
}
namespace native
{
    class ccl_device;
    class ccl_device_platform;
}

namespace ccl
{
template<>
struct generic_device_type<CCL_ENABLE_SYCL_FALSE>
{
    using device_t = device_index_type;
    using native_t = native::ccl_device;
    using native_pointer_t = std::shared_ptr<native::ccl_device>;
    using native_reference_t = std::shared_ptr<native::ccl_device>;

    generic_device_type(device_index_type id);
    device_index_type get_id() const noexcept;
    native_reference_t get() noexcept;

    device_index_type device;
};


template<>
struct generic_device_context_type<CCL_ENABLE_SYCL_FALSE>
{
    using context_t = native::ccl_device_platform;
    using native_t = context_t;
    using native_pointer_t = native_t*;
    using native_reference_t = native_t&;
    using native_const_reference_t = const native_t&;

    native_reference_t get() noexcept;
    native_const_reference_t get() const noexcept;
};
#endif

using unified_device_type = generic_device_type<CCL_ENABLE_SYCL_V>;
using unified_device_context_type = generic_device_context_type<CCL_ENABLE_SYCL_V>;

//TMP - matching device index into native device object
template<class ...Args>
unified_device_type create_from_index(Args&& ...args)
{
    return unified_device_type(std::forward<Args>(args)...);
}
#endif /* MULTI_GPU_SUPPORT */
}
