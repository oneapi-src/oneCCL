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
#include "oneapi/ccl/native_device_api/l0/utils.hpp"
#include "common/log/log.hpp"

#if defined(MULTI_GPU_SUPPORT)
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"

#if defined(CCL_ENABLE_SYCL)
//#ifdef CCL_ENABLE_SYCL
#include <CL/sycl/backend/level_zero.hpp>
//static cl::sycl::vector_class<cl::sycl::device> gpu_sycl_devices;
#endif

namespace native {
namespace detail {

adjacency_matrix::adjacency_matrix(std::initializer_list<typename base::value_type> init)
        : base(init) {}

cross_device_rating binary_p2p_rating_calculator(const native::ccl_device& lhs,
                                                 const native::ccl_device& rhs,
                                                 size_t weight) {
    return property_p2p_rating_calculator(lhs, rhs, 1);
}

size_t serialize_device_path(std::vector<uint8_t>& out,
                             const ccl::device_index_type& path,
                             size_t offset) {
    if (out.size() <= 0) {
        CCL_THROW("unexpected vector size");
    }
    if (offset > out.size() - detail::serialize_device_path_size) {
        CCL_THROW("unexpected offset size");
    }

    size_t serialized_bytes = 0;
    uint8_t* data_start = out.data() + offset;
    constexpr size_t index_type_size = sizeof(ccl::index_type);

    // store driver index
    *(reinterpret_cast<ccl::index_type*>(data_start)) =
        std::get<ccl::device_index_enum::driver_index_id>(path);
    serialized_bytes += index_type_size;

    // store device index
    data_start += index_type_size;
    *(reinterpret_cast<ccl::index_type*>(data_start)) =
        std::get<ccl::device_index_enum::device_index_id>(path);
    serialized_bytes += index_type_size;

    // store subdevice index
    data_start += index_type_size;
    *(reinterpret_cast<ccl::index_type*>(data_start)) =
        std::get<ccl::device_index_enum::subdevice_index_id>(path);
    serialized_bytes += index_type_size;

    if (serialized_bytes != detail::serialize_device_path_size) {
        CCL_THROW("unexpected serialized size");
    }

    return serialized_bytes;
}

ccl::device_index_type deserialize_device_path(const uint8_t** data, size_t& size) {
    if (data == nullptr || size < detail::serialize_device_path_size) {
        CCL_THROW("cannot deserialize path, not enough data");
    }

    constexpr size_t index_type_size = sizeof(ccl::index_type);

    // load driver index
    ccl::index_type driver_index = *(reinterpret_cast<const ccl::index_type*>(*data));
    size -= index_type_size;

    // load device index
    *data += index_type_size;
    ccl::index_type device_index = *(reinterpret_cast<const ccl::index_type*>(*data));
    size -= index_type_size;

    // load subdevice index
    *data += index_type_size;
    ccl::index_type subdevice_index = *(reinterpret_cast<const ccl::index_type*>(*data));
    size -= index_type_size;
    *data += index_type_size;

    return std::make_tuple(driver_index, device_index, subdevice_index);
}

} // namespace detail
} // namespace native
#endif //#if defined(MULTI_GPU_SUPPORT)
