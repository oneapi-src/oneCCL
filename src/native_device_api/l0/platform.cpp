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
#include <algorithm>
#include <iterator>
#include <sstream>

#include "native_device_api/l0/primitives_impl.hpp"
#include "native_device_api/l0/platform.hpp"

namespace native {
struct once_init {
    once_init() {
        platform_impl = ccl_device_platform::create();
    }
    std::shared_ptr<ccl_device_platform> platform_impl;
};

CCL_API ccl_device_platform::driver_ptr get_driver(size_t index) {
    return get_platform().get_driver(index);
}

CCL_API ccl_device_platform& get_platform() {
    static once_init init;
    return *init.platform_impl;
}

CCL_API std::shared_ptr<ccl_device_platform> ccl_device_platform::create(
    const ccl::device_indices_t& indices /* = device_indices_per_driver()*/) {
    std::shared_ptr<ccl_device_platform> platform(new ccl_device_platform);
    platform->init_drivers(indices);
    return platform;
}
/*
CCL_API std::shared_ptr<ccl_device_platform> ccl_device_platform::create(const device_affinity_per_driver& affinities)
{
    std::shared_ptr<ccl_device_platform> platform(new ccl_device_platform);
    platform->init_drivers(affinities);
    return platform;
}
*/
CCL_API ccl_device_platform::ccl_device_platform() {
    // initialize Level-Zero driver
    ze_result_t ret = zeInit(ZE_INIT_FLAG_NONE);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Cannot initialize L0: " + native::to_string(ret));
    }
}
/*
CCL_API void ccl_device_platform::init_drivers(const device_affinity_per_driver& driver_device_affinities)
{
    //collect driver indices
    ccl::device_mask_t driver_indices;
    std::transform(driver_device_affinities.begin(), driver_device_affinities.end(),
                   std::inserter(driver_indices, driver_indices.end()),
                   [](const typename device_affinity_per_driver::value_type& pair)
                   {
                        return pair.first;
                   });

    auto collected_drivers_list = ccl_device_driver::get_handles(driver_indices);
    for (const auto &val : collected_drivers_list)
    {
        auto affitinity_it = driver_device_affinities.find(val.first);
        if(affitinity_it != driver_device_affinities.end())
        {
            drivers.emplace(val.first, ccl_device_driver::create(val.second, get_ptr(), affitinity_it->second));
        }
        else
        {
            drivers.emplace(val.first, ccl_device_driver::create(val.second, get_ptr()));
        }
    }
}
*/
CCL_API void ccl_device_platform::init_drivers(
    const ccl::device_indices_t& driver_device_affinities /* = device_indices_per_driver()*/) {
    /* TODO - do we need that?

#ifdef CCL_ENABLE_SYCL
    if(gpu_sycl_devices.empty())
    {
        auto platforms = cl::sycl::platform::get_platforms();
        for(const auto& platform : platforms)
        {
            if(platform.is_host())
            {
                continue;
            }
            auto devices = platform.get_devices(cl::sycl::info::device_type::gpu);
            if (devices.empty())
            {
                continue;
            }
            gpu_sycl_devices.insert(gpu_sycl_devices.end(), devices.begin(), devices.end());
        }

        if(gpu_sycl_devices.empty())
        {
            std::cerr << "Cannot collect SYCL device! Exit.";
            abort();
        }
    }
#endif
*/

    try {
        auto collected_drivers_list = ccl_device_driver::get_handles(driver_device_affinities);
        for (const auto& val : collected_drivers_list) {
            if (driver_device_affinities.empty()) {
                drivers.emplace(val.first,
                                ccl_device_driver::create(val.second, val.first, get_ptr()));
            }
            else {
                //collect device_index only for drvier specific index
                ccl::device_indices_t per_driver_index;
                for (const auto& affitinity : driver_device_affinities) {
                    if (std::get<ccl::device_index_enum::driver_index_id>(affitinity) ==
                        val.first) {
                        per_driver_index.insert(affitinity);
                    }
                }

                drivers.emplace(
                    val.first,
                    ccl_device_driver::create(val.second, val.first, get_ptr(), per_driver_index));
            }
        }
    }
    catch (const std::exception& ex) {
        std::stringstream ss;
        ss << "Cannot create drivers by indices: ";
        for (const auto& index : driver_device_affinities) {
            ss << index << ", ";
        }

        ss << "\nError: " << ex.what();
        throw;
    }
}

void CCL_API ccl_device_platform::on_delete(ze_driver_handle_t& sub_device_handle) {
    //todo
}

CCL_API ccl_device_platform::const_driver_ptr ccl_device_platform::get_driver(
    ccl::index_type index) const {
    auto it = drivers.find(index);
    if (it == drivers.end()) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 "No driver by index: " + std::to_string(index) + " on platform");
    }
    return it->second;
}

CCL_API ccl_device_platform::driver_ptr ccl_device_platform::get_driver(ccl::index_type index) {
    auto it = drivers.find(index);
    if (it == drivers.end()) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 "No driver by index: " + std::to_string(index) + " on platform");
    }
    return it->second;
}

const ccl_device_platform::driver_storage_type& ccl_device_platform::get_drivers() const noexcept {
    return drivers;
}

CCL_API ccl_device_driver::device_ptr ccl_device_platform::get_device(
    const ccl::device_index_type& path) {
    return std::const_pointer_cast<ccl_device>(
        static_cast<const ccl_device_platform*>(this)->get_device(path));
}

CCL_API ccl_device_driver::const_device_ptr ccl_device_platform::get_device(
    const ccl::device_index_type& path) const {
    ccl::index_type driver_idx = std::get<ccl::device_index_enum::driver_index_id>(path);
    auto it = drivers.find(driver_idx);
    if (it == drivers.end()) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - incorrect driver requested: " + ccl::to_string(path) +
                                 ". Total driver count: " + std::to_string(drivers.size()));
    }

    return it->second->get_device(path);
}

std::string CCL_API ccl_device_platform::to_string() const {
    std::stringstream out;
    out << "Platform:\n{\n";
    std::string driver_prefix = "\t";
    for (const auto& driver_pair : drivers) {
        out << driver_pair.second->to_string(driver_prefix) << std::endl;
    }
    out << "\n}";
    return out.str();
}

details::adjacency_matrix ccl_device_platform::calculate_device_access_metric(
    const ccl::device_indices_t& indices,
    details::p2p_rating_function func) const {
    details::adjacency_matrix result;

    try {
        // diagonal matrix, assume symmetric cross device access
        for (typename ccl::device_indices_t::const_iterator lhs_it = indices.begin();
             lhs_it != indices.end();
             ++lhs_it) {
            for (typename ccl::device_indices_t::const_iterator rhs_it = lhs_it;
                 rhs_it != indices.end();
                 ++rhs_it) {
                ccl_device_driver::const_device_ptr lhs_dev = get_device(*lhs_it);
                ccl_device_driver::const_device_ptr rhs_dev = get_device(*rhs_it);

                details::cross_device_rating rating = func(*lhs_dev, *rhs_dev);
                result[*lhs_it][*rhs_it] = rating;
                result[*rhs_it][*lhs_it] = rating;
            }
        }
    }
    catch (const std::exception& ex) {
        throw ccl::ccl_error(std::string("Cannot calculate_device_access_metric, error: ") +
                             ex.what() + "\nCurrent platform info:\n" + to_string());
    }
    return result;
}
} // namespace native
