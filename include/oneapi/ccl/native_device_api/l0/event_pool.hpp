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

#include <atomic>
#include <mutex> //TODO use shared

#include "oneapi/ccl/native_device_api/l0/base.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives.hpp"
#include "oneapi/ccl/native_device_api/l0/utils.hpp"

namespace native {
struct ccl_context;
struct ccl_device;
class ccl_event_pool;

class event_pool_array_t {
public:
    using value_type = std::vector<std::shared_ptr<ccl_event_pool>>;
    using context_array_accessor = detail::unique_accessor<std::mutex, value_type>;
    using const_context_array_accessor = detail::unique_accessor<std::mutex, const value_type>;

    context_array_accessor access();
    const_context_array_accessor access() const;

private:
    mutable std::mutex m;
    value_type event_pools;
};

struct ccl_event_pool_holder {
    ze_event_pool_handle_t get();
    std::shared_ptr<ccl_event_pool> emplace(const std::initializer_list<ccl_device*>& devices,
                                            std::shared_ptr<ccl_event_pool> pool);

    std::vector<std::shared_ptr<ccl_event_pool>> get_event_pool_storage(
        std::initializer_list<ccl_device*> devices);
    std::vector<std::shared_ptr<ccl_event_pool>> get_event_pool_storage(
        std::initializer_list<ccl_device*> devices) const;

    void on_delete(ze_event_pool_handle_t pool_handle, ze_context_handle_t& ctx);

private:
    mutable std::mutex m;
    std::map<const ccl_device*, event_pool_array_t> contexts_pool;
};

class ccl_event_pool : public cl_base<ze_event_pool_handle_t, ccl_event_pool_holder, ccl_context>,
                       public std::enable_shared_from_this<ccl_event_pool> {
public:
    using base = cl_base<ze_event_pool_handle_t, ccl_event_pool_holder, ccl_context>;
    using handle_t = base::handle_t;
    using base::owner_t;
    using base::owner_ptr_t;
    using base::context_t;
    using base::context_ptr_t;
    using event_ptr = std::shared_ptr<event>;

    static const ze_event_desc_t& get_default_event_desc() {
        static ze_event_desc_t def = {
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            0, // index
            0, // no additional memory/cache coherency required on signal
            ZE_EVENT_SCOPE_FLAG_HOST // ensure memory coherency across device and Host after event completes
        };
        return def;
    }

    ccl_event_pool(const ze_event_pool_desc_t& descr,
                   handle_t h,
                   owner_ptr_t&& holder,
                   context_ptr_t&& ctx);
    ~ccl_event_pool();

    std::shared_ptr<ccl_event_pool> get_ptr() {
        return this->shared_from_this();
    }

    event_ptr create_event(const ze_event_desc_t& descr = ccl_event_pool::get_default_event_desc());
    void on_delete(ze_event_handle_t event_handle, ze_context_handle_t& ctx);

    const ze_event_pool_desc_t& get_pool_description() const;
    size_t get_allocated_events() const;

private:
    ze_event_pool_desc_t pool_description;
    std::atomic<size_t> allocated_event_count;
};
} // namespace native
