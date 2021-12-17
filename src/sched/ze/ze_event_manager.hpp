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

#include <list>
#include <ze_api.h>

class ccl_stream;

namespace ccl {
namespace ze {

class event_pool {
public:
    event_pool(ze_context_handle_t context);
    event_pool(ze_context_handle_t context, const ze_event_pool_desc_t& pool_desc);
    event_pool(ze_context_handle_t context,
               const ze_event_pool_desc_t& pool_desc,
               const ze_event_desc_t& event_desc);

    event_pool(const event_pool&) = delete;
    event_pool(event_pool&&) = default;
    event_pool& operator=(const event_pool&) = delete;
    event_pool& operator=(event_pool&&) = delete;

    virtual ~event_pool();

    operator ze_event_pool_handle_t() const;

    ze_event_handle_t create_event();
    ze_event_handle_t create_event(ze_event_desc_t& desc);

    size_t size() const;
    size_t capacity() const;

    void reset();
    void clear();

private:
    static constexpr size_t worker_idx{};

    const ze_context_handle_t context;
    const ze_event_pool_desc_t pool_desc;
    const ze_event_desc_t event_desc;

    ze_event_pool_handle_t pool{};
    std::list<ze_event_handle_t> events;

    void create_pool();
};

class event_manager {
public:
    event_manager(ze_context_handle_t context);
    event_manager(const ccl_stream* stream);

    event_manager(const event_manager&) = delete;
    event_manager(event_manager&&) = default;
    event_manager& operator=(const event_manager&) = delete;
    event_manager& operator=(event_manager&&) = default;

    virtual ~event_manager();

    ze_event_handle_t create(ze_event_desc_t desc = get_default_event_desc());
    std::vector<ze_event_handle_t> create(size_t count,
                                          ze_event_desc_t desc = get_default_event_desc());

    void reset();
    void clear();

    static ze_event_desc_t get_default_event_desc();
    static ze_event_pool_desc_t get_default_event_pool_desc();

protected:
    static constexpr size_t default_pool_size{ 50 };

    ze_context_handle_t context{};
    std::list<event_pool> pools;

    event_pool* add_pool(ze_event_pool_desc_t desc = get_default_event_pool_desc(),
                         ze_event_desc_t event_desc = get_default_event_desc());
};

} // namespace ze
} // namespace ccl
