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

#include "common/utils/tuple.hpp"
#include "common/log/log.hpp"

namespace native
{
/*
 * Define arguments read/write by host policy
 */
template<size_t pos, class ArgType, bool must_exist = true>
struct arg_access_policy_default
{
    using arg_type = ArgType;
    using return_t = std::pair<bool, arg_type>;
    void store(const arg_type &value)
    {
        arg_value = value;
        charged = true;
    }

    inline bool test() const noexcept
    {
        return charged;
    }

    return_t load() const
    {
        return_t ret{false, arg_type{}};
        if (!test())
        {
            if(must_exist)
            {
                abort();
                CCL_THROW("Cannot get non-existent kernel argument by index: ", pos);
            }
            return ret;
        }
        std::get<0>(ret) = true;
        std::get<1>(ret) = arg_value;
        return ret;
    }
private:

    arg_type arg_value {};
    bool charged = false;
};

template<size_t pos, class ArgType, bool must_exist = true>
struct arg_access_policy_atomic
{
    using arg_type = ArgType;
    using return_t = std::pair<bool, arg_type>;
    using throwable = std::integral_constant<bool, must_exist>;
    void store(const arg_type &value)
    {
        arg_value.store(value, std::memory_order_relaxed);  //relaxes
        charged.store(true, std::memory_order_release);
    }

    inline bool test() const noexcept
    {
        return charged.load(std::memory_order_acquire);
    }

    return_t load() const
    {
        return_t ret{false, arg_type{}};
        if(!test())
        {
            if(must_exist)
            {
                CCL_THROW("Cannot get non-existent kernel atomic argument by index:", pos);
            }
            return ret;
        }

        std::get<0>(ret) = true;
        std::get<1>(ret) = arg_value.load(std::memory_order_relaxed);
        return ret;
    }
private:
    std::atomic<arg_type> arg_value {};
    std::atomic<bool> charged{false};
};

template<size_t pos, class ArgType, bool must_exist = true>
struct arg_access_policy_atomic_move
{
    using arg_type = ArgType;
    using return_t = std::pair<bool, arg_type>;
    using throwable = std::integral_constant<bool, must_exist>;
    void store(const arg_type &value)
    {
//#ifdef DEBUG
        charged_counter.fetch_add(std::memory_order_relaxed);
//#endif
        arg_value.store(value, std::memory_order_relaxed);  //relaxes
        charged.store(true, std::memory_order_release);
    }

    inline bool test() const noexcept
    {
        return charged.load(std::memory_order_acquire);
    }

    return_t load()
    {
        return_t ret{false, arg_type{}};
        if (charged.exchange(false)) //destructive load should be done for `charge` only
        {
            std::get<0>(ret) = true;
            std::get<1>(ret) = arg_value.load(std::memory_order_relaxed);
//#ifdef DEBUG
            consumed_counter.fetch_add(std::memory_order_relaxed);
//#endif
        }
        return ret;
    }
private:
    void dump(std::ostream& out) const
    {
        out << "{ arg_value.load(std::memory_order_relaxed) , set: " << charged_counter.load() << ", get: " << consumed_counter.load() << "}";
    }

    std::atomic<arg_type> arg_value {};
    std::atomic<bool> charged{false};

    std::atomic<size_t> charged_counter {};
    std::atomic<size_t> consumed_counter {};
};

template<size_t pos>
struct arg_no_access_policy
{
    using arg_type = void;
    using return_t = bool;

    void store(...);
    bool test() const noexcept;
    return_t load() const;
};
}
