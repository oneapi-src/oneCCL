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
#include <algorithm>
#include <numeric>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "native_device_api/l0/base.hpp"

namespace native
{
inline std::ostream& operator<<(std::ostream& out, const ccl::device_index_type& index)
{
    out << ccl::to_string(index);
    return out;
}
/**
 * Base RAII L0 handles wrappper
 * support serialize/deserialize concept
 */
#define TEMPLATE_DECL_ARG         class handle_type, class resource_owner
#define TEMPLATE_DEF_ARG          handle_type, resource_owner

template<TEMPLATE_DECL_ARG>
cl_base<TEMPLATE_DEF_ARG>::cl_base(handle_t h, owner_ptr_t parent) :
    handle(h), owner(std::move(parent))
{
}

template<TEMPLATE_DECL_ARG>
cl_base<TEMPLATE_DEF_ARG>::cl_base(cl_base&& src) noexcept :
    handle(std::move(src.handle)),
    owner(std::move(src.owner))
{
}

template<TEMPLATE_DECL_ARG>
cl_base<TEMPLATE_DEF_ARG>& cl_base<TEMPLATE_DEF_ARG>::operator=(cl_base&&src) noexcept
{
    handle = std::move(src.handle);
    owner = std::move(src.owner);
    return *this;
}

template<TEMPLATE_DECL_ARG>
cl_base<TEMPLATE_DEF_ARG>::~cl_base() noexcept
{
    auto lock = owner.lock();
    if(lock)
    {
        lock->on_delete(handle);
    }
}

template<TEMPLATE_DECL_ARG>
std::shared_ptr<cl_base<TEMPLATE_DEF_ARG>> cl_base<TEMPLATE_DEF_ARG>::get_instance_ptr()
{
    return this->shared_from_this();
}

template<TEMPLATE_DECL_ARG>
typename cl_base<TEMPLATE_DEF_ARG>::handle_t cl_base<TEMPLATE_DEF_ARG>::release()
{
    handle_t ret;

    owner.reset();

    std::swap(ret, handle);
    return ret;

}
template<TEMPLATE_DECL_ARG>
typename cl_base<TEMPLATE_DEF_ARG>::handle_t& cl_base<TEMPLATE_DEF_ARG>::get() noexcept
{
    return const_cast<handle_t& >(static_cast<const self_t* >(this)->get());
}

template<TEMPLATE_DECL_ARG>
const typename cl_base<TEMPLATE_DEF_ARG>::handle_t& cl_base<TEMPLATE_DEF_ARG>::get() const noexcept
{
    return handle;
}

template<TEMPLATE_DECL_ARG>
typename cl_base<TEMPLATE_DEF_ARG>::handle_t* cl_base<TEMPLATE_DEF_ARG>::get_ptr() noexcept
{
    return &handle;
}

template<TEMPLATE_DECL_ARG>
const typename cl_base<TEMPLATE_DEF_ARG>::handle_t*cl_base<TEMPLATE_DEF_ARG>::get_ptr() const noexcept
{
    return const_cast<handle_t* >(static_cast<const self_t* >(this)->get_ptr());
}

template<TEMPLATE_DECL_ARG>
const typename cl_base<TEMPLATE_DEF_ARG>::owner_ptr_t
cl_base<TEMPLATE_DEF_ARG>::get_owner() const
{
    return owner;
}

template<TEMPLATE_DECL_ARG>
constexpr size_t cl_base<TEMPLATE_DEF_ARG>::get_size_for_serialize()
{
    return resource_owner::get_size_for_serialize() + sizeof(handle_t);
}

template<TEMPLATE_DECL_ARG>
template<class ...helpers>
size_t cl_base<TEMPLATE_DEF_ARG>::serialize(std::vector<uint8_t>& out,
                                                      size_t from_pos,
                                                      const helpers& ...args) const
{
    // check parent existence
    auto lock = owner.lock();
    if (!lock)
    {
        throw std::runtime_error("cannot serialize without owner");
    }

    constexpr size_t expected_bytes = sizeof(handle_t);

    // serialize from position
    size_t serialized_bytes = lock->serialize(out, from_pos, expected_bytes, args...); //resize vector inside

    uint8_t* data_start = out.data() + from_pos + serialized_bytes;
    *(reinterpret_cast<handle_t*>(data_start)) = handle;
    serialized_bytes += expected_bytes;
    return serialized_bytes;
}

template<TEMPLATE_DECL_ARG>
template<class type, class... helpers>
std::shared_ptr<type> cl_base<TEMPLATE_DEF_ARG>::deserialize(const uint8_t** data, size_t& size,
                                                             helpers& ...args)
{
    constexpr size_t expected_bytes = sizeof(handle);
    size_t initial_size = size;

    // recover parent handle at first
    auto owner = resource_owner::deserialize(data, size, args...).lock();

    size_t offset = initial_size - size;
    if (!owner or !offset)
    {
        throw std::runtime_error("cannot deserialize, owner if nullptr");
    }

    if (size < expected_bytes)
    {
        throw std::runtime_error("cannot deserialize, not enough data");
    }

    handle_t h = *(reinterpret_cast<const handle_t*>(*data));
    *data += expected_bytes;
    size -= expected_bytes;
    return std::shared_ptr<type>{new type(h, owner)};
}

#undef TEMPLATE_DEF_ARG
#undef TEMPLATE_DECL_ARG


namespace detail
{

template<ccl::device_index_enum index_id,
         class IndexedContainer,
         class value_type,
         class value_type_index_extractor>
indexed_storage<value_type> merge_indexed_values(const IndexedContainer& indexes,
                                                 std::vector<value_type>& values,
                                                 value_type_index_extractor functor)
{
    indexed_storage<value_type> ret;
    try
    {
        // set indices to values at first
        indexed_storage<value_type> indexed_values;
        std::transform(values.begin(), values.end(),
                       std::inserter(indexed_values, indexed_values.end()),
                       [functor](value_type& handle) -> typename indexed_storage<value_type>::value_type
                       {
                           auto index  = functor(handle);
                           return std::make_pair(index, handle);
                       });

        // find requested device index in indexed values
        std::transform(indexes.begin(), indexes.end(),
                       std::inserter(ret, ret.end()),
                       [&indexed_values](const typename IndexedContainer::value_type& index) -> typename indexed_storage<value_type>::value_type
                       {
                           auto values_it = indexed_values.find(std::get<index_id>(index));
                           if (values_it == indexed_values.end())
                           {
                               throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                                        "Cannot find index: " +
                                                        ccl::to_string(index) +
                                                        ", from collected devices");
                           }
                           return std::make_pair(std::get<index_id>(index),
                                                 values_it->second);
                       });
    }
    catch(const std::exception& ex)
    {
        std::stringstream ss;
        ss << "Cannot merge indexed values\nRequested indices: ";
        for(const auto&index : indexes)
        {
            ss << index << ", ";
        }

        ss << "\nvalues to indexed: ";
        std::copy(values.begin(), values.end(),
              std::ostream_iterator<value_type>(ss, ", "));
        ss << "\nerror: " << ex.what();
        throw std::runtime_error(ss.str());
    }
    return ret;
}

template<ccl::device_index_enum index_id, class value_type, class value_type_index_extractor>
indexed_storage<value_type> collect_indexed_data(const ccl::device_indices_t& indexes,
                                                 std::vector<value_type>& collected_values,
                                                 value_type_index_extractor functor)
{
    indexed_storage<value_type> ret;
    if(!indexes.empty())   //only requested indexes
    {
        ret = merge_indexed_values<index_id>(indexes, collected_values, functor);
    }
    else //all
    {
        std::transform(collected_values.begin(), collected_values.end(),
                       std::inserter(ret, ret.end()),
                       [&functor](value_type handle) -> typename indexed_storage<value_type>::value_type
                       {
                           auto index = functor(handle);
                           return std::make_pair(index, handle);
                       });
    }
    return ret;
}

}
}
