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

#include <iostream>
#include <stddef.h>

#include "common/log/log.hpp"

enum class ccl_buffer_type
{
    DIRECT,
    INDIRECT
};

inline std::ostream& operator << (std::ostream& os, const ccl_buffer_type& type)
{
   os << static_cast<std::underlying_type<ccl_buffer_type>::type>(type);
   return os;
}

class ccl_buffer
{

private:
    void* src;
    ssize_t size; /* max available size, for sanity checks */
    int offset;
    ccl_buffer_type type;

    bool check_offset(ssize_t access_size = 0) const
    {
        bool result = true;

        if (offset < 0)
        {
            result = false;
            LOG_ERROR("unexpected offset ", offset);
        }

        if ((size != -1) && (offset + access_size > size))
        {
            result = false;
            LOG_ERROR("unexpected (offset + access_size): ",
                      "size ", size,
                      ", offset ", offset,
                      ", access_size ", access_size);
        }

        return result;
    }

public:
    ccl_buffer(void* src, ssize_t size, int offset, ccl_buffer_type type)
        : src(src), size(size),
          offset(offset), type(type)
    {
        LOG_DEBUG("create: src ", src, ", size ", size, ", offset ", offset, ", type ", type);
        CCL_ASSERT(check_offset());
    }

    ccl_buffer() : ccl_buffer(nullptr, -1, 0, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size) : ccl_buffer(src, size, 0, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size, int offset) : ccl_buffer(src, size, offset, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size, ccl_buffer_type type) : ccl_buffer(src, size, 0, type) {}

    ccl_buffer(const ccl_buffer& buf)
        : src(buf.src),
          size(buf.size),
          offset(buf.offset),
          type(buf.type)
    {
        CCL_ASSERT(check_offset());
    };

    void set(void* src, ssize_t size, int offset, ccl_buffer_type type)
    {
        LOG_DEBUG("set: src ", src, ", size ", size, ", offset ", offset, ", type ", type);
        CCL_ASSERT(src, "new src is null");

        this->src = src;
        this->size = size;
        this->offset = offset;
        this->type = type;

        CCL_ASSERT(check_offset());
    }

    void set(void* src) { set(src, -1, 0, ccl_buffer_type::DIRECT); }
    void set(void* src, ssize_t size) { set(src, size, 0, ccl_buffer_type::DIRECT); }
    void set(void* src, ssize_t size, ccl_buffer_type type) { set(src, size, 0, type); }
    void set(void* src, ssize_t size, int offset) { set(src, size, offset, ccl_buffer_type::DIRECT); }

    void* get_src() const { return src; }
    ssize_t get_size() const { return size; }
    int get_offset() const { return offset; }
    ccl_buffer_type get_type() const { return type; }

    ccl_buffer operator+ (size_t val)
    {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator+ (size_t val) const
    {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator- (size_t val)
    {
        return ccl_buffer(src, size, offset - val, type);
    }

    ccl_buffer operator+ (int val)
    {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator- (int val)
    {
        return ccl_buffer(src, size, offset - val, type);
    }

    ccl_buffer& operator+= (size_t val)
    {
        offset += val;
        CCL_ASSERT(check_offset());
        return *this;
    }

    size_t get_difference(ccl_buffer buf)
    {
        CCL_ASSERT((get_ptr() >= buf.get_ptr()), "difference between pointers < 0");
        return (static_cast<char*>(get_ptr()) - static_cast<char*>(buf.get_ptr()));
    }

    void* get_ptr(ssize_t access_size = 0) const
    {
        CCL_ASSERT(check_offset(access_size));

        if (!src)
            return nullptr;

        if (type == ccl_buffer_type::DIRECT)
            return ((char*)src + offset);
        else
        {
            return (*((char**)src)) ? (*((char**)src) + offset) : nullptr;
        }
    }

    operator bool() const
    {
        if (type == ccl_buffer_type::DIRECT)
            return src;
        else
            return (src && (*(void**)src));
    }

    bool operator ==(ccl_buffer const& other) const
    {
        return ((get_ptr() == other.get_ptr()) &&
                (get_type() == other.get_type()));
    }

    bool operator !=(ccl_buffer const& other) const
    {
        return !(*this == other);
    }

    bool operator >(ccl_buffer const& other) const
    {
        CCL_ASSERT(get_type() == other.get_type(), "types should match");
        return (get_ptr() > other.get_ptr());
    }

    friend std::ostream& operator<< (std::ostream& out, const ccl_buffer& buf)
    {
        out << "(src: " << buf.get_src()
            << ", size " << buf.get_size()
            << ", off " << buf.get_offset()
            << ", type: " << buf.get_type()
            << ")";
        return out;
    }
};
