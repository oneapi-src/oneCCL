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
#include "ccl.h"
#include "common/utils/utils.hpp"
#include "common/stream/stream_provider_dispatcher.hpp"

namespace ccl {
class environment; //friend-zone
}

ccl_status_t CCL_API ccl_stream_create(ccl_stream_type_t type,
                                       void* native_stream,
                                       ccl_stream_t* stream);

class alignas(CACHELINE_SIZE) ccl_stream : public stream_provider_dispatcher {
public:
    friend class stream_provider_dispatcher;
    friend class ccl::environment;
    friend ccl_status_t CCL_API ccl_stream_create(ccl_stream_type_t type,
                                                  void* native_stream,
                                                  ccl_stream_t* stream);
    using stream_native_t = stream_provider_dispatcher::stream_native_t;
    using stream_native_handle_t = stream_provider_dispatcher::stream_native_handle_t;

    ccl_stream() = delete;
    ccl_stream(const ccl_stream& other) = delete;
    ccl_stream& operator=(const ccl_stream& other) = delete;

    ~ccl_stream() = default;

    using stream_provider_dispatcher::get_native_stream;
    using stream_provider_dispatcher::get_native_stream_handle;

    ccl_stream_type_t get_type() const {
        return type;
    }
    bool is_sycl_device_stream() const {
        return (type == ccl_stream_cpu || type == ccl_stream_gpu);
    }

    static std::unique_ptr<ccl_stream> create(stream_native_t& native_stream);

private:
    template <
        class NativeStream,
        typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                int>::type = 0>
    ccl_stream(ccl_stream_type_t stream_type, NativeStream& native_stream)
            : stream_provider_dispatcher(native_stream),
              type(stream_type) {}
    template <class NativeStream,
              typename std::enable_if<
                  not std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                  int>::type = 0>
    ccl_stream(ccl_stream_type_t stream_type, NativeStream native_stream)
            : stream_provider_dispatcher(native_stream),
              type(stream_type) {}

    ccl_stream_type_t type;
};
