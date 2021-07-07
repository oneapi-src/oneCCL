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
#ifdef HOST_CTX
#define __global

#include <memory>
using namespace ccl;

template <class native_type>
struct shared_event_traits {};

#else
typedef ushort bfloat16;
#endif

typedef struct __attribute__((packed)) shared_event_float {
    __global int* produced_bytes;
    __global float* mem_chunk;
} shared_event_float;

#ifdef HOST_CTX

template <>
struct shared_event_traits<float> {
    using impl_t = shared_event_float;
};

#endif
