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
#ifndef RNE_H
#define RNE_H

// bf <--> float conversion
//    bf : no igc type for bf yet. Use short as *opaque* type for it.
//
// float -> bf conversion builtins (rte rounding mode)
short __builtin_IB_ftobf_1(float a) __attribute__((const));
short2 __builtin_IB_ftobf_2(float2 a) __attribute__((const));
short4 __builtin_IB_ftobf_4(float4 a) __attribute__((const));
short8 __builtin_IB_ftobf_8(float8 a) __attribute__((const));
short16 __builtin_IB_ftobf_16(float16 a) __attribute__((const));

// bf -> float conversion builtins (precise conversion)
float __builtin_IB_bftof_1(short a) __attribute__((const));
float2 __builtin_IB_bftof_2(short2 a) __attribute__((const));
float4 __builtin_IB_bftof_4(short4 a) __attribute__((const));
float8 __builtin_IB_bftof_8(short8 a) __attribute__((const));
float16 __builtin_IB_bftof_16(short16 a) __attribute__((const));

// 2 floats --> packed 2 bf (rte rounding mode)
int __builtin_IB_2fto2bf_1(float a, float b) __attribute__((const));
int2 __builtin_IB_2fto2bf_2(float2 a, float2 b) __attribute__((const));
int4 __builtin_IB_2fto2bf_4(float4 a, float4 b) __attribute__((const));
int8 __builtin_IB_2fto2bf_8(float8 a, float8 b) __attribute__((const));
int16 __builtin_IB_2fto2bf_16(float16 a, float16 b) __attribute__((const));

float __bf16_to_fp32(ushort V) {
    return __builtin_IB_bftof_1(as_short(V));
}

ushort __fp32_to_bf16(float V) {
    return as_ushort(__builtin_IB_ftobf_1(V));
}

#endif /* RNE_H */
