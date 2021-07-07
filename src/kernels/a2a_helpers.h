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

#include "common.h"

#define DEFINE_A2A_COMM_DATA(NAME, T) \
    typedef struct __attribute__((packed)) a2a_gpu_comm_data_##NAME { \
        __global T* recv_buf; \
        __global sync_flag_type* ready_to_receive_flag; \
        __global sync_flag_type* data_sent_flag; \
    } a2a_gpu_comm_data_##NAME;

DEFINE_A2A_COMM_DATA(int8, int8_t)
DEFINE_A2A_COMM_DATA(uint8, uint8_t)
DEFINE_A2A_COMM_DATA(int16, int16_t)
DEFINE_A2A_COMM_DATA(uint16, uint16_t)
DEFINE_A2A_COMM_DATA(int32, int32_t)
DEFINE_A2A_COMM_DATA(uint32, uint32_t)
DEFINE_A2A_COMM_DATA(int64, int64_t)
DEFINE_A2A_COMM_DATA(uint64, uint64_t)
//DEFINE_A2A_COMM_DATA(float16, half)
DEFINE_A2A_COMM_DATA(float32, float)
DEFINE_A2A_COMM_DATA(float64, double)
DEFINE_A2A_COMM_DATA(bfloat16, uint16_t)
