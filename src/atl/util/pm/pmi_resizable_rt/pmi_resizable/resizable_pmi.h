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
#ifndef PMIR_H_INCLUDED
#define PMIR_H_INCLUDED

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
#define PMIR_API __attribute__((visibility("default")))

#define PMIR_SUCCESS                0
#define PMIR_FAIL                   -1
#define PMIR_ERR_INIT               1
#define PMIR_ERR_NOMEM              2
#define PMIR_ERR_INVALID_ARG        3
#define PMIR_ERR_INVALID_KEY        4
#define PMIR_ERR_INVALID_KEY_LENGTH 5
#define PMIR_ERR_INVALID_VAL        6
#define PMIR_ERR_INVALID_VAL_LENGTH 7
#define PMIR_ERR_INVALID_LENGTH     8
#define PMIR_ERR_INVALID_NUM_ARGS   9
#define PMIR_ERR_INVALID_ARGS       10
#define PMIR_ERR_INVALID_NUM_PARSED 11
#define PMIR_ERR_INVALID_KEYVALP    12
#define PMIR_ERR_INVALID_SIZE       13

typedef enum {
    KVS_RA_WAIT = 0,
    KVS_RA_RUN = 1,
    KVS_RA_FINALIZE = 2,
} kvs_resize_action_t;
typedef kvs_resize_action_t (*pmir_resize_fn_t)(int comm_size);

int PMIR_API PMIR_Main_Addr_Reserve(char* main_addr);

int PMIR_API PMIR_Init(const char* main_addr);

int PMIR_API PMIR_Finalize(void);

int PMIR_API PMIR_Get_size(int* size);

int PMIR_API PMIR_Get_rank(int* rank);

int PMIR_API PMIR_KVS_Get_my_name(char* kvs_name, size_t length);

int PMIR_API PMIR_KVS_Get_name_length_max(size_t* length);

int PMIR_API PMIR_Barrier(void);

int PMIR_API PMIR_Update(void);

int PMIR_API PMIR_KVS_Get_key_length_max(size_t* length);

int PMIR_API PMIR_KVS_Get_value_length_max(size_t* length);

int PMIR_API PMIR_KVS_Put(const char* kvs_name, const char* key, const char* value);

int PMIR_API PMIR_KVS_Commit(const char* kvs_name);

int PMIR_API PMIR_KVS_Get(const char* kvs_name, const char* key, char* value, size_t length);

int PMIR_API PMIR_set_resize_function(pmir_resize_fn_t resize_fn);

int PMIR_API PMIR_Wait_notification(void);

#ifdef __cplusplus
}
#endif
#endif
