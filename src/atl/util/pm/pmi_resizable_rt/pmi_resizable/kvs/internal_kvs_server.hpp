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
#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "internal_kvs.h"

typedef enum kvs_access_mode {
    AM_PUT = 2,
    AM_REMOVE = 3,
    AM_GET_COUNT = 4,
    AM_GET_VAL = 5,
    AM_GET_KEYS_VALUES = 6,
    AM_GET_REPLICA = 7,
    AM_FINALIZE = 8,
    AM_BARRIER = 9,
    AM_BARRIER_REGISTER = 10,
    AM_INTERNAL_REGISTER = 11,
    AM_SET_SIZE = 12,
} kvs_access_mode_t;

typedef struct kvs_request {
    kvs_access_mode_t mode;
    char name[MAX_KVS_NAME_LENGTH];
    char key[MAX_KVS_KEY_LENGTH];
    char val[MAX_KVS_VAL_LENGTH];
} kvs_request_t;

typedef struct server_args {
    int sock_listener;
    std::shared_ptr<isockaddr> args;
} server_args_t;

void* kvs_server_init(void* args);
