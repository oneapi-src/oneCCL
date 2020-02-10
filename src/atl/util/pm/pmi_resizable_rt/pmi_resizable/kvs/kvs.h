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
#ifndef KVS
#define KVS

#include <stddef.h>

size_t kvs_set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val);

size_t kvs_remove_name_key(const char* kvs_name, const char* kvs_key);

size_t kvs_get_value_by_name_key(const char* kvs_name, const char* kvs_key, char* kvs_val);

size_t kvs_init(void);

size_t kvs_get_count_names(const char* kvs_name);

size_t kvs_finalize(void);

size_t kvs_get_keys_values_by_name(const char *kvs_name, char ***kvs_keys, char ***kvs_values);

size_t kvs_get_replica_size(void);

#endif
