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
#ifndef REQUEST_WRAPPERS_H
#define REQUEST_WRAPPERS_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

kvs_status_t request_k8s_kvs_init(void);

kvs_status_t request_k8s_kvs_get_master(const char* local_host_ip,
                                        char* main_host_ip,
                                        char* port_str);

kvs_status_t request_k8s_kvs_finalize(size_t is_master);

kvs_status_t request_k8s_get_replica_size(size_t& res);

#ifdef __cplusplus
}
#endif
#endif
