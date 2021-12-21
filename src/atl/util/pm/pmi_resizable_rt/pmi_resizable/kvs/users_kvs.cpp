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
#include <string>
#include <cstring>
#include "users_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"

users_kvs::users_kvs(std::shared_ptr<ccl::kvs_interface> kvs) : kvs(kvs) {}

kvs_status_t users_kvs::kvs_set_value(const char* kvs_name,
                                      const char* kvs_key,
                                      const char* kvs_val) {
    ccl::string_class name(kvs_name), key(kvs_key);
    ccl::vector_class<char> vec_val(kvs_val, kvs_val + strlen(kvs_val) + 1);
    vec_val[strlen(kvs_val)] = '\0';
    kvs->set(name + key, vec_val);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t users_kvs::kvs_remove_name_key(const char* kvs_name, const char* kvs_key) {
    ccl::vector_class<char> kvs_val = { '\0' };
    ccl::string_class name(kvs_name), key(kvs_key);
    kvs->set(name + key, kvs_val);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t users_kvs::kvs_get_value_by_name_key(const char* kvs_name,
                                                  const char* kvs_key,
                                                  char* kvs_val) {
    ccl::string_class name(kvs_name), key(kvs_key);
    ccl::vector_class<char> res = kvs->get(name + key);

    memset(kvs_val, 0, MAX_KVS_VAL_LENGTH);
    if (res.data())
        SET_STR(kvs_val, MAX_KVS_VAL_LENGTH, "%s", res.data());
    else
        SET_STR(kvs_val, MAX_KVS_VAL_LENGTH, "%s", "");

    return KVS_STATUS_SUCCESS;
}

kvs_status_t users_kvs::kvs_get_count_names(const char* kvs_name, int& count_names) {
    /*TODO: Unsupported*/
    (void)kvs_name;
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}

kvs_status_t users_kvs::kvs_get_keys_values_by_name(const char* kvs_name,
                                                    char*** kvs_keys,
                                                    char*** kvs_values,
                                                    size_t& count) {
    /*TODO: Unsupported*/
    (void)kvs_name;
    (void)kvs_keys;
    (void)kvs_values;
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}

kvs_status_t users_kvs::kvs_get_replica_size(size_t& replica_size) {
    /*TODO: Unsupported*/
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}

kvs_status_t users_kvs::kvs_main_server_address_reserve(char* main_address) {
    /*TODO: Unsupported*/
    (void)main_address;
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}

kvs_status_t users_kvs::kvs_init(const char* main_addr) {
    /*TODO: Unsupported*/
    (void)main_addr;
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}

kvs_status_t users_kvs::kvs_finalize(void) {
    /*TODO: Unsupported*/
    LOG_ERROR("unsupported");
    return KVS_STATUS_UNSUPPORTED;
}
