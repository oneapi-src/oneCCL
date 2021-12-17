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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#include "def.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/helper.hpp"
#include "request_wrappers_k8s.hpp"

#define JOB_NAME "CCL_JOB_NAME"

#define ADDR_STR_V1_TEMPLATE "https://%s/api/v1/namespaces/default/pods/"
#define ADDR_STR_V2_TEMPLATE "https://%s/apis/apps/v1/namespaces/default/"

#define PATCH_TEMPLATE \
    "-X PATCH -d {\\\"metadata\\\":{\\\"labels\\\":{\\\"%s\\\":\\\"%s\\\"}}} -H \"Content-Type: application/merge-patch+json\""
#define PATCH_NULL_TEMPLATE \
    "-X PATCH -d {\\\"metadata\\\":{\\\"labels\\\":{\\\"%s\\\":null}}} -H \"Content-Type: application/merge-patch+json\""
#define AUTHORIZATION_TEMPLATE \
    "curl -s -H \"Authorization: Bearer `cat /var/run/secrets/kubernetes.io/serviceaccount/token`\" --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt %s%s %s"

#define MAX_KVS_STR_LENGTH       1024
#define CCL_K8S_MANAGER_TYPE_ENV "CCL_K8S_MANAGER_TYPE"
#define CCL_K8S_API_ADDR_ENV     "CCL_K8S_API_ADDR"

#define CCL_KVS_IP   "CCL_KVS_IP"
#define CCL_KVS_PORT "CCL_KVS_PORT"
#define REQ_KVS_IP   "CCL_REQ_KVS_IP"
#define MASTER_ADDR  "CCL_MASTER"

char ccl_kvs_ip[MAX_KVS_NAME_LENGTH];
char ccl_kvs_port[MAX_KVS_NAME_LENGTH];
char req_kvs_ip[MAX_KVS_NAME_LENGTH];
char master_addr[MAX_KVS_NAME_LENGTH];

#define KVS_IP   "KVS_IP"
#define KVS_PORT "KVS_PORT"

#define GET_KEY "| sed -r 's/\"[a-zA-Z0-9_]*-|: \"[a-zA-Z0-9_-]*|,|\"| |//g'"
#define GET_VAL "| sed -r 's/[a-zA-Z0-9_-]*\":|,|\"| |//g'"

#define CHECK_STR(expr, str) \
    do { \
        if (!(expr)) { \
            LOG_ERROR("wrong str: ", str); \
            return KVS_STATUS_FAILURE; \
        } \
    } while (0)

char run_get_template[RUN_TEMPLATE_SIZE];
char run_set_template[RUN_TEMPLATE_SIZE];
char job_name[MAX_KVS_NAME_LENGTH];

typedef enum manager_type {
    MT_NONE = 0,
    MT_K8S = 1,
} manager_type_t;

manager_type_t manager;

kvs_status_t request_k8s_get_keys_values_by_name(const char* kvs_name,
                                                 char*** kvs_key,
                                                 char*** kvs_values,
                                                 int& values_count);

kvs_status_t request_k8s_get_count_names(const char* kvs_name, size_t& res);

kvs_status_t request_k8s_get_val_by_name_key(const char* kvs_name,
                                             const char* kvs_key,
                                             char* kvs_val);

kvs_status_t request_k8s_remove_name_key(const char* kvs_name, const char* kvs_key);

kvs_status_t request_k8s_set_val(const char* kvs_name, const char* kvs_key, const char* kvs_val);

kvs_status_t json_get_val(FILE* fp, const char** keys, size_t keys_count, char* val) {
    char cur_kvs_str[MAX_KVS_STR_LENGTH];
    char* res;
    char last_char;
    size_t i = 0;
    size_t wrong_namespace_depth = 0;
    while (fgets(cur_kvs_str, MAX_KVS_STR_LENGTH, fp)) {
        if (wrong_namespace_depth == 0) {
            if (strstr(cur_kvs_str, keys[i])) {
                i++;
                if (i == keys_count)
                    break;
            }
            else if (strstr(cur_kvs_str, ": {") || strstr(cur_kvs_str, ": ["))
                wrong_namespace_depth++;
        }
        else {
            if (strstr(cur_kvs_str, "{") || strstr(cur_kvs_str, "["))
                wrong_namespace_depth++;
            if (strstr(cur_kvs_str, "}") || strstr(cur_kvs_str, "]"))
                wrong_namespace_depth--;
        }
    }
    CHECK_STR(res = strstr(cur_kvs_str, ":"), cur_kvs_str);
    do {
        res++;
        CHECK_STR(res, cur_kvs_str);
    } while (res[0] == ' ');

    if (res[0] == '"' || res[0] == '\'') {
        res++;
        CHECK_STR(res, cur_kvs_str);
    }

    int str_len = strlen(res) - 1;
    last_char = res[str_len];
    while (last_char == '\n' || last_char == ',' || last_char == ' ' || last_char == '"' ||
           last_char == ' ') {
        res[str_len] = '\0';
        str_len--;
        CHECK_STR(str_len, cur_kvs_str);
        last_char = res[str_len];
    }
    kvs_str_copy(val, res, MAX_KVS_VAL_LENGTH);
    while (fgets(cur_kvs_str, MAX_KVS_STR_LENGTH, fp)) {
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t k8s_init_with_manager() {
    FILE* fp;
    FILE* fp_name;
    FILE* fp_type;
    size_t i;
    size_t kind_type_size;
    char run_str[RUN_REQUEST_SIZE];
    char kind_type[MAX_KVS_NAME_LENGTH];
    char kind_name[MAX_KVS_NAME_LENGTH];
    char kind_path[MAX_KVS_NAME_KEY_LENGTH];
    char connect_api_template[RUN_TEMPLATE_SIZE];
    char* kube_api_addr = getenv(CCL_K8S_API_ADDR_ENV);
    const char* kind_type_key[] = { "metadata", "ownerReferences", "kind" };
    const char* kind_name_key[] = { "metadata", "ownerReferences", "name" };
    char pod_name[MAX_KVS_VAL_LENGTH];
    memset(pod_name, '\0', MAX_KVS_VAL_LENGTH);
    if ((fp = popen("hostname", READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get hostname\n");
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(pod_name, MAX_KVS_VAL_LENGTH, fp), pod_name);
    pclose(fp);
    int str_len = strlen(pod_name) - 1;
    CHECK_STR(str_len, "hostname");
    while (pod_name[str_len] == '\n' || pod_name[str_len] == ' ') {
        pod_name[str_len] = '\0';
        str_len--;
        CHECK_STR(str_len, "hostname");
    }
    if (kube_api_addr == NULL) {
        LOG_ERROR("%s not set\n", CCL_K8S_API_ADDR_ENV);
        return KVS_STATUS_FAILURE;
    }

    SET_STR(connect_api_template, RUN_TEMPLATE_SIZE, ADDR_STR_V1_TEMPLATE, kube_api_addr);

    /*get full pod info*/
    SET_STR(run_str, RUN_REQUEST_SIZE, AUTHORIZATION_TEMPLATE, connect_api_template, pod_name, "");

    memset(kind_type, NULL_CHAR, MAX_KVS_NAME_LENGTH);
    if ((fp_name = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get kind_type\n");
        return KVS_STATUS_FAILURE;
    }
    KVS_CHECK_STATUS(json_get_val(fp_name, kind_type_key, 3, kind_type), "failed to get type");

    /*we must use the plural to access to statefulset/deployment KVS*/
    kind_type_size = strlen(kind_type);
    kind_type[kind_type_size] = 's';
    kind_type_size++;
    for (i = 0; i < kind_type_size; i++)
        kind_type[i] = (char)tolower(kind_type[i]);

    memset(kind_name, NULL_CHAR, MAX_KVS_NAME_LENGTH);
    if ((fp_type = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get kind_name\n");
        return KVS_STATUS_FAILURE;
    }
    KVS_CHECK_STATUS(json_get_val(fp_type, kind_name_key, 3, kind_name), "filed to get name");

    SET_STR(kind_path, MAX_KVS_NAME_LENGTH, "%s/%s", kind_type, kind_name);
    SET_STR(connect_api_template, RUN_TEMPLATE_SIZE, ADDR_STR_V2_TEMPLATE, kube_api_addr);

    SET_STR(run_get_template,
            RUN_TEMPLATE_SIZE,
            AUTHORIZATION_TEMPLATE,
            connect_api_template,
            kind_path,
            "%s");
    SET_STR(run_set_template,
            RUN_TEMPLATE_SIZE,
            AUTHORIZATION_TEMPLATE,
            connect_api_template,
            kind_path,
            "%s");

    pclose(fp_name);
    pclose(fp_type);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t get_my_job_name(const char* connect_api_template) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char grep_kvs_name_key[REQUEST_POSTFIX_SIZE];
    char get_kvs_val[REQUEST_POSTFIX_SIZE];
    char pod_name[MAX_KVS_VAL_LENGTH];
    memset(pod_name, '\0', MAX_KVS_VAL_LENGTH);
    if ((fp = popen("hostname", READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get hostname\n");
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(pod_name, MAX_KVS_VAL_LENGTH, fp), pod_name);
    pclose(fp);
    int str_len = strlen(pod_name) - 1;
    CHECK_STR(str_len, "hostname");
    while (pod_name[str_len] == '\n' || pod_name[str_len] == ' ') {
        pod_name[str_len] = '\0';
        str_len--;
        CHECK_STR(str_len, "hostname");
    }

    SET_STR(grep_kvs_name_key, REQUEST_POSTFIX_SIZE, GREP_TEMPLATE, JOB_NAME);
    SET_STR(
        get_kvs_val, REQUEST_POSTFIX_SIZE, CONCAT_TWO_COMMAND_TEMPLATE, grep_kvs_name_key, GET_VAL);

    SET_STR(run_str,
            RUN_TEMPLATE_SIZE,
            AUTHORIZATION_TEMPLATE,
            connect_api_template,
            pod_name,
            get_kvs_val);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get %s", strerror(errno));
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(job_name, MAX_KVS_NAME_LENGTH, fp), job_name);
    pclose(fp);
    if (job_name[0] == NULL_CHAR) {
        job_name[0] = '0';
        job_name[1] = '_';
    }
    else {
        job_name[strlen(job_name) - 1] = '_';
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t k8s_init_without_manager() {
    FILE* fp;
    char* kube_api_addr = getenv(CCL_K8S_API_ADDR_ENV);
    char connect_api_template[RUN_TEMPLATE_SIZE];
    char pod_name[MAX_KVS_VAL_LENGTH];
    memset(pod_name, '\0', MAX_KVS_VAL_LENGTH);
    if ((fp = popen("hostname", READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get hostname\n");
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(pod_name, MAX_KVS_VAL_LENGTH, fp), pod_name);
    pclose(fp);
    int str_len = strlen(pod_name) - 1;
    CHECK_STR(str_len, "hostname");
    while (pod_name[str_len] == '\n' || pod_name[str_len] == ' ') {
        pod_name[str_len] = '\0';
        str_len--;
        CHECK_STR(str_len, "hostname");
    }

    if (kube_api_addr == NULL) {
        LOG_ERROR("%s not set\n", CCL_K8S_API_ADDR_ENV);
        return KVS_STATUS_FAILURE;
    }

    SET_STR(connect_api_template, RUN_TEMPLATE_SIZE, ADDR_STR_V1_TEMPLATE, kube_api_addr);
    SET_STR(run_get_template,
            RUN_TEMPLATE_SIZE,
            AUTHORIZATION_TEMPLATE,
            connect_api_template,
            " | sort ",
            "%s");
    SET_STR(run_set_template,
            RUN_TEMPLATE_SIZE,
            AUTHORIZATION_TEMPLATE,
            connect_api_template,
            pod_name,
            "%s");

    KVS_CHECK_STATUS(get_my_job_name(connect_api_template), "failed to get job name");

    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_kvs_init() {
    char* manager_type_env = getenv(CCL_K8S_MANAGER_TYPE_ENV);

    if (!manager_type_env || strstr(manager_type_env, "none")) {
        manager = MT_NONE;
    }
    else if (strstr(manager_type_env, "k8s")) {
        manager = MT_K8S;
    }
    else {
        LOG_WARN(
            "Unknown %s = %s, running with \"none\"\n", CCL_K8S_MANAGER_TYPE_ENV, manager_type_env);
        manager = MT_NONE;
    }

    memset(job_name, NULL_CHAR, MAX_KVS_NAME_LENGTH);

    switch (manager) {
        case MT_NONE:
            KVS_CHECK_STATUS(k8s_init_without_manager(), "failed to initialize k8z");
            break;
        case MT_K8S: KVS_CHECK_STATUS(k8s_init_with_manager(), "failed to initialize k8z"); break;
        default: LOG_ERROR("unknown k8s manager"); return KVS_STATUS_FAILURE;
    }

    memset(ccl_kvs_ip, NULL_CHAR, MAX_KVS_NAME_LENGTH);
    memset(ccl_kvs_port, NULL_CHAR, MAX_KVS_NAME_LENGTH);
    memset(req_kvs_ip, NULL_CHAR, MAX_KVS_NAME_LENGTH);
    memset(master_addr, NULL_CHAR, MAX_KVS_NAME_LENGTH);

    SET_STR(ccl_kvs_ip, MAX_KVS_NAME_LENGTH, KVS_NAME_TEMPLATE_S, job_name, CCL_KVS_IP);
    SET_STR(ccl_kvs_port, MAX_KVS_NAME_LENGTH, KVS_NAME_TEMPLATE_S, job_name, CCL_KVS_PORT);
    SET_STR(req_kvs_ip, MAX_KVS_NAME_LENGTH, KVS_NAME_TEMPLATE_S, job_name, REQ_KVS_IP);
    SET_STR(master_addr, MAX_KVS_NAME_LENGTH, KVS_NAME_TEMPLATE_S, job_name, MASTER_ADDR);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_kvs_get_master(const char* local_host_ip,
                                        char* main_host_ip,
                                        char* port_str) {
    char** kvs_values = NULL;
    char** kvs_keys = NULL;
    int values_count = 0;

    KVS_CHECK_STATUS(request_k8s_set_val(ccl_kvs_ip, my_hostname, local_host_ip),
                     "failed to set IP");
    KVS_CHECK_STATUS(request_k8s_set_val(ccl_kvs_port, my_hostname, port_str),
                     "failed to set port");
    size_t count;
    KVS_CHECK_STATUS(request_k8s_get_count_names(master_addr, count), "failed to get names count");
    if (count == 0) {
        KVS_CHECK_STATUS(
            request_k8s_get_keys_values_by_name(ccl_kvs_ip, &kvs_keys, &kvs_values, values_count),
            "failed to get keys");
        if (strstr(kvs_keys[0], my_hostname)) {
            KVS_CHECK_STATUS(request_k8s_set_val(req_kvs_ip, my_hostname, local_host_ip),
                             "failed to set IP");
            KVS_CHECK_STATUS(request_k8s_get_count_names(master_addr, count),
                             "failed to get names count");
            while (count == 0) {
                KVS_CHECK_STATUS(request_k8s_get_keys_values_by_name(
                                     req_kvs_ip, &kvs_keys, &kvs_values, values_count),
                                 "failed to get keys values");
                if (values_count > 1) {
                    if (!strstr(kvs_keys[0], my_hostname)) {
                        break;
                    }
                }
                else {
                    KVS_CHECK_STATUS(request_k8s_set_val(master_addr, KVS_IP, local_host_ip),
                                     "failed to set IP");
                    KVS_CHECK_STATUS(request_k8s_set_val(master_addr, KVS_PORT, port_str),
                                     "failed to set port");
                }
                KVS_CHECK_STATUS(request_k8s_get_count_names(master_addr, count),
                                 "failed to get names count");
            }
            KVS_CHECK_STATUS(request_k8s_remove_name_key(req_kvs_ip, my_hostname),
                             "failed to remove host info");
        }
        if (kvs_keys != NULL) {
            for (int i = 0; i < values_count; i++) {
                free(kvs_keys[i]);
            }
            free(kvs_keys);
        }
        if (kvs_values != NULL) {
            for (int i = 0; i < values_count; i++) {
                free(kvs_values[i]);
            }
            free(kvs_values);
        }
    }
    do {
        KVS_CHECK_STATUS(request_k8s_get_count_names(master_addr, count),
                         "failed to get names count");
        sleep(1);
    } while (count == 0);
    KVS_CHECK_STATUS(request_k8s_get_val_by_name_key(master_addr, KVS_IP, main_host_ip),
                     "failed to get IP");
    KVS_CHECK_STATUS(request_k8s_get_val_by_name_key(master_addr, KVS_PORT, port_str),
                     "failed to get port");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_kvs_finalize(size_t is_master) {
    KVS_CHECK_STATUS(request_k8s_remove_name_key(ccl_kvs_ip, my_hostname), "failed to remove IP");
    KVS_CHECK_STATUS(request_k8s_remove_name_key(ccl_kvs_port, my_hostname),
                     "failed to remove port");
    if (is_master) {
        KVS_CHECK_STATUS(request_k8s_remove_name_key(master_addr, KVS_IP),
                         "failed to remove master IP");
        KVS_CHECK_STATUS(request_k8s_remove_name_key(master_addr, KVS_PORT),
                         "failed to remove master IP");
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t get_by_template(char*** kvs_entry,
                             const char* request,
                             const char* template_str,
                             int count,
                             int max_count) {
    FILE* fp;
    char get_val[REQUEST_POSTFIX_SIZE];
    char run_str[RUN_REQUEST_SIZE];
    int i;

    if (*kvs_entry != NULL)
        free(*kvs_entry);

    *kvs_entry = (char**)malloc(sizeof(char*) * count);
    if (*kvs_entry == NULL) {
        LOG_ERROR("Memory allocation failed\n");
        return KVS_STATUS_FAILURE;
    }
    for (i = 0; i < count; i++) {
        (*kvs_entry)[i] = (char*)malloc(sizeof(char) * max_count);
        if ((*kvs_entry)[i] == NULL) {
            LOG_ERROR("Memory allocation failed\n");
            return KVS_STATUS_FAILURE;
        }
    }

    i = 0;

    SET_STR(get_val, REQUEST_POSTFIX_SIZE, CONCAT_TWO_COMMAND_TEMPLATE, request, template_str);
    SET_STR(run_str, RUN_REQUEST_SIZE, run_get_template, get_val);
    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get by template\n");
        return KVS_STATUS_FAILURE;
    }
    while ((fgets((*kvs_entry)[i], max_count, fp) != NULL) && (i < count)) {
        while ((*kvs_entry)[i][strlen((*kvs_entry)[i]) - 1] == '\n' ||
               (*kvs_entry)[i][strlen((*kvs_entry)[i]) - 1] == ' ')
            (*kvs_entry)[i][strlen((*kvs_entry)[i]) - 1] = NULL_CHAR;
        i++;
    }
    pclose(fp);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_get_keys_values_by_name(const char* kvs_name,
                                                 char*** kvs_keys,
                                                 char*** kvs_values,
                                                 int& values_count) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char grep_name_str[REQUEST_POSTFIX_SIZE];
    char get_name_count[REQUEST_POSTFIX_SIZE];
    char values_count_str[INT_STR_SIZE];
    values_count = 0;

    SET_STR(get_name_count, REQUEST_POSTFIX_SIZE, GREP_COUNT_TEMPLATE, kvs_name);

    /*get dead/new rank*/
    SET_STR(run_str, RUN_REQUEST_SIZE, run_get_template, get_name_count);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get keys-values by name: %s\n", kvs_name);
        return KVS_STATUS_SUCCESS;
    }
    CHECK_FGETS(fgets(values_count_str, INT_STR_SIZE, fp), values_count_str);
    pclose(fp);

    KVS_CHECK_STATUS(safe_strtol(values_count_str, values_count), "failed to convert count");
    if (values_count == 0)
        return KVS_STATUS_SUCCESS;

    SET_STR(grep_name_str, REQUEST_POSTFIX_SIZE, GREP_TEMPLATE, kvs_name);
    if (kvs_values != NULL) {
        KVS_CHECK_STATUS(
            get_by_template(kvs_values, grep_name_str, GET_VAL, values_count, MAX_KVS_VAL_LENGTH),
            "failed to get val");
    }
    if (kvs_keys != NULL) {
        KVS_CHECK_STATUS(
            get_by_template(kvs_keys, grep_name_str, GET_KEY, values_count, MAX_KVS_KEY_LENGTH),
            "failed to get key");
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_get_count_names(const char* kvs_name, size_t& res) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char get_count_str[REQUEST_POSTFIX_SIZE];
    char count_names[INT_STR_SIZE];
    res = 0;

    SET_STR(get_count_str, REQUEST_POSTFIX_SIZE, GREP_COUNT_TEMPLATE, kvs_name);

    /*get count accepted pods (like comm_world size)*/
    SET_STR(run_str, RUN_REQUEST_SIZE, run_get_template, get_count_str);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get names count: %s\n", kvs_name);
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(count_names, INT_STR_SIZE, fp), count_names);
    pclose(fp);

    KVS_CHECK_STATUS(safe_strtol(count_names, res), "failed to convert cont names");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_get_val_by_name_key(const char* kvs_name,
                                             const char* kvs_key,
                                             char* kvs_val) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char grep_kvs_name_key[REQUEST_POSTFIX_SIZE];
    char get_kvs_val[REQUEST_POSTFIX_SIZE];
    char kvs_name_key[MAX_KVS_NAME_KEY_LENGTH];

    SET_STR(kvs_name_key, MAX_KVS_NAME_KEY_LENGTH, KVS_NAME_KEY_TEMPLATE, kvs_name, kvs_key);
    SET_STR(grep_kvs_name_key, REQUEST_POSTFIX_SIZE, GREP_TEMPLATE, kvs_name_key);
    SET_STR(
        get_kvs_val, REQUEST_POSTFIX_SIZE, CONCAT_TWO_COMMAND_TEMPLATE, grep_kvs_name_key, GET_VAL);

    /*check: is me accepted*/
    SET_STR(run_str, RUN_REQUEST_SIZE, run_get_template, get_kvs_val);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't get value by name-key: %s\n", kvs_name_key);
        return KVS_STATUS_FAILURE;
    }
    CHECK_FGETS(fgets(kvs_val, MAX_KVS_VAL_LENGTH, fp), kvs_val);
    pclose(fp);
    kvs_val[strlen(kvs_val) - 1] = NULL_CHAR;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_remove_name_key(const char* kvs_name, const char* kvs_key) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char patch[REQUEST_POSTFIX_SIZE];
    char kvs_name_key[MAX_KVS_NAME_KEY_LENGTH];

    SET_STR(kvs_name_key, MAX_KVS_NAME_KEY_LENGTH, KVS_NAME_KEY_TEMPLATE, kvs_name, kvs_key);
    SET_STR(patch, REQUEST_POSTFIX_SIZE, PATCH_NULL_TEMPLATE, kvs_name_key);

    /*remove old entry*/
    SET_STR(run_str, RUN_REQUEST_SIZE, run_set_template, patch);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't remove name-key: %s\n", kvs_name_key);
        return KVS_STATUS_FAILURE;
    }
    pclose(fp);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_set_val(const char* kvs_name, const char* kvs_key, const char* kvs_val) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char patch[REQUEST_POSTFIX_SIZE];
    char kvs_name_key[MAX_KVS_NAME_KEY_LENGTH];

    SET_STR(kvs_name_key, MAX_KVS_NAME_KEY_LENGTH, KVS_NAME_KEY_TEMPLATE, kvs_name, kvs_key);
    SET_STR(patch, REQUEST_POSTFIX_SIZE, PATCH_TEMPLATE, kvs_name_key, kvs_val);

    /*insert new entry*/
    SET_STR(run_str, RUN_REQUEST_SIZE, run_set_template, patch);

    if ((fp = popen(run_str, READ_ONLY)) == NULL) {
        LOG_ERROR("Can't set name-key-val: %s-%s\n", kvs_name_key, kvs_val);
        return KVS_STATUS_FAILURE;
    }
    pclose(fp);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t request_k8s_get_replica_size(size_t& res) {
    FILE* fp;
    char run_str[RUN_REQUEST_SIZE];
    char replica_size_str[MAX_KVS_VAL_LENGTH];
    const char* replica_keys[] = { "spec", "replicas" };

    switch (manager) {
        case MT_NONE: return request_k8s_get_count_names(ccl_kvs_ip, res);
        case MT_K8S:
            /*get full output*/
            SET_STR(run_str, RUN_REQUEST_SIZE, run_get_template, "");

            if ((fp = popen(run_str, READ_ONLY)) == NULL) {
                printf("Can't get replica size\n");
                exit(1);
            }
            KVS_CHECK_STATUS(json_get_val(fp, replica_keys, 2, replica_size_str),
                             "failed to get replica size");
            pclose(fp);
            KVS_CHECK_STATUS(safe_strtol(replica_size_str, res), "failed to convert replica size");
            return KVS_STATUS_SUCCESS;
        default: LOG_ERROR("unknown k8s manager"); return KVS_STATUS_FAILURE;
    }
    return KVS_STATUS_SUCCESS;
}
