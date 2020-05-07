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
#include <algorithm>
#include <iostream>
#include <list>
#include <vector>

#include "base.h"
#include <ccl.hpp>

using namespace std;

ccl_status_t custom_reduce(const void* in_buf, size_t in_count, void* inout_buf,
                           size_t* out_count, ccl_datatype_t dtype,
                           const ccl_fn_context_t* context)
{
    size_t dtype_size;
    ccl_get_datatype_size(dtype, &dtype_size);

    ASSERT((dtype != ccl_dtype_char) &&
           (dtype != ccl_dtype_int) &&
           (dtype != ccl_dtype_bfp16) &&
           (dtype != ccl_dtype_float) &&
           (dtype != ccl_dtype_double) &&
           (dtype != ccl_dtype_int64) &&
           (dtype != ccl_dtype_uint64),
           "unexpected datatype %d", dtype);

    for (size_t idx = 0; idx < in_count; idx++)
    {
        ((float*)inout_buf)[idx] += ((float*)in_buf)[idx];
    }

    return ccl_status_success;
}

void check_predefined_datatypes()
{
    PRINT_BY_ROOT("check_predefined_datatypes");

    size_t dtype_size, expected_dtype_size;

    for (ccl_datatype_t idx = ccl_dtype_char; idx < ccl_dtype_last_value; idx++)
    {
        ccl_get_datatype_size(idx, &dtype_size);

        expected_dtype_size =
            (idx == ccl_dtype_char) ? sizeof(char) :
            (idx == ccl_dtype_int) ? sizeof(int) :
            (idx == ccl_dtype_bfp16) ? sizeof(uint16_t) :
            (idx == ccl_dtype_float) ? sizeof(float) :
            (idx == ccl_dtype_double) ? sizeof(double) :
            (idx == ccl_dtype_int64) ? sizeof(int64_t) :
            (idx == ccl_dtype_uint64) ? sizeof(uint64_t) : 0;
        
        ASSERT(dtype_size == expected_dtype_size,
               "unexpected datatype size: got %zu, expected %zu",
               dtype_size, expected_dtype_size);
    }
}

void check_create_and_free()
{
    PRINT_BY_ROOT("check_create_and_free");

    ccl_datatype_attr_t attr;
    size_t dtype_size;
    const size_t max_dtype_count = 16 * 1024;
    const size_t iter_count = 16;
    vector<ccl_datatype_t> dtypes(max_dtype_count);

    for (size_t iter = 0; iter < iter_count; iter++)
    {
        dtypes.clear();    

        for (size_t idx = 0; idx < max_dtype_count; idx++)
        {
            attr.size = idx + 1;
            ccl_datatype_create(&dtypes[idx], &attr);

            ccl_get_datatype_size(dtypes[idx], &dtype_size);

            if (dtype_size != (idx + 1))
            {
                ASSERT(0, "unexpected datatype size: got %zu, expected %zu",
                       dtype_size, (idx + 1));
            }
        }

        for (size_t idx = 0; idx < max_dtype_count; idx++)
        {
            ccl_datatype_free(dtypes[idx]);
        }
    }
}

void check_allreduce()
{
    PRINT_BY_ROOT("check_allreduce");

    ccl_datatype_attr_t attr;
    const size_t max_dtype_count = 1024;

    vector<ccl_datatype_t> dtypes(max_dtype_count);
    vector<ccl_request_t> reqs(max_dtype_count);
    vector<vector<float>> send_bufs(max_dtype_count);
    vector<vector<float>> recv_bufs(max_dtype_count);

    attr.size = sizeof(float);

    for (size_t idx = 0; idx < max_dtype_count; idx++)
    {
        ccl_datatype_create(&dtypes[idx], &attr);
        send_bufs[idx].resize(COUNT, ::rank + 1);
        recv_bufs[idx].resize(COUNT, 0);
    }

    ccl_coll_attr_t coll_attr{};
    coll_attr.reduction_fn = custom_reduce;

    for (size_t idx = 0; idx < max_dtype_count; idx++)
    {
        CCL_CALL(ccl_allreduce(send_bufs[idx].data(),
                               recv_bufs[idx].data(),
                               COUNT,
                               dtypes[idx],
                               ccl_reduction_custom,
                               &coll_attr,
                               nullptr,
                               nullptr,
                               &(reqs[idx])));
    }

    for (size_t idx = 0; idx < max_dtype_count; idx++)
    {
        CCL_CALL(ccl_wait(reqs[idx]));
    }

    float expected = (size + 1) * ((float)size / 2);

    for (size_t idx = 0; idx < max_dtype_count; idx++)
    {
        for (size_t elem_idx = 0; elem_idx < recv_bufs[idx].size(); ++elem_idx)
        {
            if (recv_bufs[idx][elem_idx] != expected)
            {
                ASSERT(0, "buf_idx %zu, elem_idx %zu: expected %f, got %f",
                       idx, elem_idx, expected, recv_bufs[idx][elem_idx]);
            }
        }
    }

    for (size_t idx = 0; idx < max_dtype_count; idx++)
    {
        ccl_datatype_free(dtypes[idx]);
    }
}

void check_invalid_actions()
{
    ccl_status_t status;

    ccl_datatype_t custom_dtype;
    ccl_datatype_create(&custom_dtype, nullptr);

    for (ccl_datatype_t idx = ccl_dtype_char; idx < ccl_dtype_last_value; idx++)
    {
        status = ccl_datatype_free(idx);
        ASSERT(status == ccl_status_invalid_arguments,
               "unexpected status %d", status);
    }

    status = ccl_datatype_free(custom_dtype);
    ASSERT(status == ccl_status_success,
           "unexpected status %d", status);

    status = ccl_datatype_free(custom_dtype);
    ASSERT(status == ccl_status_invalid_arguments,
           "unexpected status %d", status);

    status = ccl_datatype_free(custom_dtype + 1);
    ASSERT(status == ccl_status_invalid_arguments,
           "unexpected status %d", status);
}

int main()
{
    test_init();

    check_predefined_datatypes();
    check_create_and_free();
    check_allreduce();
    check_invalid_actions();

    test_finalize();

    PRINT_BY_ROOT("PASSED");

    return 0;
}
