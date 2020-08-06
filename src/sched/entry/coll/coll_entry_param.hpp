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

#include "coll/coll.hpp"

struct ccl_coll_entry_param {
    ccl_coll_type ctype;
    ccl_buffer buf;
    ccl_buffer send_buf;
    ccl_buffer recv_buf;
    size_t count;
    size_t send_count;
    const size_t* send_counts;
    const size_t* recv_counts;
    ccl_datatype dtype;
    ccl_reduction_t reduction;
    size_t root;
    ccl_comm* comm;

    ccl_coll_param to_coll_param() const {
        ccl_coll_param param;
        param.ctype = ctype;
        param.buf = buf.get_ptr();
        param.send_buf = send_buf.get_ptr();
        param.recv_buf = recv_buf.get_ptr();
        param.count = count;
        param.send_count = send_count;
        param.recv_counts = recv_counts;
        param.dtype = dtype;
        param.reduction = reduction;
        param.root = root;
        param.comm = comm;
        return param;
    }
};
