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

#include "common/comm/comm.hpp"
#include "common/utils/buffer.hpp"
#include "sched/sched.hpp"

class ccl_allreduce_2d_builder
{
public:
    ccl_allreduce_2d_builder();
    ~ccl_allreduce_2d_builder();

    ccl_status_t build(ccl_sched* sched,
                       ccl_buffer send_buf,
                       ccl_buffer recv_buf,
                       size_t count,
                       const ccl_datatype& dtype,
                       ccl_reduction_t op,
                       ccl_comm* comm);

    ccl_comm* get_first_dim_comm() const { return first_dim_comm.get(); }
    ccl_comm* get_second_dim_comm() const { return second_dim_comm.get(); }

private:
    std::shared_ptr<ccl_comm> first_dim_comm;
    std::shared_ptr<ccl_comm> second_dim_comm;

    int switch_dim = 0;
};
