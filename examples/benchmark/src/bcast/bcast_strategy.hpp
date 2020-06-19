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
#ifndef BCAST_STRATEGY_HPP
#define BCAST_STRATEGY_HPP

struct bcast_strategy_impl
{
    static constexpr const char* class_name() { return "bcast"; }

    template<class Dtype>
    void start_internal(ccl::communicator &comm, size_t count, Dtype send_buf, Dtype recv_buf,
                        const bench_coll_exec_attr& bench_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        (void)send_buf;
        reqs.push_back(comm.bcast(recv_buf, count, COLL_ROOT, &bench_attr.coll_attr, stream));
    }
};

#endif /* BCAST_STRATEGY_HPP */
