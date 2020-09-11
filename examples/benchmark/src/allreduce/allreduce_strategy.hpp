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
#ifndef ALLREDUCE_STRATEGY_HPP
#define ALLREDUCE_STRATEGY_HPP

struct allreduce_strategy_impl {
    static constexpr const char* class_name() {
        return "allreduce";
    }

    static const ccl::allreduce_attr& get_op_attr(const bench_coll_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::allreduce_attr>();
    }

    template <class Dtype, class comm_t, class... Args>
    void start_internal(comm_t& comm,
                        size_t count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_coll_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        reqs.push_back(comm.allreduce(
            send_buf, recv_buf, count, bench_attr.reduction, std::forward<Args>(args)...));
    }
};

#endif /* ALLREDUCE_STRATEGY_HPP */
