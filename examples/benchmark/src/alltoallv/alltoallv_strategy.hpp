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
#ifndef ALLTOALLV_STRATEGY_HPP
#define ALLTOALLV_STRATEGY_HPP

struct alltoallv_strategy_impl {
    size_t comm_size = 0;
    std::vector<size_t> send_counts;
    std::vector<size_t> recv_counts;

    alltoallv_strategy_impl(size_t size) : comm_size(size) {
        send_counts.resize(comm_size);
        recv_counts.resize(comm_size);
    }

    alltoallv_strategy_impl(const alltoallv_strategy_impl&) = delete;
    alltoallv_strategy_impl& operator=(const alltoallv_strategy_impl&) = delete;

    ~alltoallv_strategy_impl() {}

    static constexpr const char* class_name() {
        return "alltoallv";
    }

    static const ccl::alltoallv_attr& get_op_attr(const bench_coll_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::alltoallv_attr>();
    }

    template <class Dtype, class comm_t, class... Args>
    void start_internal(comm_t& comm,
                        size_t count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_coll_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        for (size_t idx = 0; idx < comm_size; idx++) {
            send_counts[idx] = count;
            recv_counts[idx] = count;
        }

        reqs.push_back(comm.alltoallv(
            send_buf, send_counts, recv_buf, recv_counts, std::forward<Args>(args)...));
    }
};

#endif /* ALLTOALLV_STRATEGY_HPP */
