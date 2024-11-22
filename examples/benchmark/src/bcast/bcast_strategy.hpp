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

struct bcast_strategy_impl {
    static constexpr const char* class_name() {
        return "bcast";
    }

    size_t get_send_multiplier() {
        return 1;
    }

    size_t get_recv_multiplier() {
        return 1;
    }

    static const ccl::broadcast_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::broadcast_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t count,
                        Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        (void)send_buf;
        reqs.push_back(ccl::broadcast(
            recv_buf, count, get_ccl_dtype<Dtype>(), COLL_ROOT, comm, std::forward<Args>(args)...));
    }
};

struct broadcast_strategy_impl {
    static constexpr const char* class_name() {
        return "broadcast";
    }

    size_t get_send_multiplier() {
        return 1;
    }

    size_t get_recv_multiplier() {
        return 1;
    }

    static const ccl::broadcast_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::broadcast_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t count,
                        Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        reqs.push_back(ccl::broadcast(send_buf,
                                      recv_buf,
                                      count,
                                      get_ccl_dtype<Dtype>(),
                                      COLL_ROOT,
                                      comm,
                                      std::forward<Args>(args)...));
    }
};
