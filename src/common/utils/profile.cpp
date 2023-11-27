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
#include "common/utils/utils.hpp"
#include "common/log/log.hpp"

ccl::profile::metrics_manager::~metrics_manager() {
    finalize();
}

void ccl::profile::metrics_manager::init() {
    allreduce_pipe_nonparallel_calls_per_count.clear();
    allreduce_pipe_parallel_calls_per_count.clear();
}

void ccl::profile::metrics_manager::finalize() {
    std::string allreduce_pipe_metrics;

    for (auto calls_per_count : allreduce_pipe_nonparallel_calls_per_count) {
        allreduce_pipe_metrics += "nonparallel_calls_per_count[" +
                                  std::to_string(calls_per_count.first) +
                                  "]=" + std::to_string(calls_per_count.second) + ",\n";
    }

    for (auto calls_per_count : allreduce_pipe_parallel_calls_per_count) {
        allreduce_pipe_metrics += "   parallel_calls_per_count[" +
                                  std::to_string(calls_per_count.first) +
                                  "]=" + std::to_string(calls_per_count.second) + ",\n";
    }

    if (!allreduce_pipe_metrics.empty()) {
        LOG_INFO("allreduce_pipe_metrics: [\n", allreduce_pipe_metrics, "]");
    }
}
