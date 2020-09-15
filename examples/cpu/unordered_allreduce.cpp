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
#include "base.h"

#include <vector>
#include <list>
#include <string>
#include <utility>
#include <thread>
#include <random>

ccl_request_t start_allreduce_with_tensor_name(const std::string& tensor_name,
                                               const float* send_buff,
                                               float* recv_buff) {
    coll_attr.to_cache = true;
    coll_attr.match_id = tensor_name.c_str();

    ccl_request_t req = nullptr;

    CCL_CALL(ccl_allreduce(send_buff,
                           recv_buff,
                           COUNT,
                           ccl_dtype_float,
                           ccl_reduction_sum,
                           &coll_attr,
                           nullptr,
                           nullptr,
                           &req));
    return req;
}

int main() {
    setenv("CCL_UNORDERED_COLL", "1", 1);

    const size_t iterations_count = 64;
    std::vector<std::string> tensor_names;
    // request, operation idx (for example purpose)
    std::list<std::pair<ccl_request_t, size_t>> started_ops;
    std::vector<std::vector<float>> allreduce_send_bufs;
    std::vector<std::vector<float>> allreduce_recv_bufs;

    test_init();

    std::random_device rand_dev;
    std::uniform_int_distribution<size_t> distribution(0, size - 1);

    for (size_t rank_idx = 0; rank_idx < size; ++rank_idx) {
        tensor_names.emplace_back("tensor_number_" + std::to_string(rank_idx));
        allreduce_send_bufs.emplace_back(COUNT, rank_idx + 1);
        allreduce_recv_bufs.emplace_back(COUNT, 0.0f);
    }

    if (rank != 0) {
        // delay non-root ranks to check that delayed comm creation works
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    for (size_t iteration = 0; iteration < iterations_count; ++iteration) {
        printf("## starting iteration #%zu\n", iteration);

        size_t start_idx = distribution(rand_dev);
        size_t rank_idx = start_idx;
        size_t operations_count = 0;

        for (; operations_count < size; ++operations_count, rank_idx = (rank_idx + 1) % size) {
            // start allreduce with shift in tensor names
            printf("   submit allreduce #%zu for tensor %s\n",
                   rank_idx,
                   tensor_names[rank_idx].c_str());
            started_ops.emplace_back(
                start_allreduce_with_tensor_name(tensor_names[rank_idx],
                                                 allreduce_send_bufs[rank_idx].data(),
                                                 allreduce_recv_bufs[rank_idx].data()),
                rank_idx);
        }

        int test_completed = 0;

        while (!started_ops.empty()) {
            for (auto it = started_ops.begin(); it != started_ops.end();) {
                ccl_test(it->first, &test_completed);
                if (test_completed) {
                    float expected = (it->second + 1) * size;
                    printf(
                        "   completed allreduce #%zu for tensor %s. Actual %3.2f, expected %3.2f\n",
                        it->second,
                        tensor_names[it->second].c_str(),
                        allreduce_recv_bufs[it->second][0],
                        expected);
                    for (size_t idx = 0; idx < COUNT; ++idx) {
                        if (allreduce_recv_bufs[it->second][idx] != expected) {
                            fprintf(
                                stderr,
                                "!! wrong result, rank %zu, result %3.2f, intitial %3.2f, exp %3.f\n",
                                it->second,
                                allreduce_recv_bufs[it->second][idx],
                                allreduce_send_bufs[it->second][idx],
                                expected);
                            printf("FAILED\n");
                            exit(1);
                        }
                    }

                    it = started_ops.erase(it);
                }
                else {
                    ++it;
                }
            }
        }

        printf("## iteration #%zu has been finished\n", iteration);
    }

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
