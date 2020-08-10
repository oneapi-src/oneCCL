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
#include "base.hpp"

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    std::vector<float>& recv_buf,
                    ccl::communicator_t& comm,
                    ccl::stream_t& stream,
                    ccl::coll_attr& coll_attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = (comm->size() - 1) * (static_cast<float>(comm->size()) / 2);
    float received;

    comm->barrier(stream);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        comm->reduce(send_buf.data(),
                     recv_buf.data(),
                     recv_buf.size(),
                     ccl::reduction::sum,
                     COLL_ROOT,
                     &coll_attr,
                     stream)
            ->wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < recv_buf.size(); idx++) {
        received = recv_buf[idx];
        if ((comm->rank() == COLL_ROOT) && (received != expected)) {
            fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, received);

            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    comm->barrier(stream);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

int main() {
    auto comm = ccl::environment::instance().create_communicator();
    auto stream = ccl::environment::instance().create_stream();
    ccl::coll_attr coll_attr{};

    MSG_LOOP(comm, std::vector<float> send_buf(msg_count, static_cast<float>(comm->rank()));
             std::vector<float> recv_buf(msg_count);
             coll_attr.to_cache = 0;
             run_collective("warmup_reduce", send_buf, recv_buf, comm, stream, coll_attr);
             coll_attr.to_cache = 1;
             run_collective("persistent_reduce", send_buf, recv_buf, comm, stream, coll_attr);
             coll_attr.to_cache = 0;
             run_collective("regular_reduce", send_buf, recv_buf, comm, stream, coll_attr););

    return 0;
}
