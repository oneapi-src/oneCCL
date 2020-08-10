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

#include "base.hpp"
#include <ccl.hpp>

#define COUNT (8192)

using namespace std;

ccl::communicator_t global_comm;

void check_allreduce_on_comm(ccl::communicator_t& comm) {
    size_t cur_comm_rank = comm->rank();
    size_t cur_comm_size = comm->size();

    vector<float> send_buf(COUNT, cur_comm_rank);
    vector<float> recv_buf(COUNT, 0.0f);

    PRINT_BY_ROOT(global_comm, "allreduce on %zu ranks", cur_comm_size);

    comm->allreduce(
            send_buf.data(), recv_buf.data(), COUNT, ccl::datatype::dt_float, ccl::reduction::sum)
        ->wait();

    float expected = (cur_comm_size - 1) * ((float)cur_comm_size / 2);

    for (size_t i = 0; i < recv_buf.size(); ++i) {
        if (recv_buf[i] != expected) {
            printf("FAILED\n");
            throw std::runtime_error("recv[" + to_string(i) + "]= " + to_string(recv_buf[i]) +
                                     ", expected " + to_string(expected));
        }
    }
}

void check_allreduce() {
    PRINT_BY_ROOT(
        global_comm,
        "create new communicator as a copy of the global one and check that allreduce works");

    auto comm = ccl::environment::instance().create_communicator();
    check_allreduce_on_comm(comm);
}

void check_max_comm_number() {
    PRINT_BY_ROOT(global_comm, "create max number of communicators");

    size_t user_comms = 0;
    std::vector<ccl::communicator_t> communicators;

    do {
        try {
            auto new_comm = ccl::environment::instance().create_communicator();
            ++user_comms;
            communicators.push_back(std::move(new_comm));
        }
        catch (...) {
            break;
        }
    } while (true);

    PRINT_BY_ROOT(global_comm, "created %zu communicators", user_comms);
    PRINT_BY_ROOT(global_comm, "try to create one more communicator, it should fail");

    try {
        auto comm = ccl::environment::instance().create_communicator();
        printf("FAILED\n");
        throw std::runtime_error("extra communicator has been created");
    }
    catch (...) {
    }

    PRINT_BY_ROOT(global_comm, "free one comm, try to create again");
    size_t comm_idx = user_comms / 2;

    try {
        communicators[comm_idx].reset();
    }
    catch (...) {
        printf("FAILED\n");
        throw std::runtime_error("can't free communicator");
    }

    try {
        communicators[comm_idx] = ccl::environment::instance().create_communicator();
    }
    catch (...) {
        printf("FAILED\n");
        throw std::runtime_error("can't create communicator after free");
    }
}

void check_comm_create_colored() {
    PRINT_BY_ROOT(global_comm,
                  "create comm with color, comm_size should be a power of 2 for test purpose");

    for (size_t split_by = 2; split_by <= global_comm->size(); split_by *= 2) {
        ccl::comm_attr_t comm_attr = ccl::environment::instance().create_host_comm_attr();
        ccl::comm_attr_t comm_attr_inside = ccl::environment::instance().create_host_comm_attr();

        comm_attr->set_value<ccl_host_color, int>(global_comm->rank() % split_by);
        comm_attr_inside->set_value<ccl_host_color, int>(global_comm->rank() / split_by);

        auto comm = ccl::environment::instance().create_communicator(comm_attr);
        auto comm_inside = ccl::environment::instance().create_communicator(comm_attr_inside);

        size_t comm_size = comm->size();
        size_t comm_size_inside = comm_inside->size();
        size_t comm_rank = comm->rank();
        size_t comm_rank_inside = comm_inside->rank();

        size_t expected_comm_size = global_comm->size() / split_by;
        size_t expected_comm_size_inside = split_by;

        if (comm_size != expected_comm_size) {
            printf("FAILED\n");

            throw std::runtime_error("mismatch in size, expected " + to_string(expected_comm_size) +
                                     " received " + to_string(comm_size));
        }

        if (comm_size_inside != expected_comm_size_inside) {
            printf("FAILED\n");

            throw std::runtime_error("mismatch in size, expected " +
                                     to_string(expected_comm_size_inside) + " received " +
                                     to_string(comm_size_inside));
        }

        PRINT_BY_ROOT(global_comm,
                      "global comm: idx = %zu, count = %zu; "
                      "new comms: rank = %zu, size = %zu; "
                      "rank_inside = %zu, size_inside = %zu\n",
                      global_comm->rank(),
                      global_comm->size(),
                      comm_rank,
                      comm_size,
                      comm_rank_inside,
                      comm_size_inside);

        check_allreduce_on_comm(comm);
        check_allreduce_on_comm(comm_inside);
    }
}

void check_comm_create_identical_color() {
    size_t comm_size{};
    size_t comm_rank{};

    PRINT_BY_ROOT(global_comm,
                  "create comm as a copy of the global one by settings identical colors");

    ccl::comm_attr_t comm_attr = ccl::environment::instance().create_host_comm_attr();
    comm_attr->set_value<ccl_host_color>(123);
    auto comm = ccl::environment::instance().create_communicator(comm_attr);

    comm_size = comm->size();
    comm_rank = comm->rank();

    if (comm_size != global_comm->size()) {
        printf("FAILED\n");
        throw std::runtime_error("mismatch in size, expected " + to_string(global_comm->size()) +
                                 " received " + to_string(comm_size));
    }

    if (comm_rank != global_comm->rank()) {
        printf("FAILED\n");
        throw std::runtime_error("mismatch in rank, expected " + to_string(global_comm->rank()) +
                                 " received " + to_string(comm_rank));
    }

    PRINT_BY_ROOT(global_comm,
                  "global comm: rank = %zu, size = %zu; "
                  "new comm: rank = %zu, size = %zu",
                  global_comm->rank(),
                  global_comm->size(),
                  comm_rank,
                  comm_size);

    check_allreduce_on_comm(comm);
}

int main() {
    global_comm = ccl::environment::instance().create_communicator();

    PRINT_BY_ROOT(global_comm, "run communicator test on %zu ranks", global_comm->size());

    check_allreduce();
    check_max_comm_number();
    check_comm_create_colored();
    check_comm_create_identical_color();

    global_comm.reset();

    PRINT("PASSED");

    return 0;
}
