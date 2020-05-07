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

#include "base.h"
#include <ccl.hpp>

using namespace std;

void check_allreduce_on_comm(ccl_comm_t comm)
{
    size_t cur_comm_rank{};
    size_t cur_comm_size{};
    CCL_CALL(ccl_get_comm_rank(comm, &cur_comm_rank));
    CCL_CALL(ccl_get_comm_size(comm, &cur_comm_size));
    vector<float> send_buf(COUNT, cur_comm_rank);
    vector<float> recv_buf(COUNT, 0.0f);

    PRINT_BY_ROOT("allreduce on %zu ranks", cur_comm_size);

    coll_attr.to_cache = 0;

    CCL_CALL(ccl_allreduce(send_buf.data(),
                           recv_buf.data(),
                           COUNT,
                           ccl_dtype_float,
                           ccl_reduction_sum,
                           &coll_attr,
                           comm,
                           nullptr,
                           &request));
    CCL_CALL(ccl_wait(request));

    float expected = (cur_comm_size - 1) * ((float) cur_comm_size / 2);

    for (size_t i = 0; i < recv_buf.size(); ++i)
    {
        if (recv_buf[i] != expected)
        {
            printf("FAILED\n");
            throw runtime_error("Recv[ " + to_string(i) + "]= " + to_string(recv_buf[i]) +
                    ", expected " + to_string(expected));
        }
    }
}

void check_allreduce()
{
    PRINT_BY_ROOT("create new communicator as a copy of the global one and check that allreduce works");

    ccl_comm_t comm;
    CCL_CALL(ccl_comm_create(&comm, nullptr));
    check_allreduce_on_comm(comm);
    CCL_CALL(ccl_comm_free(comm));
}

void check_max_comm_number()
{
    PRINT_BY_ROOT("create max number of communicators");

    size_t user_comms = 0;
    std::vector<ccl_comm_t> communicators;
    ccl_status_t status = ccl_status_success;

    do
    {
        ccl_comm_t new_comm;
        status = ccl_comm_create(&new_comm, nullptr);

        ++user_comms;
        if (status != ccl_status_success)
        {
            break;
        }

        communicators.push_back(new_comm);

    } while (status == ccl_status_success);

    PRINT_BY_ROOT("created %zu communicators\n", user_comms);
    PRINT_BY_ROOT("try to create one more communicator, it should fail");

    ccl_comm_t comm;
    status = ccl_comm_create(&comm, nullptr);
    if (status == ccl_status_success)
    {
        ccl_comm_free(comm);
        printf("FAILED\n");
        throw runtime_error("extra communicator has been created");
    }

    PRINT_BY_ROOT("free one comm, try to create again");
    size_t comm_idx = user_comms / 2;
    status = ccl_comm_free(communicators[comm_idx]);
    if (status != ccl_status_success)
    {
        printf("FAILED\n");
        throw runtime_error("can't to free communicator");
    }

    status = ccl_comm_create(&communicators[comm_idx], nullptr);
    if (status != ccl_status_success)
    {
        printf("FAILED\n");
        throw runtime_error("can't create communicator after free");
    }

    for (auto& comm_elem: communicators)
    {
        CCL_CALL(ccl_comm_free(comm_elem));
    }
}

void check_comm_create_colored()
{
    PRINT_BY_ROOT("create comm with color, ranks_count should be a power of 2 for test purpose");

    for (size_t split_by = 2; split_by <= size; split_by *= 2)
    {
        ccl_comm_t comm;
        ccl_comm_t comm_inside;
        ccl_comm_attr_t comm_attr;
        ccl_comm_attr_t comm_attr_inside;
        comm_attr.color = ::rank % split_by;
        comm_attr_inside.color = ::rank / split_by;
        size_t comm_size{};
        size_t comm_size_inside{};
        size_t comm_rank{};
        size_t comm_rank_inside{};

        PRINT_BY_ROOT("splitting global comm into %zu parts", split_by);
        CCL_CALL(ccl_comm_create(&comm, &comm_attr));
        CCL_CALL(ccl_comm_create(&comm_inside, &comm_attr_inside));

        CCL_CALL(ccl_get_comm_size(comm, &comm_size));
        CCL_CALL(ccl_get_comm_size(comm_inside, &comm_size_inside));

        CCL_CALL(ccl_get_comm_rank(comm, &comm_rank));
        CCL_CALL(ccl_get_comm_rank(comm_inside, &comm_rank_inside));
        size_t expected_ranks_count = size / split_by;
        size_t expected_ranks_count_inside = split_by;

        if (comm_size != expected_ranks_count)
        {
            if (comm != nullptr)
                CCL_CALL(ccl_comm_free(comm));

            if (comm_inside != nullptr)
                CCL_CALL(ccl_comm_free(comm_inside));

            printf("FAILED\n");

            throw runtime_error("mismatch in size, expected " +
                                to_string(expected_ranks_count) +
                                " received " + to_string(comm_size));
        }

        if (comm_size_inside != expected_ranks_count_inside)
        {
            if (comm_inside != nullptr)
                CCL_CALL(ccl_comm_free(comm_inside));

            printf("FAILED\n");

            throw runtime_error("mismatch in size, expected " +
                                to_string(expected_ranks_count_inside) +
                                " received " + to_string(comm_size_inside));
        }

        PRINT_BY_ROOT("global comm: idx=%zu, count=%zu; new comms: rank=%zu, size=%zu; rank_inside=%zu, size_inside=%zu\n", ::rank,
                      size, comm_rank, comm_size, comm_rank_inside, comm_size_inside);

        check_allreduce_on_comm(comm);
        check_allreduce_on_comm(comm_inside);

        CCL_CALL(ccl_comm_free(comm));
        CCL_CALL(ccl_comm_free(comm_inside));
    }
}

void check_comm_create_identical_color()
{
    ccl_comm_t comm;
    ccl_comm_attr_t comm_attr;
    comm_attr.color = 123;
    size_t comm_size{};
    size_t comm_rank{};

    PRINT_BY_ROOT("create comm as a copy of the global one by settings identical colors");

    CCL_CALL(ccl_comm_create(&comm, &comm_attr));
    CCL_CALL(ccl_get_comm_size(comm, &comm_size));
    CCL_CALL(ccl_get_comm_rank(comm, &comm_rank));

    if (comm_size != size)
    {
        if (comm != nullptr)
            CCL_CALL(ccl_comm_free(comm));

        printf("FAILED\n");
        throw runtime_error("mismatch in size, expected " +
                            to_string(size) +
                            " received " + to_string(comm_size));
    }

    if (comm_rank != ::rank)
    {
        if (comm != nullptr)
            CCL_CALL(ccl_comm_free(comm));

        printf("FAILED\n");
        throw runtime_error("mismatch in rank, expected " +
                            to_string(::rank) +
                            " received " + to_string(comm_rank));
    }

    PRINT_BY_ROOT("global comm: rank=%zu, size=%zu; new comm: rank=%zu, size=%zu", ::rank,
                  size, comm_rank, comm_size);

    check_allreduce_on_comm(comm);

    CCL_CALL(ccl_comm_free(comm));
}

int main()
{
    test_init();

    PRINT_BY_ROOT("run communicator test on %zu ranks", size);

    check_allreduce();
    check_max_comm_number();
    check_comm_create_colored();
    check_comm_create_identical_color();

    test_finalize();

    PRINT_BY_ROOT("PASSED");

    return 0;
}
