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
#include <vector>

#include "base.hpp"

void check_allreduce_on_comm(const ccl::communicator& comm) {
    const int count = 1000;

    std::vector<int> send_buf(count);
    std::vector<int> recv_buf(count, 0);
    std::vector<int> expected_buf(count);

    for (size_t i = 0; i < count; i++) {
        send_buf.at(i) = i;
        expected_buf.at(i) = i * comm.size();
    }

    auto req = ccl::allreduce(send_buf.data(), recv_buf.data(), count, ccl::reduction::sum, comm);
    req.wait();

    for (size_t i = 0; i < count; i++) {
        if (recv_buf.at(i) != expected_buf.at(i)) {
            printf("FAILED (allreduce test)\n");
            throw std::runtime_error("recv_buf[" + std::to_string(i) +
                                     "]= " + std::to_string(recv_buf.at(i)) + ", expected " +
                                     std::to_string(expected_buf.at(i)));
        }
    }
}

void check_max_comm_number(const ccl::communicator& comm,
                           std::shared_ptr<ccl::kvs> kvs_instance,
                           int mpi_size,
                           int mpi_rank) {
    size_t user_comms = 0;
    std::vector<ccl::communicator> communicators;

    do {
        try {
            auto new_comm = ccl::create_communicator(mpi_size, mpi_rank, kvs_instance);
            ++user_comms;
            communicators.push_back(std::move(new_comm));
        }
        catch (...) {
            break;
        }
    } while (true);

    PRINT_BY_ROOT(comm, "created %zu communicators", user_comms);
    // PRINT_BY_ROOT(comm, "try to create one more communicator, it should fail");

    // try
    // {
    //     auto comm = ccl::environment::instance().create_communicator();
    //     printf("FAILED\n");
    //     throw std::runtime_error("extra communicator has been created");
    // }
    // catch(...)
    // {}

    // PRINT_BY_ROOT(comm, "free one comm, try to create again");
    // size_t comm_idx = user_comms / 2;

    // try
    // {
    //     communicators[comm_idx].reset();
    // }
    // catch (...)
    // {
    //     printf("FAILED\n");
    //     throw std::runtime_error("can't free communicator");
    // }

    // try
    // {
    //     communicators[comm_idx] = ccl::environment::instance().create_communicator();
    // }
    // catch (...)
    // {
    //     printf("FAILED\n");
    //     throw std::runtime_error("can't create communicator after free");
    // }
}

// void check_comm_create_identical_color()
// {
//     size_t comm_size{};
//     size_t comm_rank{};

//     PRINT_BY_ROOT(global_comm,
//         "create comm as a copy of the global one by settings identical colors");

//     ccl::comm_attr_t comm_attr = ccl::environment::instance().create_host_comm_attr();
//     comm_attr->set_value<ccl_host_color>(123);
//     auto comm = ccl::environment::instance().create_communicator(comm_attr);

//     comm_size = comm->size();
//     comm_rank = comm->rank();

//     if (comm_size != global_comm->size())
//     {
//         printf("FAILED\n");
//         throw std::runtime_error("mismatch in size, expected " +
//             to_string(global_comm->size()) +
//             " received " + to_string(comm_size));
//     }

//     if (comm_rank != global_comm->rank())
//     {
//         printf("FAILED\n");
//         throw std::runtime_error("mismatch in rank, expected " +
//             to_string(global_comm->rank()) +
//             " received " + to_string(comm_rank));
//     }

//     PRINT_BY_ROOT(global_comm,
//         "global comm: rank = %zu, size = %zu; "
//         "new comm: rank = %zu, size = %zu",
//         global_comm->rank(), global_comm->size(),
//         comm_rank, comm_size);

//     check_allreduce_on_comm(comm);
// }

bool isPowerOfTwo(unsigned int x) {
    return x && !(x & (x - 1));
}

void check_comm_split_by_color(ccl::communicator& comm, int mpi_size, int mpi_rank) {
    if (!isPowerOfTwo(comm.size())) {
        PRINT_BY_ROOT(
            comm,
            "split comm by color: number of processes should be a power of 2 for test purpose");
        return;
    }

    for (size_t split_by = 2; split_by <= comm.size(); split_by *= 2) {
        int color = comm.rank() % split_by;
        auto attr =
            ccl::create_comm_split_attr(ccl::attr_val<ccl::comm_split_attr_id::color>(color));
        auto new_comm = comm.split(attr);

        size_t comm_size = comm.size();
        size_t new_comm_size = new_comm.size();
        size_t comm_rank = comm.rank();
        size_t new_comm_rank = new_comm.rank();

        size_t expected_new_comm_size = comm_size / split_by;

        if (new_comm_size != expected_new_comm_size) {
            printf("FAILED (split)\n");

            throw std::runtime_error("mismatch in size, expected " +
                                     std::to_string(expected_new_comm_size) + " received " +
                                     std::to_string(new_comm_size));
        }

        PRINT_BY_ROOT(comm,
                      "base comm: rank = %zu, size = %zu; "
                      "new comm: rank = %zu, size = %zu",
                      comm_rank,
                      comm_size,
                      new_comm_rank,
                      new_comm_size);

        check_allreduce_on_comm(new_comm);
    }
}

int main() {

    ccl::init();

    int mpi_size, mpi_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (mpi_rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(mpi_size, mpi_rank, kvs);

    PRINT_BY_ROOT(comm, "\n- Basic communicator allreduce test");
    check_allreduce_on_comm(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    // PRINT_BY_ROOT(comm, "\n- Create max number of communicators");
    // check_max_comm_number(comm, kvs, mpi_size, mpi_rank);
    // PRINT_BY_ROOT(comm, "PASSED");

    PRINT_BY_ROOT(comm, "\n- Communicator split test");
    check_comm_split_by_color(comm, mpi_size, mpi_rank);
    PRINT_BY_ROOT(comm, "PASSED");

    // check_comm_create_identical_color();

    MPI_Finalize();

    return 0;
}
