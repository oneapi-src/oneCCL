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
#include <mpi.h>
#include "base.hpp"
#include "transport.hpp"

transport_settings::transport_settings() {
    init_by_mpi();
}

transport_settings::~transport_settings() {
    deinit_by_mpi();
}

transport_settings &transport_settings::instance() {
    static transport_settings inst;
    return inst;
}

int transport_settings::get_rank() const noexcept {
    return rank;
}

int transport_settings::get_size() const noexcept {
    return size;
}

ccl::shared_ptr_class<ccl::kvs> transport_settings::get_kvs() {
    return kvs;
}

void transport_settings::init_by_mpi() {
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* create CCL internal KVS */
    auto &env = ccl::environment::instance();
    (void)env;
    ccl::shared_ptr_class<ccl::kvs> kvs_candidate;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs_candidate = ccl::environment::instance().create_main_kvs();
        main_addr = kvs_candidate->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs_candidate = ccl::environment::instance().create_kvs(main_addr);
    }
    kvs = kvs_candidate;
}

void transport_settings::deinit_by_mpi() {
    MPI_Finalize();
}
