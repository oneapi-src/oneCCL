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
#include <stdio.h>
#include <sstream>
#include <vector>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <limits>
#include <thread>
#include <numeric>

#include <mpi.h>

#include "base.hpp"
#include "base_utils.hpp"
#include "oneapi/ccl/ccl_gpu_modules.h"
#include "oneapi/ccl/native_device_api/export_api.hpp"

#define COUNT     512
#define COLL_ROOT (0)

#ifdef CCL_ENABLE_SYCL
template <class processing_type>
void user_thread_idx(size_t thread_idx,
                     const std::vector<std::pair<size_t, cl::sycl::device>>& devices,
                     cl::sycl::context ctx,
                     size_t total_devices_in_cluster,
                     std::shared_ptr<ccl::kvs_interface> kvs);
#else
template <class processing_type>
void user_thread_idx(size_t thread_idx,
                     ccl::device_indices_t thread_device_idx,
                     size_t total_devices_in_process);

#endif
int main(int argc, char** argv) {
    using namespace ::native;
    setenv("L0_CLUSTER_AFFINITY_MASK", "[0:0],[0:0]|[0:0],[0:0]", 0);
    const char* affinity_env_value = getenv("L0_CLUSTER_AFFINITY_MASK");

    if (argc == 2) {
        std::cout << "Not supported at now, only 'process_ring' built" << std::endl;
    }

    //Use:
    // SYCL_BE=PI_OTHER SYCL_PI_TRACE=1 ZE_DEBUG=1  SYCL_DEVICE_WHITE_LIST="" CCL_LOG_LEVEL=1 gdb examples/level_zero/l0_thread_allreduce_cpp_test

    // determine GPu device affinity
    /*
     * Affinity mask description:
     *
     *  "0,1,2|3,4,5"     (thread_0 has 0,1,2 devices; thread_1 has 3,4,5)
     *
     *  "0,1,2|3,4,5#6,7,8|9,10,11"    per host group separation
     */
    std::vector<std::thread> thread_group;
    std::vector<std::string> process_group_gpu_affinity;
    std::map<size_t, std::vector<std::string>> thread_group_gpu_affinity_per_process;

    using thread_device_indices_t = ccl::process_device_indices_t;
    std::map<size_t, thread_device_indices_t> node_device_indices;

    // extract GPU affinities by processes using '#' separator from L0_CLUSTER_AFFINITY_MASK
    utils::str_to_array<std::string>(affinity_env_value, process_group_gpu_affinity, '#');

    //get addresses from MPI
    int mpi_rank = 0, mpi_size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size != process_group_gpu_affinity.size()) {
        std::cerr << "L0_CLUSTER_AFFINITY_MASK is configured for processes: "
                  << process_group_gpu_affinity.size()
                  << ", but MPI requested world size: " << mpi_size << ". Both should to be equal"
                  << std::endl;
        return -1;
    }

    std::cout << "MPI process rank: " << mpi_rank << ", size: " << mpi_size << std::endl;

    // build CCL internal KVS
    auto& env = ccl::environment::instance();
    (void)env;
    std::shared_ptr<ccl::kvs> kvs_instance;
    ccl::kvs::address_type main_addr;
    if (mpi_rank == 0) {
        kvs_instance = ccl::environment::instance().create_main_kvs();
        main_addr = kvs_instance->get_address();

        std::cout << "Master KVS  hast build on addr: " /*<< main_addr*/ << std::endl;
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs_instance = ccl::environment::instance().create_kvs(main_addr);

        std::cout << "Slave KVS hast connected on addr: " /* << main_addr*/ << std::endl;
    }

    size_t total_device_in_cluster = 0;
    std::cout << "Expected process count: " << process_group_gpu_affinity.size() << std::endl;
    std::vector<size_t> total_devices_in_process(process_group_gpu_affinity.size(), 0);

#ifdef CCL_ENABLE_SYCL
    std::map<size_t, std::vector<cl::sycl::device>> sycl_devices_in_mpi_rank;
#endif

    size_t device_rank_for_mpi_rank_id_offset = 0;
    for (size_t process_index = 0; process_index < process_group_gpu_affinity.size();
         process_index++) {
        // extract  GPU affinities by thread inside process using '|' separator from L0_CLUSTER_AFFINITY_MASK
        utils::str_to_array<std::string>(process_group_gpu_affinity[process_index].c_str(),
                                         thread_group_gpu_affinity_per_process[process_index],
                                         '|');

        const std::vector<std::string>& thread_gpu_affinity =
            thread_group_gpu_affinity_per_process.find(process_index)->second;
        thread_device_indices_t thread_group_affinity;

        if (process_index == mpi_rank) {
            device_rank_for_mpi_rank_id_offset = total_device_in_cluster;
        }

        std::cout << "For process by id: " << process_index
                  << ", expected threads in process count: " << thread_gpu_affinity.size()
                  << std::endl;
        for (size_t thread_index = 0; thread_index < thread_gpu_affinity.size(); thread_index++) {
            ccl::device_indices_t device_group_affinity;
            utils::str_to_mset<ccl::device_index_type>(
                thread_gpu_affinity[thread_index].c_str(), device_group_affinity, ',');

            std::cout << " Extracted GPU indices for thread by id: " << thread_index
                      << ", devices in threads count: " << device_group_affinity.size()
                      << std::endl;
            total_device_in_cluster += device_group_affinity.size();
            total_devices_in_process[process_index] += device_group_affinity.size();

            thread_group_affinity[thread_index] = device_group_affinity;

#ifdef CCL_ENABLE_SYCL
            if (process_index == mpi_rank) {
                for (auto device_vendor_id : device_group_affinity) {
                    sycl_devices_in_mpi_rank[thread_index].push_back(
                        ccl::create_from_index(device_vendor_id).device);
                }
            }
#endif
        }

        node_device_indices[process_index] = thread_group_affinity;
    }

    // calculate total devices for cluster
    std::cout << "Devices in cluster count: " << total_device_in_cluster
              << ", for rank: " << mpi_rank << " devices count"
              << total_devices_in_process[mpi_rank] << ", thread count"
              << node_device_indices[mpi_rank].size() << std::endl;

    // Register algorithm from kernel source
    register_allreduce_gpu_module_source("kernels/ring_allreduce.spv",
                                         ccl::device_topology_type::ring);
    register_allreduce_gpu_module_source("kernels/a2a_allreduce.spv",
                                         ccl::device_topology_type::a2a);

    // launch user threads
#ifdef CCL_ENABLE_SYCL
    const auto& thread_group_affinity = sycl_devices_in_mpi_rank;
    std::vector<cl::sycl::device> devices_in_process;
    for (auto& thread_devices : sycl_devices_in_mpi_rank) {
        devices_in_process.insert(
            devices_in_process.end(), thread_devices.second.begin(), thread_devices.second.end());
    }
    //TODO: terminate called after throwing an instance of 'cl::sycl::invalid_parameter_error'
    //what():  Can't add devices across platforms to a single context. -33 (CL_INVALID_DEVICE)
    //auto ctx = cl::sycl::context(devices_in_process);
    auto ctx = cl::sycl::context(*devices_in_process.begin()); //use single device
#else
    const auto &thread_group_affinity = node_device_indices[mpi_rank];
    auto ctx = std::make_shared<::native::ccl_context>(); //TODO stub at moment
#endif
    for (auto thread_affinity_it = thread_group_affinity.begin();
         thread_affinity_it != thread_group_affinity.end();
         ++thread_affinity_it) {
#ifdef CCL_ENABLE_SYCL
        size_t thread_id;
        std::vector<cl::sycl::device> devices;
        std::tie(thread_id, devices) = *thread_affinity_it;

        std::vector<std::pair<size_t, cl::sycl::device>> ranked_devices;
        ranked_devices.reserve(devices.size());
        std::transform(devices.begin(),
                       devices.end(),
                       std::back_inserter(ranked_devices),
                       [&device_rank_for_mpi_rank_id_offset](const cl::sycl::device& dev) {
                           return std::make_pair(device_rank_for_mpi_rank_id_offset++, dev);
                       });

        std::cout << "Launch thread: " << thread_id
                  << " with expected local thread device communicators count: " << devices.size()
                  << std::endl;
        thread_group.emplace_back(&user_thread_idx<float>,
                                  thread_id,
                                  std::cref(ranked_devices),
                                  ctx,
                                  total_device_in_cluster,
                                  kvs_instance);
#else
        size_t thread_id;
        ccl::device_indices_t devices;
        std::tie(thread_id, devices) = *thread_affinity_it;

        std::cout << "Launch thread: " << thread_id
                  << " with expected local thread device communicators count: " << devices.size()
                  << std::endl;
        thread_group.emplace_back(&user_thread_idx<float>,
                                  thread_id,
                                  std::cref(devices),
                                  ctx,
                                  total_device_in_cluster,
                                  kvs_instance);
#endif
    }

    //wait finishing
    for (auto& t : thread_group) {
        t.join();
    }

    return 0;
}

#ifdef CCL_ENABLE_SYCL
template <class processing_type>
void user_thread_idx(size_t thread_idx,
                     const std::vector<std::pair<size_t, cl::sycl::device>>& devices,
                     cl::sycl::context ctx,
                     size_t total_devices_in_cluster,
                     std::shared_ptr<ccl::kvs_interface> kvs_instance) {
    using namespace ::native;

    // test data
    using allocated_memory_array = std::vector<processing_type*>;
    using rank_allocated_memory = std::map<size_t, allocated_memory_array>;
    //using native_queue_storage       = std::map<size_t, ccl_device::device_queue>;
    using stream_storage = std::map<size_t, ccl::stream>;

    rank_allocated_memory memory_storage;
    //native_queue_storage  queues;
    stream_storage streams;
    std::vector<processing_type> send_values(COUNT);
    std::iota(send_values.begin(), send_values.end(), 1);
    std::vector<processing_type> recv_values(COUNT, 0);

    // Create device communicators
    std::vector<ccl::device_communicator> comms =
        ccl::environment::instance().create_device_communicators(
            total_devices_in_cluster, devices, ctx, kvs_instance);

    std::cout << "Create device communicators, expected count: " << devices.size() << std::endl;

    // alloc memory specific to devices
    for (auto& comm : comms) {
        // get native l0* /
        ccl::device_communicator::ccl_device_t dev = comm.get_device();
        size_t rank = comm.rank();

        // create comm split attr
        auto device_spilt_attr = ccl::environment::instance().create_device_comm_split_attr();
        (void)device_spilt_attr;

        // create stream from device communicator directly
        streams.emplace(rank, comm.create_stream());
        const cl::sycl::queue& q =
            streams.find(rank)->second.get<ccl::stream_attr_id::native_handle>();

        // allocate memory
        processing_type* mem_send = static_cast<processing_type*>(
            cl::sycl::aligned_alloc_device(sizeof(processing_type), COUNT, q));
        processing_type* mem_recv = static_cast<processing_type*>(
            cl::sycl::aligned_alloc_device(sizeof(processing_type), COUNT, q));

        // set initial memory
        {
            static std::mutex memory_mutex;

            std::lock_guard<std::mutex> lock(memory_mutex);

            // TODO fill USM pointers
        }

        if (memory_storage[rank].empty()) {
            memory_storage[rank].reserve(100);
        }
        memory_storage[rank].push_back(std::move(mem_send));
        memory_storage[rank].push_back(std::move(mem_recv));
    }

    //allreduce
    std::vector<std::shared_ptr<ccl::request>> reqs;
    for (auto& comm : comms) {
        size_t rank = comm.rank();

        /*
        if (!comm.is_ready())
        {
            std::cerr << "Communicator by rank: " << rank << " should be ready already" << std::endl;
            abort();
        }
*/
        allocated_memory_array& mem_objects = memory_storage.find(rank)->second;

        // create operation attributes
        auto attr = ccl::environment::instance().create_operation_attr<ccl::allreduce_attr>();

        // invoke operation
        reqs.push_back(comm.allreduce(mem_objects[0],
                                      mem_objects[1],
                                      COUNT,
                                      ccl::reduction::sum,
                                      streams.find(rank)->second,
                                      attr));
    }

    //wait
    for (auto& req : reqs) {
        req->wait();
    }

//gpu_comm->barrier(stream);
//printout
#if 0
    static std::mutex printout_mutex;
    {
        std::unique_lock<std::mutex> lock(printout_mutex);
        for(auto &dev_it : memory_storage)
        {
            size_t rank = dev_it.first;
            const auto& handles = dev_it.second;
            std::cout << "rank : "  << rank << std::endl;
            for(const auto& mem : handles)
            {
                // TODO: check correctness
                std::cout << "\n\n" << std::endl;
            }
        }
    }
#endif
}
#else
template <class processing_type>
void user_thread_idx(size_t thread_idx,
                     ccl::device_indices_t thread_device_idx,
                     std::shared_ptr<::native::ccl_context> ctx,
                     size_t total_devices_in_cluster,
                     std::shared_ptr<ccl::kvs_interface> kvs) {
    using namespace ::native;

    // test data
    using allocated_memory_array = std::vector<ccl_device::device_memory<processing_type>>;
    using rank_allocated_memory = std::map<size_t, allocated_memory_array>;
    using native_queue_storage = std::map<size_t, ccl_device::device_queue>;
    using stream_storage = std::map<size_t, ccl::stream>;

    rank_allocated_memory memory_storage;
    native_queue_storage queues;
    stream_storage streams;
    std::vector<processing_type> send_values(COUNT);
    std::iota(send_values.begin(), send_values.end(), 1);
    std::vector<processing_type> recv_values(COUNT, 0);

    // Create device communicators
    std::vector<ccl::device_communicator> comms =
        ccl::environment::instance().create_device_communicators(
            total_devices_in_cluster, thread_device_idx, ctx, kvs);

    std::cout << "Create device communicators, expected count: " << thread_device_idx.size()
              << std::endl;

    // alloc memory specific to devices
    for (auto &comm : comms) {
        // get native l0* /
        ccl::device_communicator::ccl_device_t dev = comm.get_device();
        size_t rank = comm.rank();

        // wrapped L0-native API for devices: create native buffers
        auto mem_send = dev->alloc_memory<processing_type>(COUNT, sizeof(processing_type));
        auto mem_recv = dev->alloc_memory<processing_type>(COUNT, sizeof(processing_type));

        // set initial memory
        {
            static std::mutex memory_mutex;

            std::lock_guard<std::mutex> lock(memory_mutex);

            // wrapped L0-native API for memory: fill device buffers
            mem_send.enqueue_write_sync(send_values);
            mem_recv.enqueue_write_sync(recv_values);
        }

        if (memory_storage[rank].empty()) {
            memory_storage[rank].reserve(100);
        }
        memory_storage[rank].push_back(std::move(mem_send));
        memory_storage[rank].push_back(std::move(mem_recv));

        // create native stream
        enum { INSERTED_ITER, RESULT };
        auto queue_it = std::get<INSERTED_ITER>(queues.emplace(rank, dev->create_cmd_queue()));
        streams.emplace(rank, ccl::environment::instance().create_stream(queue_it->second.get()));
    }

    //allreduce
    std::vector<std::shared_ptr<ccl::request>> reqs;
    for (auto &comm : comms) {
        size_t rank = comm.rank();
        /*
        if (!comm.is_ready())
        {
            std::cerr << "Communicator by rank: " << rank << " should be ready already" << std::endl;
            abort();
        }
*/
        allocated_memory_array &mem_objects = memory_storage.find(rank)->second;
        reqs.push_back(comm.allreduce(mem_objects[0].get(),
                                      mem_objects[1].get(),
                                      mem_objects[1].count(),
                                      ccl::reduction::sum,
                                      streams[rank]));
    }

    //wait
    for (auto &req : reqs) {
        req->wait();
    }

    //gpu_comm->barrier(stream);
    //printout
    static std::mutex printout_mutex;
    {
        std::unique_lock<std::mutex> lock(printout_mutex);
        for (auto &dev_it : memory_storage) {
            size_t rank = dev_it.first;
            const auto &handles = dev_it.second;
            std::cout << "rank : " << rank << std::endl;
            for (const auto &mem : handles) {
                std::vector<processing_type> tmp = mem.enqueue_read_sync();
                std::copy(
                    tmp.begin(), tmp.end(), std::ostream_iterator<processing_type>(std::cout, ","));
                std::cout << "\n\n" << std::endl;
            }
        }
    }
}
#endif
