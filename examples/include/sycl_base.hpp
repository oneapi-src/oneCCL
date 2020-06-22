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
#ifndef SYCL_BASE_HPP
#define SYCL_BASE_HPP

/* sycl-specific base implementation and its help functions */
#include <iostream>
#include <string>

#include <CL/sycl.hpp>
#include "ccl.hpp"

#define COUNT     (10 * 1024 * 1024)
#define COLL_ROOT (0)

using namespace std;
using namespace cl::sycl;
using namespace cl::sycl::access;

/* help functions for sycl-specific base implementation */
inline bool has_gpu()
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    for (const auto& device : devices)
    {
        if (device.is_gpu())
        {
            return true;
        }
    }
    return false;
}

inline bool has_accelerator()
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    for (const auto& device : devices)
    {
        if (device.is_accelerator())
        {
            return true;
        }
    }
    return false;
}

inline int create_sycl_queue(int argc, char **argv, cl::sycl::queue &queue, ccl_stream_type_t& stream_type)
{
    stream_type = ccl_stream_cpu;
    auto exception_handler = [&](cl::sycl::exception_list elist) {
      for (std::exception_ptr const& e : elist) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
          std::cout << "failure" << std::endl;
          std::terminate();
        }
      }
    };

    std::unique_ptr<cl::sycl::device_selector> selector;
    if (argc == 2)
    {
        if (strcmp(argv[1], "cpu") == 0)
        {
            selector.reset(new cl::sycl::cpu_selector());
            stream_type = ccl_stream_cpu;
        }
        else if (strcmp(argv[1], "gpu") == 0)
        {
            stream_type = ccl_stream_gpu;
            if (has_gpu()) 
            {
                selector.reset(new cl::sycl::gpu_selector());
            }
            else if (has_accelerator()) 
            {
                selector.reset(new cl::sycl::host_selector());
                std::cout << "Accelerator is the first in device list, but unavailable for multiprocessing, host_selector has been created instead of default_selector." << std::endl;
            }
            else
            {
                selector.reset(new cl::sycl::default_selector());
                std::cout << "GPU is unavailable, default_selector has been created instead of gpu_selector." << std::endl;
            }
        }
        else if (strcmp(argv[1], "host") == 0)
        {
            stream_type = ccl_stream_cpu;
            selector.reset(new cl::sycl::host_selector());
        }
        else if (strcmp(argv[1], "default") == 0)
        {
            stream_type = ccl_stream_cpu;
            if (!has_accelerator())
            {
                selector.reset(new cl::sycl::default_selector());
            }
            else
            {
                selector.reset(new cl::sycl::host_selector());
                std::cout << "Accelerator is the first in device list, but unavailable for multiprocessing, host_selector has been created instead of default_selector." << std::endl;
            }
        }
        else
        {
            std::cerr << "Please provide device type: cpu | gpu | host | default " << std::endl;
            return -1;
        }
        queue = cl::sycl::queue(*selector, exception_handler);
        std::cout << "Provided device type " << argv[1] << "\nRunning on "
                  << queue.get_device().get_info<cl::sycl::info::device::name>()
                  << "\n";
    }
    else
    {
        std::cerr << "Please provide device type: cpu | gpu | host | default " << std::endl;
        return -1;
    }
    return 0;
}

void handle_exception(cl::sycl::queue &q)
{
    try
    {
        q.wait_and_throw();
    }
    catch (cl::sycl::exception const& e)
    {
        std::cout << "Caught synchronous SYCL exception:\n"
          << e.what() << std::endl;
    }
}

#endif /* SYCL_BASE_HPP */
