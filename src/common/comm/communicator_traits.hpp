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

namespace ccl
{
struct empty {};
struct host_sign {};
struct cpu_sign {};
struct gpu_sign {};
struct accel_sign {};

template<class host, class cpu, class gpu, class accel>
struct base_communicator_traits
{
    static constexpr bool is_host()
    {
        return std::is_same<host, host_sign>::value;
    }

    static constexpr bool is_cpu()
    {
        return std::is_same<cpu, cpu_sign>::value;
    }

    static constexpr bool is_gpu()
    {
        return std::is_same<gpu, gpu_sign>::value;
    }

    static constexpr bool is_accelerator()
    {
        return std::is_same<accel, accel_sign>::value;
    }
};

struct host_communicator_traits : base_communicator_traits<host_sign, empty, empty, empty>
{
    static constexpr const char* name()
    {
        return "host communicator";
    }
};

struct cpu_communicator_traits : base_communicator_traits<empty, cpu_sign, empty, empty>
{
    static constexpr const char* name()
    {
        return "cpu communicator";
    }
};

struct gpu_communicator_traits : base_communicator_traits<empty, empty, gpu_sign, empty>
{
    static constexpr const char* name()
    {
        return "gpu communicator";
    }
};
}
