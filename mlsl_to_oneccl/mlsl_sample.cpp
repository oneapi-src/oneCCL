/*
 Copyright 2016-2019 Intel Corporation

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

#include <iostream>
#include <stdio.h>

#include "mlsl.hpp"

using namespace MLSL;

#define COUNT 128

int main(int argc, char** argv)
{
    int i, size, rank;

    auto sendbuf = new float[COUNT];
    auto recvbuf = new float[COUNT];

    Environment::GetEnv().Init(&argc, &argv);
    rank = Environment::GetEnv().GetProcessIdx();
    size = Environment::GetEnv().GetProcessCount();
    auto dist = Environment::GetEnv().CreateDistribution(size, 1);

    /* initialize sendbuf */
    for (i = 0; i < COUNT; i++)
        sendbuf[i] = rank;

    /* invoke allreduce */
    auto req = dist->AllReduce(sendbuf, recvbuf, COUNT,
                               DT_FLOAT, RT_SUM, GT_GLOBAL);
    Environment::GetEnv().Wait(req);

    /* check correctness of recvbuf */
    float expected = (size - 1) * ((float)size / 2);
    for (i = 0; i < COUNT; i++)
    {
        if (recvbuf[i] != expected)
        {
            std::cout << "idx " << i
                      << ": got " << recvbuf[i]
                      << " but expected " << expected
                      << std::endl;
            break;
        }
    }

    if (i == COUNT && rank == 0)
        std::cout << "PASSED" << std::endl;

    Environment::GetEnv().DeleteDistribution(dist);
    Environment::GetEnv().Finalize();

    delete[] sendbuf;
    delete[] recvbuf;

    return 0;
}

