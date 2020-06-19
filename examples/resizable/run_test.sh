#!/bin/bash
#
# Copyright 2016-2020 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
ccl_extra_env="CCL_ATL_TRANSPORT=ofi CCL_KVS_IP_EXCHANGE=env CCL_KVS_IP_PORT=`hostname -i`_1244 CCL_PM_TYPE=resizable CCL_DEFAULT_RESIZABLE=1"

function run_test()
{
    test_case=$1
    world_size=$2
    logname="$test_case""_$3"
    echo $logname
    eval `echo $ccl_extra_env "CCL_WORLD_SIZE=$world_size" ./resizable.out $test_case;`  &> $logname.out
    if [[ "1" -eq "$?" ]] && [[ "$test_case" == "reconnect" ]];
    then
        eval `echo $ccl_extra_env "CCL_WORLD_SIZE=$world_size" ./resizable.out $test_case;`
    fi
}

run_test $1 $2 $3
