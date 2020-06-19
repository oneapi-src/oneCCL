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
test_timeout=120

function run_test()
{
    test_case=$1
    for ((i = 0; i < $world_size ; i++ )); do
         timeout -k $((test_timeout))s $((test_timeout))s ./run_test.sh $test_case $world_size $i&
    done
    wait
}

run()
{
    world_size=4
    test_cases=("simple")
    test_cases+=("reinit")
#    test_cases+=("reconnect")
    PUR='\033[0;35m'
    RED='\033[0;31m'
    NC='\033[0m'
    for test_case in "${test_cases[@]}"; do
        echo -e "${PUR}Start $test_case test${NC}"
        exec_time=`date +%s`

        run_test $test_case

        exec_time="$((`date +%s`-$exec_time))"
        if [ "$exec_time" -ge "$test_timeout" ];
        then
            echo -e "${RED}FAIL: Timeout $test_case($exec_time > $test_timeout)${NC}"
        fi
    done
}

run
