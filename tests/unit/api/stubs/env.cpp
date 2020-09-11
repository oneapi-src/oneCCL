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
#include <iterator>
#include <sstream>
#include <unistd.h>

#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace ccl {

env_data::env_data() : log_level(static_cast<int>(ccl_log_level::DEBUG)) {}

} /* namespace ccl */
