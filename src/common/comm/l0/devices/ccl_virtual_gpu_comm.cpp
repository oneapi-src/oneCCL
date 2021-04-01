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
#include "common/comm/l0/devices/ccl_virtual_gpu_comm.hpp"
#include "common/comm/l0/modules/specific_modules_source_data.hpp"

namespace native {
ccl_virtual_gpu_comm::ccl_virtual_gpu_comm(ccl_device& device,
                                           comm_rank_t idx,
                                           ccl_gpu_comm& real_gpu)
        : base(device, idx),
          real_gpu_comm(real_gpu) {
    //TODO increase reference count
    real_gpu.register_virtual_gpu(this);

    //compile and load modules from all sources
    load_modules(specific_modules_source_data_storage::instance());
}

std::string ccl_virtual_gpu_comm::to_string_impl() const {
    std::string ret(name_impl());
    ret = ret + ", comm:\n" + comm_to_str();
    return ret;
}

cmd_list_proxy ccl_virtual_gpu_comm::get_cmd_list(std::shared_ptr<ccl_context> ctx,
                                                  const ze_command_list_desc_t& properties) {
    return real_gpu_comm.get_cmd_list(ctx, properties);
}

fence_proxy ccl_virtual_gpu_comm::get_fence(const ccl_device::device_queue& cmd_queue,
                                            std::shared_ptr<ccl_context> ctx) {
    return real_gpu_comm.get_fence(cmd_queue, ctx);
}

} // namespace native
