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
#include <sstream>

#include "common/comm/l0/context/scale/scale_out/scale_out_session.hpp"
#include "common/log/log.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

namespace native {
namespace observer {

std::string scale_out_session_iface::to_string() const {
    std::stringstream ss;
    ss << "sess: " << reinterpret_cast<const void*>(this);
    return ss.str();
}

size_t scale_out_session_iface::get_send_tag() const {
    return send_tag;
}

void ccl_worker_adapter::submit_coll_work(std::shared_ptr<ccl::host_communicator>& comm,
                                          const session_notification& in,
                                          session_notification_handle& out,
                                          const coll_param_gpu& kernel_params) {
    // allreduce
    if (kernel_params.get_coll_type() == ccl_coll_allreduce) {
        out.output_buffer.resize(in.src_size_bytes);
        ccl::stream::impl_value_t empty_stream{};

        // notice: not thread-safe
        out.op_handle = comm->allreduce_impl(in.host_src_ptr,
                                             out.output_buffer.data(),
                                             in.src_size_bytes,
                                             kernel_params.get_datatype(),
                                             kernel_params.get_reduction(),
                                             empty_stream,
                                             ccl::default_allreduce_attr,
                                             {});
        out.op_handle_ready = true;
    }
}

} // namespace observer
} // namespace native
