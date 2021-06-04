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
#include "coll/coll_param.hpp"

bool operator==(const coll_param_gpu& lhs, const coll_param_gpu& rhs) {
    CCL_ASSERT((lhs.is_reduction() && rhs.is_reduction()) ||
               (!lhs.is_reduction() && !rhs.is_reduction()));

    bool res =
        lhs.get_coll_type() == rhs.get_coll_type() && lhs.get_datatype() == rhs.get_datatype();

    if (lhs.is_reduction()) {
        res = res && (lhs.get_reduction() == rhs.get_reduction());
    }

    return res;
}

void copy_deps(const std::vector<ccl::event>& in, std::vector<ccl::event>& out) {
#ifdef CCL_ENABLE_SYCL
    out.clear();
    for (size_t idx = 0; idx < in.size(); idx++) {
        try {
            auto sycl_event = in[idx].get_native();
            out.push_back(ccl::create_event(sycl_event));
        }
        catch (ccl::exception&) {
        }
    }
#else /* CCL_ENABLE_SYCL */
    CCL_THROW_IF_NOT(in.size() == 0, "host deps are not supported yet");
#endif /* CCL_ENABLE_SYCL */
}

ccl_coll_param::ccl_coll_param(const ccl_coll_param& other) {
    ctype = other.ctype;
    send_buf = other.send_buf;
    recv_buf = other.recv_buf;
    count = other.count;
    send_count = other.send_count;
    send_counts = other.send_counts;
    recv_counts = other.recv_counts;
    dtype = other.dtype;
    reduction = other.reduction;
    root = other.root;
    stream = other.stream;
    copy_deps(other.deps, deps);
    comm = other.comm;
    sparse_param = other.sparse_param;

#ifdef CCL_ENABLE_SYCL
    device_send_buf = other.device_send_buf;
    device_recv_buf = other.device_recv_buf;
#endif /* CCL_ENABLE_SYCL */
}
