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
#include "common/global/global.hpp"
#include "comp/comp.hpp"
#include "sched/entry/entry.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
#include "sched/entry/gpu/ze_base_entry.hpp"
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
class reduce_local_entry : public ze_base_entry {
#else
class reduce_local_entry : public sched_entry {
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
public:
    static constexpr const char* class_name() noexcept {
        return "REDUCE_LOCAL";
    }

    reduce_local_entry() = delete;
    reduce_local_entry(ccl_sched* sched,
                       const ccl_buffer in_buf,
                       size_t in_cnt,
                       ccl_buffer inout_buf,
                       size_t* out_cnt,
                       const ccl_datatype& dtype,
                       ccl::reduction reduction_op)
            :
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
              ze_base_entry(sched),
#else // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
              sched_entry(sched),
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
              in_buf(in_buf),
              in_cnt(in_cnt),
              inout_buf(inout_buf),
              out_cnt(out_cnt),
              dtype(dtype),
              op(reduction_op),
              fn(sched->coll_attr.reduction_fn),
              use_device(false) {
        CCL_THROW_IF_NOT(op != ccl::reduction::custom || fn,
                         "custom reduction requires user provided callback");
    }

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
    ~reduce_local_entry() override {
        finalize();
    }
    void init();
    void finalize();
    void update() override;
    void check_use_device();
    void start_on_device();
#else // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
    void check_use_device() {}
    void start_on_device() {}
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
    void start_on_host() {
        size_t bytes = in_cnt * dtype.size();
        size_t offset = inout_buf.get_offset();
        const ccl::fn_context context = { sched->coll_attr.match_id.c_str(), offset };
        ccl::status comp_status = ccl_comp_reduce(sched,
                                                  in_buf.get_ptr(bytes),
                                                  in_cnt,
                                                  inout_buf.get_ptr(bytes),
                                                  out_cnt,
                                                  dtype,
                                                  op,
                                                  fn,
                                                  &context);
        CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);

        status = ccl_sched_entry_status_complete;
    }

    void start() override {
        check_use_device();
        if (use_device) {
            LOG_DEBUG("start on device");
            start_on_device();
        }
        else {
            LOG_DEBUG("start on host");
            start_on_host();
        }
    }

    const char* name() const noexcept override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", in_buf ",
                           in_buf,
                           ", in_cnt ",
                           in_cnt,
                           ", inout_buf ",
                           inout_buf,
                           ", out_cnt ",
                           out_cnt,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", red_fn ",
                           fn,
                           "\n");
    }

private:
    ccl_buffer in_buf;
    size_t in_cnt;
    ccl_buffer inout_buf;
    size_t* out_cnt;
    ccl_datatype dtype;
    ccl::reduction op;
    ccl::reduction_fn fn;
    void* in_buf_ptr;
    void* inout_buf_ptr;

    bool use_device;

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
    ze_module_handle_t module;
    ze_kernel_handle_t kernel;
    std::string kernel_name;
    ze_group_count_t group_count;
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
};
