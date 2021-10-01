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

#include <vector>

#include "coll/algorithms/algorithms_enum.hpp"
#include "common/datatype/datatype.hpp"
#include "oneapi/ccl.hpp"

class ccl_comm;

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>

template <class native_type>
using ccl_sycl_typed_buffer_t = cl::sycl::buffer<native_type, 1>;

/* ordering should be aligned with ccl::datatype */
using ccl_sycl_buffer_one_dim_types = std::tuple<ccl_sycl_typed_buffer_t<int8_t>,
                                                 ccl_sycl_typed_buffer_t<uint8_t>,
                                                 ccl_sycl_typed_buffer_t<int16_t>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>,
                                                 ccl_sycl_typed_buffer_t<int32_t>,
                                                 ccl_sycl_typed_buffer_t<uint32_t>,
                                                 ccl_sycl_typed_buffer_t<int64_t>,
                                                 ccl_sycl_typed_buffer_t<uint64_t>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>,
                                                 ccl_sycl_typed_buffer_t<float>,
                                                 ccl_sycl_typed_buffer_t<double>,
                                                 ccl_sycl_typed_buffer_t<uint16_t>>;
#endif // CCL_ENABLE_SYCL

#define CCL_INVALID_PROC_IDX (-1)

struct ccl_coll_attr {
    ccl_coll_attr() = default;
    ccl_coll_attr(const ccl_coll_attr&) = default;
    ccl_coll_attr& operator=(const ccl_coll_attr&) = default;

    ccl_coll_attr(const ccl::allgatherv_attr& attr);
    ccl_coll_attr(const ccl::allreduce_attr& attr);
    ccl_coll_attr(const ccl::alltoall_attr& attr);
    ccl_coll_attr(const ccl::alltoallv_attr& attr);
    ccl_coll_attr(const ccl::barrier_attr& attr);
    ccl_coll_attr(const ccl::broadcast_attr& attr);
    ccl_coll_attr(const ccl::reduce_attr& attr);
    ccl_coll_attr(const ccl::reduce_scatter_attr& attr);
    ccl_coll_attr(const ccl::sparse_allreduce_attr& attr);

    ccl_coll_attr(ccl_coll_attr&&) = default;
    ccl_coll_attr& operator=(ccl_coll_attr&&) = default;

    std::string to_string() const;

    ccl::prologue_fn prologue_fn = nullptr;
    ccl::epilogue_fn epilogue_fn = nullptr;
    ccl::reduction_fn reduction_fn = nullptr;

    size_t priority = 0;
    int synchronous = 0;
    int to_cache = 0;
    std::string match_id{};

    /* change how user-supplied buffers have to be interpreted */
    int is_vector_buf = 0;

#ifdef CCL_ENABLE_SYCL
    int is_sycl_buf = 0;
#endif // CCL_ENABLE_SYCL

    ccl::sparse_allreduce_completion_fn sparse_allreduce_completion_fn = nullptr;
    ccl::sparse_allreduce_alloc_fn sparse_allreduce_alloc_fn = nullptr;
    const void* sparse_allreduce_fn_ctx = nullptr;
    ccl::sparse_coalesce_mode sparse_coalesce_mode = ccl::sparse_coalesce_mode::regular;
};

struct ccl_coll_sparse_param {
    const void* send_ind_buf;
    size_t send_ind_count;
    const void* send_val_buf;
    size_t send_val_count;
    void* recv_ind_buf;
    size_t recv_ind_count;
    void* recv_val_buf;
    size_t recv_val_count;
    ccl_datatype itype;
};

struct ccl_coll_param {
    enum class buf_type { regular, device };

    ccl_coll_type ctype;

    std::vector<void*> send_bufs;
    std::vector<void*> recv_bufs;

    /*
        filled if pre-post copy is used
        to keep original send/recv buffers
        send_buf and recv_buf fields are replaced by staging buffers
    */
    std::vector<void*> device_send_bufs;
    std::vector<void*> device_recv_bufs;

    std::vector<size_t> send_counts;
    std::vector<size_t> recv_counts;

    ccl_datatype dtype;
    ccl::reduction reduction;
    int root;
    ccl_stream* stream;
    ccl_comm* comm;
    std::vector<ccl::event> deps;

    ccl_coll_sparse_param sparse_param;

    ccl_coll_param();
    ccl_coll_param(const ccl_coll_param& other);

    std::string to_string() const;

    void* get_send_buf(size_t idx = 0, buf_type type = buf_type::regular) const;
    void* get_recv_buf(size_t idx = 0, buf_type type = buf_type::regular) const;

    void* get_send_buf_ptr(size_t idx = 0, buf_type type = buf_type::regular) const;
    void* get_recv_buf_ptr(size_t idx = 0, buf_type type = buf_type::regular) const;

    size_t get_send_count(size_t idx = 0) const;
    size_t get_recv_count(size_t idx = 0) const;

    bool is_inplace(buf_type type = buf_type::regular) const;

    std::vector<void*> get_all_non_zero_bufs() const;

    void validate() const;

    void copy_deps(const std::vector<ccl::event>& d, ccl::event* extra = nullptr);
    void set_common_fields(ccl::datatype dtype,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps);
    void sync_deps(const ccl_stream* s, const std::vector<ccl::event>& ds);

    static ccl_coll_param create_allgatherv_param(const void* send_buf,
                                                  size_t send_count,
                                                  void* recv_buf,
                                                  const size_t* recv_counts,
                                                  ccl::datatype dtype,
                                                  const ccl_coll_attr& attr,
                                                  ccl_comm* comm,
                                                  const ccl_stream* stream,
                                                  const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_allreduce_param(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_alltoall_param(const void* send_buf,
                                                void* recv_buf,
                                                size_t count,
                                                ccl::datatype dtype,
                                                const ccl_coll_attr& attr,
                                                ccl_comm* comm,
                                                const ccl_stream* stream,
                                                const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_alltoallv_param(const void* send_buf,
                                                 const size_t* send_counts,
                                                 void* recv_buf,
                                                 const size_t* recv_counts,
                                                 ccl::datatype dtype,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_barrier_param(ccl_comm* comm,
                                               const ccl_stream* stream,
                                               const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_broadcast_param(void* buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 int root,
                                                 const ccl_coll_attr& attr,
                                                 ccl_comm* comm,
                                                 const ccl_stream* stream,
                                                 const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_reduce_param(const void* send_buf,
                                              void* recv_buf,
                                              size_t count,
                                              ccl::datatype dtype,
                                              ccl::reduction reduction,
                                              int root,
                                              const ccl_coll_attr& attr,
                                              ccl_comm* comm,
                                              const ccl_stream* stream,
                                              const std::vector<ccl::event>& deps = {});

    static ccl_coll_param create_reduce_scatter_param(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t recv_count,
                                                      ccl::datatype dtype,
                                                      ccl::reduction reduction,
                                                      const ccl_coll_attr& attr,
                                                      ccl_comm* comm,
                                                      const ccl_stream* stream,
                                                      const std::vector<ccl::event>& deps = {});
};

class coll_param_gpu {
    ccl_coll_type ctype;
    ccl::datatype dtype;
    ccl::reduction red;

public:
    coll_param_gpu(ccl_coll_type ctype, ccl::datatype dtype, ccl::reduction red)
            : ctype{ ctype },
              dtype{ dtype },
              red{ red } {}

    coll_param_gpu(ccl_coll_type ctype, ccl::datatype dtype)
            : ctype{ ctype },
              dtype{ dtype },
              red{ (ccl::reduction)-1 } {
        assert(!is_reduction() && "This constructor is invalid for reduction types");
    }

    ccl_coll_type get_coll_type() const {
        return ctype;
    }

    ccl::datatype get_datatype() const {
        return dtype;
    }

    bool is_reduction() const {
        return ccl_coll_type_is_reduction(get_coll_type());
    }

    ccl::reduction get_reduction() const {
        if (!is_reduction()) {
            throw ccl::exception(
                "get_ruduction(): is not supported for non-reduction collective type, i.e. bcast");
        }
        return red;
    }
};

bool operator==(const coll_param_gpu& lhs, const coll_param_gpu& rhs);
