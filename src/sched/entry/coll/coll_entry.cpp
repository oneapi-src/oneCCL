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
#include "sched/entry/coll/coll_entry.hpp"

ccl::status coll_entry::build_sched(ccl_sched* sched, const ccl_coll_entry_param& param) {
    ccl::status res = ccl::status::success;

    sched->hint_algo = param.hint_algo;

    switch (param.ctype) {
        case ccl_coll_allgatherv: {
            res = ccl_coll_build_allgatherv(sched,
                                            param.send_buf,
                                            param.send_count,
                                            param.recv_buf,
                                            param.recv_counts,
                                            param.dtype,
                                            param.comm);
            break;
        }
        case ccl_coll_allreduce: {
            res = ccl_coll_build_allreduce(sched,
                                           param.send_buf,
                                           param.recv_buf,
                                           param.count,
                                           param.dtype,
                                           param.reduction,
                                           param.comm,
                                           param.is_scaleout);
            break;
        }
        case ccl_coll_alltoall: {
            res = ccl_coll_build_alltoall(
                sched, param.send_buf, param.recv_buf, param.count, param.dtype, param.comm);
            break;
        }
        case ccl_coll_alltoallv: {
            res = ccl_coll_build_alltoallv(sched,
                                           param.send_buf,
                                           param.send_counts,
                                           param.recv_buf,
                                           param.recv_counts,
                                           param.dtype,
                                           param.comm);
            break;
        }
        case ccl_coll_barrier: {
            res = ccl_coll_build_barrier(sched, param.comm);
            break;
        }
        case ccl_coll_bcast: {
            res = ccl_coll_build_bcast(
                sched, param.recv_buf, param.count, param.dtype, param.root, param.comm);
            break;
        }
        case ccl_coll_reduce: {
            res = ccl_coll_build_reduce(sched,
                                        param.send_buf,
                                        param.recv_buf,
                                        param.count,
                                        param.dtype,
                                        param.reduction,
                                        param.root,
                                        param.comm);
            break;
        }
        case ccl_coll_reduce_scatter: {
            res = ccl_coll_build_reduce_scatter(sched,
                                                param.send_buf,
                                                param.recv_buf,
                                                param.count,
                                                param.dtype,
                                                param.reduction,
                                                param.comm);
            break;
        }
        default: CCL_FATAL("not supported coll_type ", param.ctype); break;
    }
    return res;
}

void coll_entry::start() {
    if (update_fields()) {
        LOG_DEBUG("updated fields in COLL entry: ", this);
        subsched.reset();
    }

    if (!subsched) {
        ccl_coll_param coll_param{};
        coll_param.ctype = sched->coll_param.ctype;
        coll_param.comm = sched->coll_param.comm;
        coll_param.stream = sched->coll_param.stream;

        LOG_DEBUG("building COLL entry: ",
                  this,
                  ", subsched: ",
                  subsched.get(),
                  ", coll: ",
                  ccl_coll_type_to_str(param.ctype),
                  ", count: ",
                  param.count);
        subsched_entry::build_subsched({ sched->sched_id, coll_param });
        LOG_DEBUG("built COLL entry: ",
                  this,
                  ", subsched: ",
                  subsched.get(),
                  ", coll: ",
                  ccl_coll_type_to_str(param.ctype),
                  ", count: ",
                  param.count);
    }

    LOG_DEBUG("starting COLL entry: sched_id ",
              sched->sched_id,
              ", this: ",
              this,
              ", subsched: ",
              subsched.get());
    subsched_entry::start();
    LOG_DEBUG("started COLL entry: sched_id ",
              sched->sched_id,
              ", this: ",
              this,
              ", subsched: ",
              subsched.get());
}
