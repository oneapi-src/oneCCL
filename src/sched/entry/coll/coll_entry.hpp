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

#pragma once

#include "sched/entry/entry.hpp"
#include "common/request/request.hpp"
#include "sched/extra_sched.hpp"

class coll_entry : public sched_entry,
                   public postponed_fields<coll_entry,
                                           ccl_sched_entry_field_buf,
                                           ccl_sched_entry_field_send_buf,
                                           ccl_sched_entry_field_recv_buf,
                                           ccl_sched_entry_field_cnt,
                                           ccl_sched_entry_field_dtype>
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "COLL";
    }

    coll_entry() = delete;
    coll_entry(ccl_sched* sched,
               ccl_coll_entry_param& param)
        : sched_entry(sched), param(param), coll_sched()
    {
    }

    void start() override
    {
        update_fields();

        create_schedule();
        status = ccl_sched_entry_status_started;
    }

    void update() override
    {
        CCL_THROW_IF_NOT(coll_sched, "empty request");
        if (coll_sched->is_completed())
        {
            LOG_DEBUG("COLL entry, completed sched: ", coll_sched.get());
            coll_sched.reset();
            status = ccl_sched_entry_status_complete;
        }
    }

    bool is_strict_order_satisfied() override
    {
        return is_completed();
    }

    const char* name() const override
    {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_buf> id)
    {
        return param.recv_buf;
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_send_buf> id)
    {
        return param.send_buf;
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_recv_buf> id)
    {
        return param.recv_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id)
    {
        return param.count;
    }

    ccl_datatype_internal_t& get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id)
    {
        return param.dtype;
    }
protected:
    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                            "dt ", ccl_datatype_get_name(param.dtype),
                            ", coll_type ", ccl_coll_type_to_str(param.ctype),
                            ", send_buf ", param.send_buf,
                            ", recv_buf ", param.recv_buf,
                            ", cnt ", param.count,
                            ", op ", ccl_reduction_to_str(param.reduction),
                            ", comm ", sched->coll_param.comm,
                            ", coll sched ", coll_sched.get(),
                            "\n");
    }

private:

    void create_schedule()
    {
        size_t bytes = param.count * ccl_datatype_get_size(param.dtype);
        switch (param.ctype)
        {
            case ccl_coll_barrier:
                break;
            case ccl_coll_bcast:
            {
                ccl_coll_param coll_param{};
                coll_param.ctype = ccl_coll_bcast;
                coll_param.buf = param.recv_buf.get_ptr(bytes);
                coll_param.count = param.count;
                coll_param.dtype = param.dtype;
                coll_param.root = param.root;
                coll_param.comm = sched->coll_param.comm;
                coll_sched.reset(new ccl_extra_sched(coll_param, sched->sched_id));

                auto result = ccl_coll_build_bcast(coll_sched.get(),
                                                   param.recv_buf,
                                                   coll_sched->coll_param.count,
                                                   coll_sched->coll_param.dtype,
                                                   coll_sched->coll_param.root);

                CCL_ASSERT(result == ccl_status_success, "bad result ", result);

                break;
            }
            case ccl_coll_reduce:
                break;
            case ccl_coll_allreduce:
            {
                ccl_coll_param coll_param{};
                coll_param.ctype = ccl_coll_allreduce;
                coll_param.send_buf = param.send_buf.get_ptr(bytes);
                coll_param.recv_buf = param.recv_buf.get_ptr(bytes);
                coll_param.count = param.count;
                coll_param.dtype = param.dtype;
                coll_param.reduction = param.reduction;
                coll_param.comm = sched->coll_param.comm;
                coll_sched.reset(new ccl_extra_sched(coll_param, sched->sched_id));
                coll_sched->coll_attr.reduction_fn = sched->coll_attr.reduction_fn;
                coll_sched->coll_attr.match_id = sched->coll_attr.match_id;

                auto result = ccl_coll_build_allreduce(coll_sched.get(),
                                                       param.send_buf,
                                                       param.recv_buf,
                                                       coll_sched->coll_param.count,
                                                       coll_sched->coll_param.dtype,
                                                       coll_sched->coll_param.reduction);

                CCL_ASSERT(result == ccl_status_success, "bad result ", result);

                break;
            }
            case ccl_coll_allgatherv:
            {
                ccl_coll_param coll_param{};
                coll_param.ctype = ccl_coll_allgatherv;
                coll_param.send_buf = param.send_buf.get_ptr(bytes);
                coll_param.recv_counts = static_cast<size_t*>(param.recv_counts.get_ptr(sizeof(size_t) * sched->coll_param.comm->size()));
                size_t recv_bytes = 0;
                for (size_t i = 0; i < sched->coll_param.comm->size(); i++)
                {
                    recv_bytes += coll_param.recv_counts[i];
                }
                coll_param.recv_buf = param.recv_buf.get_ptr(recv_bytes);
                coll_param.count = param.count;
                coll_param.dtype = param.dtype;
                coll_param.comm = sched->coll_param.comm;
                coll_sched.reset(new ccl_extra_sched(coll_param, sched->sched_id));

                auto result = ccl_coll_build_allgatherv(coll_sched.get(),
                                                       param.send_buf,
                                                       coll_sched->coll_param.count,
                                                       param.recv_buf,
                                                       coll_sched->coll_param.recv_counts,
                                                       coll_sched->coll_param.dtype);

                CCL_ASSERT(result == ccl_status_success, "bad result ", result);

                break;
            }
            default:
                CCL_FATAL("not supported type ", param.ctype);
                break;
        }

        if (coll_sched)
        {
            LOG_DEBUG("starting COLL entry");
            auto req = sched->start_subsched(coll_sched.get());
            LOG_DEBUG("COLL entry: sched ", coll_sched.get(), ", req ", req);
            // TODO: insert into per-worker sched cache
        }
        else
        {
            CCL_ASSERT(0);
        }
    }

    ccl_coll_entry_param param;
    std::unique_ptr<ccl_extra_sched> coll_sched;
};
