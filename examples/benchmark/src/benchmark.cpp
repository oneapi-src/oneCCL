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
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "benchmark.hpp"
#include "declarations.hpp"

void do_regular(ccl::communicator* comm,
                ccl::coll_attr& coll_attr,
                coll_list_t& all_colls,
                req_list_t& reqs,
                const user_options_t& options)
{
    char* match_id = (char*)coll_attr.match_id;
    for (auto dtype : all_dtypes)
    {
        coll_list_t colls;
        std::string dtype_name;

        std::copy_if(all_colls.begin(), all_colls.end(), std::back_inserter(colls),
                    [dtype](const typename coll_list_t::value_type coll)
                    {  return dtype == coll->get_dtype(); });
        if (colls.empty())
            continue;

        dtype_name = find_str_val(dtype_names, dtype);

        PRINT_BY_ROOT(comm, "\ndtype: %s", dtype_name.c_str());

        reqs.reserve(colls.size() * options.buf_count);

        /* warm up */
        PRINT_BY_ROOT(comm, "do warm up");

        coll_attr.to_cache = 0;
        for (size_t count = 1; count < ELEM_COUNT; count *= 2)
        {
            for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
            {
                auto& coll = colls[coll_idx];
                for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++)
                {
                    // snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                    //          coll->name(), coll_idx, count, buf_idx);
                    // PRINT_BY_ROOT(comm, "start_coll: %s, count %zu, buf_idx %zu", coll->name(), count, buf_idx);
                    coll->start(count, buf_idx, coll_attr, reqs);
                }
            }
            for (auto &req : reqs)
            {
                req->wait();
            }
            reqs.clear();
        }

        /* benchmark with multiple equal sized buffer per collective */
        PRINT_BY_ROOT(comm, "do multi-buffers benchmark");
        coll_attr.to_cache = 1;
        for (size_t count = 1; count <= ELEM_COUNT; count *= 2)
        {
            try
            {
                double t = 0;
                for (size_t iter_idx = 0; iter_idx < options.iters; iter_idx++)
                {
                    if (options.check_values)
                    {
                        for (auto& coll : colls)
                        {
                            coll->prepare(count);
                        }
                    }

                    double t1 = when();
                    for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                    {
                        auto& coll = colls[coll_idx];
                        for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++)
                        {
                            snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                     coll->name(), coll_idx, count, buf_idx);
                            coll->start(count, buf_idx, coll_attr, reqs);
                        }
                    }
                    for (auto &req : reqs)
                    {
                        req->wait();
                    }
                    double t2 = when();
                    t += (t2 - t1);
                }

                reqs.clear();

                if (options.check_values)
                {
                    for (auto& coll : colls)
                    {
                        coll->finalize(count);
                    }
                }

                print_timings(*comm, &t, count,
                              ccl::datatype_get_size(dtype),
                              options.buf_count, comm->rank(),
                              comm->size());
            }
            catch(const std::exception& ex)
            {
                ASSERT(0, "error on count %zu, reason: %s", count, ex.what());
            }
        }

        comm->barrier();

        /* benchmark with single buffer per collective */
        PRINT_BY_ROOT(comm, "do single-buffer benchmark");
        coll_attr.to_cache = 1;
        for (size_t count = options.buf_count; count <= SINGLE_ELEM_COUNT; count *= 2)
        {
            try
            {
                double t = 0;
                for (size_t iter_idx = 0; iter_idx < options.iters; iter_idx++)
                {
                    double t1 = when();
                    for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                    {
                        auto& coll = colls[coll_idx];
                        snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_single_count_%zu",
                                 coll->name(), coll_idx, count);
                        coll->start_single(count, coll_attr, reqs);
                    }
                    for (auto &req : reqs)
                    {
                        req->wait();
                    }
                    double t2 = when();
                    t += (t2 - t1);

                    reqs.clear();
                }
                print_timings(*comm, &t, count,
                              ccl::datatype_get_size(dtype),
                              1, comm->rank(), comm->size());
            } catch (...)
            {
                ASSERT(0, "error on count %zu", count);
            }
        }
    }

    PRINT_BY_ROOT(comm, "PASSED\n");
}

void do_unordered(ccl::communicator* comm,
                  ccl::coll_attr& coll_attr,
                  coll_list_t& all_colls,
                  req_list_t& reqs,
                  const user_options_t& options)
{
    std::set<std::string> match_ids;
    char* match_id = (char*)coll_attr.match_id;

    for (auto dtype : all_dtypes)
    {
        coll_list_t colls;
        std::string dtype_name;

        std::copy_if(all_colls.begin(), all_colls.end(), std::back_inserter(colls),
                    [dtype](const typename coll_list_t::value_type coll)
                    {  return dtype == coll->get_dtype(); });

        if (colls.empty())
            continue;

        dtype_name = find_str_val(dtype_names, dtype);

        PRINT_BY_ROOT(comm, "\ndtype: %s", dtype_name.c_str());

        size_t rank = comm->rank();

        reqs.reserve(colls.size() * options.buf_count * (log2(ELEM_COUNT) + 1));

        PRINT_BY_ROOT(comm, "do unordered test");
        coll_attr.to_cache = 1;

        for (size_t count = 1; count <= ELEM_COUNT; count *= 2)
        {
            try
            {
                if (rank % 2)
                {
                    for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                    {
                        auto& coll = colls[coll_idx];
                        for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++)
                        {
                            snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                     coll->name(), coll_idx, count, buf_idx);
                            coll->start(count, buf_idx, coll_attr, reqs);
                            match_ids.emplace(match_id);
                        }
                    }
                }
                else
                {
                    for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                    {
                        size_t real_coll_idx = colls.size() - coll_idx - 1;
                        auto& coll = colls[real_coll_idx];
                        for (size_t buf_idx = 0; buf_idx < options.buf_count; buf_idx++)
                        {
                            size_t real_buf_idx = options.buf_count - buf_idx - 1;
                            snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                     coll->name(), real_coll_idx, count, real_buf_idx);
                            coll->start(count, real_buf_idx, coll_attr, reqs);
                            match_ids.insert(std::string(match_id));
                        }
                    }
                }
            }
            catch (...)
            {
                ASSERT(0, "error on count %zu", count);
            }
        }

        ASSERT(match_ids.size() == reqs.size(),
               "unexpected match_ids.size %zu, expected %zu",
               match_ids.size(), reqs.size());

        try
        {
            for (auto &req : reqs)
            {
                req->wait();
            }
        }
        catch (...)
        {
            ASSERT(0, "error on coll completion");
        }

        PRINT_BY_ROOT(comm, "PASSED\n");
    }
}

template<class Dtype>
void create_cpu_colls(std::list<std::string>& names, coll_list_t& colls)
{
    using namespace sparse_detail;
    using incremental_index_int_sparse_strategy = 
                                sparse_allreduce_strategy_impl<int, 
                                sparse_detail::incremental_indices_distributor>;
    using incremental_index_bfp16_sparse_strategy = 
                                sparse_allreduce_strategy_impl<ccl::bfp16, 
                                sparse_detail::incremental_indices_distributor>;

    std::stringstream error_messages_stream;
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream();
    for (auto names_it = names.begin(); names_it != names.end(); )
    {
        const std::string& name = *names_it;
        if (name == allgatherv_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_allgatherv_coll<Dtype>());
        }
        else if (name == allreduce_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_allreduce_coll<Dtype>());
        }
        else if (name == bcast_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_bcast_coll<Dtype>());
        }
        else if (name == reduce_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_reduce_coll<Dtype>());
        }
        else if (name == alltoall_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_alltoall_coll<Dtype>);
        }
        else if (name == alltoallv_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_alltoallv_coll<Dtype>);
        }
        else if (name.find(incremental_index_int_sparse_strategy::class_name()) != std::string::npos)
        {
            if (name.find(incremental_index_bfp16_sparse_strategy::class_name()) != std::string::npos)
            {
                if (is_bfp16_enabled() == 0)
                {
                    error_messages_stream << "BFP16 is not supported for current CPU, skipping " << name << ".\n";
                    names_it = names.erase(names_it);
                    continue;
                }
#ifdef CCL_BFP16_COMPILER
                const std::string args = name.substr(name.find(incremental_index_bfp16_sparse_strategy::class_name()) +
                                                     std::strlen(incremental_index_bfp16_sparse_strategy::class_name()));
                colls.emplace_back(new cpu_sparse_allreduce_coll<ccl::bfp16, int64_t,
                                                                 sparse_detail::incremental_indices_distributor>(args,
                                                                                  sizeof(float) / sizeof(ccl::bfp16),
                                                                                  sizeof(float) / sizeof(ccl::bfp16)));
#else
                error_messages_stream << "BFP16 is not supported by current compiler, skipping " << name << ".\n";
                names_it = names.erase(names_it);
                continue;
#endif
            }
            else
            {
                const std::string args = name.substr(name.find(incremental_index_int_sparse_strategy::class_name()) +
                                                     std::strlen(incremental_index_int_sparse_strategy::class_name()));
                colls.emplace_back(new cpu_sparse_allreduce_coll<Dtype, int>(args));
            }
        }
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
        ++names_it;
    }

    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty())
    {
        std::cerr << "WARNING:\n" << coll_processing_log << std::endl;
    }
    
    if (colls.empty())
    {
        throw std::logic_error(std::string(__FUNCTION__) + 
                               " - empty colls, reason: " + coll_processing_log);
    }
}

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
void create_sycl_colls(std::list<std::string>& names, coll_list_t& colls)
{

    using incremental_index_int_sparse_strategy = 
                            sparse_allreduce_strategy_impl<int, 
                                sparse_detail::incremental_indices_distributor>;
    using incremental_index_bfp16_sparse_strategy = 
                            sparse_allreduce_strategy_impl<ccl::bfp16, 
                                sparse_detail::incremental_indices_distributor>;
            
    std::stringstream error_messages_stream;
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream(sycl_queue);

    for (auto names_it = names.begin(); names_it != names.end(); )
    {
        const std::string& name = *names_it;
        
        if (name == allgatherv_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_allgatherv_coll<Dtype>());
        }
        else if (name == allreduce_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_allreduce_coll<Dtype>());
        }
        else if (name == alltoall_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_alltoall_coll<Dtype>());
        }
        else if (name == alltoallv_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_alltoallv_coll<Dtype>());
        }
        else if (name == bcast_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_bcast_coll<Dtype>());
        }
        else if (name == reduce_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_reduce_coll<Dtype>());
        }
        else if (name.find(incremental_index_int_sparse_strategy::class_name()) != std::string::npos)
        {
            // TODO case is not supported yet
            if (true)
            {
                error_messages_stream << "SYCL coll: skipping " << name << ", because it is not supported yet.\n";
                names_it = names.erase(names_it);
                continue;
            }
            
            const std::string args = name.substr(name.find(incremental_index_int_sparse_strategy::class_name()) +
                                                 std::strlen(incremental_index_int_sparse_strategy::class_name()));
            colls.emplace_back(new sycl_sparse_allreduce_coll<Dtype, int>(args));
        }
        else if (name.find(incremental_index_bfp16_sparse_strategy::class_name()) != std::string::npos)
        {
            // TODO case is not supported yet
            if (true)
            {
                error_messages_stream << "SYCL coll: skipping " << name << ", because it is not supported yet.\n";
                names_it = names.erase(names_it);
                continue;
            }
            
            if (is_bfp16_enabled() == 0)
            {
                error_messages_stream << "SYCL BFP16 is not supported for current CPU, skipping " << name << ".\n";
                names_it = names.erase(names_it);
                continue;
            }
#ifdef CCL_BFP16_COMPILER
            const std::string args = name.substr(name.find(incremental_index_bfp16_sparse_strategy::class_name()) +
                                                 std::strlen(incremental_index_bfp16_sparse_strategy::class_name()));
            colls.emplace_back(new sycl_sparse_allreduce_coll<ccl::bfp16, int64_t,
                                                              sparse_detail::incremental_indices_distributor>(args,
                                                                               sizeof(float) / sizeof(ccl::bfp16),
                                                                               sizeof(float) / sizeof(ccl::bfp16)));
#else
            error_messages_stream << "SYCL BFP16 is not supported by current compiler, skipping " << name << ".\n";
            names_it = names.erase(names_it);
            continue;
#endif
        }
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
        
        ++names_it;
    }
    
    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty())
    {
        std::cerr << "WARNING: " << coll_processing_log << std::endl;
    }
    
    if (colls.empty())
    {
        throw std::logic_error(std::string(__FUNCTION__) + 
                               " - empty colls, reason: " + coll_processing_log);
    }
}
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
void create_colls(std::list<std::string>& coll_names, ccl::stream_type backend,
                  coll_list_t& colls)
{
    switch (backend)
    {
        case ccl::stream_type::host:
            create_cpu_colls<Dtype>(coll_names, colls);
            break;
        case ccl::stream_type::gpu:
#ifdef CCL_ENABLE_SYCL
            create_sycl_colls<Dtype>(coll_names, colls);
#else
            ASSERT(0, "SYCL backend is requested but CCL_ENABLE_SYCL is not defined");
#endif
            break;
        default:
            ASSERT(0, "unknown backend %d", (int)backend);
            break;
    }

}

int main(int argc, char *argv[])
{
    user_options_t options;
    coll_list_t colls;
    req_list_t reqs;

    ccl::coll_attr coll_attr{};
    char match_id[MATCH_ID_SIZE] {'\0'};
    coll_attr.match_id = match_id;

    if (parse_user_options(argc, argv, options))
        return -1;

    try
    {
        ccl_tuple_for_each(launch_dtypes,
                           set_types_func(options.dtypes));

        ccl_tuple_for_each(launch_dtypes,
                           create_colls_func(options.coll_names,
                                             options.backend, colls));
    }
    catch (const std::runtime_error& e)
    {
        ASSERT(0, "cannot create coll objects: %s\n", e.what());
    }
    catch (const std::logic_error& e)
    {
        std::cerr << "Cannot launch benchmark: " << e.what() << std::endl;
        return -1;
    }

    ccl::communicator* comm = base_coll::comm.get();

    print_user_options(options, comm);

    if (options.coll_names.empty())
    {
        PRINT_BY_ROOT(comm, "empty coll list");
        print_help_usage(argv[0]);
        return -1;
    }

    comm->barrier();

    switch (options.loop)
    {
        case LOOP_REGULAR:
            do_regular(comm, coll_attr, colls, reqs, options);
            break;
        case LOOP_UNORDERED:
            do_unordered(comm, coll_attr, colls, reqs, options);
            break;
        default:
            ASSERT(0, "unknown loop %d", options.loop);
            break;
    }

    base_coll::comm.reset();
    base_coll::stream.reset();

    return 0;
}

template
void create_colls<int>(std::list<std::string>& coll_names, ccl::stream_type backend,
                  coll_list_t& colls);
template
void create_colls<float>(std::list<std::string>& coll_names, ccl::stream_type backend,
                  coll_list_t& colls);
