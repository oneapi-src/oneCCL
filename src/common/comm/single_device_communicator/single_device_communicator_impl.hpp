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
#include "common/comm/single_device_communicator/single_device_communicator.hpp"
#include "common/comm/single_device_communicator/single_device_base_impl.hpp"

#include "oneapi/ccl/native_device_api/interop_utils.hpp"
#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/event/impls/scoped_event.hpp"

#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"

/* allgatherv */

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allgatherv_base_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl_coll_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    usm_support_mode send_buf_result, recv_buf_result;
    std::string send_buf_error, recv_buf_error;
    std::tie(send_buf_result, std::ignore, send_buf_error) =
        ::native::details::check_assoc_device_memory(send_buf, get_device(), get_context());
    std::tie(recv_buf_result, std::ignore, recv_buf_error) =
        ::native::details::check_assoc_device_memory(recv_buf, get_device(), get_context());
    if ((send_buf_result == usm_support_mode::direct or
         send_buf_result == usm_support_mode::shared) and
        (recv_buf_result == usm_support_mode::direct or
         recv_buf_result == usm_support_mode::shared)) {
        LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for both buffers");
        scoped_req = ccl::details::make_unique_scoped_event<
            ccl::host_event_impl>(ccl_allgatherv_impl(
            reinterpret_cast<const void*>(send_buf),
            send_count,
            reinterpret_cast<void*>(recv_buf),
            recv_counts.data(),
            ccl::native_type_info<buffer_type>::ccl_datatype_value,
            attr,
            comm_impl.get(),
            nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
            /*, stream.get()*/));
    }
    else if (send_buf_result == usm_support_mode::need_conversion and
             recv_buf_result == usm_support_mode::need_conversion) {
        size_t recv_total_size = std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{});
        ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
        auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
            nullptr,
            /*send_buf*/
            cl::sycl::buffer<buffer_type>{
                send_buf, send_count, cl::sycl::property::buffer::use_host_ptr{} },
            /*recv_buf*/
            cl::sycl::buffer<buffer_type>{
                recv_buf, recv_total_size, cl::sycl::property::buffer::use_host_ptr{} });

        req = ccl_allgatherv_impl(
            reinterpret_cast<const void*>(
                &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
            send_count,
            reinterpret_cast<void*>(&scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
            recv_counts.data(),
            ccl::native_type_info<buffer_type>::ccl_datatype_value,
            attr,
            comm_impl.get(),
            stream.get());
#else
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                        " - USM convertation is not supported for such configuration");
#endif
        scoped_req_sycl->charge(req);
        scoped_req = std::move(scoped_req_sycl);
    }
    else {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + "\nsend_buf check result:\n" +
                             send_buf_error + "\nrecv_buf check result:\n" + recv_buf_error);
    }

    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {

    ccl_coll_attr internal_attr(attr);
    return allgatherv_base_impl(send_buf, send_count, recv_buf, recv_counts, stream, internal_attr, deps);
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    ccl::vector_class<buffer_type*>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {

    ccl_coll_attr internal_attr(attr);
    internal_attr.vector_buf = 1;
    return allgatherv_base_impl(send_buf, send_count, (buffer_type*)(recv_bufs.data()), recv_counts, stream, internal_attr, deps);
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(&recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           stream.get());
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allreduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    using namespace native;

    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    usm_support_mode send_buf_result, recv_buf_result;
    std::string send_buf_error, recv_buf_error;
    std::tie(send_buf_result, std::ignore, send_buf_error) =
        ::native::details::check_assoc_device_memory(send_buf, get_device(), get_context());
    std::tie(recv_buf_result, std::ignore, recv_buf_error) =
        ::native::details::check_assoc_device_memory(recv_buf, get_device(), get_context());
    if ((send_buf_result == usm_support_mode::direct or
         send_buf_result == usm_support_mode::shared) and
        (recv_buf_result == usm_support_mode::direct or
         recv_buf_result == usm_support_mode::shared)) {
        LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for both buffers");
        scoped_req = ccl::details::make_unique_scoped_event<
            ccl::host_event_impl>(ccl_allreduce_impl(
            reinterpret_cast<const void*>(send_buf),
            reinterpret_cast<void*>(recv_buf),
            count,
            ccl::native_type_info<buffer_type>::ccl_datatype_value,
            reduction,
            attr,
            comm_impl.get(),
            nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
            /*, stream.get()*/));
    }
    else if (send_buf_result == usm_support_mode::need_conversion and
             recv_buf_result == usm_support_mode::need_conversion) {
        ccl_request* req = nullptr;
        LOG_TRACE("comm: ", to_string(), " - use USM pointers convertation for both buffers");
#ifdef CCL_ENABLE_SYCL
        auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
            nullptr,
            /*send_buf*/
            cl::sycl::buffer<buffer_type>{
                send_buf, count, cl::sycl::property::buffer::use_host_ptr{} },
            /*recv_buf*/
            cl::sycl::buffer<buffer_type>{
                recv_buf, count, cl::sycl::property::buffer::use_host_ptr{} });

        req = ccl_allreduce_impl(
            reinterpret_cast<const void*>(
                &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
            reinterpret_cast<void*>(&scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
            count,
            ccl::native_type_info<buffer_type>::ccl_datatype_value,
            reduction,
            attr,
            comm_impl.get(),
            stream.get());
#else
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                        " - USM convertation is not supported for such configuration");
#endif
        scoped_req_sycl->charge(req);
        scoped_req = std::move(scoped_req_sycl);
    }
    else {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + "\nsend_buf check result:\n" +
                             send_buf_error + "\nrecv_buf check result:\n" + recv_buf_error);
    }

    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::allreduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(&send_buf),
                                          reinterpret_cast<void*>(&recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          reduction,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoall */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoall_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (unlikely(is_ready() == false)) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using alltoall_usm_check_result = multiple_assoc_result<2>;
    alltoall_usm_check_result usm_assoc_results =
        check_multiple_assoc_device_memory(get_device(), get_context(), send_buf, recv_buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    bool ret = std::all_of(usm_assoc_results.begin(),
                           usm_assoc_results.end(),
                           [test_value](const typename alltoall_usm_check_result::value_type& v) {
                               return test_value == std::get<assoc_result_index::SUPPORT_MODE>(v);
                           });
    if (!ret) {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - invalid USM arguments:\n" +
                             ::native::details::to_string(usm_assoc_results) +
                             "\nMixed types are not supported as well");
    }
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_alltoall_impl(
                reinterpret_cast<const void*>(send_buf),
                reinterpret_cast<void*>(recv_buf),
                count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            LOG_TRACE(
                "comm: ", to_string(), " - use USM pointers convertation to SYCL for both buffers");
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*send_buf*/
                cl::sycl::buffer<buffer_type>{
                    send_buf, count * size(), cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_buf*/
                cl::sycl::buffer<buffer_type>{
                    recv_buf, count * size(), cl::sycl::property::buffer::use_host_ptr{} });

            req = ccl_alltoall_impl(
                reinterpret_cast<const void*>(
                    &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
                reinterpret_cast<void*>(
                    &scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
                count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                attr,
                comm_impl.get(),
                stream.get());
#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoall_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<buffer_type*>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoall_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(&send_buf),
                                         reinterpret_cast<void*>(&recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         stream.get());
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoallv_impl(
    const buffer_type* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using alltoallv_usm_check_result = multiple_assoc_result<2>;
    alltoallv_usm_check_result usm_assoc_results =
        check_multiple_assoc_device_memory(get_device(), get_context(), send_buf, recv_buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    bool ret = std::all_of(usm_assoc_results.begin(),
                           usm_assoc_results.end(),
                           [test_value](const typename alltoallv_usm_check_result::value_type& v) {
                               return test_value == std::get<assoc_result_index::SUPPORT_MODE>(v);
                           });
    if (!ret) {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - invalid USM arguments:\n" +
                             ::native::details::to_string(usm_assoc_results));
    }
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_alltoallv_impl(
                reinterpret_cast<const void*>(send_buf),
                send_counts.data(),
                reinterpret_cast<void*>(recv_buf),
                recv_counts.data(),
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            size_t send_total_size =
                std::accumulate(send_counts.begin(), send_counts.end(), size_t{});
            size_t recv_total_size =
                std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{});
            LOG_TRACE(
                "comm: ",
                to_string(),
                " - use USM pointers convertation to SYCL for both buffers, send_total_size: ",
                send_total_size,
                ", recv_total_size: ",
                recv_total_size);
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*send_buf*/
                cl::sycl::buffer<buffer_type>{
                    send_buf, send_total_size, cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_buf*/
                cl::sycl::buffer<buffer_type>{
                    recv_buf, recv_total_size, cl::sycl::property::buffer::use_host_ptr{} });

            req = ccl_alltoallv_impl(
                reinterpret_cast<const void*>(
                    &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
                send_counts.data(),
                reinterpret_cast<void*>(
                    &scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
                recv_counts.data(),
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                attr,
                comm_impl.get(),
                stream.get());
#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoallv_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoallv_impl(
    const buffer_type& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(&send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(&recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::broadcast_impl(
    buffer_type* buf,
    size_t count,
    size_t root,
    const ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using broadcast_usm_check_result = multiple_assoc_result<1>;
    broadcast_usm_check_result usm_assoc_results =
        check_multiple_assoc_device_memory(get_device(), get_context(), buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_broadcast_impl(
                reinterpret_cast<void*>(buf),
                count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                root,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            LOG_TRACE(
                "comm: ", to_string(), " - use USM pointers convertation to SYCL for both buffers");
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*buf*/
                cl::sycl::buffer<buffer_type>{
                    buf, count, cl::sycl::property::buffer::use_host_ptr{} });

            req = ccl_broadcast_impl(
                reinterpret_cast<void*>(&scoped_req_sycl->template get_arg_by_index<0>() /*buf*/),
                count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                root,
                attr,
                comm_impl.get(),
                stream.get());
#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::broadcast_impl(
    buffer_type& buf,
    size_t count,
    size_t root,
    const ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(&buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          root,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::reduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using reduce_usm_check_result = multiple_assoc_result<2>;
    reduce_usm_check_result usm_assoc_results =
        check_multiple_assoc_device_memory(get_device(), get_context(), send_buf, recv_buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    bool ret = std::all_of(usm_assoc_results.begin(),
                           usm_assoc_results.end(),
                           [test_value](const typename reduce_usm_check_result::value_type& v) {
                               return test_value == std::get<assoc_result_index::SUPPORT_MODE>(v);
                           });
    if (!ret) {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - invalid USM arguments:\n" +
                             ::native::details::to_string(usm_assoc_results));
    }
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_reduce_impl(
                reinterpret_cast<const void*>(send_buf),
                reinterpret_cast<void*>(recv_buf),
                count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                reduction,
                root,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            LOG_TRACE(
                "comm: ", to_string(), " - use USM pointers convertation to SYCL for both buffers");
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*send_buf*/
                cl::sycl::buffer<buffer_type>{
                    send_buf, count, cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_buf*/
                cl::sycl::buffer<buffer_type>{
                    recv_buf, count, cl::sycl::property::buffer::use_host_ptr{} });

            req =
                ccl_reduce_impl(reinterpret_cast<const void*>(
                                    &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
                                reinterpret_cast<void*>(
                                    &scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
                                count,
                                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                reduction,
                                root,
                                attr,
                                comm_impl.get(),
                                stream.get());
#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::reduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(&send_buf),
                                       reinterpret_cast<void*>(&recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                       reduction,
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce_scatter */
template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::reduce_scatter_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {

    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using reduce_scatter_usm_check_result = multiple_assoc_result<2>;
    reduce_scatter_usm_check_result usm_assoc_results =
        check_multiple_assoc_device_memory(get_device(), get_context(), send_buf, recv_buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    bool ret = std::all_of(usm_assoc_results.begin(),
                           usm_assoc_results.end(),
                           [test_value](const typename reduce_scatter_usm_check_result::value_type& v) {
                               return test_value == std::get<assoc_result_index::SUPPORT_MODE>(v);
                           });
    if (!ret) {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - invalid USM arguments:\n" +
                             ::native::details::to_string(usm_assoc_results));
    }
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_reduce_scatter_impl(
                reinterpret_cast<const void*>(send_buf),
                reinterpret_cast<void*>(recv_buf),
                recv_count,
                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                reduction,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            LOG_TRACE(
                "comm: ", to_string(), " - use USM pointers convertation to SYCL for both buffers");
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*send_buf*/
                cl::sycl::buffer<buffer_type>{
                    send_buf, recv_count, cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_buf*/
                cl::sycl::buffer<buffer_type>{
                    recv_buf, recv_count, cl::sycl::property::buffer::use_host_ptr{} });

            req =
                ccl_reduce_scatter_impl(reinterpret_cast<const void*>(
                                    &scoped_req_sycl->template get_arg_by_index<0>() /*send_buf*/),
                                reinterpret_cast<void*>(
                                    &scoped_req_sycl->template get_arg_by_index<1>() /*recv_buf*/),
                                recv_count,
                                ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                reduction,
                                attr,
                                comm_impl.get(),
                                stream.get());
#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
}

template <class buffer_type>
single_device_communicator::coll_request_t single_device_communicator::reduce_scatter_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {

    const ccl_stream* stream_ptr = stream.get();
    ccl_request* req = ccl_reduce_scatter_impl(reinterpret_cast<const void*>(&send_buf),
                                               reinterpret_cast<void*>(&recv_buf),
                                               recv_count,
                                               ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                               reduction,
                                               attr,
                                               comm_impl.get(),
                                               stream_ptr);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
single_device_communicator::coll_request_t single_device_communicator::sparse_allreduce_impl(
    const index_buffer_type* send_ind_buf,
    size_t send_ind_count,
    const value_buffer_type* send_val_buf,
    size_t send_val_count,
    index_buffer_type* recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_type* recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    LOG_DEBUG("device idx: ", get_device_path(), ", rank: (", rank(), "/", size(), ")");

    std::unique_ptr<ccl::chargeable_event> scoped_req;
    using namespace ::native::details;

    // test USM arguments on validity
    using sparse_allreduce_usm_check_result = multiple_assoc_result<4>;
    sparse_allreduce_usm_check_result usm_assoc_results = check_multiple_assoc_device_memory(
        get_device(), get_context(), send_ind_buf, send_val_buf, recv_ind_buf, recv_val_buf);
    usm_support_mode test_value = std::get<assoc_result_index::SUPPORT_MODE>(usm_assoc_results[0]);
    bool ret =
        std::all_of(usm_assoc_results.begin(),
                    usm_assoc_results.end(),
                    [test_value](const typename sparse_allreduce_usm_check_result::value_type& v) {
                        return test_value == std::get<assoc_result_index::SUPPORT_MODE>(v);
                    });
    if (!ret) {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - invalid USM arguments:\n" +
                             ::native::details::to_string(usm_assoc_results));
    }
    switch (test_value) {
        case usm_support_mode::shared: /*the same as `direct` at now*/
            LOG_TRACE("comm: ", to_string(), " - use USM shared pointers for buffers");
        case usm_support_mode::direct: {
            LOG_TRACE("comm: ", to_string(), " - use USM direct pointers for buffers");
            scoped_req = ccl::details::make_unique_scoped_event<
                ccl::host_event_impl>(ccl_sparse_allreduce_impl(
                (const void*)send_ind_buf,
                send_ind_count,
                (const void*)send_val_buf,
                send_val_count,
                (void*)recv_ind_buf,
                recv_ind_count,
                (void*)recv_val_buf,
                recv_val_count,
                ccl::native_type_info<index_buffer_type>::ccl_datatype_value,
                ccl::native_type_info<value_buffer_type>::ccl_datatype_value,
                reduction,
                attr,
                comm_impl.get(),
                nullptr /*TODO fix core part, because stream existance use force cast to sycl::buffer*/
                /*, stream.get()*/));
            break;
        }
        case need_conversion: {
            ccl_request* req = nullptr;
#ifdef CCL_ENABLE_SYCL
            LOG_TRACE("comm: ",
                      to_string(),
                      " - use USM pointers convertation to SYCL for every buffers");
            auto scoped_req_sycl = ccl::details::make_unique_scoped_event<ccl::host_event_impl>(
                nullptr,
                /*send_ind_buf*/
                cl::sycl::buffer<index_buffer_type>{
                    send_ind_buf, send_ind_count, cl::sycl::property::buffer::use_host_ptr{} },
                /*send_val_buf*/
                cl::sycl::buffer<value_buffer_type>{
                    send_val_buf, send_val_count, cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_ind_buf*/
                cl::sycl::buffer<index_buffer_type>{
                    recv_ind_buf, recv_ind_count, cl::sycl::property::buffer::use_host_ptr{} },
                /*recv_val_buf*/
                cl::sycl::buffer<value_buffer_type>{
                    recv_val_buf, recv_val_count, cl::sycl::property::buffer::use_host_ptr{} });
            req = ccl_sparse_allreduce_impl(
                reinterpret_cast<const void*>(&scoped_req_sycl->template get_arg_by_index<0>()),
                send_ind_count,
                reinterpret_cast<const void*>(&scoped_req_sycl->template get_arg_by_index<1>()),
                send_val_count,
                reinterpret_cast<void*>(&scoped_req_sycl->template get_arg_by_index<2>()),
                recv_ind_count,
                reinterpret_cast<void*>(&scoped_req_sycl->template get_arg_by_index<3>()),
                recv_val_count,
                ccl::native_type_info<index_buffer_type>::ccl_datatype_value,
                ccl::native_type_info<value_buffer_type>::ccl_datatype_value,
                reduction,
                attr,
                comm_impl.get(),
                stream.get());

#else
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                            " - USM convertation is not supported for such configuration");
#endif
            scoped_req_sycl->charge(req);
            scoped_req = std::move(scoped_req_sycl);
            break;
        }
        default:
            throw ccl::exception(std::string(__PRETTY_FUNCTION__) +
                                 " - USM category is not supported for such configuration:\n" +
                                 ::native::details::to_string(usm_assoc_results[0]));
    }
    return std::unique_ptr<ccl::event_impl>(scoped_req.release());
    ;
}

template <class index_buffer_container_type, class value_buffer_container_type>
single_device_communicator::coll_request_t single_device_communicator::sparse_allreduce_impl(
    const index_buffer_container_type& send_ind_buf,
    size_t send_ind_count,
    const value_buffer_container_type& send_val_buf,
    size_t send_val_count,
    index_buffer_container_type& recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_container_type& recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_sparse_allreduce_impl(
        reinterpret_cast<const void*>(&send_ind_buf),
        send_ind_count,
        reinterpret_cast<const void*>(&send_val_buf),
        send_val_count,
        reinterpret_cast<void*>(&recv_ind_buf),
        recv_ind_count,
        reinterpret_cast<void*>(&recv_val_buf),
        recv_val_count,
        ccl::native_type_info<index_buffer_container_type>::ccl_datatype_value,
        ccl::native_type_info<value_buffer_container_type>::ccl_datatype_value,
        reduction,
        attr,
        comm_impl.get(),
        stream_ptr);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
