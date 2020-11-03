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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_event.hpp"
#include "common/utils/tuple.hpp"
#include "common/event/impls/event_impl.hpp"

class ccl_request;

namespace ccl {

struct chargeable_event : public event_impl {
    virtual void charge(ccl_request* r) = 0;
};

template <class event_impl_t, class... scoped_args>
class scoped_event_impl final : public chargeable_event {
public:
    using base_t = chargeable_event;
    using impl_t = event_impl_t;
    using scoped_args_storage_t = std::tuple<scoped_args...>;

    explicit scoped_event_impl(ccl_request* r, scoped_args&&... args)
            : base_t(),
              storage(std::forward<scoped_args>(args)...),
              impl(r) {}

    explicit scoped_event_impl(ccl_request* r, scoped_args_storage_t&& args)
            : base_t(),
              storage(std::forward<scoped_args_storage_t>(args)),
              impl(r) {}

    ~scoped_event_impl() override = default;

    void wait() override {
        return impl.wait();
    }

    bool test() override {
        return impl.test();
    }

    bool cancel() override {
        return impl.cancel();
    }

    event::native_t& get_native() override {
        throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
    }

    // scoped interface
    template <int index>
    typename std::tuple_element<index, scoped_args_storage_t>::type& get_arg_by_index() {
        return std::get<index>(storage);
    }

    void charge(ccl_request* r) override {
        impl_t charged_req(r);
        std::swap(impl, charged_req);
    }

    template <class OP, class... Args>
    void charge_by_op(OP op, Args&&... args) {
        //ccl_request* req = op(ccl_tuple_get<scoped_args>(storage)..., std::forward<Args>(args)...);
        //charge(req);

        charge_by_op_impl(
            op,
            typename sequence_generator<std::tuple_size<scoped_args_storage_t>::value>::type(),
            std::forward<Args>(args)...);
    }

private:
    template <class OP, int... connected_arguments_indices, class... Args>
    void charge_by_op_impl(OP op,
                           numeric_sequence<connected_arguments_indices...>,
                           Args&&... args) {
        ccl_request* req =
            op(std::get<connected_arguments_indices>(storage)..., std::forward<Args>(args)...);
        charge(req);
    }

    scoped_args_storage_t storage;
    impl_t impl;
};

namespace details {
template <class event_impl_t, class... scoped_args>
std::unique_ptr<scoped_event_impl<event_impl_t, scoped_args...>> make_unique_scoped_event(
    ccl_request* r,
    scoped_args&&... args) {
    return std::unique_ptr<scoped_event_impl<event_impl_t, scoped_args...>>(
        new scoped_event_impl<event_impl_t, scoped_args...>(
            r, std::forward<scoped_args>(args)...));
}

template <class event_impl_t, class... scoped_args>
std::unique_ptr<scoped_event_impl<event_impl_t, scoped_args...>> make_unique_scoped_event(
    ccl_request* r,
    std::tuple<scoped_args...>&& args) {
    return std::unique_ptr<scoped_event_impl<event_impl_t, scoped_args...>>(
        new scoped_event_impl<event_impl_t, scoped_args...>(
            r, std::forward<std::tuple<scoped_args...>>(args)));
}

template <class event_impl_t, class operation, class... scoped_args, class... non_scoped_args>
std::unique_ptr<chargeable_event> make_and_charge_scoped_event(
    operation op,
    std::tuple<scoped_args...>&& args,
    non_scoped_args&&... elapsed_args) {
    auto typed_arg = make_unique_scoped_event<event_impl_t>(
        nullptr, std::forward<std::tuple<scoped_args...>>(args));
    typed_arg->charge_by_op(op, std::forward<non_scoped_args>(elapsed_args)...);
    return typed_arg;
}

} // namespace details
} // namespace ccl
