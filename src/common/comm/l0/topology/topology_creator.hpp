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
namespace native
{
namespace details
{

template<class F>
struct device_group_container_functor
{
    template<class ...Args>
    device_group_container_functor(Args&& ...args):
        operation(std::forward<Args>(args)...)
    {
    }

    template<class device_container_t>
    void operator() (device_container_t& container)
    {
        operation(container);
    }
    F& get_functor()
    {
        return operation;
    }
private:
    F operation;
};
}

template<class F, class ...Args>
details::device_group_container_functor<F> create_device_functor(Args&& ...args)
{
    return details::device_group_container_functor<F> (std::forward<Args>(args)...);
}
}
