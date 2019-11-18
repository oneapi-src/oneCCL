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

#include <utility>
#include <tuple>


template <int CurIndex, class T, class U, class... Args>
struct get_tuple_elem_index
{
    static constexpr int index = get_tuple_elem_index<CurIndex + 1, T, Args...>::index;
};

template <int CurIndex, class T, class... Args>
struct get_tuple_elem_index<CurIndex, T, T, Args...>
{
    static constexpr int index = CurIndex;
};

template <class T, class... Args>
typename std::remove_reference<typename std::remove_cv<T>::type>::type& ccl_tuple_get(std::tuple<Args...>& t)
{
    using non_cv_type = typename std::remove_cv<T>::type;
    using non_ref_type = typename std::remove_reference<non_cv_type>::type;
    return std::get<get_tuple_elem_index<0, non_ref_type, Args...>::index>(t);
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each(TupleType&&, FunctionType,
                        std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>)
{}

template<std::size_t I, typename TupleType, typename FunctionType
       , typename = typename std::enable_if<I!=std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
void ccl_tuple_for_each(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
    f(std::get<I>(std::forward<TupleType>(t)));
    ccl_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each(TupleType&& t, FunctionType f)
{
    ccl_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each_indexed(FunctionType,
                        std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>)
{}

template<typename TupleType, typename FunctionType, std::size_t I,
       typename = typename std::enable_if<I!=std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
void ccl_tuple_for_each_indexed(FunctionType f, std::integral_constant<size_t, I>)
{
    f.template invoke<I, typename std::tuple_element<I, TupleType>::type>();
    ccl_tuple_for_each_indexed<TupleType, FunctionType>(f, std::integral_constant<size_t, I + 1>());
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each_indexed(FunctionType f)
{
    ccl_tuple_for_each_indexed<TupleType, FunctionType, 0>(f, std::integral_constant<size_t, 0>());
}
