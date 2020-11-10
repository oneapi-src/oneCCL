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
#ifndef BASE_UTILS_HPP
#define BASE_UTILS_HPP

#include <algorithm>
#include <iterator>
#include <sstream>
#include <tuple>
#include <utility>

template <int CurIndex, class T, class U, class... Args>
struct get_tuple_elem_index {
    static constexpr int index = get_tuple_elem_index<CurIndex + 1, T, Args...>::index;
};

template <int CurIndex, class T, class... Args>
struct get_tuple_elem_index<CurIndex, T, T, Args...> {
    static constexpr int index = CurIndex;
};

template <class T, class... Args>
typename std::remove_reference<typename std::remove_cv<T>::type>::type& ccl_tuple_get(
    std::tuple<Args...>& t) {
    using non_cv_type = typename std::remove_cv<T>::type;
    using non_ref_type = typename std::remove_reference<non_cv_type>::type;
    return std::get<get_tuple_elem_index<0, non_ref_type, Args...>::index>(t);
}

template <class T, class... Args>
const typename std::remove_reference<typename std::remove_cv<T>::type>::type& ccl_tuple_get(
    const std::tuple<Args...>& t) {
    using non_cv_type = typename std::remove_cv<T>::type;
    using non_ref_type = typename std::remove_reference<non_cv_type>::type;
    return std::get<get_tuple_elem_index<0, non_ref_type, Args...>::index>(t);
}

template <class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, std::true_type tuple_finished) {
    // nothing to do
}

template <class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, std::false_type tuple_not_finished) {
    f(std::get<cur_index>(std::forward<specific_tuple>(t)));

    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_impl<specific_tuple, functor, cur_index + 1>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{});
}

template <class specific_tuple, class functor, size_t cur_index = 0>
void ccl_tuple_for_each(specific_tuple&& t, functor f) {
    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::integral_constant<bool, cur_index >= tuple_size>;
    ccl_tuple_for_each_impl<specific_tuple, functor, cur_index>(
        std::forward<specific_tuple>(t), f, is_tuple_finished_t{});
}

template <typename specific_tuple, size_t cur_index, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor,
                                     std::true_type tuple_finished,
                                     const FunctionArgs&... args) {}

template <typename specific_tuple, size_t cur_index, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor f,
                                     std::false_type tuple_not_finished,
                                     const FunctionArgs&... args) {
    using tuple_element_t = typename std::tuple_element<cur_index, specific_tuple>::type;

    f.template invoke<cur_index, tuple_element_t>(args...);

    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_indexed_impl<specific_tuple, cur_index + 1, functor>(
        f, is_tuple_finished_t{}, args...);
}

template <typename specific_tuple, typename functor, class... FunctionArgs>
void ccl_tuple_for_each_indexed(functor f, const FunctionArgs&... args) {
    constexpr std::size_t tuple_size =
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::false_type; //non-empty tuple started
    ccl_tuple_for_each_indexed_impl<specific_tuple, 0, functor, FunctionArgs...>(
        f, is_tuple_finished_t{}, args...);
}

namespace utils {

template <typename T>
void str_to_array(const char* input, std::vector<T>& output, char delimiter) {
    if (!input) {
        return;
    }
    std::stringstream ss(input);
    T temp{};
    while (ss >> temp) {
        output.push_back(temp);
        if (ss.peek() == delimiter) {
            ss.ignore();
        }
    }
}
template <>
void str_to_array(const char* input, std::vector<std::string>& output, char delimiter) {
    std::string processes_input(input);

    processes_input.erase(std::remove_if(processes_input.begin(),
                                         processes_input.end(),
                                         [](unsigned char x) {
                                             return std::isspace(x);
                                         }),
                          processes_input.end());

    std::replace(processes_input.begin(), processes_input.end(), delimiter, ' ');
    std::stringstream ss(processes_input);

    while (ss >> processes_input) {
        output.push_back(processes_input);
    }
}

template <typename T>
void str_to_mset(const char* input, std::multiset<T>& output, char delimiter) {
    if (!input) {
        return;
    }
    std::stringstream ss(input);
    T temp{};
    while (ss >> temp) {
        output.insert(temp);
        if (ss.peek() == delimiter) {
            ss.ignore();
        }
    }
}

#ifdef MULTI_GPU_SUPPORT
template <>
void str_to_mset(const char* input, std::multiset<ccl::device_index_type>& output, char delimiter) {
    std::string processes_input(input);

    processes_input.erase(std::remove_if(processes_input.begin(),
                                         processes_input.end(),
                                         [](unsigned char x) {
                                             return std::isspace(x);
                                         }),
                          processes_input.end());

    std::replace(processes_input.begin(), processes_input.end(), delimiter, ' ');
    std::stringstream ss(processes_input);

    while (ss >> processes_input) {
        output.insert(ccl::from_string(processes_input));
    }
}

std::shared_ptr<ccl::kvs> build_kvs(int mpi_rank) {
    std::shared_ptr<ccl::kvs> kvs_instance;
    ccl::kvs::address_type main_addr;
    if (mpi_rank == 0) {
        kvs_instance = ccl::create_main_kvs();
        main_addr = kvs_instance->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs_instance = ccl::create_kvs(main_addr);
    }
    return kvs_instance;
}

inline size_t take_mpi_rank_id_offest(const size_t mpi_rank_in_cluster,
                                      const int mpi_size,
                                      const size_t total_device_in_cluster) {
    if (mpi_size > 2) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - Only TWO processes support case !\n");
    }
    return total_device_in_cluster;
}

ccl::process_device_indices_type extract_indices_for_threads(
    const size_t mpi_rank_in_cluster,
    const int current_mpi_rank,
    std::vector<std::string> thread_gpu_affinity,
    size_t& total_device_in_cluster,
    std::vector<size_t>& total_devices_in_process,
    std::map<size_t, std::vector<ccl::communicator::device_type>>& devices_for_current_mpi_rank) {
    ccl::process_device_indices_type thread_group_affinity;

    for (size_t thread_index = 0; thread_index < thread_gpu_affinity.size(); thread_index++) {
        ccl::device_indices_type device_group_affinity;
        str_to_mset<ccl::device_index_type>(
            thread_gpu_affinity[thread_index].c_str(), device_group_affinity, ',');

        std::cout << " Extracted GPU indices for thread by id: " << thread_index
                  << ", devices in threads count: " << device_group_affinity.size() << std::endl;
        total_device_in_cluster += device_group_affinity.size();
        total_devices_in_process[mpi_rank_in_cluster] += device_group_affinity.size();
        thread_group_affinity[thread_index] = device_group_affinity;

        if (mpi_rank_in_cluster == static_cast<size_t>(current_mpi_rank)) {
            for (auto device_vendor_id : device_group_affinity) {
                devices_for_current_mpi_rank[thread_index].push_back(
                    ccl::create_from_index(device_vendor_id).device);
            }
        }
    }
    return thread_group_affinity;
}

std::vector<ccl::communicator::device_type> set_union_devices_in_current_process(
    const std::map<size_t, std::vector<ccl::communicator::device_type>>& devices_for_mpi_rank) {
    std::vector<ccl::communicator::device_type> devices_in_process;
    for (auto& thread_devices : devices_for_mpi_rank) {
        devices_in_process.insert(
            devices_in_process.end(), thread_devices.second.begin(), thread_devices.second.end());
    }
    return devices_in_process;
}

#endif //MULTI_GPU_SUPPORT
} // namespace utils
#endif /* BASE_UTILS_HPP */
