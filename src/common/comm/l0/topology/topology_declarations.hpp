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
#include <list>
#include <map>
#include <vector>

#include "oneapi/ccl/native_device_api/l0/utils.hpp"

namespace native {
namespace details {
struct marked_idx : std::pair<bool, ccl::device_index_type> {
    marked_idx(bool m, ccl::device_index_type i) : std::pair<bool, ccl::device_index_type>(m, i) {}
};

using color_t = size_t; //consider std::optional

struct colored_index {
    colored_index(color_t c, const ccl::device_index_type& i) : color(c), index(i) {}
    color_t color;
    ccl::device_index_type index;

    bool operator==(const colored_index& rhs) const noexcept {
        return (color == rhs.color) and (index == rhs.index);
    }
};

template <class data_t>
struct colored_indexed_data : public colored_index {
    using payload_t = data_t;

    colored_indexed_data(color_t c,
                         const ccl::device_index_type& i,
                         const payload_t& t = payload_t{})
            : colored_index(c, i),
              payload(t) {}

    const payload_t& get_payload() const {
        return payload;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "color: " << color << ", index:" << index << ", data: "
           << "STUB";
        ;
        return ss.str();
    }

private:
    payload_t payload;
};

template <>
struct colored_indexed_data<void> : public colored_index {
    colored_indexed_data(color_t c, const ccl::device_index_type& i) : colored_index(c, i) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << "color: " << color << ", index:" << index;
        return ss.str();
    }
};

using colored_idx = colored_indexed_data<void>;

using plain_graph = std::vector<ccl::device_index_type>;
using plain_graph_list = std::list<plain_graph>;
using colored_plain_graph = std::vector<colored_idx>;
using colored_plain_graph_list = std::list<colored_plain_graph>;

using process_index_t = size_t;
using global_sorted_plain_graphs = std::map<process_index_t, plain_graph_list>;
using global_plain_graphs = std::vector<std::pair<process_index_t, plain_graph_list>>;
using global_sorted_colored_plain_graphs = std::map<process_index_t, colored_plain_graph_list>;
using global_plain_colored_graphs =
    std::vector<std::pair<process_index_t, colored_plain_graph_list>>;
using global_colored_plain_graphs = global_plain_colored_graphs;

std::string to_string(const plain_graph& cont);
std::string to_string(const plain_graph_list& lists, const std::string& prefix = std::string());
std::string to_string(const global_sorted_plain_graphs& cluster);
std::string to_string(const global_plain_graphs& cluster);
std::string to_string(const colored_plain_graph& cont);
std::string to_string(const colored_plain_graph_list& lists,
                      const std::string& prefix = std::string());
std::string to_string(const global_sorted_colored_plain_graphs& cluster);
std::string to_string(const global_plain_colored_graphs& cluster);

std::ostream& operator<<(std::ostream& out, const colored_idx& idx);
} // namespace details

template <class payload_type>
std::ostream& operator<<(std::ostream& out,
                         const details::colored_indexed_data<payload_type>& data) {
    out << data.to_string();
    return out;
}
} // namespace native
