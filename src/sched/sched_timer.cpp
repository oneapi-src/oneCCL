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
#include <iomanip>
#include <numeric>
#include <sstream>

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "sched_timer.hpp"

namespace ccl {

void sched_timer::start() noexcept {
    start_time = std::chrono::high_resolution_clock::now();
}

void sched_timer::stop() {
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> time_span = stop_time - start_time;
    time_usec = time_span.count();
}

std::string sched_timer::str() const {
    std::stringstream ss;
    ss.precision(2);
    ss << std::fixed << get_time();
    return ss.str();
}

void sched_timer::print(std::string title) const {
    logger.info(title, ": ", this->str());
}

void sched_timer::reset() noexcept {
    time_usec = 0;
}

long double sched_timer::get_time() const noexcept {
    return time_usec;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

kernel_timer::kernel_timer()
        : kernel_time{ get_uninit_values() },
          operation_event_time{ get_uninit_values() },
          operation_start_time{ std::numeric_limits<uint64_t>::max() },
          operation_end_time{ std::numeric_limits<uint64_t>::max() },
          kernel_submit_time{ std::numeric_limits<uint64_t>::max() } {}

// Returns true if we have all the necessary data to print
bool kernel_timer::print(bool delay) const {
    auto is_value_set = [](std::pair<uint64_t, uint64_t> val) {
        return val != get_uninit_values();
    };

    auto is_val_set = [](uint64_t val) {
        return val != std::numeric_limits<uint64_t>::max();
    };

    // Convert value from ns to usec and format it.
    auto convert_output = [](uint64_t val) {
        std::stringstream ss;
        ss << (val / 1000) << "." << (val % 1000) / 100;

        return ss.str();
    };

    // currently there are 3 levels:
    // 0 - profiling and output is disabled
    // 1 - print time dureation for intervals
    // 2 - level 1 + raw timestamp that we collected
    int profile_level = ccl::global_data::env().enable_kernel_profile;

    // Make sure we have all the measurements
    bool all_measurements_are_ready =
        is_value_set(kernel_time) && is_val_set(operation_create_time) &&
        is_val_set(operation_start_time) && is_val_set(operation_end_time) &&
        is_val_set(kernel_submit_time) && is_val_set(deps_start_time) && is_val_set(deps_end_time);

    // operation_event_time is only required if we use output event, otherwise just
    // skip it
    if (ccl::global_data::env().enable_sycl_output_event) {
        all_measurements_are_ready =
            all_measurements_are_ready && is_value_set(operation_event_time);
    }

    if (!all_measurements_are_ready) {
        // need more data
        return false;
    }

    if (delay && all_measurements_are_ready) {
        // the output will be printed later
        ccl::global_data::get().timer_printer->add_timer(*this);
        return true;
    }

    std::stringstream ss;
    ss << "kernel: " << name << " time(usec)" << std::endl;
    if (profile_level > 1) {
        ss << "timestamps: " << std::endl;
        ss << "  operation create: " << operation_create_time << std::endl;
        ss << "  operation start: " << operation_start_time << std::endl;
        ss << "  deps wait start: " << deps_start_time << std::endl;
        ss << "  deps wait end: " << deps_end_time << std::endl;
        ss << "  kernel submit: " << kernel_submit_time << std::endl;
        ss << "  kernel start: " << kernel_time.first << std::endl;
        ss << "  kernel end: " << kernel_time.first << std::endl;
        if (ccl::global_data::env().enable_sycl_output_event) {
            ss << "  operation event start: " << operation_event_time.first << std::endl;
            ss << "  operation event end: " << operation_event_time.second << std::endl;
        }
        ss << "  operation end: " << operation_end_time << std::endl;
    }

    ss << "operation: " << convert_output(operation_end_time - operation_create_time) << std::endl;
    ss << "  api call: " << convert_output(operation_start_time - operation_create_time)
       << std::endl;
    ss << "  preparation: " << convert_output(kernel_submit_time - operation_start_time)
       << std::endl;
    ss << "    deps handling: " << convert_output(deps_end_time - deps_start_time) << std::endl;
    ss << "  device scheduling: " << convert_output(kernel_time.first - kernel_submit_time)
       << std::endl;
    ss << "  device execution: " << convert_output(kernel_time.second - kernel_time.first)
       << std::endl;
    if (ccl::global_data::env().enable_sycl_output_event) {
        ss << "  event completion: "
           << convert_output(operation_event_time.second - kernel_time.second) << std::endl;
        ss << "  completion: " << convert_output(operation_end_time - operation_event_time.second)
           << std::endl;
    }
    else {
        ss << "  completion: " << convert_output(operation_end_time - kernel_time.second)
           << std::endl;
    }

    ss << std::endl;

    std::cout << ss.str() << std::endl;

    return true;
}

void kernel_timer::reset() {
    kernel_time = { get_uninit_values() };
    operation_event_time = { get_uninit_values() };
    operation_start_time = std::numeric_limits<uint64_t>::max();
    operation_end_time = std::numeric_limits<uint64_t>::max();
    kernel_submit_time = std::numeric_limits<uint64_t>::max();
}

uint64_t kernel_timer::get_current_time() {
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now())
        .time_since_epoch()
        .count();
}

void kernel_timer::set_name(const std::string new_name) {
    name = new_name;
}
const std::string& kernel_timer::get_name() const {
    return name;
}

void kernel_timer::set_kernel_time(std::pair<uint64_t, uint64_t> val) {
    kernel_time = val;
}

void kernel_timer::set_operation_event_time(std::pair<uint64_t, uint64_t> val) {
    operation_event_time = val;
}

void kernel_timer::set_operation_create_time(uint64_t val) {
    operation_create_time = val;
}

void kernel_timer::set_operation_start_time(uint64_t val) {
    operation_start_time = val;
}

void kernel_timer::set_operation_end_time(uint64_t val) {
    operation_end_time = val;
}

void kernel_timer::set_deps_start_time(uint64_t val) {
    deps_start_time = val;
}

void kernel_timer::set_deps_end_time(uint64_t val) {
    deps_end_time = val;
}

void kernel_timer::set_kernel_submit_time(uint64_t val) {
    kernel_submit_time = val;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

} // namespace ccl
