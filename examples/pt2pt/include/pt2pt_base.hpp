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

#include <getopt.h>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "types.hpp"

typedef struct user_options_t {
    uint32_t cache;
    uint32_t iters;

    std::vector<int> peers;
    uint32_t queue;
    int min_elem_count;
    int max_elem_count;
    validate_values_t validate;
    uint32_t warmup_iters;
    uint32_t wait;
    int window_size;

    user_options_t() {
        iters = DEFAULT_ITERS;
        warmup_iters = DEFAULT_WARMUP_ITERS;
        cache = DEFAULT_CACHE_OPS;
        queue = DEFAULT_QUEUE;
        wait = DEFAULT_WAIT;
        min_elem_count = DEFAULT_MIN_ELEM_COUNT;
        max_elem_count = DEFAULT_MAX_ELEM_COUNT;
        validate = DEFAULT_VALIDATE;
        // for bw benchmark
        window_size = DEFAULT_WINDOW_SIZE;

        peers.reserve(2);
        // filling out with the default values
        peers.push_back(0);
        peers.push_back(1);
    }
} user_options_t;

int parse_user_options(int& argc, char**(&argv), user_options_t& options) {
    int ch;
    int errors = 0;

    char short_options[1024] = { 0 };
    const char* base_options = "i:w:c:q:s:f:t:v:m:h";
    memcpy(short_options, base_options, strlen(base_options));

    struct option getopt_options[] = {
        { "iters", required_argument, nullptr, 'i' },
        { "warmup_iters", required_argument, nullptr, 'w' },
        { "cache", required_argument, nullptr, 'c' },
        { "queue", required_argument, nullptr, 'q' },
        { "wait", required_argument, nullptr, 's' },
        { "min_elem_count", required_argument, nullptr, 'f' },
        { "max_elem_count", required_argument, nullptr, 't' },
        { "validate", required_argument, nullptr, 'v' },
        { "window", required_argument, nullptr, 'm' },
        { "help", no_argument, nullptr, 'h' },
        //TODO: { "peers", required_argument, nullptr, 'p' },
        { nullptr, 0, nullptr, 0 } // required at end of array.
    };

    while ((ch = getopt_long(argc, argv, short_options, getopt_options, nullptr)) != -1) {
        switch (ch) {
            case 'i':
                if (is_valid_integer_option(optarg)) {
                    options.iters = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'w':
                if (is_valid_integer_option(optarg)) {
                    options.warmup_iters = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'c':
                if (is_valid_integer_option(optarg)) {
                    options.cache = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'q':
                if (is_valid_integer_option(optarg)) {
                    options.queue = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'f':
                if (is_valid_integer_option(optarg)) {
                    options.min_elem_count = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 't':
                if (is_valid_integer_option(optarg)) {
                    options.max_elem_count = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 's':
                if (is_valid_integer_option(optarg)) {
                    options.wait = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'v':
                if (set_validate_values(optarg, options.validate)) {
                    PRINT("failed to parse 'check' option");
                    errors++;
                }
                break;
            case 'm':
                if (is_valid_integer_option(optarg)) {
                    options.window_size = atoll(optarg);
                }
                else
                    errors++;
                break;
            case 'h': return INVALID_RETURN;
            default:
                PRINT("failed to parse unknown option");
                errors++;
                break;
        }
    }

    if (optind < argc) {
        PRINT("non-option ARGV-elements given");
        errors++;
    }

    if (errors > 0) {
        PRINT("found %d errors while parsing user options", errors);
        for (int idx = 0; idx < argc; idx++) {
            PRINT("arg %d: %s", idx, argv[idx]);
        }
        return -1;
    }
    return 0;
}

auto create_attr(const bool is_cache, const int count, const std::string& match_id_suffix) {
    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    if (is_cache) {
        std::string matchId = "_len_" + std::to_string(count) + match_id_suffix;
        attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(matchId));
        attr.set<ccl::operation_attr_id::to_cache>(true);
    }
    return attr;
}

void print_timings(ccl::communicator& comm,
                   const user_options_t& options,
                   const double total_time,
                   const int count,
                   const std::string mesure_str) {
    static bool print_once = false;

    if (!print_once && comm.rank() == 0) {
        std::stringstream ss;
        ss << std::right << std::setw(COL_WIDTH - 4) << "#bytes" << std::setw(COL_WIDTH)
           << "#repetitions" << std::setw(COL_WIDTH) << mesure_str << std::endl;
        std::cout << ss.str();
        print_once = true;
    }

    std::stringstream ss;

    ss << std::right << std::fixed << std::setw(COL_WIDTH - 4) << count << std::setw(COL_WIDTH)
       << options.iters << std::setw(COL_WIDTH) << std::setprecision(COL_PRECISION) << total_time
       << std::setw(COL_WIDTH) << std::endl;
    std::cout << ss.str();
}

template <class Dtype>
void check_buffers(sycl::queue q,
                   const user_options_t& options,
                   const int count,
                   const size_t iter_idx,
                   Dtype buf_recv) {
    bool failed = false;
    sycl::buffer<int> check_buf(count);

    auto e = q.submit([&](auto& h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            if (buf_recv[id] != static_cast<int>(id + iter_idx)) {
                check_buf_acc[id] = INVALID_VALUE;
            }
        });
    });

    if (options.wait) {
        e.wait_and_throw();
    }

    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        for (int j = 0; j < count; j++) {
            if (check_buf_acc[j] == INVALID_VALUE) {
                failed = true;
                break;
            }
        }
    }

    if (failed) {
        std::cout << "FAILED: iter_idx: " << iter_idx << ", count: " << count << std::endl;
        ASSERT(0, "unexpected value");
    }
}

void print_help_usage(const char* app) {
    PRINT("\nUSAGE:\n"
          "\t%s [OPTIONS]\n\n"
          "OPTIONS:\n"
          "\t[-i,--iters <iteration count>]: %d\n"
          "\t[-w,--warmup_iters <warm up iteration count>]: %d\n"
          "\t[-c,--cache <use persistent operations>]: %d\n"
          "\t[-q,--queue <sycl queue mode in/out order>]: %d\n"
          "\t[-s,--wait <enable synchronization on sycl and pt2pt level>]: %d\n"
          "\t[-f,--min_elem_count <minimum element count>]: %d\n"
          "\t[-t,--max_elem_count <maximum element count>]: %d\n"
          "\t[-v,--validate <validate result correctness>]: %s\n"
          "\t[-h,--help]\n\n"
          "example:\n\t--queue 1 --cache 0 --validate 1\n",
          app,
          DEFAULT_ITERS,
          DEFAULT_WARMUP_ITERS,
          DEFAULT_CACHE_OPS,
          DEFAULT_QUEUE,
          DEFAULT_WAIT,
          DEFAULT_MIN_ELEM_COUNT,
          DEFAULT_MAX_ELEM_COUNT,
          validate_values_names[DEFAULT_VALIDATE].c_str());
}

void print_user_options(const std::string benchmark,
                        const user_options_t& options,
                        const ccl::communicator& comm) {
    std::stringstream ss;

    std::string validate_values_str = find_str_val(validate_values_names, options.validate);

    ss << "\noptions:"
       << "\n  iters:          " << options.iters << "\n  warmup_iters:   " << options.warmup_iters
       << "\n  cache:          " << options.cache << "\n  queue:          " << options.queue
       << "\n  wait:           " << options.wait << "\n  min_elem_count: " << options.min_elem_count
       << "\n  max_elem_count: " << options.max_elem_count
       << "\n  validate:       " << validate_values_str;

    if (benchmark == "Bandwidth") {
        ss << "\n  window_size:    " << options.window_size;
    }

    if (comm.rank() == 0) {
        std::cout << ss.str() << std::endl;
    }

    ss.str("");

    ss << "#------------------------------------------------------------\n"
       << "# Benchmarking: " << benchmark << "\n"
       << "# #processes: " << comm.size() << "\n"
       << "#------------------------------------------------------------\n";

    if (comm.rank() == 0) {
        std::cout << ss.str() << std::endl;
    }
}
