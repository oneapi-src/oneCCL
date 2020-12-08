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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <getopt.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <fstream>

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace cl::sycl::access;
#endif /* CCL_ENABLE_SYCL */

#include "base.hpp"
#include "base_utils.hpp"
#include "bf16.hpp"
#include "coll.hpp"
#include "sparse_allreduce/sparse_detail.hpp"

void print_help_usage(const char* app) {
    PRINT("\nUSAGE:\n"
          "\t%s [OPTIONS]\n\n"
          "OPTIONS:\n"
          "\t[-b,--backend <backend>]: %s\n"
          "\t[-e,--loop <execution loop>]: %s\n"
          "\t[-i,--iters <iteration count>]: %d\n"
          "\t[-w,--warmup_iters <warm up iteration count>]: %d\n"
          "\t[-p,--buf_count <number of parallel operations within single collective>]: %d\n"
          "\t[-f,--min_elem_count <minimum number of elements for single collective>]: %d\n"
          "\t[-t,--max_elem_count <maximum number of elements for single collective>]: %d\n"
          "\t[-c,--check <check result correctness>]: %d\n"
          "\t[-a,--sycl_dev_type <sycl device type>]: %s\n"
          "\t[-m,--sycl_mem_type <sycl memory type>]: %s\n"
          "\t[-u,--sycl_usm_type <sycl usm type>]: %s\n"
          "\t[-k,--ranks_per_proc <rank count per process>]: %d\n"
          "\t[-l,--coll <collectives list/all>]: %s\n"
          "\t[-d,--dtype <datatypes list/all>]: %s\n"
          "\t[-r,--reduction <reductions list/all>]: %s\n"
          "\t[-o,--csv_filepath <file to store CSV-formatted data into>]: %s\n"
          "\t[-h,--help]\n\n"
          "example:\n\t--coll allgatherv,allreduce --backend host --loop regular\n"
          "example:\n\t--coll bcast,reduce --backend sycl --loop unordered \n",
          app,
          backend_names[DEFAULT_BACKEND].c_str(),
          loop_names[DEFAULT_LOOP].c_str(),
          DEFAULT_ITERS,
          DEFAULT_WARMUP_ITERS,
          DEFAULT_BUF_COUNT,
          DEFAULT_MIN_ELEM_COUNT,
          DEFAULT_MAX_ELEM_COUNT,
          DEFAULT_CHECK_VALUES,
          sycl_dev_names[DEFAULT_SYCL_DEV_TYPE].c_str(),
          sycl_mem_names[DEFAULT_SYCL_MEM_TYPE].c_str(),
          sycl_usm_names[DEFAULT_SYCL_USM_TYPE].c_str(),
          DEFAULT_RANKS_PER_PROC,
          DEFAULT_COLL_LIST,
          DEFAULT_DTYPES_LIST,
          DEFAULT_REDUCTIONS_LIST,
          DEFAULT_CSV_FILEPATH);
}

template <class Dtype, class Container>
std::string find_str_val(Container& mp, const Dtype& key) {
    typename std::map<Dtype, std::string>::iterator it;
    it = mp.find(key);
    if (it != mp.end())
        return it->second;
    return NULL;
}

template <class Dtype, class Container>
bool find_key_val(ccl::reduction& key, Container& mp, const Dtype& val) {
    for (auto& i : mp) {
        if (i.second == val) {
            key = i.first;
            return true;
        }
    }
    return false;
}

int check_supported_options(const std::string& option_name,
                            const std::string& option_value,
                            const std::set<std::string>& supported_option_values) {
    std::stringstream sstream;

    if (supported_option_values.find(option_value) == supported_option_values.end()) {
        PRINT("unsupported %s: %s", option_name.c_str(), option_value.c_str());

        std::copy(supported_option_values.begin(),
                  supported_option_values.end(),
                  std::ostream_iterator<std::string>(sstream, " "));
        PRINT("supported %s: %s", option_name.c_str(), sstream.str().c_str());
        return -1;
    }

    return 0;
}

int set_backend(const std::string& option_value, backend_type_t& backend) {
    std::string option_name = "backend";
    std::set<std::string> supported_option_values{ backend_names[BACKEND_HOST] };

#ifdef CCL_ENABLE_SYCL
    supported_option_values.insert(backend_names[BACKEND_SYCL]);
#endif

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    backend = (option_value == backend_names[BACKEND_SYCL]) ? BACKEND_SYCL : BACKEND_HOST;

    return 0;
}

int set_loop(const std::string& option_value, loop_type_t& loop) {
    std::string option_name = "loop";
    std::set<std::string> supported_option_values{ loop_names[LOOP_REGULAR],
                                                   loop_names[LOOP_UNORDERED] };

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    loop = (option_value == loop_names[LOOP_REGULAR]) ? LOOP_REGULAR : LOOP_UNORDERED;

    if (loop == LOOP_UNORDERED) {
        setenv("CCL_UNORDERED_COLL", "1", 1);
    }

    return 0;
}

int set_sycl_dev_type(const std::string& option_value, sycl_dev_type_t& dev) {
    std::string option_name = "sycl_dev_type";
    std::set<std::string> supported_option_values{ sycl_dev_names[SYCL_DEV_HOST],
                                                   sycl_dev_names[SYCL_DEV_CPU],
                                                   sycl_dev_names[SYCL_DEV_GPU] };

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    if (option_value == sycl_dev_names[SYCL_DEV_HOST])
        dev = SYCL_DEV_HOST;
    else if (option_value == sycl_dev_names[SYCL_DEV_CPU])
        dev = SYCL_DEV_CPU;
    else if (option_value == sycl_dev_names[SYCL_DEV_GPU])
        dev = SYCL_DEV_GPU;

    return 0;
}

int set_sycl_mem_type(const std::string& option_value, sycl_mem_type_t& mem) {
    std::string option_name = "sycl_mem_type";
    std::set<std::string> supported_option_values{ sycl_mem_names[SYCL_MEM_USM],
                                                   sycl_mem_names[SYCL_MEM_BUF] };

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    mem = (option_value == sycl_mem_names[SYCL_MEM_USM]) ? SYCL_MEM_USM : SYCL_MEM_BUF;

    return 0;
}

int set_sycl_usm_type(const std::string& option_value, sycl_usm_type_t& usm) {
    std::string option_name = "sycl_usm_type";
    std::set<std::string> supported_option_values{ sycl_usm_names[SYCL_USM_SHARED],
                                                   sycl_usm_names[SYCL_USM_DEVICE] };

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    usm = (option_value == sycl_usm_names[SYCL_USM_SHARED]) ? SYCL_USM_SHARED : SYCL_USM_DEVICE;

    return 0;
}

size_t get_iter_count(size_t bytes, size_t max_iter_count) {
    size_t n, res = max_iter_count;
    n = bytes >> 18;
    while (n) {
        res >>= 1;
        n >>= 1;
    }

    if (!res && max_iter_count)
        res = 1;

    return res;
}

/* timer array contains one number per collective, one collective corresponds to rank_per_proc ranks */
void print_timings(ccl::communicator& comm,
                   const std::vector<double>& local_timers,
                   const user_options_t& options,
                   const size_t elem_count,
                   const size_t iter_count,
                   ccl::datatype dtype,
                   ccl::reduction op) {
    const size_t buf_count = options.buf_count;
    const size_t ncolls = options.coll_names.size();
    std::vector<double> all_timers(ncolls * comm.size());
    std::vector<size_t> recv_counts(comm.size());

    int idx;
    for (idx = 0; idx < comm.size(); idx++)
        recv_counts[idx] = ncolls;

    ccl::allgatherv(local_timers.data(), ncolls, all_timers.data(), recv_counts, comm).wait();

    if (comm.rank() == 0) {
        std::vector<double> timers(comm.size(), 0);
        for (int r = 0; r < comm.size(); ++r) {
            for (size_t c = 0; c < ncolls; ++c) {
                timers[r] += all_timers[r * ncolls + c];
            }
        }

        double avg_timer(0);
        double avg_timer_per_buf(0);
        for (idx = 0; idx < comm.size(); idx++) {
            avg_timer += timers[idx];
        }
        avg_timer /= (iter_count * comm.size());
        avg_timer_per_buf = avg_timer / buf_count;

        double stddev_timer = 0;
        double sum = 0;
        for (idx = 0; idx < comm.size(); idx++) {
            double val = timers[idx] / iter_count;
            sum += (val - avg_timer) * (val - avg_timer);
        }

        stddev_timer = sqrt(sum / comm.size()) / avg_timer * 100;
        if (buf_count == 1) {
            printf("%10zu %12.2lf %11.1lf\n",
                   elem_count * ccl::get_datatype_size(dtype) * buf_count,
                   avg_timer,
                   stddev_timer);
        }
        else {
            printf("%10zu %13.2lf %18.2lf %11.1lf\n",
                   elem_count * ccl::get_datatype_size(dtype) * buf_count,
                   avg_timer,
                   avg_timer_per_buf,
                   stddev_timer);
        }

        // in case csv export is requested
        // we write one line per collop, dtype and reduction
        // hence average is per collop, not the aggregate over all
        if (!options.csv_filepath.empty()) {
            std::ofstream csvf;
            csvf.open(options.csv_filepath, std::ios::app);

            if (csvf.is_open()) {
                std::vector<double> avg_timer(ncolls, 0);

                for (int r = 0; r < comm.size(); ++r) {
                    for (size_t c = 0; c < ncolls; ++c) {
                        avg_timer[c] += all_timers[r * ncolls + c];
                    }
                }

                for (size_t c = 0; c < ncolls; ++c) {
                    avg_timer[c] /= (iter_count * comm.size());
                }

                int i = 0;
                for (auto cop = options.coll_names.begin(); cop != options.coll_names.end();
                     ++cop, ++i) {
                    csvf << comm.size() << "," << (*cop) << "," << reduction_names[op] << ","
                         << dtype_names[dtype] << "," << ccl::get_datatype_size(dtype) << ","
                         << elem_count << "," << buf_count << "," << avg_timer[i] << std::endl;
                }
                csvf.close();
            }
        }
    }

    ccl::barrier(comm);
}

int parse_user_options(int& argc, char**(&argv), user_options_t& options) {
    int ch;
    int errors = 0;

    const char* const short_options = "b:e:i:w:p:f:t:c:v:o:a:m:u:k:l:d:r:h";

    struct option getopt_options[] = {
        { "backend", required_argument, 0, 'b' },
        { "loop", required_argument, 0, 'e' },
        { "iters", required_argument, 0, 'i' },
        { "warmup_iters", required_argument, 0, 'w' },
        { "buf_count", required_argument, 0, 'p' },
        { "min_elem_count", required_argument, 0, 'f' },
        { "max_elem_count", required_argument, 0, 't' },
        { "check", required_argument, 0, 'c' },
        { "v2i_ratio", required_argument, 0, 'v' },
        { "sycl_dev_type", required_argument, 0, 'a' },
        { "sycl_mem_type", required_argument, 0, 'm' },
        { "sycl_usm_type", required_argument, 0, 'u' },
        { "ranks", required_argument, 0, 'k' },
        { "coll", required_argument, 0, 'l' },
        { "dtype", required_argument, 0, 'd' },
        { "reduction", required_argument, 0, 'r' },
        { "csv_filepath", required_argument, 0, 'o' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 } // required at end of array.
    };

    while ((ch = getopt_long(argc, argv, short_options, getopt_options, NULL)) != -1) {
        switch (ch) {
            case 'b':
                if (set_backend(optarg, options.backend)) {
                    PRINT("failed to parse 'backend' option");
                    errors++;
                }
                break;
            case 'e':
                if (set_loop(optarg, options.loop)) {
                    PRINT("failed to parse 'loop' option");
                    errors++;
                }
                break;
            case 'i': options.iters = atoll(optarg); break;
            case 'w': options.warmup_iters = atoll(optarg); break;
            case 'p': options.buf_count = atoll(optarg); break;
            case 'f': options.min_elem_count = atoll(optarg); break;
            case 't': options.max_elem_count = atoll(optarg); break;
            case 'c': options.check_values = atoi(optarg); break;
            case 'v': options.v2i_ratio = atoll(optarg); break;
            case 'a':
                if (set_sycl_dev_type(optarg, options.sycl_dev_type)) {
                    PRINT("failed to parse 'sycl_dev_type' option");
                    errors++;
                }
                break;
            case 'm':
                if (set_sycl_mem_type(optarg, options.sycl_mem_type)) {
                    PRINT("failed to parse 'sycl_mem_type' option");
                    errors++;
                }
                break;
            case 'u':
                if (set_sycl_usm_type(optarg, options.sycl_usm_type)) {
                    PRINT("failed to parse 'sycl_usm_type' option");
                    errors++;
                }
                break;
            case 'k': options.ranks_per_proc = atoll(optarg); break;
            case 'l':
                if (strcmp("all", optarg) == 0) {
                    options.coll_names = tokenize(ALL_COLLS_LIST, ',');
                }
                else
                    options.coll_names = tokenize(optarg, ',');
                break;
            case 'd':
                if (strcmp("all", optarg) == 0) {
                    options.dtypes = tokenize(ALL_DTYPES_LIST, ',');
                }
                else
                    options.dtypes = tokenize(optarg, ',');
                break;
            case 'r':
                if (strcmp("all", optarg) == 0) {
                    options.reductions = tokenize(ALL_REDUCTIONS_LIST, ',');
                }
                else
                    options.reductions = tokenize(optarg, ',');
                break;
            case 'o': options.csv_filepath = std::string(optarg); break;
            case 'h': return -1;
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

    /* adjust user options */
    if (!options.min_elem_count)
        options.min_elem_count = 1;

    if (options.max_elem_count < options.min_elem_count)
        options.max_elem_count = options.min_elem_count;

    return 0;
}

void print_user_options(const user_options_t& options, const ccl::communicator& comm) {
    std::stringstream ss;
    ss << "colls:          ";
    std::copy(options.coll_names.begin(),
              options.coll_names.end(),
              std::ostream_iterator<std::string>(ss, " "));
    ss << "\n  dtypes:         ";
    std::copy(
        options.dtypes.begin(), options.dtypes.end(), std::ostream_iterator<std::string>(ss, " "));
    ss << "\n  reductions:     ";
    std::copy(options.reductions.begin(),
              options.reductions.end(),
              std::ostream_iterator<std::string>(ss, " "));

    std::string backend_str = find_str_val(backend_names, options.backend);
    std::string loop_str = find_str_val(loop_names, options.loop);

    std::string sycl_dev_type_str = find_str_val(sycl_dev_names, options.sycl_dev_type);
    std::string sycl_mem_type_str = find_str_val(sycl_mem_names, options.sycl_mem_type);
    std::string sycl_usm_type_str = find_str_val(sycl_usm_names, options.sycl_usm_type);

    PRINT_BY_ROOT(comm,
                  "options:"
                  "\n  processes:      %d"
                  "\n  backend:        %s"
                  "\n  loop:           %s"
                  "\n  iters:          %zu"
                  "\n  warmup_iters:   %zu"
                  "\n  buf_count:      %zu"
                  "\n  min_elem_count: %zu"
                  "\n  max_elem_count: %zu"
                  "\n  check:          %d"
                  "\n  v2i_ratio:      %zu"
                  "\n  sycl_dev_type:  %s"
                  "\n  sycl_mem_type:  %s"
                  "\n  sycl_usm_type:  %s"
                  "\n  ranks_per_proc: %zu"
                  "\n  %s"
                  "\n  csv_filepath:   %s",
                  comm.size(),
                  backend_str.c_str(),
                  loop_str.c_str(),
                  options.iters,
                  options.warmup_iters,
                  options.buf_count,
                  options.min_elem_count,
                  options.max_elem_count,
                  options.check_values,
                  options.v2i_ratio,
                  sycl_dev_type_str.c_str(),
                  sycl_mem_type_str.c_str(),
                  sycl_usm_type_str.c_str(),
                  options.ranks_per_proc,
                  ss.str().c_str(),
                  options.csv_filepath.c_str());
}
