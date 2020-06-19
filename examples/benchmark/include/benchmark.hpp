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
#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

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

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace cl::sycl::access;
#endif /* CCL_ENABLE_SYCL */

#include "base_utils.hpp"
#include "bfp16.h"
#include "coll.hpp"
#include "sparse_allreduce/sparse_detail.hpp"

/* required declarations */
template<class Dtype>
void create_colls(std::list<std::string>& coll_names, ccl::stream_type backend,
                  coll_list_t& colls);

/* specific benchmark variables */
// TODO: add ccl::bfp16
constexpr std::initializer_list<ccl::datatype> all_dtypes = {ccl::dt_char,
                                                        ccl::dt_int,
                                                        ccl::dt_float,
                                                        ccl::dt_double,
                                                        ccl::dt_int64,
                                                        ccl::dt_uint64};

/* specific benchmark defines */
// different collectives with duplications
#define DEFAULT_COLL_LIST "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
                          "sparse_allreduce,sparse_allreduce_bfp16," \
                          "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
                          "sparse_allreduce,sparse_allreduce_bfp16"

#define DEFAULT_DTYPES_LIST "float"
#define ALL_DTYPES_LIST "char,int,float,double,int64_t,uint64_t"

#define PRINT(fmt, ...)             \
    printf(fmt"\n", ##__VA_ARGS__); \

#define PRINT_BY_ROOT(comm, fmt, ...)   \
    if (comm->rank() == 0)              \
    {                                   \
        printf(fmt"\n", ##__VA_ARGS__); \
    }

#define ASSERT(cond, fmt, ...)                            \
  do                                                      \
  {                                                       \
      if (!(cond))                                        \
      {                                                   \
          printf("FAILED\n");                             \
          fprintf(stderr, "ASSERT '%s' FAILED " fmt "\n", \
                  #cond, ##__VA_ARGS__);                  \
          throw std::runtime_error("ASSERT FAILED");      \
      }                                                   \
  } while (0)

/* specific benchmark dtypes */
typedef enum
{
    LOOP_REGULAR,
    LOOP_UNORDERED
} loop_type_t;

#define DEFAULT_BACKEND ccl::stream_type::host
#define DEFAULT_LOOP    LOOP_REGULAR

std::map<ccl::stream_type, std::string> backend_names =
  {
    std::make_pair(ccl::stream_type::host, "cpu"),
    std::make_pair(ccl::stream_type::gpu, "sycl") /* TODO: align names */
  };

std::map<loop_type_t, std::string> loop_names =
  {
    std::make_pair(LOOP_REGULAR, "regular"),
    std::make_pair(LOOP_UNORDERED, "unordered")
  };

// TODO: add ccl::bfp16
std::map<ccl::datatype, std::string> dtype_names =
  {
    std::make_pair(ccl::datatype::dt_char,   "char"),
    std::make_pair(ccl::datatype::dt_int,    "int"),
    std::make_pair(ccl::datatype::dt_float,  "float"),
    std::make_pair(ccl::datatype::dt_double, "double"),
    std::make_pair(ccl::datatype::dt_int64,  "int64_t"),
    std::make_pair(ccl::datatype::dt_uint64, "uint64_t"),
  };

// variables for setting dtypes to launch benchmark
// TODO: add ccl::bfp16
template<class native_type>
using checked_dtype_t    = std::pair<bool, native_type>;
using supported_dtypes_t = std::tuple< checked_dtype_t<char>,
                                       checked_dtype_t<int>,
                                       checked_dtype_t<float>,
                                       checked_dtype_t<double>,
                                       checked_dtype_t<int64_t>,
                                       checked_dtype_t<uint64_t> >;
supported_dtypes_t launch_dtypes;

/* specific benchmark functors */
class create_colls_func
{
private:
    std::list<std::string>& coll_names;
    ccl::stream_type backend;
    coll_list_t& colls;
public:
    create_colls_func(std::list<std::string>& coll_names, ccl::stream_type backend,
                      coll_list_t& colls)
                      : coll_names(coll_names), backend(backend), colls(colls)
    { }

    template<class Dtype>
    void operator() (const Dtype& value)
    {
        if (true == std::get<0>(value))
        {
            create_colls<typename Dtype::second_type>(coll_names, backend, colls);
        }
    }
};

class set_types_func
{
private:
    const std::list<std::string>& dtypes;
public:
    set_types_func(const std::list<std::string>& dtypes) : dtypes(dtypes)
    { }

    template<class Dtype>
    void operator() (checked_dtype_t<Dtype>& val)
    {
        auto it = std::find(dtypes.begin(), dtypes.end(),
                            ccl::native_type_info<Dtype>::name());
        if (it != std::end(dtypes))
        {
            val.first  = true;
        }
    }
};

/* specific benchmark functions */
void print_help_usage(const char* app)
{
    PRINT("USAGE: %s [OPTIONS]\n\n\t"
          "[-b,--backend <backend>]\n\t"
          "[-e,--loop <execution loop>]\n\t"
          "[-l,--coll <collectives list>]\n\t"
          "[-i,--iters <iteration count>]\n\t"
          "[-p,--buf_count <number of parallel operations within single collective>]\n\t"
          "[-c,--check <check result correctness>]\n\t"
          "[-d,--dtype <datatypes list/all>]\n\t"
          "[-h,--help]\n\n"
          "example:\n\t--coll allgatherv,allreduce,sparse_allreduce,sparse_allreduce_bfp16 --backend cpu --loop regular\n"
          "example:\n\t--coll bcast,reduce --backend sycl --loop unordered \n"
          "\n\n\tThe collectives \"sparse_*\" support additional configuration parameters:\n"
          "\n\t\t\"indices_to_value_ratio\" - to produce indices count not more than 'elem_count/indices_to_value_ratio\n"
          "\t\t\t(default value is 3)\n"
          "\n\t\t\"vdim_count\" - maximum value counts for index\n"
          "\t\t\t(default values determines all elapsed elements after \"indices_to_value_ratio\" recalculation application)\n"
          "\n\tUser can set this additional parameters to sparse collective in the way:\n"
          "\n\t\tsparse_allreduce[4:99]\n"
          "\t\t\t - to set \"indices_to_value_ratio\" in 4 and \"vdim_cout\" in 99\n"
          "\t\tsparse_allreduce[6]\n"
          "\t\t\t - to set \"indices_to_value_ratio\" in 6 and \"vdim_cout\" in default\n"
          "\n\tPlease use default configuration in most cases! You do not need to change it in general benchmark case\n",
          app);
}

double when(void)
{
    struct timeval tv;
    static struct timeval tv_base;
    static int is_first = 1;

    if (gettimeofday(&tv, NULL))
    {
        perror("gettimeofday");
        return 0;
    }

    if (is_first)
    {
        tv_base = tv;
        is_first = 0;
    }
    return (double)(tv.tv_sec - tv_base.tv_sec) * 1.0e6 +
           (double)(tv.tv_usec - tv_base.tv_usec);
}

void print_timings(ccl::communicator& comm,
                  double* timer, size_t elem_count,
                  size_t elem_size, size_t buf_count,
                  size_t rank, size_t size)
{
    double* timers = (double*)malloc(size * sizeof(double));
    size_t* recv_counts = (size_t*)malloc(size * sizeof(size_t));

    size_t idx;
    for (idx = 0; idx < size; idx++)
        recv_counts[idx] = 1;

    ccl::coll_attr attr;
    memset(&attr, 0, sizeof(ccl_coll_attr_t));

    comm.allgatherv(timer,
                    1,
                    timers,
                    recv_counts,
                    &attr,
                    nullptr)->wait();

    if (rank == 0)
    {
        double avg_timer = 0;
        double avg_timer_per_buf = 0;
        for (idx = 0; idx < size; idx++)
        {
            avg_timer += timers[idx];
        }
        avg_timer /= (ITERS * size);
        avg_timer_per_buf = avg_timer / buf_count;

        double stddev_timer = 0;
        double sum = 0;
        for (idx = 0; idx < size; idx++)
        {
            double val = timers[idx] / ITERS;
            sum += (val - avg_timer) * (val - avg_timer);
        }
        stddev_timer = sqrt(sum / size) / avg_timer * 100;
        printf("size %10zu x %5zu bytes, avg %10.2lf us, avg_per_buf %10.2f, stddev %5.1lf %%\n",
                elem_count * elem_size, buf_count, avg_timer, avg_timer_per_buf, stddev_timer);
    }
    comm.barrier();

    free(timers);
    free(recv_counts);
}

std::list<std::string> tokenize(const std::string& input, char delimeter)
{
    std::stringstream ss(input);
    std::list<std::string> ret;
    std::string value;
    while (std::getline(ss, value, delimeter))
    {
        ret.push_back(value);
    }
    return ret;
}

template<class Dtype, class Container>
std::string find_str_val(Container& mp, const Dtype& key)
{
    typename std::map<Dtype, std::string>::iterator it;
    it = mp.find(key);
    if (it != mp.end())
         return it->second;
    return NULL;
}

int check_supported_options(const std::string& option_name, const std::string& option_value,
                            const std::set<std::string>& supported_option_values)
{
    std::stringstream sstream;

    if (supported_option_values.find(option_value) == supported_option_values.end())
    {
        PRINT("unsupported %s: %s", option_name.c_str(), option_value.c_str());

        std::copy(supported_option_values.begin(), supported_option_values.end(),
                  std::ostream_iterator<std::string>(sstream, " "));
        PRINT("supported %s: %s", option_name.c_str(), sstream.str().c_str());
        return -1;
    }

    return 0;
}

int set_backend(const std::string& option_value, ccl::stream_type& backend)
{
    std::string option_name = "backend";
    std::set<std::string> supported_option_values { backend_names[ccl::stream_type::host] };

#ifdef CCL_ENABLE_SYCL
    supported_option_values.insert(backend_names[ccl::stream_type::gpu]);
#endif

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    backend = (option_value == backend_names[ccl::stream_type::gpu]) ?
               ccl::stream_type::gpu : ccl::stream_type::host;

    return 0;
}

int set_loop(const std::string& option_value, loop_type_t& loop)
{
    std::string option_name = "loop";
    std::set<std::string> supported_option_values { loop_names[LOOP_REGULAR], loop_names[LOOP_UNORDERED] };

    if (check_supported_options(option_name, option_value, supported_option_values))
        return -1;

    loop = (option_value == loop_names[LOOP_REGULAR]) ?
        LOOP_REGULAR : LOOP_UNORDERED;

    if (loop == LOOP_UNORDERED)
    {
        setenv("CCL_UNORDERED_COLL", "1", 1);
    }

    return 0;
}

// leave this dtype here because of tokenize() call
typedef struct user_options_t
{
    ccl::stream_type backend;
    loop_type_t loop;
    size_t iters;
    size_t buf_count;
    size_t check_values;
    std::list<std::string> coll_names;
    std::list<std::string> dtypes;

    user_options_t()
    {
        backend = ccl::stream_type::host;
        loop = LOOP_REGULAR;
        coll_names = tokenize(DEFAULT_COLL_LIST, ',');
        iters = ITERS;
        buf_count = BUF_COUNT;
        check_values = 1;
        dtypes = tokenize(DEFAULT_DTYPES_LIST, ','); // default: float
    }
} user_options_t;

int parse_user_options(int& argc, char** (&argv), user_options_t& options)
{
    int ch;
    int errors = 0;

    // values needed by getopt
    const char* const short_options = "b:e:i:p:c:l:d:h:";
    struct option getopt_options[] =
    {
        { "backend",      required_argument, 0, 'b' },
        { "loop",         required_argument, 0, 'e' },
        { "iters",        required_argument, 0, 'i' },
        { "buf_count",    required_argument, 0, 'p' },
        { "check",        required_argument, 0, 'c' },
        { "coll",         required_argument, 0, 'l' },
        { "dtype",        required_argument, 0, 'd' },
        { "help",         no_argument,       0, 'h' },
        {  0,             0,                 0,  0 } // required at end of array.
    };

    while ((ch = getopt_long(argc, argv, short_options,
                             getopt_options, NULL)) != -1)
    {
        switch (ch)
        {
            case 'b':
                if (set_backend(optarg, options.backend))
                    errors++;
                break;
            case 'e':
                if (set_loop(optarg, options.loop))
                    errors++;
                break;
            case 'i':
                options.iters = atoi(optarg);
                break;
            case 'p':
                options.buf_count = atoi(optarg);
                break;
            case 'c':
                options.check_values = atoi(optarg);
                break;
            case 'l':
                options.coll_names = tokenize(optarg, ',');
                break;
            case 'd':
                if (strcmp("all", optarg) == 0)
                {
                    options.dtypes = tokenize(ALL_DTYPES_LIST, ',');
                }
                else
                    options.dtypes = tokenize(optarg, ',');
                break;
            case 'h':
                print_help_usage(argv[0]);
                return -1;
            default:
                errors++;
                break;
        }
    }

    if (optind < argc)
    {
        PRINT("non-option ARGV-elements given");
        errors++;
    }

    if (errors > 0)
    {
        PRINT("failed to parse user options, errors %d", errors);
        print_help_usage(argv[0]);
        return -1;
    }

    return 0;
}

void print_user_options(const user_options_t& options, ccl::communicator* comm)
{
    std::stringstream ss;
    ss << "colls:     ";
    std::copy(options.coll_names.begin(), options.coll_names.end(),
              std::ostream_iterator<std::string>(ss, " "));
    ss << "\n  dtypes:    ";
    std::copy(options.dtypes.begin(), options.dtypes.end(),
              std::ostream_iterator<std::string>(ss, " "));

    std::string backend_str = find_str_val(backend_names, options.backend);
    std::string loop_str = find_str_val(loop_names, options.loop);

    PRINT_BY_ROOT(comm,
                  "options:"
                  "\n  ranks:     %zu"
                  "\n  backend:   %s"
                  "\n  loop:      %s"
                  "\n  iters:     %zu"
                  "\n  buf_count: %zu"
                  "\n  check:     %zu"
                  "\n  %s",
                  comm->size(),
                  backend_str.c_str(),
                  loop_str.c_str(),
                  options.iters,
                  options.buf_count,
                  options.check_values,
                  ss.str().c_str());
}

#endif /* BENCHMARK_HPP */
