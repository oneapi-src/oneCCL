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

#ifdef CCL_ENABLE_MPI

#include <mpi.h>

#include "atl/atl_def.h"
#include "comp/bf16/bf16_intrisics.hpp"
#include "comp/fp16/fp16_intrisics.hpp"

#ifdef CCL_BF16_COMPILER
#define ATL_MPI_BF16
#endif // CCL_BF16_COMPILER

#ifdef CCL_FP16_COMPILER
#define ATL_MPI_FP16
#endif // CCL_FP16_COMPILER

class atl_mpi_global_data {
public:
    typedef enum { ATL_MPI_LIB_IMPI, ATL_MPI_LIB_MPICH, ATL_MPI_LIB_NONE } atl_mpi_lib_type_t;

private:
    typedef struct {
        atl_mpi_lib_type_t type;
        int hmem;
    } atl_mpi_lib_attr_t;
    typedef struct {
        // custom MPI operations for BF16
        MPI_Op sum_op;
        MPI_Op prod_op;
        MPI_Op min_op;
        MPI_Op max_op;
        // custom MPI dtype for BF16
        MPI_Datatype dtype;
    } atl_mpi_bf16_data_t;

    typedef struct {
        // custom MPI operations for FP16
        MPI_Op sum_op;
        MPI_Op prod_op;
        MPI_Op min_op;
        MPI_Op max_op;
        // custom MPI dtype for FP16
        MPI_Datatype dtype;
    } atl_mpi_fp16_data_t;

    typedef struct {
        atl_mpi_lib_type_t type;
        const char* name;

        /* string prefix before numerical version of library, mandatory */
        const char* version_prefix_1;

        /* string prefix before numerical version of library, following prefix_1, optional */
        const char* version_prefix_2;

        /* minimal expected version of library, mandatory */
        int min_version_value;

        /* minimal expected version of library with hmem support, mandatory */
        int min_hmem_version_value;

        /* string prefix before library kind, optional */
        const char* kind_prefix;

        /* library kind, optional */
        const char* kind_value;
    } atl_mpi_lib_info_t;

#define MPI_LIB_INFO_MAX_COUNT 3

    const atl_mpi_lib_info_t mpi_lib_infos[MPI_LIB_INFO_MAX_COUNT] = {
        { ATL_MPI_LIB_IMPI,
          "impi",
          "Intel(R) MPI Library",
          NULL,
          2019,
          2021,
          "library kind:",
          "release" },
        { ATL_MPI_LIB_MPICH, "mpich", "MPICH Custom Information:", "drop", 34, -1, NULL, NULL },
        { ATL_MPI_LIB_NONE, "none", "", NULL, 0, -1, NULL, NULL },
    };

    size_t get_nic_count(const char* nic_count_key);

public:
    const char* EP_IDX_KEY = "vci";

    const char* NIC_IDX_KEY = "multi_nic_pref_nic";
    const char* GLOBAL_NIC_COUNT_KEY = "num_nics";
    const char* LOCAL_NIC_COUNT_KEY = "num_close_nics";

    int is_external_init;
    size_t ctx_count;
    int extra_ep;
    atl_mnic_t mnic_type;
    size_t mnic_count;
    atl_mnic_offset_t mnic_offset;
    atl_mpi_lib_attr_t mpi_lib_attr;
    atl_mpi_bf16_data_t bf16;
    atl_mpi_fp16_data_t fp16;

    atl_mpi_global_data()
            : is_external_init(0),
              ctx_count(0),
              extra_ep(0),
              mnic_type(ATL_MNIC_NONE),
              mnic_count(1),
              mnic_offset(ATL_MNIC_OFFSET_NONE) {
        mpi_lib_attr.type = ATL_MPI_LIB_NONE;
        mpi_lib_attr.hmem = 0;

        bf16.dtype = MPI_DATATYPE_NULL;
        bf16.sum_op = MPI_OP_NULL;
        bf16.prod_op = MPI_OP_NULL;
        bf16.min_op = MPI_OP_NULL;
        bf16.max_op = MPI_OP_NULL;

        fp16.dtype = MPI_DATATYPE_NULL;
        fp16.sum_op = MPI_OP_NULL;
        fp16.prod_op = MPI_OP_NULL;
        fp16.min_op = MPI_OP_NULL;
        fp16.max_op = MPI_OP_NULL;
    }
    atl_mpi_lib_attr_t get_lib_attr();
    size_t get_ep_count(const atl_attr_t& attr);
    atl_status_t set_impi_env(const atl_attr_t& attr, const atl_mpi_lib_attr_t& lib_attr);
    int bf16_init();
    void bf16_finalize();
    int fp16_init();
    void fp16_finalize();
    void print_error(int error);
    atl_status_t check_impi_env(const atl_attr_t& attr);
    atl_status_t update_global_data(atl_attr_t* attr);
    atl_status_t set_mpich_env(const atl_attr_t& attr);
    atl_status_t set_base_env(const atl_attr_t& attr);
    atl_status_t check_mpich_env(const atl_attr_t& attr);
    atl_status_t set_env(const atl_attr_t& attr);
    MPI_Op atl2mpi_op_fp16(atl_reduction_t rtype);
    MPI_Op atl2mpi_op_bf16(atl_reduction_t rtype);
    void print_log_info();
};

#endif
