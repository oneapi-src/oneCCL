#include "common.h"
#include "shared.h"

__kernel void empty_kernel(int my_rank,
                           int comm_size,
                           ulong count,
                           const __global void* input_buffer,
                           __global void* output_buffer,
                           const __global void* right_input_buffer,
                           __global void* right_output_buffer) {
    return;
}

// Name - unique name suffix for the kernel
// T - type parameter(e.g. float, int4, etc)
// VecSize - vector size of the type. E.g. if float4 is used, VecSize is 4. Note: if just float is used,
// the value must be one as it's used for division inside the kernel.
// Op - A operation parameter(e.g. add(x, y))
// OpName - Operator name which goes to the kernel name, e.g. OpName = add, Op = __add_int(actual function)
#define DEFINE_ALLREDUCE_KERNEL(Name, T, VecSize, Op, OpName) \
    __kernel void allreduce_kernel_##Name##_##OpName(int my_rank, \
                                                     int comm_size, \
                                                     ulong count, \
                                                     const __global T* input_buffer, \
                                                     __global T* output_buffer, \
                                                     const __global T* right_input_buffer, \
                                                     __global T* right_output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
\
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            output_buffer[idx] = Op(input_buffer[idx], right_input_buffer[idx]); \
            right_output_buffer[idx] = output_buffer[idx]; \
        } \
    }

#define DEFINE_REDUCE_LOCAL_OUTOFPLACE_KERNEL(Name, T, VecSize, Op, OpName) \
    __kernel void reduce_local_outofplace_kernel_##Name##_##OpName( \
        int my_rank, \
        int comm_size, \
        ulong count, \
        const __global T* input_buffer_1, \
        const __global T* input_buffer_2, \
        __global T* output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
\
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            output_buffer[idx] = Op(input_buffer_1[idx], input_buffer_2[idx]); \
        } \
    }

#define DEFINE_REDUCE_LOCAL_INPLACE_KERNEL(Name, T, VecSize, Op, OpName) \
    __kernel void reduce_local_inplace_kernel_##Name##_##OpName( \
        ulong count, const __global T* input_buffer, __global T* inoutput_buffer) { \
        DEBUG_BLOCK(/* int sg_id = get_sub_group_id(); */ \
                    printf("in reduce_local_inplace_kernel_\n")); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
\
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            inoutput_buffer[idx] = Op(input_buffer[idx], inoutput_buffer[idx]); \
        } \
    }

// Define kernels for a specific operation for all the supported types.
// Note: for op function we use convention __<OpName>_<type>, where type is the actual type(e.g. int4, float)
// FIXME: Temporary use scalar types instead of vector ones. This is a workaround for issues in case when
// elems_count % VecSize != 0. Need to find a proper fix with a good performance.
#define VEC_SIZE RING_ALLREDUCE_VEC_SIZE

#define DEFINE_KERNELS_WITH_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(int8, char, VEC_SIZE, __##OpName##_##char, OpName) \
        DEFINE_##KernelName##_KERNEL(uint8, uchar, VEC_SIZE, __##OpName##_##uchar, OpName) \
\
            DEFINE_##KernelName##_KERNEL(int16, short, VEC_SIZE, __##OpName##_##short, OpName) \
                DEFINE_##KernelName##_KERNEL( \
                    uint16, ushort, VEC_SIZE, __##OpName##_##ushort, OpName) \
\
                    DEFINE_##KernelName##_KERNEL(int32, int, VEC_SIZE, __##OpName##_##int, OpName) \
                        DEFINE_##KernelName##_KERNEL( \
                            uint32, uint, VEC_SIZE, __##OpName##_##uint, OpName) \
\
                            DEFINE_##KernelName##_KERNEL( \
                                int64, long, VEC_SIZE, __##OpName##_##long, OpName) \
                                DEFINE_##KernelName##_KERNEL( \
                                    uint64, ulong, VEC_SIZE, __##OpName##_##ulong, OpName) \
\
                                    DEFINE_##KernelName##_KERNEL( \
                                        float32, float, VEC_SIZE, __##OpName##_##float, OpName) \
                                        DEFINE_##KernelName##_KERNEL(float64, \
                                                                     double, \
                                                                     VEC_SIZE, \
                                                                     __##OpName##_##double, \
                                                                     OpName)

#define DEFINE_KERNELS_WITH_LP_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(bfloat16, ushort, VEC_SIZE, __bf16_##OpName##_##ushort, OpName) \
        DEFINE_##KernelName##_KERNEL(float16, half, VEC_SIZE, __##OpName##_##half, OpName)

#define DEFINE_OPS(T) \
    DEFINE_SUM_OP(T) \
    DEFINE_PROD_OP(T) \
    DEFINE_MIN_OP(T) \
    DEFINE_MAX_OP(T)

#define DEFINE_BF16OPS(T) \
    DEFINE_BF16SUM_OP(T) \
    DEFINE_BF16PROD_OP(T) \
    DEFINE_BF16MIN_OP(T) \
    DEFINE_BF16MAX_OP(T)

#define DEFINE_FP16OPS(T) \
    DEFINE_FP16SUM_OP(T) \
    DEFINE_FP16PROD_OP(T) \
    DEFINE_FP16MIN_OP(T) \
    DEFINE_FP16MAX_OP(T)

// Define Op function for each supported type(use vector types for some of them as required by the kernel)
DEFINE_OPS(char)
DEFINE_OPS(uchar)

DEFINE_OPS(short)
DEFINE_OPS(ushort)

DEFINE_OPS(int)
DEFINE_OPS(uint)

DEFINE_OPS(long)
DEFINE_OPS(ulong)

DEFINE_OPS(float)
DEFINE_OPS(double)

DEFINE_BF16OPS(ushort)
DEFINE_FP16OPS(half)

// Define the actual kernels
#define DEFINE_ALL_KERNELS(KernelName) \
    DEFINE_KERNELS_WITH_OP(KernelName, sum) \
    DEFINE_KERNELS_WITH_OP(KernelName, prod) \
    DEFINE_KERNELS_WITH_OP(KernelName, min) \
    DEFINE_KERNELS_WITH_OP(KernelName, max) \
\
    DEFINE_KERNELS_WITH_LP_OP(KernelName, sum) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, prod) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, min) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, max)

DEFINE_ALL_KERNELS(ALLREDUCE)
DEFINE_ALL_KERNELS(REDUCE_LOCAL_OUTOFPLACE)
DEFINE_ALL_KERNELS(REDUCE_LOCAL_INPLACE)
