#include "common.h"

__kernel void empty_kernel(int my_rank,
                           int comm_size,
                           ulong count,
                           const __global void* input_buffer,
                           __global void* output_buffer,
                           const __global void* peer_input_buffer,
                           __global void* peer_output_buffer) {
    return;
}

// DtypeName - e.g. float32, uint16
// Dtype - e.g. float, ushort
// OpName - e.g. sum, prod
// OpFunc - e.g. __sum_int, __prod_float (convention __<OpName>_<Dtype>)

#define DEFINE_ALLREDUCE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void allreduce_kernel_##DtypeName##_##OpName(int my_rank, \
                                                          int comm_size, \
                                                          ulong count, \
                                                          const __global Dtype* input_buffer, \
                                                          __global Dtype* output_buffer, \
                                                          const __global Dtype* peer_input_buffer, \
                                                          __global Dtype* peer_output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            Dtype ret = OpFunc(input_buffer[idx], peer_input_buffer[idx]); \
            output_buffer[idx] = ret; \
            peer_output_buffer[idx] = ret; \
        } \
    }

#define DEFINE_REDUCE_LOCAL_OUTOFPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_local_outofplace_kernel_##DtypeName##_##OpName( \
        int my_rank, \
        int comm_size, \
        ulong count, \
        const __global Dtype* input_buffer_1, \
        const __global Dtype* input_buffer_2, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            output_buffer[idx] = OpFunc(input_buffer_1[idx], input_buffer_2[idx]); \
        } \
    }

#define DEFINE_REDUCE_LOCAL_INPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_local_inplace_kernel_##DtypeName##_##OpName( \
        ulong count, const __global Dtype* input_buffer, __global Dtype* inoutput_buffer) { \
        DEBUG_BLOCK(printf("in reduce_local_inplace_kernel\n")); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            inoutput_buffer[idx] = OpFunc(input_buffer[idx], inoutput_buffer[idx]); \
        } \
    }

#define DEFINE_REDUCE_SINGLE_LOCAL_INPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_single_local_inplace_kernel_##DtypeName##_##OpName( \
        ulong count, \
        int peer_count, \
        const __global Dtype* input_buffer, \
        __global Dtype* inoutput_buffer) { \
        DEBUG_BLOCK(printf("in reduce_single_local_inplace_kernel\n")); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            Dtype ret = OpFunc(input_buffer[idx], inoutput_buffer[idx]); \
            for (int j = 1; j < peer_count; j++) { \
                ret = OpFunc(inoutput_buffer[j * count + idx], ret); \
            } \
            inoutput_buffer[idx] = ret; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_1_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_1_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_1\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_2_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_2_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_2\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_3_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_3_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_3\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_4_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_4_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_4\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_5_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_5_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        const __global Dtype* peer_buffer5, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_5\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            sum = OpFunc(sum, peer_buffer5[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_1_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_1_##DtypeName##_##OpName( \
        ulong count, const __global Dtype* input_buffer, __global Dtype* peer_buffer1) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_1 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_2_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_2_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_2 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_3_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_3_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_3 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_4_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_4_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_4 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_5_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_5_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4, \
        __global Dtype* peer_buffer5) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_5 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
            peer_buffer5[idx] = val; \
        } \
    }

// Define kernels for a specific reduction operation for all supported datatypes
#define DEFINE_KERNELS_WITH_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(int8, char, OpName, __##OpName##_##char) \
        DEFINE_##KernelName##_KERNEL(uint8, uchar, OpName, __##OpName##_##uchar) \
\
            DEFINE_##KernelName##_KERNEL(int16, short, OpName, __##OpName##_##short) \
                DEFINE_##KernelName##_KERNEL(uint16, ushort, OpName, __##OpName##_##ushort) \
\
                    DEFINE_##KernelName##_KERNEL(int32, int, OpName, __##OpName##_##int) \
                        DEFINE_##KernelName##_KERNEL(uint32, uint, OpName, __##OpName##_##uint) \
\
                            DEFINE_##KernelName##_KERNEL(int64, long, OpName, __##OpName##_##long) \
                                DEFINE_##KernelName##_KERNEL( \
                                    uint64, ulong, OpName, __##OpName##_##ulong) \
\
                                    DEFINE_##KernelName##_KERNEL( \
                                        float32, float, OpName, __##OpName##_##float) \
                                        DEFINE_##KernelName##_KERNEL( \
                                            float64, double, OpName, __##OpName##_##double)

#define DEFINE_KERNELS_WITH_LP_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(bfloat16, ushort, OpName, __bf16_##OpName##_##ushort) \
        DEFINE_##KernelName##_KERNEL(float16, half, OpName, __##OpName##_##half)

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

// Define reduction operation function for each supported datatype
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
DEFINE_ALL_KERNELS(REDUCE_SINGLE_LOCAL_INPLACE)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_1)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_2)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_3)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_4)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_5)

DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_1, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_2, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_3, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_4, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_5, custom)

DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_1, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_2, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_3, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_4, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_5, custom)
