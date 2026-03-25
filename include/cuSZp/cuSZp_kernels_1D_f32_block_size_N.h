#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_BLOCK_SIZE_N_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_BLOCK_SIZE_N_H

#include "cuSZp_kernels_1D_f32.h"

template <int DBLOCK>
__global__ void cuSZp_compress_kernel_1D_plain_f32_block_size_N(
    const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

template <int DBLOCK>
__global__ void cuSZp_decompress_kernel_1D_plain_f32_block_size_N(
    float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

template <int DBLOCK>
__global__ void cuSZp_compress_kernel_1D_outlier_f32_block_size_N(
    const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

template <int DBLOCK>
__global__ void cuSZp_decompress_kernel_1D_outlier_f32_block_size_N(
    float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData,
    volatile unsigned int* const __restrict__ cmpOffset,
    volatile unsigned int* const __restrict__ locOffset,
    volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_BLOCK_SIZE_N_H
