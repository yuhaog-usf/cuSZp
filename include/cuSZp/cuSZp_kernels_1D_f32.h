#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_H

static const int tblock_size = 32; // Fixed to 32, cannot be modified.
static const int thread_chunk = 1024;

__global__ void cuSZp_compress_kernel_1D_fixed_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_1D_fixed_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_compress_kernel_1D_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_1D_plain_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_compress_kernel_1D_outlier_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_1D_outlier_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

// Template kernels for arbitrary block sizes (32, 64, 128, 256).
// BLOCK_SIZE must be a multiple of 32.
template <int BLOCK_SIZE>
__global__ void cuSZp_compress_kernel_1D_plain_f32_blkN(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
template <int BLOCK_SIZE>
__global__ void cuSZp_decompress_kernel_1D_plain_f32_blkN(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_1D_F32_H