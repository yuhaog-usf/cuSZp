#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F32_BLOCK_SIZE_N_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F32_BLOCK_SIZE_N_H

#include <cstddef>
#include <cuda_runtime.h>

void cuSZp_compress_1D_plain_f32_block_size_N(float* d_oriData, unsigned char* d_cmpBytes,
                                              size_t nbEle, size_t* cmpSize, float errorBound,
                                              int dblockSize, cudaStream_t stream = 0);
void cuSZp_decompress_1D_plain_f32_block_size_N(float* d_decData, unsigned char* d_cmpBytes,
                                                size_t nbEle, size_t cmpSize, float errorBound,
                                                int dblockSize, cudaStream_t stream = 0);

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_ENTRY_1D_F32_BLOCK_SIZE_N_H
