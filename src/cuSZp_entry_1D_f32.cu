#include "cuSZp_entry_1D_f32.h"
#include "cuSZp_kernels_1D_f32.h"
#include <stdio.h>

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_1D_fixed_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

 /** ************************************************************************
 * @brief cuSZp end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 * 
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_decompress_1D_fixed_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_1D_fixed_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    
    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_1D_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;
    const size_t rate_ofs =
        (nbEle + tblock_size * thread_chunk - 1) / (tblock_size * thread_chunk) *
        (tblock_size * thread_chunk) / 32;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    unsigned int glob_sync = 0;
    cudaError_t err = cudaSuccess;
    dim3 blockSize;
    dim3 gridSize;

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        return;
    }
    err = cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMemsetAsync(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    err = cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMemsetAsync(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    err = cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMemsetAsync(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }

    // cuSZp GPU compression.
    blockSize = dim3(bsize);
    gridSize = dim3(gsize);
    cuSZp_compress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }

    // Obtain compression ratio and move data back to CPU.  
    err = cudaMemcpyAsync(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaMemcpyAsync(glob_sync) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_compress_1D_plain_f32: cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_plain_cmp;
    }
    if (cmpSize) *cmpSize = (size_t)glob_sync + rate_ofs;

    // Free memory that is used.
cleanup_plain_cmp:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

 /** ************************************************************************
 * @brief cuSZp end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 * 
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_decompress_1D_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaSuccess;
    dim3 blockSize;
    dim3 gridSize;

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        return;
    }
    err = cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMemsetAsync(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }
    err = cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMemsetAsync(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }
    err = cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: cudaMemsetAsync(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }

    // cuSZp GPU decompression.
    blockSize = dim3(bsize);
    gridSize = dim3(gsize);
    cuSZp_decompress_kernel_1D_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuSZp_decompress_1D_plain_f32: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_plain_dec;
    }
    
    // Free memory that is used.
cleanup_plain_dec:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

/** ************************************************************************
 * @brief cuSZp end-to-end compression API for device pointers
 *        Compression is executed in GPU.
 *        Original data is stored as device pointers (in GPU).
 *        Compressed data is stored back as device pointers (in GPU).
 * 
 * @param   d_oriData       original data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           original data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_compress_1D_outlier_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+tblock_size*thread_chunk-1)/(tblock_size*thread_chunk)*(tblock_size*thread_chunk)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

 /** ************************************************************************
 * @brief cuSZp end-to-end decompression API for device pointers
 *        Decompression is executed in GPU.
 *        Compressed data is stored as device pointers (in GPU).
 *        Reconstructed data is stored as device pointers (in GPU).
 *        P.S. Reconstructed data and original data have the same shape.
 * 
 * @param   d_decData       reconstructed data (device pointer)
 * @param   d_cmpBytes      compressed data (device pointer)
 * @param   nbEle           reconstructed data size (number of floating point)
 * @param   cmpSize         compressed data size (number of unsigned char)
 * @param   errorBound      user-defined error bound
 * @param   stream          CUDA stream for executing compression kernel
 * *********************************************************************** */
void cuSZp_decompress_1D_outlier_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = tblock_size;
    int gsize = (nbEle + bsize * thread_chunk - 1) / (bsize * thread_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_1D_outlier_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


// ====================================================================
//  Generic block-size variants (template dispatch)
// ====================================================================

// Helper macro to launch the correct template instantiation.
#define LAUNCH_COMPRESS_BLKN(BS) \
    cuSZp_compress_kernel_1D_plain_f32_blkN<BS><<<gsize, bsize, 0, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)
#define LAUNCH_DECOMPRESS_BLKN(BS) \
    cuSZp_decompress_kernel_1D_plain_f32_blkN<BS><<<gsize, bsize, 0, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)

void cuSZp_compress_1D_plain_f32_blkN(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, int blockSize, cudaStream_t stream)
{
    int bsize = blockSize;
    int num_warps_per_block = bsize / 32;
    int total_warps = (nbEle + 32 * thread_chunk - 1) / (32 * thread_chunk);
    int gsize = (total_warps + num_warps_per_block - 1) / num_warps_per_block;
    int cmpOffSize = total_warps + 1;
    const size_t rate_ofs =
        (nbEle + (size_t)bsize * thread_chunk - 1) / ((size_t)bsize * thread_chunk) *
        ((size_t)bsize * thread_chunk) / 32;

    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    unsigned int glob_sync = 0;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; return; }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; goto cleanup_blkN_cmp; }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; goto cleanup_blkN_cmp; }
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize, stream);

    switch (blockSize) {
        case 32:  LAUNCH_COMPRESS_BLKN(32);  break;
        case 64:  LAUNCH_COMPRESS_BLKN(64);  break;
        case 128: LAUNCH_COMPRESS_BLKN(128); break;
        case 256: LAUNCH_COMPRESS_BLKN(256); break;
        default:
            fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: unsupported blockSize=%d\n", blockSize);
            if (cmpSize) *cmpSize = 0;
            goto cleanup_blkN_cmp;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: kernel launch failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; goto cleanup_blkN_cmp; }

    err = cudaMemcpyAsync(&glob_sync, d_cmpOffset+total_warps, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; goto cleanup_blkN_cmp; }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_compress_1D_plain_f32_blkN: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err)); if (cmpSize) *cmpSize = 0; goto cleanup_blkN_cmp; }
    if (cmpSize) *cmpSize = (size_t)glob_sync + rate_ofs;

cleanup_blkN_cmp:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuSZp_decompress_1D_plain_f32_blkN(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, int blockSize, cudaStream_t stream)
{
    int bsize = blockSize;
    int num_warps_per_block = bsize / 32;
    int total_warps = (nbEle + 32 * thread_chunk - 1) / (32 * thread_chunk);
    int gsize = (total_warps + num_warps_per_block - 1) / num_warps_per_block;
    int cmpOffSize = total_warps + 1;

    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_decompress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); return; }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_decompress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); goto cleanup_blkN_dec; }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_decompress_1D_plain_f32_blkN: cudaMalloc failed: %s\n", cudaGetErrorString(err)); goto cleanup_blkN_dec; }
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize, stream);

    switch (blockSize) {
        case 32:  LAUNCH_DECOMPRESS_BLKN(32);  break;
        case 64:  LAUNCH_DECOMPRESS_BLKN(64);  break;
        case 128: LAUNCH_DECOMPRESS_BLKN(128); break;
        case 256: LAUNCH_DECOMPRESS_BLKN(256); break;
        default:
            fprintf(stderr, "cuSZp_decompress_1D_plain_f32_blkN: unsupported blockSize=%d\n", blockSize);
            goto cleanup_blkN_dec;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "cuSZp_decompress_1D_plain_f32_blkN: kernel launch failed: %s\n", cudaGetErrorString(err)); goto cleanup_blkN_dec; }

cleanup_blkN_dec:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

#undef LAUNCH_COMPRESS_BLKN
#undef LAUNCH_DECOMPRESS_BLKN
