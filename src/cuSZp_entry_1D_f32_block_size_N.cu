#include "cuSZp_entry_1D_f32_block_size_N.h"
#include "cuSZp_kernels_1D_f32_block_size_N.h"

#include <stdio.h>

#define LAUNCH_COMPRESS_BLOCK_SIZE_N(DBS)                                                       \
    cuSZp_compress_kernel_1D_plain_f32_block_size_N<DBS><<<gridSize, blockSize,                 \
                                                           sizeof(unsigned int) * 2, stream>>>( \
        d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)

#define LAUNCH_COMPRESS_OUTLIER_BLOCK_SIZE_N(DBS)                                               \
    cuSZp_compress_kernel_1D_outlier_f32_block_size_N<DBS><<<gridSize, blockSize,               \
                                                             sizeof(unsigned int) * 2, stream>>>(\
        d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)

void cuSZp_compress_1D_plain_f32_block_size_N(float* d_oriData, unsigned char* d_cmpBytes,
                                              size_t nbEle, size_t* cmpSize, float errorBound,
                                              int dblockSize, cudaStream_t stream)
{
    int bsize = tblock_size;
    int gsize = (nbEle + (size_t)bsize * thread_chunk - 1) / ((size_t)bsize * thread_chunk);
    int cmpOffSize = gsize + 1;
    size_t rate_ofs;
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    unsigned int glob_sync = 0;
    cudaError_t err = cudaSuccess;

    switch (dblockSize) {
        case 32:
        case 64:
        case 128:
        case 256:
            rate_ofs = (size_t)gsize * bsize * thread_chunk / dblockSize;
            break;
        default:
            fprintf(stderr,
                    "cuSZp_compress_1D_plain_f32_block_size_N: unsupported dblockSize=%d\n",
                    dblockSize);
            if (cmpSize) *cmpSize = 0;
            return;
    }

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_plain_f32_block_size_N: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        return;
    }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_plain_f32_block_size_N: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_cmp;
    }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_plain_f32_block_size_N: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_cmp;
    }
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    {
        dim3 blockSize(bsize);
        dim3 gridSize(gsize);
        switch (dblockSize) {
            case 32:  LAUNCH_COMPRESS_BLOCK_SIZE_N(32);  break;
            case 64:  LAUNCH_COMPRESS_BLOCK_SIZE_N(64);  break;
            case 128: LAUNCH_COMPRESS_BLOCK_SIZE_N(128); break;
            case 256: LAUNCH_COMPRESS_BLOCK_SIZE_N(256); break;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "cuSZp_compress_1D_plain_f32_block_size_N: kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            if (cmpSize) *cmpSize = 0;
            goto cleanup_cmp;
        }
    }

    cudaMemcpyAsync(&glob_sync, d_cmpOffset + cmpOffSize - 1, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (cmpSize) *cmpSize = (size_t)glob_sync + rate_ofs;

cleanup_cmp:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuSZp_compress_1D_outlier_f32_block_size_N(float* d_oriData, unsigned char* d_cmpBytes,
                                                size_t nbEle, size_t* cmpSize, float errorBound,
                                                int dblockSize, cudaStream_t stream)
{
    int bsize = tblock_size;
    int gsize = (nbEle + (size_t)bsize * thread_chunk - 1) / ((size_t)bsize * thread_chunk);
    int cmpOffSize = gsize + 1;
    size_t rate_ofs;
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    unsigned int glob_sync = 0;
    cudaError_t err = cudaSuccess;

    switch (dblockSize) {
        case 32:
        case 64:
        case 128:
        case 256:
            rate_ofs = (size_t)gsize * bsize * thread_chunk / dblockSize;
            break;
        default:
            fprintf(stderr,
                    "cuSZp_compress_1D_outlier_f32_block_size_N: unsupported dblockSize=%d\n",
                    dblockSize);
            if (cmpSize) *cmpSize = 0;
            return;
    }

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_outlier_f32_block_size_N: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        return;
    }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_outlier_f32_block_size_N: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_cmp_outlier;
    }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_compress_1D_outlier_f32_block_size_N: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        if (cmpSize) *cmpSize = 0;
        goto cleanup_cmp_outlier;
    }
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    {
        dim3 blockSize(bsize);
        dim3 gridSize(gsize);
        switch (dblockSize) {
            case 32:  LAUNCH_COMPRESS_OUTLIER_BLOCK_SIZE_N(32);  break;
            case 64:  LAUNCH_COMPRESS_OUTLIER_BLOCK_SIZE_N(64);  break;
            case 128: LAUNCH_COMPRESS_OUTLIER_BLOCK_SIZE_N(128); break;
            case 256: LAUNCH_COMPRESS_OUTLIER_BLOCK_SIZE_N(256); break;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "cuSZp_compress_1D_outlier_f32_block_size_N: kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            if (cmpSize) *cmpSize = 0;
            goto cleanup_cmp_outlier;
        }
    }

    cudaMemcpyAsync(&glob_sync, d_cmpOffset + cmpOffSize - 1, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (cmpSize) *cmpSize = (size_t)glob_sync + rate_ofs;

cleanup_cmp_outlier:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

#define LAUNCH_DECOMPRESS_BLOCK_SIZE_N(DBS)                                                     \
    cuSZp_decompress_kernel_1D_plain_f32_block_size_N<DBS><<<gridSize, blockSize,               \
                                                             sizeof(unsigned int) * 2, stream>>>(\
        d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)

#define LAUNCH_DECOMPRESS_OUTLIER_BLOCK_SIZE_N(DBS)                                             \
    cuSZp_decompress_kernel_1D_outlier_f32_block_size_N<DBS><<<gridSize, blockSize,             \
                                                               sizeof(unsigned int) * 2, stream>>>(\
        d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle)

void cuSZp_decompress_1D_plain_f32_block_size_N(float* d_decData, unsigned char* d_cmpBytes,
                                                size_t nbEle, size_t cmpSize, float errorBound,
                                                int dblockSize, cudaStream_t stream)
{
    (void)cmpSize;

    int bsize = tblock_size;
    int gsize = (nbEle + (size_t)bsize * thread_chunk - 1) / ((size_t)bsize * thread_chunk);
    int cmpOffSize = gsize + 1;
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaSuccess;

    if (dblockSize != 32 && dblockSize != 64 && dblockSize != 128 && dblockSize != 256) {
        fprintf(stderr,
                "cuSZp_decompress_1D_plain_f32_block_size_N: unsupported dblockSize=%d\n",
                dblockSize);
        return;
    }

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_plain_f32_block_size_N: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        return;
    }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_plain_f32_block_size_N: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_dec;
    }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_plain_f32_block_size_N: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_dec;
    }
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    {
        dim3 blockSize(bsize);
        dim3 gridSize(gsize);
        switch (dblockSize) {
            case 32:  LAUNCH_DECOMPRESS_BLOCK_SIZE_N(32);  break;
            case 64:  LAUNCH_DECOMPRESS_BLOCK_SIZE_N(64);  break;
            case 128: LAUNCH_DECOMPRESS_BLOCK_SIZE_N(128); break;
            case 256: LAUNCH_DECOMPRESS_BLOCK_SIZE_N(256); break;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "cuSZp_decompress_1D_plain_f32_block_size_N: kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            goto cleanup_dec;
        }
    }

cleanup_dec:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuSZp_decompress_1D_outlier_f32_block_size_N(float* d_decData, unsigned char* d_cmpBytes,
                                                  size_t nbEle, size_t cmpSize, float errorBound,
                                                  int dblockSize, cudaStream_t stream)
{
    (void)cmpSize;

    int bsize = tblock_size;
    int gsize = (nbEle + (size_t)bsize * thread_chunk - 1) / ((size_t)bsize * thread_chunk);
    int cmpOffSize = gsize + 1;
    unsigned int* d_cmpOffset = NULL;
    unsigned int* d_locOffset = NULL;
    int* d_flag = NULL;
    cudaError_t err = cudaSuccess;

    if (dblockSize != 32 && dblockSize != 64 && dblockSize != 128 && dblockSize != 256) {
        fprintf(stderr,
                "cuSZp_decompress_1D_outlier_f32_block_size_N: unsupported dblockSize=%d\n",
                dblockSize);
        return;
    }

    err = cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_outlier_f32_block_size_N: cudaMalloc(d_cmpOffset) failed: %s\n",
                cudaGetErrorString(err));
        return;
    }
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_locOffset, sizeof(unsigned int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_outlier_f32_block_size_N: cudaMalloc(d_locOffset) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_dec_outlier;
    }
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int) * cmpOffSize, stream);
    err = cudaMalloc((void**)&d_flag, sizeof(int) * cmpOffSize);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cuSZp_decompress_1D_outlier_f32_block_size_N: cudaMalloc(d_flag) failed: %s\n",
                cudaGetErrorString(err));
        goto cleanup_dec_outlier;
    }
    cudaMemsetAsync(d_flag, 0, sizeof(int) * cmpOffSize, stream);

    {
        dim3 blockSize(bsize);
        dim3 gridSize(gsize);
        switch (dblockSize) {
            case 32:  LAUNCH_DECOMPRESS_OUTLIER_BLOCK_SIZE_N(32);  break;
            case 64:  LAUNCH_DECOMPRESS_OUTLIER_BLOCK_SIZE_N(64);  break;
            case 128: LAUNCH_DECOMPRESS_OUTLIER_BLOCK_SIZE_N(128); break;
            case 256: LAUNCH_DECOMPRESS_OUTLIER_BLOCK_SIZE_N(256); break;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "cuSZp_decompress_1D_outlier_f32_block_size_N: kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            goto cleanup_dec_outlier;
        }
    }

cleanup_dec_outlier:
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}
