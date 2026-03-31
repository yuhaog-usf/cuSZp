/*
 * cuSZp_bench.cpp — cuSZp plain-1D-f32 dataset benchmark
 *
 * Usage:
 *   ./cuSZp_bench -i <datafile> -eb <rel|abs> <bound> [-w <warmup>] [-r <repeat>]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuSZp/cuSZp_entry_1D_f32.h>
#include <cuSZp/cuSZp_utility.h>

static int cmp_float(const void *a, const void *b)
{
    float da = *(const float *)a, db = *(const float *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv)
{
    char oriFilePath[640] = {0};
    char errorBoundMode[4] = {0};
    float userBound = 0.0f;
    int warmup = 3;
    int repeat = 10;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s -i <datafile> -eb <rel|abs> <bound> [-w <warmup>] [-r <repeat>]\n", argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            strncpy(oriFilePath, argv[++i], sizeof(oriFilePath) - 1);
        } else if (strcmp(argv[i], "-eb") == 0 && i + 2 < argc) {
            strncpy(errorBoundMode, argv[++i], sizeof(errorBoundMode) - 1);
            userBound = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        }
    }
    if (strlen(oriFilePath) == 0 || strlen(errorBoundMode) == 0 || userBound == 0.0f) {
        fprintf(stderr, "Error: -i and -eb are required.\n");
        return 1;
    }
    if (warmup < 0) warmup = 0;
    if (repeat < 1) repeat = 1;

    size_t nbEle = 0;
    int status = 0;
    float *oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    if (status != 0 || !oriData || nbEle == 0) {
        fprintf(stderr, "ERROR: cannot read %s\n", oriFilePath);
        return 1;
    }

    float vmin = oriData[0], vmax = oriData[0];
    for (size_t i = 1; i < nbEle; i++) {
        if (oriData[i] < vmin) vmin = oriData[i];
        if (oriData[i] > vmax) vmax = oriData[i];
    }
    float valueRange = vmax - vmin;
    float absErr = (strcmp(errorBoundMode, "rel") == 0) ? userBound * valueRange : userBound;

    size_t rawBytes = nbEle * sizeof(float);
    double rawMiB = (double)rawBytes / 1024.0 / 1024.0;

    printf("======================================================\n");
    printf("  CUDA cuSZp Benchmark\n");
    printf("======================================================\n");
    printf("  Dataset   : %s\n", oriFilePath);
    printf("  Elements  : %zu (%.2f MB)\n", nbEle, (double)rawBytes / 1e6);
    printf("  Range     : [%e, %e]  (%.6e)\n", vmin, vmax, (double)valueRange);
    printf("  ErrMode   : %s\n", errorBoundMode);
    printf("  UserBound : %e\n", (double)userBound);
    printf("  AbsErr    : %e\n", (double)absErr);
    printf("  Warmup    : %d\n", warmup);
    printf("  Repeat    : %d\n", repeat);
    printf("======================================================\n\n");

    float *d_oriData, *d_decData;
    unsigned char *d_cmpBytes;
    cudaMalloc((void **)&d_oriData, rawBytes);
    cudaMemcpy(d_oriData, oriData, rawBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_decData, rawBytes);
    cudaMalloc((void **)&d_cmpBytes, rawBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *comp_times = (float *)malloc((size_t)repeat * sizeof(float));
    float *decomp_times = (float *)malloc((size_t)repeat * sizeof(float));

    size_t cmpSize = 0;
    float kernelTimeMs = 0.0f;

    // Warmup.
    for (int w = 0; w < warmup; w++) {
        cudaMemsetAsync(d_cmpBytes, 0, rawBytes, stream);
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream);
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, absErr, stream);
        cudaStreamSynchronize(stream);
    }

    // Timed compression runs (kernel-only).
    for (int r = 0; r < repeat; r++) {
        cudaMemsetAsync(d_cmpBytes, 0, rawBytes, stream);
        cudaStreamSynchronize(stream);
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream, &kernelTimeMs);
        comp_times[r] = kernelTimeMs;
    }

    if (cmpSize == 0 || cmpSize > rawBytes) {
        fprintf(stderr, "Invalid cmpSize=%zu\n", cmpSize);
        free(oriData); free(comp_times); free(decomp_times);
        cudaFree(d_oriData); cudaFree(d_decData); cudaFree(d_cmpBytes);
        return 1;
    }

    // Round-trip D2H -> zero -> H2D.
    unsigned char *cmpBytes_dup = (unsigned char *)malloc(cmpSize);
    cudaMemcpyAsync(cmpBytes_dup, d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    cudaMemsetAsync(d_cmpBytes, 0, rawBytes, stream);
    cudaMemcpyAsync(d_cmpBytes, cmpBytes_dup, cmpSize, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Timed decompression runs (kernel-only).
    for (int r = 0; r < repeat; r++) {
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, absErr, stream, &kernelTimeMs);
        decomp_times[r] = kernelTimeMs;
    }

    // Compute median.
    qsort(comp_times, (size_t)repeat, sizeof(float), cmp_float);
    qsort(decomp_times, (size_t)repeat, sizeof(float), cmp_float);
    float compMedian = comp_times[repeat / 2];
    float decMedian = decomp_times[repeat / 2];

    double compGBs = rawMiB / compMedian;
    double decGBs = rawMiB / decMedian;
    double cr = (double)rawBytes / (double)cmpSize;

    printf("Compress speed:   %.2f GB/s  (median of %d runs)\n", compGBs, repeat);
    printf("Decompress speed: %.2f GB/s  (median of %d runs)\n", decGBs, repeat);
    printf("Compression ratio: %.2f\n", cr);

    // Error check.
    float *decData = (float *)malloc(rawBytes);
    cudaMemcpy(decData, d_decData, rawBytes, cudaMemcpyDeviceToHost);

    int not_bound = 0;
    for (size_t i = 0; i < nbEle; i++) {
        if (fabs(oriData[i] - decData[i]) > absErr * 1.1f)
            not_bound++;
    }
    if (!not_bound)
        printf("\033[0;32mPass error check!\033[0m\n");
    else {
        printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    }

    free(oriData);
    free(decData);
    free(cmpBytes_dup);
    free(comp_times);
    free(decomp_times);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return not_bound ? 1 : 0;
}
