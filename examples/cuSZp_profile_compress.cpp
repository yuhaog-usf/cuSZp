/*
 * cuSZp_profile_compress.cpp — Nsight Systems profile-only target for compress.
 *
 * Flow:  load data -> H2D -> 3 warmup iterations -> cudaProfilerStart()
 *        -> 1 compress call + stream sync -> cudaProfilerStop() -> cleanup.
 *
 * Pair with: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
 *
 * Usage:
 *   ./cuSZp_profile_compress -i <datafile> -eb <rel|abs> <bound>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuSZp/cuSZp_entry_1D_f32.h>
#include <cuSZp/cuSZp_utility.h>

int main(int argc, char **argv)
{
    char oriFilePath[640] = {0};
    char errorBoundMode[4] = {0};
    float userBound = 0.0f;
    const int warmup = 3;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s -i <datafile> -eb <rel|abs> <bound>\n", argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            strncpy(oriFilePath, argv[++i], sizeof(oriFilePath) - 1);
        } else if (strcmp(argv[i], "-eb") == 0 && i + 2 < argc) {
            strncpy(errorBoundMode, argv[++i], sizeof(errorBoundMode) - 1);
            userBound = (float)atof(argv[++i]);
        }
    }
    if (strlen(oriFilePath) == 0 || strlen(errorBoundMode) == 0 || userBound == 0.0f) {
        fprintf(stderr, "Error: -i and -eb are required.\n");
        return 1;
    }

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

    float *d_oriData, *d_decData;
    unsigned char *d_cmpBytes;
    cudaMalloc((void **)&d_oriData, rawBytes);
    cudaMemcpy(d_oriData, oriData, rawBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_decData, rawBytes);
    cudaMalloc((void **)&d_cmpBytes, rawBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t cmpSize = 0;

    // Warmup (not profiled).
    for (int w = 0; w < warmup; w++) {
        cudaMemsetAsync(d_cmpBytes, 0, rawBytes, stream);
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream);
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, absErr, stream);
        cudaStreamSynchronize(stream);
    }

    cudaMemsetAsync(d_cmpBytes, 0, rawBytes, stream);
    cudaStreamSynchronize(stream);

    // ===== Profiled region: exactly one compress call =====
    cudaProfilerStart();
    cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream);
    cudaStreamSynchronize(stream);
    cudaProfilerStop();
    // =======================================================

    free(oriData);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}
