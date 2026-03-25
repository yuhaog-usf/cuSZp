#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <cuSZp.h>
#include "cuSZp/cuSZp_entry_1D_f32_block_size_N.h"

static int cmp_float(const void* a, const void* b)
{
    float da = *(const float*)a;
    float db = *(const float*)b;
    return (da > db) - (da < db);
}

typedef struct {
    float median;
    float p95;
    float min;
    float max;
} pstats;

typedef struct {
    double max_abs_err;
    double max_rel_err;
    double psnr;
} quality_t;

static pstats compute_pstats(float* samples, int n)
{
    pstats s = {0};
    if (n <= 0) return s;
    qsort(samples, (size_t)n, sizeof(float), cmp_float);
    s.min = samples[0];
    s.max = samples[n - 1];
    if (n == 1) {
        s.median = samples[0];
        s.p95 = samples[0];
        return s;
    }
    s.median = samples[n / 2];
    s.p95 = samples[(int)((n - 1) * 0.95)];
    return s;
}

static quality_t check_quality(const float* orig, const float* dec, size_t n)
{
    quality_t q = {0};
    double sum_sq = 0.0;
    float vmin = orig[0];
    float vmax = orig[0];

    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)orig[i] - (double)dec[i]);
        if (diff > q.max_abs_err) q.max_abs_err = diff;
        if (fabs((double)orig[i]) > 1e-30) {
            double rel_err = diff / fabs((double)orig[i]);
            if (rel_err > q.max_rel_err) q.max_rel_err = rel_err;
        }
        sum_sq += diff * diff;
        if (orig[i] < vmin) vmin = orig[i];
        if (orig[i] > vmax) vmax = orig[i];
    }

    double mse = sum_sq / (double)n;
    double range = (double)(vmax - vmin);
    q.psnr = (mse > 0.0 && range > 0.0) ? 20.0 * log10(range / sqrt(mse)) : 999.0;
    return q;
}

static size_t compute_cmp_buffer_size(size_t nbEle)
{
    const size_t chunk_size = 32768;
    const size_t blocks_per_chunk = 1024;
    const size_t worst_block_bytes = 133;
    size_t gsize = (nbEle + chunk_size - 1) / chunk_size;
    return gsize * blocks_per_chunk * worst_block_bytes;
}

int main(int argc, char** argv)
{
    char oriFilePath[640] = {0};
    char errorBoundMode[8] = {0};
    float userBound = 0.0f;
    int warmup = 3;
    int repeat = 10;

    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s -i <datafile> -eb <rel|abs> <bound> [-w <warmup>] [-r <repeat>]\n",
                argv[0]);
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
    float* oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    if (status != 0 || !oriData || nbEle == 0) {
        fprintf(stderr, "ERROR: cannot read %s\n", oriFilePath);
        return 1;
    }

    float vmin = oriData[0];
    float vmax = oriData[0];
    for (size_t i = 1; i < nbEle; i++) {
        if (oriData[i] < vmin) vmin = oriData[i];
        if (oriData[i] > vmax) vmax = oriData[i];
    }

    float valueRange = vmax - vmin;
    float absErr = (strcmp(errorBoundMode, "rel") == 0) ? userBound * valueRange : userBound;
    size_t rawBytes = nbEle * sizeof(float);
    double rawMiB = (double)rawBytes / 1024.0 / 1024.0;
    size_t cmpBufSize = compute_cmp_buffer_size(nbEle);

    printf("======================================================\n");
    printf("  cuSZp Variable Datablock-Size GPU Benchmark\n");
    printf("======================================================\n");
    printf("  Dataset    : %s\n", oriFilePath);
    printf("  Elements   : %zu (%.2f MB)\n", nbEle, (double)rawBytes / 1e6);
    printf("  Range      : [%e, %e]  (%.6e)\n", vmin, vmax, (double)valueRange);
    printf("  ErrMode    : %s\n", errorBoundMode);
    printf("  UserBound  : %e\n", (double)userBound);
    printf("  AbsErr     : %e\n", (double)absErr);
    printf("  Warmup     : %d\n", warmup);
    printf("  Repeat     : %d\n", repeat);
    printf("  CmpBufSize : %zu bytes\n", cmpBufSize);
    printf("======================================================\n\n");

    float* d_oriData = NULL;
    float* d_decData = NULL;
    unsigned char* d_cmpBytes = NULL;
    float* decData = (float*)malloc(rawBytes);
    unsigned char* cmpBytesDup = (unsigned char*)malloc(cmpBufSize);
    float* comp_times = (float*)malloc((size_t)repeat * sizeof(float));
    float* decomp_times = (float*)malloc((size_t)repeat * sizeof(float));

    if (!decData || !cmpBytesDup || !comp_times || !decomp_times) {
        fprintf(stderr, "ERROR: host allocation failed.\n");
        free(oriData);
        free(decData);
        free(cmpBytesDup);
        free(comp_times);
        free(decomp_times);
        return 1;
    }

    cudaMalloc((void**)&d_oriData, rawBytes);
    cudaMemcpy(d_oriData, oriData, rawBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, rawBytes);
    cudaMalloc((void**)&d_cmpBytes, cmpBufSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int dblock_sizes[] = {32, 64, 128, 256};
    int num_dblock_sizes = 4;
    size_t dblock32_cmpSize = 0;

    printf("%-10s  %10s  %12s  %12s  %12s  %12s  %12s  %12s  %s\n",
           "DBlk", "CR", "Comp(GB/s)", "Comp_P95", "Decomp(GB/s)", "Dec_P95",
           "MaxAbsErr", "PSNR(dB)", "Check");
    printf("----------  ----------  ------------  ------------  ------------  ------------  ------------  ------------  -----\n");

    printf("\n# CSV_START\n");
    printf("dataset,elements,valueRange,errMode,userBound,absErr,warmup,repeat,"
           "dblockSize,cmpBytes,CR,"
           "comp_median_GBs,comp_p95_GBs,comp_min_GBs,"
           "decomp_median_GBs,decomp_p95_GBs,decomp_min_GBs,"
           "maxAbsErr,maxRelErr,PSNR\n");

    for (int bi = 0; bi < num_dblock_sizes; bi++) {
        int dbs = dblock_sizes[bi];
        size_t cmpSize = 0;
        float elapsed = 0.0f;

        for (int w = 0; w < warmup; w++) {
            cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
            cuSZp_compress_1D_plain_f32_block_size_N(d_oriData, d_cmpBytes, nbEle, &cmpSize,
                                                     absErr, dbs, stream);
            cuSZp_decompress_1D_plain_f32_block_size_N(d_decData, d_cmpBytes, nbEle, cmpSize,
                                                       absErr, dbs, stream);
            cudaStreamSynchronize(stream);
        }

        for (int r = 0; r < repeat; r++) {
            cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
            cudaStreamSynchronize(stream);
            cudaEventRecord(start, stream);
            cuSZp_compress_1D_plain_f32_block_size_N(d_oriData, d_cmpBytes, nbEle, &cmpSize,
                                                     absErr, dbs, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            comp_times[r] = elapsed;
        }

        if (cmpSize == 0 || cmpSize > cmpBufSize) {
            printf("%-10d  SKIP (invalid cmpSize=%zu)\n", dbs, cmpSize);
            continue;
        }

        cudaMemcpyAsync(cmpBytesDup, d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
        cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
        cudaMemcpyAsync(d_cmpBytes, cmpBytesDup, cmpSize, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        for (int r = 0; r < repeat; r++) {
            cudaEventRecord(start, stream);
            cuSZp_decompress_1D_plain_f32_block_size_N(d_decData, d_cmpBytes, nbEle, cmpSize,
                                                       absErr, dbs, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            decomp_times[r] = elapsed;
        }

        cudaMemcpy(decData, d_decData, rawBytes, cudaMemcpyDeviceToHost);
        quality_t q = check_quality(oriData, decData, nbEle);
        pstats cs = compute_pstats(comp_times, repeat);
        pstats ds = compute_pstats(decomp_times, repeat);

        if (dbs == 32) dblock32_cmpSize = cmpSize;

        double cr = (double)rawBytes / (double)cmpSize;
        double comp_gbs = rawMiB / cs.median;
        double comp_p95_gbs = rawMiB / cs.p95;
        double comp_min_gbs = rawMiB / cs.max;
        double dec_gbs = rawMiB / ds.median;
        double dec_p95_gbs = rawMiB / ds.p95;
        double dec_min_gbs = rawMiB / ds.max;
        const char* check_str = (q.max_abs_err <= absErr * 1.01f) ? "PASS" : "FAIL";
        const char* dblk_label = (dbs == 32) ? "32(base)" : (dbs == 64) ? "64" : (dbs == 128) ? "128" : "256";

        printf("%-10s  %10.2f  %12.2f  %12.2f  %12.2f  %12.2f  %12.2e  %12.1f  %s\n",
               dblk_label, cr, comp_gbs, comp_p95_gbs, dec_gbs, dec_p95_gbs,
               q.max_abs_err, q.psnr, check_str);

        printf("%s,%zu,%.6e,%s,%e,%e,%d,%d,"
               "%d,%zu,%.4f,"
               "%.4f,%.4f,%.4f,"
               "%.4f,%.4f,%.4f,"
               "%.6e,%.6e,%.1f\n",
               oriFilePath, nbEle, (double)valueRange,
               errorBoundMode, (double)userBound, (double)absErr, warmup, repeat,
               dbs, cmpSize, cr,
               comp_gbs, comp_p95_gbs, comp_min_gbs,
               dec_gbs, dec_p95_gbs, dec_min_gbs,
               q.max_abs_err, q.max_rel_err, q.psnr);
    }

    {
        size_t baselineCmpSize = 0;
        cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &baselineCmpSize, absErr, stream);
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, baselineCmpSize, absErr, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(decData, d_decData, rawBytes, cudaMemcpyDeviceToHost);
        quality_t baselineQ = check_quality(oriData, decData, nbEle);
        printf("# BASELINE_CHECK master_plain_cmpBytes=%zu block_size_N_32_cmpBytes=%zu cmpSizeMatch=%s maxAbsErr=%.6e\n",
               baselineCmpSize, dblock32_cmpSize,
               (baselineCmpSize == dblock32_cmpSize) ? "YES" : "NO",
               baselineQ.max_abs_err);
    }

    printf("# CSV_END\n");

    free(oriData);
    free(decData);
    free(cmpBytesDup);
    free(comp_times);
    free(decomp_times);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}
