/*
 * cuSZp_blk64.cu — cuSZp block-size=64 GPU benchmark
 *
 * Benchmarks cuSZp plain-1D-f32 with tblock_size=64.
 *
 * Usage:
 *   ./cuSZp_blk64 -i <datafile> -eb <rel|abs> <bound> [-w <warmup>] [-r <repeat>]
 *
 * Examples:
 *   ./cuSZp_blk64 -i pressure_3000 -eb rel 1e-3
 *   ./cuSZp_blk64 -i pressure_3000 -eb abs 1e-4 -w 5 -r 20
 *
 * Output: human-readable table + CSV on stdout.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuSZp.h>

/* ====================================================================
 *  Percentile helpers
 * ==================================================================== */
static int cmp_float(const void *a, const void *b)
{
    float da = *(const float *)a, db = *(const float *)b;
    return (da > db) - (da < db);
}

typedef struct {
    float median;
    float p95;
    float min;
    float max;
} pstats;

static pstats compute_pstats(float *samples, int n)
{
    pstats s = {0};
    if (n <= 0) return s;
    qsort(samples, (size_t)n, sizeof(float), cmp_float);
    s.min = samples[0];
    s.max = samples[n - 1];
    if (n == 1) { s.median = s.p95 = samples[0]; return s; }
    s.median = samples[n / 2];
    s.p95 = samples[(int)((n - 1) * 0.95)];
    return s;
}

/* ====================================================================
 *  Quality check
 * ==================================================================== */
typedef struct {
    double max_abs_err;
    double max_rel_err;
    double psnr;
} quality_t;

static quality_t check_quality(const float *orig, const float *dec, size_t n)
{
    quality_t q = {0};
    double sum_sq = 0.0;
    float vmin = orig[0], vmax = orig[0];
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)orig[i] - (double)dec[i]);
        if (diff > q.max_abs_err) q.max_abs_err = diff;
        if (fabs((double)orig[i]) > 1e-30) {
            double re = diff / fabs((double)orig[i]);
            if (re > q.max_rel_err) q.max_rel_err = re;
        }
        sum_sq += diff * diff;
        if (orig[i] < vmin) vmin = orig[i];
        if (orig[i] > vmax) vmax = orig[i];
    }
    double mse = sum_sq / (double)n;
    double range = (double)(vmax - vmin);
    q.psnr = (mse > 0 && range > 0) ? 20.0 * log10(range / sqrt(mse)) : 999.0;
    return q;
}

/* ====================================================================
 *  Main
 * ==================================================================== */
int main(int argc, char **argv)
{
    char oriFilePath[640] = {0};
    char errorBoundMode[4] = {0};
    float userBound = 0.0f;
    int warmup = 3;
    int repeat = 10;

    /* Parse arguments */
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

    /* Load data using cuSZp utility */
    size_t nbEle = 0;
    int status = 0;
    float *oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    if (status != 0 || !oriData || nbEle == 0) {
        fprintf(stderr, "ERROR: cannot read %s\n", oriFilePath);
        return 1;
    }

    /* Compute value range and error bound */
    float vmin = oriData[0], vmax = oriData[0];
    for (size_t i = 1; i < nbEle; i++) {
        if (oriData[i] < vmin) vmin = oriData[i];
        if (oriData[i] > vmax) vmax = oriData[i];
    }
    float valueRange = vmax - vmin;
    float absErr;
    if (strcmp(errorBoundMode, "rel") == 0) {
        absErr = userBound * valueRange;
    } else {
        absErr = userBound;
    }

    size_t rawBytes = nbEle * sizeof(float);
    /* Throughput unit: MiB / ms = GB/s (matching cuSZp.cpp / cuSZp_test convention) */
    double rawMiB = (double)rawBytes / 1024.0 / 1024.0;

    printf("======================================================\n");
    printf("  cuSZp Block-Size=64 GPU Benchmark\n");
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

    /* GPU memory */
    float *d_oriData, *d_decData;
    unsigned char *d_cmpBytes;
    size_t cmpBufSize = rawBytes;  /* match cuSZp.cpp: malloc(nbEle*sizeof(float)) */
    cudaMalloc((void **)&d_oriData, rawBytes);
    cudaMemcpy(d_oriData, oriData, rawBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_decData, rawBytes);
    cudaMalloc((void **)&d_cmpBytes, cmpBufSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *decData = (float *)malloc(rawBytes);
    unsigned char *cmpBytes_dup = (unsigned char *)malloc(rawBytes);
    float *comp_times = (float *)malloc((size_t)repeat * sizeof(float));
    float *decomp_times = (float *)malloc((size_t)repeat * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t cmpSize = 0;
    float elapsed = 0.0f;

    /* Warmup */
    printf("Warming up (%d iterations)...\n", warmup);
    for (int w = 0; w < warmup; w++) {
        cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
        cuSZp_compress_1D_plain_f32_blk64(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream);
        cuSZp_decompress_1D_plain_f32_blk64(d_decData, d_cmpBytes, nbEle, cmpSize, absErr, stream);
        cudaStreamSynchronize(stream);
    }

    /* Timed compress */
    printf("Benchmarking (%d iterations)...\n", repeat);
    for (int r = 0; r < repeat; r++) {
        cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(start, stream);
        cuSZp_compress_1D_plain_f32_blk64(d_oriData, d_cmpBytes, nbEle, &cmpSize, absErr, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        comp_times[r] = elapsed;
    }

    if (cmpSize == 0 || cmpSize > rawBytes) {
        printf("SKIP (invalid cmpSize=%zu)\n", cmpSize);
        goto cleanup;
    }

    /* D2H -> zero -> H2D round-trip (matching cuSZp.cpp / cuSZp_test pattern) */
    cudaMemcpyAsync(cmpBytes_dup, d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    cudaMemsetAsync(d_cmpBytes, 0, cmpBufSize, stream);
    cudaMemcpyAsync(d_cmpBytes, cmpBytes_dup, cmpSize, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Timed decompress */
    for (int r = 0; r < repeat; r++) {
        cudaEventRecord(start, stream);
        cuSZp_decompress_1D_plain_f32_blk64(d_decData, d_cmpBytes, nbEle, cmpSize, absErr, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        decomp_times[r] = elapsed;
    }

    /* Quality check */
    cudaMemcpy(decData, d_decData, rawBytes, cudaMemcpyDeviceToHost);
    {
        quality_t q = check_quality(oriData, decData, nbEle);

        /* Compute stats: throughput = MiB / ms (matching cuSZp convention) */
        pstats cs = compute_pstats(comp_times, repeat);
        pstats ds = compute_pstats(decomp_times, repeat);

        double cr = (double)rawBytes / (double)cmpSize;
        double comp_gbs = rawMiB / cs.median;
        double comp_p95_gbs = rawMiB / cs.p95;
        double comp_min_gbs = rawMiB / cs.max;  /* worst time = min throughput */
        double dec_gbs = rawMiB / ds.median;
        double dec_p95_gbs = rawMiB / ds.p95;
        double dec_min_gbs = rawMiB / ds.max;

        /* Error bound check */
        if (q.max_abs_err <= absErr * 1.01)
            printf("\033[0;32mPass error check!\033[0m\n");
        else
            printf("\033[0;31mFail error check! maxAbsErr=%.6e > bound=%.6e\033[0m\n",
                   q.max_abs_err, (double)absErr);

        /* Human-readable table */
        printf("\n%-8s  %10s  %12s  %12s  %12s  %12s  %12s  %12s\n",
               "BlkSize", "CR", "Comp(GB/s)", "Comp_P95", "Decomp(GB/s)", "Dec_P95", "MaxAbsErr", "PSNR(dB)");
        printf("--------  ----------  ------------  ------------  ------------  ------------  ------------  ------------\n");
        printf("%-8d  %10.2f  %12.2f  %12.2f  %12.2f  %12.2f  %12.2e  %12.1f\n",
               64, cr, comp_gbs, comp_p95_gbs, dec_gbs, dec_p95_gbs,
               q.max_abs_err, q.psnr);

        /* CSV block */
        printf("\n# CSV_START\n");
        printf("dataset,elements,valueRange,errMode,userBound,absErr,warmup,repeat,"
               "blockSize,CR,cmpBytes,"
               "comp_median_GBs,comp_p95_GBs,comp_min_GBs,"
               "decomp_median_GBs,decomp_p95_GBs,decomp_min_GBs,"
               "maxAbsErr,maxRelErr,PSNR\n");
        printf("%s,%zu,%.6e,%s,%e,%e,%d,%d,"
               "%d,%.4f,%zu,"
               "%.4f,%.4f,%.4f,"
               "%.4f,%.4f,%.4f,"
               "%.6e,%.6e,%.1f\n",
               oriFilePath, nbEle, (double)valueRange,
               errorBoundMode, (double)userBound, (double)absErr, warmup, repeat,
               64, cr, cmpSize,
               comp_gbs, comp_p95_gbs, comp_min_gbs,
               dec_gbs, dec_p95_gbs, dec_min_gbs,
               q.max_abs_err, q.max_rel_err, q.psnr);
        printf("# CSV_END\n");
    }

cleanup:
    free(oriData);
    free(decData);
    free(cmpBytes_dup);
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
