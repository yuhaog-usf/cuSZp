#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>

int main()
{
    // Input data preparation on CPU.
    float* oriData = NULL;
    float* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 1024*1024*512; // 2 GB fp32 data.
    size_t cmpSize = 0;
    float kernelTimeMs = 0.0f;
    oriData = (float*)malloc(nbEle*sizeof(float));
    decData = (float*)malloc(nbEle*sizeof(float));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));

    // Initialize oriData.
    printf("Generating test data...\n\n");
    float startValue = -20.0f;
    float step = 0.1f;
    float endValue = 20.0f;
    size_t idx = 0;
    float value = startValue;
    while (idx < nbEle)
    {
        oriData[idx++] = value;
        value += step;
        if (value > endValue)
        {
            value = startValue;
        }
    }

    // Get value range, making it a REL errMode test.
    float max_val = oriData[0];
    float min_val = oriData[0];
    for(size_t i=0; i<nbEle; i++)
    {
        if(oriData[i]>max_val)
            max_val = oriData[i];
        else if(oriData[i]<min_val)
            min_val = oriData[i];
    }
    float errorBound = (max_val - min_val) * 1E-2f;

    // Input data preparation on GPU.
    float* d_oriData;
    float* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, sizeof(float)*nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(float)*nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, sizeof(float)*nbEle);
    cudaMalloc((void**)&d_cmpBytes, sizeof(float)*nbEle);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for plain kernel (no prints).
    for(int i=0; i<3; i++)
    {
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
        cudaMemset(d_cmpBytes, 0, sizeof(float)*nbEle);
    }

    // cuSZp-p testing.
    printf("=================================================\n");
    printf("========Testing cuSZp-p-1D-f32 on REL 1E-2=======\n");
    printf("=================================================\n");

    // cuSZp compression with kernel-only timing.
    cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream, &kernelTimeMs);
    printf("cuSZp-p compression   kernel-only speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/kernelTimeMs);

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, sizeof(float)*nbEle);
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cuSZp decompression with kernel-only timing.
    cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream, &kernelTimeMs);
    printf("cuSZp-p decompression kernel-only speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/kernelTimeMs);

    // Print result.
    printf("cuSZp-p compression ratio: %f\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    printf("cuSZp-p finished!\n");

    // Error check.
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    printf("\033[1mDone with testing cuSZp-p on REL 1E-2!\033[0m\n");

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}
