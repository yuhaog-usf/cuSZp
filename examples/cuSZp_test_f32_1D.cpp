#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuSZp.h>

int main()
{
    // Input data preparation on CPU.
    float* oriData = NULL;
    float* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 1024*1024*512; // 2 GB fp32 data.
    size_t cmpSize2 = 0;
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

    // Get value range, making it a REL errMode test -- remove this will be ABS errMode.
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

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++)
    {
        cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize2, errorBound, stream);
        cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize2, errorBound, stream);
        cudaStreamSynchronize(stream);
        cudaMemsetAsync(d_cmpBytes, 0, sizeof(float)*nbEle, stream);
        cudaStreamSynchronize(stream);
    }

    // cuSZp-p testing.
    printf("=================================================\n");
    printf("========Testing cuSZp-p-1D-f32 on REL 1E-2=======\n");
    printf("=================================================\n");

    cudaEvent_t cmpStart, cmpStop, decStart, decStop;
    cudaEventCreate(&cmpStart);
    cudaEventCreate(&cmpStop);
    cudaEventCreate(&decStart);
    cudaEventCreate(&decStop);

    // cuSZp compression.
    cudaEventRecord(cmpStart, stream);
    cuSZp_compress_1D_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize2, errorBound, stream);
    cudaEventRecord(cmpStop, stream);
    cudaEventSynchronize(cmpStop);
    float cmpTime = 0.0f;
    cudaEventElapsedTime(&cmpTime, cmpStart, cmpStop);

    if (cmpSize2 == 0 || cmpSize2 > nbEle * sizeof(float))
    {
        fprintf(stderr, "Invalid cmpSize2=%zu\n", cmpSize2);
        cudaEventDestroy(cmpStart);
        cudaEventDestroy(cmpStop);
        cudaEventDestroy(decStart);
        cudaEventDestroy(decStop);
        cudaFree(d_oriData);
        cudaFree(d_decData);
        cudaFree(d_cmpBytes);
        cudaStreamDestroy(stream);
        free(oriData);
        free(decData);
        free(cmpBytes);
        return EXIT_FAILURE;
    }

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup2 = (unsigned char*)malloc(cmpSize2*sizeof(unsigned char));
    cudaMemcpyAsync(cmpBytes_dup2, d_cmpBytes, cmpSize2*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    cudaMemsetAsync(d_cmpBytes, 0, sizeof(float)*nbEle, stream); // set to zero for double check.
    cudaMemcpyAsync(d_cmpBytes, cmpBytes_dup2, cmpSize2*sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // cuSZp decompression.
    cudaEventRecord(decStart, stream);
    cuSZp_decompress_1D_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize2, errorBound, stream);
    cudaEventRecord(decStop, stream);
    cudaEventSynchronize(decStop);
    float decTime = 0.0f;
    cudaEventElapsedTime(&decTime, decStart, decStop);

    // Print result.
    printf("cuSZp-p finished!\n");
    printf("cuSZp-p compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/cmpTime);
    printf("cuSZp-p decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/decTime);
    printf("cuSZp-p compression ratio: %f\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize2*sizeof(unsigned char)/1024.0/1024.0));

    // Error check.
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize2*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i++)
    {
        if(fabs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], fabs(oriData[i]-decData[i]), errorBound);
        }
    }
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);
    printf("\033[1mDone with testing cuSZp-p on REL 1E-2!\033[0m\n");

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup2);
    cudaEventDestroy(cmpStart);
    cudaEventDestroy(cmpStop);
    cudaEventDestroy(decStart);
    cudaEventDestroy(decStop);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);

    return 0;
}
