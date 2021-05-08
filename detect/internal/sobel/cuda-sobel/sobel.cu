#include <stdint.h>

#include <cstdio>
#include <cstring>

#include "sobel.h"
#include "utils.cu"

int ApplySobel(uint32_t* data, int w, int h) {
    // Copy data to device
    cudaArray_t gpuData;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    CHECK_CUDART_ERROR(cudaMallocArray(&gpuData, &channel_desc, w, h));

    const size_t spitch = w * sizeof(uint32_t);
    CHECK_CUDART_ERROR(cudaMemcpy2DToArray(gpuData, 0, 0, data, spitch, w * sizeof(uchar4), h, cudaMemcpyHostToDevice));
    
    // Convert data to float values
    cudaArray_t gpuDataFloat4;
    createCudaArray<float4>(gpuDataFloat4, nullptr, w, h);
    convertToFloat(gpuData, gpuDataFloat4, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuData));

    // Applying greyscale filter
    cudaArray_t greyscaleOut;
    createCudaArray<float>(greyscaleOut, nullptr, w, h);
    applyKernel(greyscaleKernel, gpuDataFloat4, greyscaleOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuDataFloat4));

    // Applying gaussian blur filter
    cudaArray_t gaussianOut;
    createCudaArray<float>(gaussianOut, nullptr, w, h);
    applyKernel(gaussianKernel, greyscaleOut, gaussianOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(greyscaleOut));

    // Applying edge detecting filter
    cudaArray_t edgeOut;
    createCudaArray<float2>(edgeOut, nullptr, w, h);
    applyKernel(sobelKernel, gaussianOut, edgeOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gaussianOut));

    // Convert back to RGBA
    cudaArray_t output;
    createCudaArray<uchar4>(output, nullptr, w, h);
    applyKernel(convertToRGBaKernel, edgeOut, output, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(edgeOut));

    // Copy data back to host
    CHECK_CUDART_ERROR(cudaMemcpy2DFromArray(data, w * sizeof(uint32_t), output, 0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));
    CHECK_CUDART_ERROR(cudaFreeArray(output));

    return 0;
}
