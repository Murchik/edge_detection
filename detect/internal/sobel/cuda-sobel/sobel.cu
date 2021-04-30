#include <stdint.h>

#include <cstdio>
#include <cstring>

#include "kernels.h"
#include "sobel.h"

#define CHECK_CUDART_ERROR(call)                                       \
    do {                                                               \
        cudaError_t status = call;                                     \
        if (status != cudaSuccess) {                                   \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(status));             \
            return 1;                                                  \
        }                                                              \
    } while (0)

int createTextureObject(cudaTextureObject_t& TexObj, const cudaArray_t& cuArray) {
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CHECK_CUDART_ERROR(cudaCreateTextureObject(&TexObj, &resDesc, &texDesc, NULL));

    return 0;
}

int createSurfaceObject(cudaSurfaceObject_t& SurfObj, cudaArray_t& cuArray) {
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    CHECK_CUDART_ERROR(cudaCreateSurfaceObject(&SurfObj, &resDesc));

    return 0;
}

template <typename T>
int createCudaArray(cudaArray_t& cuArray, const void* data, int width, int height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channel_desc, width, height));
    if (data) {
        const size_t spitch = width * sizeof(T);
        CHECK_CUDART_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(T), height, cudaMemcpyHostToDevice));
    }

    return 0;
}

int applyToFloat(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width  + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = inCuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CHECK_CUDART_ERROR(cudaCreateTextureObject(&inTexObj, &resDesc, &texDesc, NULL));

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    toFloatKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

int applyGreyscale(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width  + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    createTextureObject(inTexObj, inCuArray);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    greyscaleKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

int applyGaussian(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    createTextureObject(inTexObj, inCuArray);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    gaussianKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

int applySobelFilter(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    createTextureObject(inTexObj, inCuArray);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    sobelKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

int convertToRGBA(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    createTextureObject(inTexObj, inCuArray);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    toRGBaKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));
    
    return 0;
}

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
    applyToFloat(gpuData, gpuDataFloat4, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuData));

    // Applying greyscale filter
    cudaArray_t greyscaleOut;
    createCudaArray<float>(greyscaleOut, nullptr, w, h);
    applyGreyscale(gpuDataFloat4, greyscaleOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuDataFloat4));

    // Applying gaussian blur filter
    cudaArray_t gaussianOut;
    createCudaArray<float>(gaussianOut, nullptr, w, h);
    applyGaussian(greyscaleOut, gaussianOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(greyscaleOut));

    // Applying sobel filter
    cudaArray_t sobelOut;
    createCudaArray<float2>(sobelOut, nullptr, w, h);
    applySobelFilter(gaussianOut, sobelOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gaussianOut));

    // Convert back to RGBA
    cudaArray_t output;
    createCudaArray<uchar4>(output, nullptr, w, h);
    convertToRGBA(sobelOut, output, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(sobelOut));

    // Copy data back to host
    CHECK_CUDART_ERROR(cudaMemcpy2DFromArray(data, w * sizeof(uint32_t), output, 0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));
    CHECK_CUDART_ERROR(cudaFreeArray(output));

    return 0;
}
