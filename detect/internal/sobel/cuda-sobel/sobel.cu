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
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
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
int createCudaArray(cudaArray_t& cuArray, int width, int height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channel_desc, width, height));
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

// int applySobel(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
//     dim3 threadsperBlock(16, 16);
//     dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
//                    (height + threadsperBlock.y - 1) / threadsperBlock.y);

//     // Applying vertical kernel of the Sobel filter
//     cudaTextureObject_t inTexObj = 0;
//     createTextureObject(inTexObj, inCuArray);    

//     cudaArray_t outSobelHorizontal;
//     createCudaArray<float>(outSobelHorizontal, nullptr, width, height);
//     cudaSurfaceObject_t outSobelHorizontalSurfObj;
//     createSurfaceObject(outSobelHorizontalSurfObj, outSobelHorizontal);

//     sobelHorizontalKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSobelHorizontalSurfObj, width, height);
//     CHECK_CUDART_ERROR(cudaDeviceSynchronize());

//     CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSobelHorizontalSurfObj));
//     CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

//     // Applying horizontal kernel of the Sobel filter
//     cudaArray_t inCuArrayCopy;
//     createCudaArray<float>(inCuArrayCopy, inCuArray, width, height);
//     cudaTextureObject_t inTexObjCopy = 0;
//     createTextureObject(inTexObjCopy, inCuArrayCopy);

//     cudaArray_t outSobelVertical;
//     createCudaArray<float>(outSobelVertical, nullptr, width, height);
//     cudaSurfaceObject_t outSobelVerticalSurfObj;
//     createSurfaceObject(outSobelVerticalSurfObj, outSobelVertical);

//     sobelVerticalKernel<<<numBlocks, threadsperBlock>>>(inTexObjCopy, outSobelVerticalSurfObj, width, height);
//     CHECK_CUDART_ERROR(cudaDeviceSynchronize());

//     CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSobelVerticalSurfObj));
//     CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObjCopy));

//     // Getting result of the Sobel filter
//     cudaSurfaceObject_t outSurfObj;
//     createSurfaceObject(outSurfObj, outCuArray);

//     cudaTextureObject_t inHorizontalTexObj = 0;
//     createTextureObject(inHorizontalTexObj, outSobelHorizontal);

//     cudaTextureObject_t inVerticalTexObj = 0;
//     createTextureObject(inVerticalTexObj, outSobelVertical);

//     rootKernel<<<numBlocks, threadsperBlock>>>(inHorizontalTexObj, inVerticalTexObj, outSurfObj, width, height);
//     CHECK_CUDART_ERROR(cudaDeviceSynchronize());

//     CHECK_CUDART_ERROR(cudaDestroyTextureObject(inVerticalTexObj));
//     CHECK_CUDART_ERROR(cudaDestroyTextureObject(inHorizontalTexObj));
//     CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));

//     CHECK_CUDART_ERROR(cudaFreeArray(outSobelVertical));
//     CHECK_CUDART_ERROR(cudaFreeArray(inCuArrayCopy));
//     CHECK_CUDART_ERROR(cudaFreeArray(outSobelHorizontal));
// }

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
        
    cudaArray_t gpuDataFloat4;
    createCudaArray<float4>(gpuDataFloat4, w, h);
    applyToFloat(gpuData, gpuDataFloat4, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuData));

    // Applying greyscale filter
    cudaArray_t greyscaleOut;
    createCudaArray<float>(greyscaleOut, w, h);
    applyGreyscale(gpuDataFloat4, greyscaleOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gpuDataFloat4));

    // Applying gaussian blur filter
    cudaArray_t gaussianOut;
    createCudaArray<float>(gaussianOut, w, h);
    applyGaussian(greyscaleOut, gaussianOut, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(greyscaleOut));

    // // Applying sobel filter
    // cudaArray_t sobelOut;
    // createCudaArray<float>(sobelOut, w, h);
    // applySobel(gaussianOut, sobelOut, w, h);
    // CHECK_CUDART_ERROR(cudaFreeArray(gaussianOut));

    // Convert back to RGBA
    cudaArray_t output;
    createCudaArray<uchar4>(output, w, h);
    convertToRGBA(gaussianOut, output, w, h);
    CHECK_CUDART_ERROR(cudaFreeArray(gaussianOut));

    // Copy data back to host
    CHECK_CUDART_ERROR(cudaMemcpy2DFromArray(data, w * sizeof(uint32_t), output, 0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));
    CHECK_CUDART_ERROR(cudaFreeArray(output));

    return 0;
}
