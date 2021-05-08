#ifndef UTILS_CU
#define UTILS_CU

#include <iostream>

#include "timer.hpp"
#include "kernels.cu"

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
int createCudaArray(cudaArray_t& cuArray, const void* data, int width, int height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channel_desc, width, height));
    if (data) {
        const size_t spitch = width * sizeof(T);
        CHECK_CUDART_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(T), height, cudaMemcpyHostToDevice));
    }

    return 0;
}

/*  
    A function that applies the selected kernel to the input cudaArray_t array and writes the result to the output cudaArray_t
    Please make sure the input and output array types are the same as the kernel input and output arrays. 
        @param K kernel function.
        @param inCuArray cudaArray_t of input data.
        @param outCuArray cudaArray_t of output data.
        @param width width of the image.
        @param height height of the image.
*/
template <typename K>
int applyKernel(K kernel, const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks(( width + threadsperBlock.x - 1) / threadsperBlock.x,
                   (height + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaTextureObject_t inTexObj = 0;
    createTextureObject(inTexObj, inCuArray);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, outCuArray);

    std::cout << "Invoking a kernel ..." << std::endl;
    {
        timer t;
        kernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
        CHECK_CUDART_ERROR(cudaDeviceSynchronize());
    }

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

/*  
    Conversion from uchar4 to float4 format. A separate function is needed because cudaFilterModeLinear cannot be applied to an array of non floats
        @param inCuArray cudaArray_t of uchar4.
        @param outCuArray cudaArray_t of float4.
*/
int convertToFloat(const cudaArray_t& inCuArray, cudaArray_t& outCuArray, int width, int height) {
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

    convertToFloatKernel<<<numBlocks, threadsperBlock>>>(inTexObj, outSurfObj, width, height);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(inTexObj));

    return 0;
}

#endif // UTILS_CU
