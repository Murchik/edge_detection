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

int createTextureObject(cudaTextureObject_t& TexObj, cudaArray_t& cuArray)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CHECK_CUDART_ERROR(cudaCreateTextureObject(&TexObj, &resDesc, &texDesc, NULL));

    return 0;
}

int createSurfaceObject(cudaSurfaceObject_t& SurfObj, cudaArray_t& cuArray)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = cuArray;
    CHECK_CUDART_ERROR(cudaCreateSurfaceObject(&SurfObj, &resDesc));

    return 0;
}

template <typename T>
int initializeCudaArray(cudaArray_t& cuArray, const void* data, int width, int height) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channel_desc, width, height));

    if (data) {
        const size_t spitch = width * sizeof(T);
        CHECK_CUDART_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, width * sizeof(T), height, cudaMemcpyHostToDevice));
    }

    return 0;
}

int ApplySobel(uint32_t *data, int w, int h) {
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((w + threadsperBlock.x - 1) / threadsperBlock.x,
                   (h + threadsperBlock.y - 1) / threadsperBlock.y);

    cudaArray_t gpuData;
    initializeCudaArray<uchar4>(gpuData, data, w, h);

    cudaTextureObject_t gpuDataTexObj = 0;
    createTextureObject(gpuDataTexObj, gpuData);

    cudaArray_t output;
    initializeCudaArray<uchar4>(output, nullptr, w, h);

    cudaSurfaceObject_t outSurfObj = 0;
    createSurfaceObject(outSurfObj, output);

    greyscaleKernel<<<numBlocks, threadsperBlock>>>(gpuDataTexObj, outSurfObj, w, h);
    CHECK_CUDART_ERROR(cudaDeviceSynchronize());

    CHECK_CUDART_ERROR(cudaMemcpy2DFromArray(data, w * sizeof(uint32_t), output, 0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));

    CHECK_CUDART_ERROR(cudaDestroySurfaceObject(outSurfObj));

    CHECK_CUDART_ERROR(cudaFreeArray(output));
    CHECK_CUDART_ERROR(cudaFreeArray(gpuData));

    return 0; 
}
