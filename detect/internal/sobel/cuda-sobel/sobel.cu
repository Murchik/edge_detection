#include <stdint.h>

#include <cstdio>
#include <cstring>

#include "cuda_kernels.h"
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

int ApplySobel(uint32_t *data, int w, int h) {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channelDesc, w, h));

    // Set pitch of the source 
    // (the width in memory in bytes of the 2D array pointed to by src, including padding)
    const size_t spitch = w * sizeof(uchar4);
    // Copy data located in host memory to device memory
    CHECK_CUDART_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDART_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Allocate result of transformation in device memory
    uchar4 *output;
    CHECK_CUDART_ERROR(cudaMalloc(&output, w * h * sizeof(uchar4)));

    // Invoke kernel
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((w + threadsperBlock.x - 1) / threadsperBlock.x,
                   (h + threadsperBlock.y - 1) / threadsperBlock.y);
    greyscaleKernel<<<numBlocks, threadsperBlock>>>(output, texObj, w, h);

    // Copy data from device back to host
    CHECK_CUDART_ERROR(cudaMemcpy(data, output, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Destroy texture object
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(texObj));

    // Free device memory
    CHECK_CUDART_ERROR(cudaFreeArray(cuArray));
    CHECK_CUDART_ERROR(cudaFree(output));

    return 0;
}
