#include <stdint.h>

#include <cstdio>

#include "cuda_runtime.h"
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

/* The index of a thread and its thread ID relate to each other in a
 * straightforward way: For a one-dimensional block, they are the same;
 * for a two-dimensional block of size (Dx, Dy),
 * the thread ID of a thread of index (x, y) is (x + y Dx); */

__global__ void sobel_kernel(uchar4 *data, int w, int h) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tidx; i < w * h; i += stride) {
        float Y = 0.299 * (float)data[i].x + 0.587 * (float)data[i].y +
                  0.114 * (float)data[i].z;
        data[i] = make_uchar4(Y, Y, Y, 0);
    }
}

int ApplySobel(uint32_t *data, int w, int h) {
    cudaStream_t stream;
    CHECK_CUDART_ERROR(cudaStreamCreate(&stream));
    uchar4 *gpu_data;
    size_t image_byte_size = sizeof(uchar4) * w * h;
    CHECK_CUDART_ERROR(cudaMalloc(&gpu_data, image_byte_size));
    CHECK_CUDART_ERROR(cudaMemcpyAsync(gpu_data, data, image_byte_size,
                                       cudaMemcpyHostToDevice, stream));
    sobel_kernel<<<256, 256, 0, stream>>>(gpu_data, w, h);
    CHECK_CUDART_ERROR(cudaMemcpyAsync(data, gpu_data, image_byte_size,
                                       cudaMemcpyDeviceToHost, stream));
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDART_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDART_ERROR(cudaFree(gpu_data));
    return 0;
}
