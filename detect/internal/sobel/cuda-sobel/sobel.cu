#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <cmath>

#include "cuda_runtime.h"
#include "sobel.h"

#define DEBUG 0

#define CHECK_CUDART_ERROR(call)                                       \
    do {                                                               \
        cudaError_t status = call;                                     \
        if (status != cudaSuccess) {                                   \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(status));             \
            return 1;                                                  \
        }                                                              \
    } while (0)

__global__ void monochrome_kernel(uchar4 *data, int w, int h) {
    int image_size = w * h;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < image_size) {
        float Y = 0.299 * (float)data[i].x + 0.587 * (float)data[i].y +
                  0.114 * (float)data[i].z;
        data[i] = make_uchar4(Y, Y, Y, 0);
    }
}

__global__ void sobel_vertical_kernel(const uchar4 *input, uchar4 *output,
                                      int w, int h) {
    int image_size = w * h;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < image_size) {
        int i = idx % w;
        int j = idx / w;
        unsigned char Y =   (*(input + (j - 1) + w * (i - 1))).x *  1 +
                            (*(input + (j    ) + w * (i - 1))).x *  0 +
                            (*(input + (j + 1) + w * (i - 1))).x * -1 +

                            (*(input + (j - 1) + w * (i    ))).x *  2 +
                            (*(input + (j    ) + w * (i    ))).x *  0 +
                            (*(input + (j + 1) + w * (i    ))).x * -2 +

                            (*(input + (j - 1) + w * (i + 1))).x *  1 +
                            (*(input + (j    ) + w * (i + 1))).x *  0 +
                            (*(input + (j + 1) + w * (i + 1))).x * -1;
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}

__global__ void sobel_horizontal_kernel(const uchar4 *input, uchar4 *output,
                                      int w, int h) {
    int image_size = w * h;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < image_size) {
        int i = idx % w;
        int j = idx / w;
        unsigned char Y =   (*(input + (j - 1) + w * (i - 1))).x *  1 +
                            (*(input + (j    ) + w * (i - 1))).x *  2 +
                            (*(input + (j + 1) + w * (i - 1))).x *  1 +

                            (*(input + (j - 1) + w * (i    ))).x *  0 +
                            (*(input + (j    ) + w * (i    ))).x *  0 +
                            (*(input + (j + 1) + w * (i    ))).x *  0 +

                            (*(input + (j - 1) + w * (i + 1))).x * -1 +
                            (*(input + (j    ) + w * (i + 1))).x * -2 +
                            (*(input + (j + 1) + w * (i + 1))).x * -1;
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}

__global__ void root_kernel(const uchar4 *input_vertical,
                            const uchar4 *input_horizontal, uchar4 *output,
                            int w, int h) {
    int image_size = w * h;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < image_size) {
        int i = idx % w;
        int j = idx / w;
        unsigned char Y = sqrt(pow((*(input_vertical + j + w * i)).x, 2.0) +
                               pow((*(input_horizontal + j + w * i)).x, 2.0));
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}

int ApplySobel(uint32_t *data, int w, int h) {
    cudaStream_t stream;
    CHECK_CUDART_ERROR(cudaStreamCreate(&stream));

    size_t image_byte_size = sizeof(uchar4) * w * h;

    uchar4 *gpu_data;
    CHECK_CUDART_ERROR(cudaMalloc(&gpu_data, image_byte_size));

    CHECK_CUDART_ERROR(cudaMemcpyAsync(gpu_data, data, image_byte_size,
                                       cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;
    int numBlocks = w * h / threadsPerBlock + 1;

    monochrome_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(gpu_data, w,
                                                                 h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    uchar4 *output_vertical;
    CHECK_CUDART_ERROR(cudaMalloc(&output_vertical, image_byte_size));
    sobel_vertical_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data, output_vertical, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    uchar4 *output_horizontal;
    CHECK_CUDART_ERROR(cudaMalloc(&output_horizontal, image_byte_size));
    sobel_horizontal_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data, output_horizontal, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    uchar4 *output;
    CHECK_CUDART_ERROR(cudaMalloc(&output, image_byte_size));
    root_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_vertical, output_horizontal, output, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDART_ERROR(cudaMemcpyAsync(data, output, image_byte_size,
                                       cudaMemcpyDeviceToHost, stream));
                                       
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDART_ERROR(cudaStreamDestroy(stream));

    CHECK_CUDART_ERROR(cudaFree(output));
    CHECK_CUDART_ERROR(cudaFree(output_horizontal));
    CHECK_CUDART_ERROR(cudaFree(output_vertical));
    CHECK_CUDART_ERROR(cudaFree(gpu_data));

    return 0;
}
