#include <stdint.h>

#include <cmath>
#include <cstdio>
#include <cstring>

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

__global__ void monochrome_kernel(uchar4 *input, float *output, int image_size,
                                  int w) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < image_size) {
        output[i] = 0.299 * (float)input[i].x + 0.587 * (float)input[i].y +
                    0.114 * (float)input[i].z;
    }
}

__global__ void gaussian_kernel(const float *input, float *output, int image_size, int w, int h) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (i < image_size && i > 1 && j > 1 && i < h - 1 && j < w - 1) {
        *(output + j + w * i) = (*(input + (j - 1) + w * (i - 1)) * 1 +
                                *(input + (j    ) + w * (i - 1)) * 2 +
                                *(input + (j + 1) + w * (i - 1)) * 1 +

                                *(input + (j - 1) + w * (i    )) * 2 +
                                *(input + (j    ) + w * (i    )) * 4 +
                                *(input + (j + 1) + w * (i    )) * 2 +

                                *(input + (j - 1) + w * (i + 1)) * 1 +
                                *(input + (j    ) + w * (i + 1)) * 2 +
                                *(input + (j + 1) + w * (i + 1)) * 1) / 16.0;
    }
}

__global__ void sobel_vertical_kernel(const float *input, float *output,
                                      int image_size, int w, int h) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size && i > 1 && j > 1 && i < h - 1 && j < w - 1) {
        *(output + j + w * i) = *(input + (j - 1) + w * (i - 1)) *  1 +
                                *(input + (j    ) + w * (i - 1)) *  0 +
                                *(input + (j + 1) + w * (i - 1)) * -1 +

                                *(input + (j - 1) + w * (i    )) *  2 +
                                *(input + (j    ) + w * (i    )) *  0 +
                                *(input + (j + 1) + w * (i    )) * -2 +

                                *(input + (j - 1) + w * (i + 1)) *  1 +
                                *(input + (j    ) + w * (i + 1)) *  0 +
                                *(input + (j + 1) + w * (i + 1)) * -1;
    }
}

__global__ void sobel_horizontal_kernel(const float *input, float *output,
                                        int image_size, int w, int h) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size && i > 1 && j > 1 && i < h - 1 && j < w - 1) {
        *(output + j + w * i) = *(input + (j - 1) + w * (i - 1)) *  1 +
                                *(input + (j    ) + w * (i - 1)) *  2 +
                                *(input + (j + 1) + w * (i - 1)) *  1 +

                                *(input + (j - 1) + w * (i    )) *  0 +
                                *(input + (j    ) + w * (i    )) *  0 +
                                *(input + (j + 1) + w * (i    )) *  0 +

                                *(input + (j - 1) + w * (i + 1)) * -1 +
                                *(input + (j    ) + w * (i + 1)) * -2 +
                                *(input + (j + 1) + w * (i + 1)) * -1;
    }
}

__global__ void root_kernel(const float *input_vertical,
                            const float *input_horizontal, float *output,
                            int image_size, int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size) {
        *(output + j + w * i) = sqrtf(powf(*(input_vertical + j + w * i), 2.0) +
                        powf(*(input_horizontal + j + w * i), 2.0));
    }
}

__global__ void conv_float_uchar4(const float *input, uchar4 *output,
                                  int image_size, int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size) {
        float Y = *(input + j + w * i);
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}

int ApplySobel(uint32_t *data, int w, int h) {
    cudaStream_t stream;
    CHECK_CUDART_ERROR(cudaStreamCreate(&stream));

    int image_size = w * h;

    size_t image_byte_size = sizeof(uchar4) * image_size;
    int image_float_size = sizeof(float) * image_size;

    int threadsPerBlock = 256;
    int numBlocks = image_size / threadsPerBlock + 1;

    // Copying input data into GPU
    uchar4 *gpu_data;
    CHECK_CUDART_ERROR(cudaMalloc(&gpu_data, image_byte_size));
    CHECK_CUDART_ERROR(cudaMemcpyAsync(gpu_data, data, image_byte_size,
                                       cudaMemcpyHostToDevice, stream));

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Сolor (RGBA) to grayscale (float value) conversion
    float *output_monochrome;
    CHECK_CUDART_ERROR(cudaMalloc(&output_monochrome, image_float_size));

    monochrome_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data, output_monochrome, image_size, w);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Gaussian blur
    float *output_gaussian;
    CHECK_CUDART_ERROR(cudaMalloc(&output_gaussian, image_float_size));

    gaussian_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_monochrome, output_gaussian, image_size, w, h);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Sobel vertical core
    float *output_vertical;
    CHECK_CUDART_ERROR(cudaMalloc(&output_vertical, image_float_size));

    sobel_vertical_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_monochrome, output_vertical, image_size, w, h);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Sobel horizontal core
    float *output_horizontal;
    CHECK_CUDART_ERROR(cudaMalloc(&output_horizontal, image_float_size));

    sobel_horizontal_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_monochrome, output_horizontal, image_size, w, h);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Composing vertical and horizontal outputs in an image
    float *output_sobel;
    CHECK_CUDART_ERROR(cudaMalloc(&output_sobel, image_float_size));

    root_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_vertical, output_horizontal, output_sobel, image_size, w);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Convert float data back to RGBA
    uchar4 *output;
    CHECK_CUDART_ERROR(cudaMalloc(&output, image_byte_size));

    conv_float_uchar4<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_sobel, output, image_size, w);

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Copying output data into CPU
    CHECK_CUDART_ERROR(cudaMemcpyAsync(data, output, image_byte_size,
                                       cudaMemcpyDeviceToHost, stream));

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDART_ERROR(cudaStreamDestroy(stream));

    CHECK_CUDART_ERROR(cudaFree(gpu_data));
    CHECK_CUDART_ERROR(cudaFree(output_monochrome));
    CHECK_CUDART_ERROR(cudaFree(output_gaussian));
    CHECK_CUDART_ERROR(cudaFree(output_vertical));
    CHECK_CUDART_ERROR(cudaFree(output_horizontal));
    CHECK_CUDART_ERROR(cudaFree(output_sobel));
    CHECK_CUDART_ERROR(cudaFree(output));

    return 0;
}
