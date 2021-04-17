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

__global__ void monochrome_kernel(const uchar4 *input, float *output,
                                  int image_size, int w) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < image_size) {
        float R = (float)input[i].x / 255.0;
        float G = (float)input[i].y / 255.0;
        float B = (float)input[i].z / 255.0;

        float Y_linear = 0.2126 * R + 0.7152 * G + 0.0722 * B;

        if (Y_linear > 0.0031308) {
            output[i] = 1.055 * powf(Y_linear, 1.0 / 2.4) - 0.055;
        } else {
            output[i] = 12.92 * Y_linear;
        }
    }
}

__global__ void gaussian_kernel(const float *input, float *output,
                                int image_size, int w, int h) {
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
        *(output + j + w * i) =
            sqrtf(powf(*(input_vertical + j + w * i), 2.0) +
                  powf(*(input_horizontal + j + w * i), 2.0));
    }
}

__global__ void conv_float_uchar4(const float *input, uchar4 *output,
                                  int image_size, int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size) {
        float Y = *(input + j + w * i) * 255.0;
        if (Y > 255.0) Y = 255.0;
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}

int ApplySobel(uint32_t *data, int w, int h) {
    cudaStream_t stream;
    CHECK_CUDART_ERROR(cudaStreamCreate(&stream));

    int image_size = w * h;

    int threadsPerBlock = 256;
    int numBlocks = image_size / threadsPerBlock + 1;

    uchar4 *gpu_data_uchar4;
    size_t image_byte_size = sizeof(uchar4) * image_size;
    CHECK_CUDART_ERROR(cudaMalloc(&gpu_data_uchar4, image_byte_size));
    float *gpu_data_float;
    int image_float_size = sizeof(float) * image_size;
    CHECK_CUDART_ERROR(cudaMalloc(&gpu_data_float, image_byte_size));

    float *output_gaussian;
    CHECK_CUDART_ERROR(cudaMalloc(&output_gaussian, image_float_size));
    float *output_vertical;
    CHECK_CUDART_ERROR(cudaMalloc(&output_vertical, image_float_size));
    float *output_horizontal;
    CHECK_CUDART_ERROR(cudaMalloc(&output_horizontal, image_float_size));

    // Copying data into GPU
    CHECK_CUDART_ERROR(cudaMemcpyAsync(gpu_data_uchar4, data, image_byte_size,
                                       cudaMemcpyHostToDevice, stream));

    // Color (RGBA) to grayscale (float value) conversion
    monochrome_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data_uchar4, gpu_data_float, image_size, w);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Gaussian blur
    gaussian_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data_float, output_gaussian, image_size, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Sobel vertical core
    sobel_vertical_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_gaussian, output_vertical, image_size, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Sobel horizontal core
    sobel_horizontal_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_gaussian, output_horizontal, image_size, w, h);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Composing vertical and horizontal outputs in an image
    root_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        output_vertical, output_horizontal, gpu_data_float, image_size, w);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Convert float data back to RGBA
    conv_float_uchar4<<<numBlocks, threadsPerBlock, 0, stream>>>(
        gpu_data_float, gpu_data_uchar4, image_size, w);
    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));

    // Copying output data into CPU
    CHECK_CUDART_ERROR(cudaMemcpyAsync(data, gpu_data_uchar4, image_byte_size,
                                       cudaMemcpyDeviceToHost, stream));

    CHECK_CUDART_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDART_ERROR(cudaStreamDestroy(stream));

    CHECK_CUDART_ERROR(cudaFree(gpu_data_uchar4));
    CHECK_CUDART_ERROR(cudaFree(gpu_data_float));
    CHECK_CUDART_ERROR(cudaFree(output_gaussian));
    CHECK_CUDART_ERROR(cudaFree(output_vertical));
    CHECK_CUDART_ERROR(cudaFree(output_horizontal));

    return 0;
}
