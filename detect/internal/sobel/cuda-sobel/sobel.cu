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
/*
__global__ void monochrome_kernel(const uchar4 *input, float *output, int image_size, int w) {
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

__global__ void sobel_vertical_kernel(const float *input, float *output, int image_size, int w, int h) {
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

__global__ void sobel_horizontal_kernel(const float *input, float *output, int image_size, int w, int h) {
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

__global__ void root_kernel(const float *input_vertical, const float *input_horizontal, float *output, int image_size, int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size) {
        *(output + j + w * i) =
            sqrtf(powf(*(input_vertical + j + w * i), 2.0) + powf(*(input_horizontal + j + w * i), 2.0));
    }
}

__global__ void conv_float_uchar4(const float *input, uchar4 *output, int image_size, int w) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = idx / w;
    int j = idx % w;
    if (idx < image_size) {
        float Y = *(input + j + w * i) * 255.0;
        if (Y > 255.0) Y = 255.0;
        *(output + j + w * i) = make_uchar4(Y, Y, Y, 0);
    }
}
*/

__global__ void transformKernel(uchar4 *output, cudaTextureObject_t texObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / width;
    float v = y / height;

    uchar4 pixel = tex2D<uchar4>(texObj, u, v);

    float R = pixel.x / 255.0f;
    float G = pixel.y / 255.0f;
    float B = pixel.z / 255.0f;

    float Y_linear = 0.2126f * R + 0.7152f * G + 0.0722f * B;

    float Y_sRGB;
    if (Y_linear > 0.0031308f) {
        Y_sRGB = 1.055f * powf(Y_linear, 1.0f / 2.4f) - 0.055f;
    } else {
        Y_sRGB = 12.92f * Y_linear;
    }

    Y_sRGB *= 255.0f;
    output[y * width + x] = make_uchar4(Y_sRGB, Y_sRGB, Y_sRGB, 0);
}

int ApplySobel(uint32_t *data, int w, int h) {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    CHECK_CUDART_ERROR(cudaMallocArray(&cuArray, &channelDesc, w, h));

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = w * sizeof(uchar4);
    // Copy data located at address h_data in host memory to device memory
    CHECK_CUDART_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDART_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Allocate result of transformation in device memory
    uchar4 *output;
    CHECK_CUDART_ERROR(cudaMalloc(&output, w * h * sizeof(uchar4)));

    // Invoke kernel
    // int threadsperBlock = 256;
    // int numBlocks = (w * h) / threadsperBlock + 1;

    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((w + threadsperBlock.x - 1) / threadsperBlock.x,
                   (h + threadsperBlock.y - 1) / threadsperBlock.y);

    transformKernel<<<numBlocks, threadsperBlock>>>(output, texObj, w, h);

    // Copy data from device back to host
    CHECK_CUDART_ERROR(cudaMemcpy(data, output, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Destroy texture object
    CHECK_CUDART_ERROR(cudaDestroyTextureObject(texObj));

    // Free device memory
    CHECK_CUDART_ERROR(cudaFreeArray(cuArray));
    CHECK_CUDART_ERROR(cudaFree(output));

    return 0;
}
