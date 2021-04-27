#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>

#include "cuda_runtime.h"

__global__ void greyscaleKernel(cudaTextureObject_t texObj, cudaSurfaceObject_t outputSurfObj, int width, int height) {
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / static_cast<float>(width);
    float v = y / static_cast<float>(height);

    if (x < width && y < height) {
        uchar4 pixel = tex2D<uchar4>(texObj, u, v);;
        
        float R = static_cast<float>(pixel.x) / 255.0f;
        float G = static_cast<float>(pixel.y) / 255.0f;
        float B = static_cast<float>(pixel.z) / 255.0f;

        float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;

        if (Y > 0.0031308f) {
            Y = 1.055f * powf(Y, 1.0f / 2.4f) - 0.055f;
        } else {
            Y = 12.92f * Y;
        }
        Y *= 255.0f;

        pixel = make_uchar4(Y, Y, Y, 0);
        surf2Dwrite(pixel, outputSurfObj, x * 4, y);
    }
}

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

#endif // KERNELS_H
