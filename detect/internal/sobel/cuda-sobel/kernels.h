#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>

#include "cuda_runtime.h"

__device__ float gaussianFilterKernel[3][3] = { 1.0f, 2.0f, 1.0f, 
                                                2.0f, 4.0f, 2.0f, 
                                                1.0f, 2.0f, 1.0f };
                                                
__device__ float sobelHorizontalFilterKernel[3][3] = { -1.0f, 0.0f, 1.0f, 
                                                       -2.0f, 0.0f, 2.0f, 
                                                       -1.0f, 0.0f, 1.0f };

__device__ float sobelVerticalFilterKernel[3][3] = { -1.0f, -2.0f, -1.0f, 
                                                      0.0f,  0.0f,  0.0f, 
                                                      1.0f,  2.0f,  1.0f };

__global__ void toFloatKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        uchar4 pixel_uchar4 = tex2D<uchar4>(inTexObj, u, v);
        float4 pixel_float4 = make_float4(pixel_uchar4.x, pixel_uchar4.y, pixel_uchar4.z, pixel_uchar4.w);
        surf2Dwrite(pixel_float4, outSurfObj, x * sizeof(float4), y);
    }
}

__global__ void greyscaleKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        float4 pixel = tex2D<float4>(inTexObj, u, v);

        float R =  pixel.x / 255.0f;
        float G =  pixel.y / 255.0f;
        float B =  pixel.z / 255.0f;

        float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;

        if (Y > 0.0031308f) {
            Y = 1.055f * powf(Y, 1.0f / 2.4f) - 0.055f;
        } else {
            Y = 12.92f * Y;
        }
        Y *= 255.0f;

        surf2Dwrite(Y, outSurfObj, x * sizeof(float), y);
    }
}

__global__ void gaussianKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i, j;
        float u, v;

        float result = 0.0f;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                u = (x + j - 1) / static_cast<float>(width);
                v = (y + i - 1) / static_cast<float>(height);
                result += tex2D<float>(inTexObj, u, v) * gaussianFilterKernel[i][j];
            }
        }
        result /= 16.0f;

        surf2Dwrite(result, outSurfObj, x * sizeof(float), y);
    }
}

__global__ void sobelKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i, j;
        float u, v;

        float resultHorizontal = 0.0f;
        float resultVertical = 0.0f;
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                u = (x + j - 1) / static_cast<float>(width);
                v = (y + i - 1) / static_cast<float>(height);
                resultHorizontal += tex2D<float>(inTexObj, u, v) * sobelHorizontalFilterKernel[i][j];
                resultVertical += tex2D<float>(inTexObj, u, v) * sobelVerticalFilterKernel[i][j];
            }
        }
        float result = sqrtf(powf(resultHorizontal, 2.0) + powf(resultVertical, 2.0));
        float resultDirection = atanf(resultVertical / resultHorizontal);

        surf2Dwrite(result, outSurfObj, x * sizeof(float), y);
    }
}

__global__ void toRGBaKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        float pixel = tex2D<float>(inTexObj, u, v);
        uchar4 pixel_uchar4 = make_uchar4(pixel, pixel, pixel, 0);
        surf2Dwrite(pixel_uchar4, outSurfObj, x * sizeof(uchar4), y);
    }
}

#endif // KERNELS_H
