#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>

#define __CUDACC__
#include "cuda_runtime.h"

__device__ float max_brightness = 1.0f;

/*  
    Conversion from uint32_t RGBA format (8bit, 8bit, 8bit, 8bit) to float4 format (float, float, float, float)
        @param inTexObj texture object of cudaArray_t of uchar4.
        @param outSurfObj surface object of cudaArray_t of float4.
*/
__global__ void convertToFloatKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
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

/*  
    Conversion from float4 format (float, float, float, float) to single float value of pixel brightness
        @param inTexObj texture object of cudaArray_t of float4.
        @param outSurfObj surface object of cudaArray_t of float.
*/
__global__ void greyscaleKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        float4 pixel = tex2D<float4>(inTexObj, u, v);

        float R = pixel.x;
        float G = pixel.y;
        float B = pixel.z;

        float Y = sqrtf(0.299 * powf(R, 2.0f) + 0.587 * powf(G, 2.0f) + 0.114 * powf(B, 2.0f));
        // float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;

        // if (Y > 0.0031308f) {
        //     Y = 1.055f * powf(Y, 1.0f / 2.4f) - 0.055f;
        // } else {
        //     Y = 12.92f * Y;
        // }
        // Y *= 255.0f;

        surf2Dwrite(Y, outSurfObj, x * sizeof(float), y);
    }
}

__device__ float gaussianFilterKernel[3][3] = { 1.0f, 2.0f, 1.0f, 
                                                2.0f, 4.0f, 2.0f, 
                                                1.0f, 2.0f, 1.0f };

/*  
    Applying Gaussian blur to an array of floats representing the brightness of pixels
        @param inTexObj texture object of cudaArray_t of float.
        @param outSurfObj surface object of cudaArray_t of float.
*/
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

__device__ float sobelHorizontalKernel[3][3] = { -1.0f, 0.0f, 1.0f, 
                                                 -2.0f, 0.0f, 2.0f, 
                                                 -1.0f, 0.0f, 1.0f };

__device__ float sobelVerticalKernel[3][3] = { -1.0f, -2.0f, -1.0f, 
                                                0.0f,  0.0f,  0.0f, 
                                                1.0f,  2.0f,  1.0f };

/*  
    Applying Sobel operator to an array of floats representing the brightness of pixels
        @param inTexObj texture object of cudaArray_t of float.
        @param outSurfObj surface object of cudaArray_t of float2. The first float is the gradient's magnitude, and the second is the gradient's direction.
*/
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
                resultHorizontal += tex2D<float>(inTexObj, u, v) * sobelHorizontalKernel[i][j];
                resultVertical += tex2D<float>(inTexObj, u, v) * sobelVerticalKernel[i][j];
            }
        }
        float result = sqrtf(powf(resultHorizontal, 2.0) + powf(resultVertical, 2.0));
        float resultDirection = atanf(resultVertical / resultHorizontal);

        if (result > max_brightness) {
            max_brightness = result;
        }

        float2 result_float2 = make_float2(result, resultDirection);

        surf2Dwrite(result_float2, outSurfObj, x * sizeof(float2), y);
    }
}

__device__ float robertsHorizontalKernel[2][2] = { 1.0f,  0.0f,
                                                   0.0f, -1.0f};

__device__ float robertsVerticalKernel[2][2] = { 0.0f, 1.0f,
                                                -1.0f, 0.0f};

/*  
    Applying Roberts cross operator to an array of floats representing the brightness of pixels
        @param inTexObj texture object of cudaArray_t of float.
        @param outSurfObj surface object of cudaArray_t of float2. The first float is the gradient's magnitude, and the second is the gradient's direction.
*/
__global__ void robertsKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i, j;
        float u, v;

        float resultHorizontal = 0.0f;
        float resultVertical = 0.0f;
        for (i = 0; i < 2; i++) {
            for (j = 0; j < 2; j++) {
                u = (x + j) / static_cast<float>(width);
                v = (y + i) / static_cast<float>(height);
                resultHorizontal += tex2D<float>(inTexObj, u, v) * robertsHorizontalKernel[i][j];
                resultVertical += tex2D<float>(inTexObj, u, v) * robertsVerticalKernel[i][j];
            }
        }
        float result = sqrtf(powf(resultHorizontal, 2.0) + powf(resultVertical, 2.0));
        float resultDirection = atanf(resultVertical / resultHorizontal) + M_PI / 4.0f;

        if (result > max_brightness) {
            max_brightness = result;
        }

        float2 result_float2 = make_float2(result, resultDirection);

        surf2Dwrite(result_float2, outSurfObj, x * sizeof(float2), y);
    }
}

__device__ float prewittHorizontalKernel[3][3] = { 1.0f, 0.0f, -1.0f, 
                                                   1.0f, 0.0f, -1.0f, 
                                                   1.0f, 0.0f, -1.0f };

__device__ float prewittVerticalKernel[3][3] = {  1.0f,  1.0f,  1.0f, 
                                                  0.0f,  0.0f,  0.0f, 
                                                 -1.0f, -1.0f, -1.0f };

/*  
    Applying Prewitt operator  to an array of floats representing the brightness of pixels
        @param inTexObj texture object of cudaArray_t of float.
        @param outSurfObj surface object of cudaArray_t of float2. The first float is the gradient's magnitude, and the second is the gradient's direction.
*/
__global__ void prewittKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
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
                resultHorizontal += tex2D<float>(inTexObj, u, v) * prewittHorizontalKernel[i][j];
                resultVertical += tex2D<float>(inTexObj, u, v) * prewittVerticalKernel[i][j];
            }
        }
        float result = sqrtf(powf(resultHorizontal, 2.0) + powf(resultVertical, 2.0));
        float resultDirection = atanf(resultVertical / resultHorizontal);

        if (result > max_brightness) {
            max_brightness = result;
        }

        float2 result_float2 = make_float2(result, resultDirection);

        surf2Dwrite(result_float2, outSurfObj, x * sizeof(float2), y);
    }
}
/*  
    Converting float data to RGBA format where color of the pixel based on a direction of the gradient
        @param inTexObj texture object of cudaArray_t of float2. The first float is the gradient's magnitude, and the second is the gradient's direction.
        @param outSurfObj surface object of cudaArray_t of uchar4.
*/
__global__ void convertToRGBaKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        float2 pixel_float2 = tex2D<float2>(inTexObj, u, v);

        float pixel = pixel_float2.x;
        pixel *= (255.0f / max_brightness);

        float pixel_atan = pixel_float2.y;
        float R, G, B;
        if (pixel_atan > (M_PI / 6.0f)) {
            R = (pixel_atan - M_PI / 6.0f) / (M_PI / 3.0f);
            G = 0.0f;
            B = 1.0f - (pixel_atan - M_PI / 6.0f) / (M_PI / 3.0f);
        } else if (pixel_atan < (M_PI / 6.0f) && pixel_atan > -(M_PI / 6.0f)) {
            R = 0.0f;
            G = 1.0f - (pixel_atan + M_PI / 6.0f) / (M_PI / 3.0f);
            B = (pixel_atan + M_PI / 6.0f) / (M_PI / 3.0f);
        } else {
            R = 1.0f - (pixel_atan + M_PI / 2.0f) / (M_PI / 3.0f);
            G = (pixel_atan + M_PI / 2.0f) / (M_PI / 3.0f);
            B = 0.0f;
        }
        R *= 255.0f * (pixel / 255.0f);
        G *= 255.0f * (pixel / 255.0f);
        B *= 255.0f * (pixel / 255.0f);

        uchar4 pixel_uchar4 = make_uchar4(R, G, B, 0);
        surf2Dwrite(pixel_uchar4, outSurfObj, x * sizeof(uchar4), y);
    }
}

__global__ void convertToGreyKernel(cudaTextureObject_t inTexObj, cudaSurfaceObject_t outSurfObj, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = x / static_cast<float>(width);
        float v = y / static_cast<float>(height);

        float2 pixel_float2 = tex2D<float2>(inTexObj, u, v);

        float pixel = pixel_float2.x;
        pixel *= (255.0f / max_brightness);

        uchar4 pixel_uchar4 = make_uchar4(pixel, pixel, pixel, 0);
        surf2Dwrite(pixel_uchar4, outSurfObj, x * sizeof(uchar4), y);
    }
}

#endif // KERNELS_H
