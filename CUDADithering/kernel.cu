
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "BmpUtil.h"
#include "Common.h"

#define	f7_16	112	
#define	f5_16	 80	
#define	f3_16	 48	
#define	f1_16	 16	

byte imgRes[256 * 256 * 4];
byte imgRes2[256 * 256 * 4];
const int width = 256;
const int height = 256;
int imageW, imageH;

cudaError_t makeDitherThreshold(const byte* h_Src, int imageW, int imageH, int bytesPerPixel);
cudaError_t makeDitherFSRgbNbpp(const byte* h_Src, int imageW, int imageH, int bytesPerPixel);

__global__ void test(byte* a)
{
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            for (int k = 0; k < 4; k++) {

                if ((a[i * 256 * 4 + j * 4] + a[i * 256 * 4 + j * 4 + 1] + a[i * 256 * 4 + j * 4 + 2]) / 3 < 127) {
                    a[i * 256 * 4 + j * 4 + 0] = 0;
                    a[i * 256 * 4 + j * 4 + 1] = 0;
                    a[i * 256 * 4 + j * 4 + 2] = 0;
                    a[i * 256 * 4 + j * 4 + 3] = 255;
                }

                else {
                    a[i * 256 * 4 + j * 4 + 0] = 255;
                    a[i * 256 * 4 + j * 4 + 1] = 255;
                    a[i * 256 * 4 + j * 4 + 2] = 255;
                    a[i * 256 * 4 + j * 4 + 3] = 255;
                }

            }
        }
    }

}

__global__ void fsd2(byte* a)
{
    const int size = 256 * 256;

    int* error = (int*)malloc(size * sizeof(int));

    memset(error, 0, size * sizeof(int));

    int	i = 0;

    for (int y = 0; y < height; y++)
    {
        byte* prow = a + (y * width * 4);

        for (int x = 0; x < width; x++, i++)
        {
            const int	blue = prow[x * 4 + 0];
            const int	green = prow[x * 4 + 1];
            const int	red = prow[x * 4 + 2];

            //	Get the pixel gray value.
            int	newVal = (red + green + blue) / 3 + (error[i] >> 8);	//	PixelGray + error correction

            int	newc = (newVal < 128 ? 0 : 255);
            prow[x * 4 + 0] = newc;	//	blue
            prow[x * 4 + 1] = newc;	//	green
            prow[x * 4 + 2] = newc;	//	red

            //	Correction - the new error
            const int	cerror = newVal - newc;

            int idx = i + 1;
            if (x + 1 < width)
                error[idx] += (cerror * f7_16);

            idx += width - 2;
            if (x - 1 > 0 && y + 1 < height)
                error[idx] += (cerror * f3_16);

            idx++;
            if (y + 1 < height)
                error[idx] += (cerror * f5_16);

            idx++;
            if (x + 1 < width && y + 1 < height)
                error[idx] += (cerror * f1_16);
        }
    }

    free(error);
}

int main()
{
    char imageName[] = "peppers.bmp";

    byte* pixels;
    int32 width;
    int32 height;
    int32 bytesPerPixel;
    ReadImage(imageName, &pixels, &width, &height, &bytesPerPixel);

    cudaError_t status = makeDitherThreshold(pixels, width, height, bytesPerPixel);

    status = makeDitherFSRgbNbpp(pixels, width, height, bytesPerPixel);

    free(pixels);

    return 0;
}

cudaError_t makeDitherThreshold(const byte* h_Src, int imageW, int imageH, int bytesPerPixel) {

    byte* imgArray = 0;
    byte resultLocal[256 * 256 * 4];

    cudaError_t error;

    error = cudaMalloc((void**)&imgArray, imageW * imageH * 4 * sizeof(byte));
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(imgArray, h_Src, imageW * imageH * 4 * sizeof(byte), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        error = cudaGetLastError();
        fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(error));
    }

    fsd2 << <1, 1 >> > (imgArray);

    // Check for any errors launching the kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
    }

    error = cudaMemcpy(imgRes, imgArray, imageW * imageH * 4 * sizeof(byte), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Device2Host failed: %s\n", cudaGetErrorString(error));
    }


    char resultName[] = "output2.bmp";

    WriteImage(resultName, imgRes2, imageW, imageH, bytesPerPixel);

    cudaFree(imgArray);
    return error;

}

cudaError_t makeDitherFSRgbNbpp(const byte* h_Src, int imageW, int imageH, int bytesPerPixel) {

    byte* imgArray = 0;
    byte resultLocal[256 * 256 * 4];

    cudaError_t error;

    error = cudaMalloc((void**)&imgArray, imageW * imageH * 4 * sizeof(byte));
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(imgArray, h_Src, imageW * imageH * 4 * sizeof(byte), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        error = cudaGetLastError();
        fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(error));
    }

    fsd2 << <1, 1 >> > (imgArray);

    // Check for any errors launching the kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
    }

    error = cudaMemcpy(imgRes, imgArray, imageW * imageH * 4 * sizeof(byte), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Device2Host failed: %s\n", cudaGetErrorString(error));
    }


    char resultName[] = "output1.bmp";

    WriteImage(resultName, imgRes, imageW, imageH, bytesPerPixel);

    char resultName2[] = "output2.bmp";

    error = cudaMalloc((void**)&imgArray, imageW * imageH * 4 * sizeof(byte));
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! %s\n", cudaGetErrorString(error));
    }

    error = cudaMemcpy(imgArray, h_Src, imageW * imageH * 4 * sizeof(byte), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        error = cudaGetLastError();
        fprintf(stderr, "cudaMemcpy failed! %s\n", cudaGetErrorString(error));
    }

    test << <1, 1 >> > (imgArray);

    // Check for any errors launching the kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
    }

    error = cudaMemcpy(imgRes, imgArray, imageW * imageH * 4 * sizeof(byte), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Device2Host failed: %s\n", cudaGetErrorString(error));
    }

    WriteImage(resultName2, imgRes, imageW, imageH, bytesPerPixel);

    cudaFree(imgArray);
    return error;

};


