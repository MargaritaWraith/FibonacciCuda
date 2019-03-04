#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Math.h>
#include <stdio.h>
#include <locale>

cudaError_t PhibWithCuda(unsigned long long *Phib, unsigned int size);


__global__ void PhibKernel(unsigned long long *Phib)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Phib[i] = unsigned long long(pow(1.6180339887498948482045868343656, i) / 2.2360679774997896964091736687313 + 0.5);

	// 1/sqrt(5) = 0.44721359549995793928183473374626
	// Phi = (1+sqrt(5))/2 = 1.6180339887498948482045868343656  Золотое сечение!  +/-Phi - являются корнями характеристического уравнения x^{2}-x-1=0
	// sqrt(5) = 2.2360679774997896964091736687313
}

int main()
{
	setlocale(LC_ALL, "Russian");
	const int arraySize = 64;
	unsigned long long Phib[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = PhibWithCuda(Phib, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i < arraySize; i++)
	{
		printf("Phib[%d] = %llu\n", i + 1, Phib[i]);
	}


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t PhibWithCuda(unsigned long long *Phib, unsigned int size)
{
	unsigned long long *dev_Phib = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers.
	cudaStatus = cudaMalloc((void**)&dev_Phib, size * sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	PhibKernel << <2, 32 >> > (dev_Phib);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Phib, dev_Phib, size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_Phib);

	return cudaStatus;
}
