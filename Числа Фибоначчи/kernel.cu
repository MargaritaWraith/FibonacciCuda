
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Math.h>
#include <stdio.h>

cudaError_t PhibWithCuda(int *Phib, unsigned int size);


__global__ void PhibKernel(int *Phib)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Phib[i] = int(pow(1.6180339887, (int)i) / 2.236067977 + 0.5); // pow(1.6180339887,(int)i)
}

int main()
{
	const int arraySize = 100;
	int Phib[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = PhibWithCuda(Phib, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i < arraySize; i++)
	{
		printf("Phib[%d] = %d\n", i + 1, Phib[i]);
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t PhibWithCuda(int *Phib, unsigned int size)
{
	int *dev_Phib = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_Phib, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 block(32, 1);
	dim3 grid((size / 32), 1);
	PhibKernel << <grid, block >> > (dev_Phib);

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
	cudaStatus = cudaMemcpy(Phib, dev_Phib, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_Phib);

	return cudaStatus;
}
