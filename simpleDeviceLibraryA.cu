/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 // System includes.
#include <stdio.h>
#include <iostream>

 // STL.
#include <vector>

 // CUDA runtime.
#include <cuda_runtime.h>

 // Helper functions and utilities to work with CUDA.
#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned int uint;

using std::cout;
using std::endl;
using std::vector;

__device__ float multiplyByTwo(float number)
{
    return number * 2.0f;
}


__global__ void someInternalKernel(float *v, uint size)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < size)
	{
		v[tid] = v[tid]+10;
	}
}



void internalFunctionLaunchingKernel(int argc, const char **argv)
{
	try
	{
		int devID;

		//cudaError_t error;

		// This will pick the best possible CUDA capable device.
		devID = findCudaDevice(argc, (const char **)argv);

		// Create host vector.
		const uint kVectorSize = 1000;

		vector<float> hVector(kVectorSize);

		for (uint i = 0; i < kVectorSize; ++i)
		{
			hVector[i] = rand() / static_cast<float>(RAND_MAX);
		}

		// Create and populate device vector.
		float *dVector;
		checkCudaErrors(cudaMalloc(&dVector, kVectorSize * sizeof(float)));

		checkCudaErrors(cudaMemcpy(dVector,
			&hVector[0],
			kVectorSize * sizeof(float),
			cudaMemcpyHostToDevice));

		// Kernel configuration, where a one-dimensional
		// grid and one-dimensional blocks are configured.
		const int nThreads = 1024;
		const int nBlocks = 1;

		dim3 dimGrid(nBlocks);
		dim3 dimBlock(nThreads);


		someInternalKernel << <dimGrid, dimBlock >> >
			(dVector, kVectorSize);
		checkCudaErrors(cudaGetLastError());

		

		// Download results.
		vector<float> hResultVector(kVectorSize);

		checkCudaErrors(cudaMemcpy(&hResultVector[0],
			dVector,
			kVectorSize * sizeof(float),
			cudaMemcpyDeviceToHost));


		// Free resources.
		if (dVector) checkCudaErrors(cudaFree(dVector));
	}
	catch (...)
	{
		cout << "Error occured, exiting..." << endl;

		exit(EXIT_FAILURE);
	}
}
