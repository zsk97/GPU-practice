#include <cuda_runtime.h>

#include "../helper_cuda.h"

#define SECTION_SIZE 256

template <typename T>
__global__ void scan0(T* input, T* output, int n) {
	__shared__ T sdata[SECTION_SIZE];
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		sdata[threadIdx.x] = input[idx];
	} else {
		sdata[threadIdx.x] = 0.0;
	}

	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride <<= 1) {
		float sum = 0;
		if (threadIdx.x >= stride) {
			sum = sdata[threadIdx.x] + sdata[threadIdx.x - stride];
		}
		__syncthreads();

		if (threadIdx.x >= stride) {
			sdata[threadIdx.x] = sum;
		}
		__syncthreads();
	}

	if (idx < n) {
		output[idx] = sdata[threadIdx.x];
	}
}

int main() {

	int n = 1 << 20;

	float* data_D, *output;
	cudaMalloc((void**)&data_D, n * sizeof(float));
	cudaMalloc((void**)&output, n * sizeof(float));

	int gridSize = (n + SECTION_SIZE - 1) / SECTION_SIZE;
	scan0<<<gridSize, SECTION_SIZE>>>(data_D, output, n);
}