#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 512

template <typename T>
__global__ void reduce0(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx];
	__syncthreads();

	for (size_t i = 0; i < blockDim.x; i *= 2) {
		if (tid%(2*i) == 0) {
			sdata[tid] += sdata[tid+i];
		}
		__syncthreads();
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

int main() {
	int n = THREAD_PER_BLOCK * 100;

	float* A_H = (float*) malloc(n * sizeof(float));
	for (int i = 0; i < n; i++) {
		A_H[i] = rand() / (float)RAND_MAX;;
	}

	float* A_D = nullptr;
	cudaMalloc((void**)&A_D, n * sizeof(float));
	cudaMemcpy(A_D, A_H, n * sizeof(float), cudaMemcpyHostToDevice);

	float* res_D = nullptr;
	cudaMalloc((void**)&res_D, n * sizeof(float));
	
	int num_block = (n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	int iter = 10;
	for (int i = 0; i < iter; i++) {
		reduce0<<<num_block, THREAD_PER_BLOCK>>>(A_D, res_D);
	}

	return 0;
}