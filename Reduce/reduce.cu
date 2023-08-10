#include <cuda_runtime.h>

#include "../helper_cuda.h"

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template <typename T>
__global__ void reduce0(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx];
	__syncthreads();

	for (size_t i = 1; i < blockDim.x; i *= 2) {
		if (tid%(2*i) == 0) {
			sdata[tid] += sdata[tid+i];
		}
		__syncthreads();
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__ void reduce1(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx];
	__syncthreads();
	
	for (size_t i = 1; i < blockDim.x; i *= 2) {
		size_t index = tid * 2 * i;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + i];
		}
		__syncthreads();
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__ void reduce2(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx];
	__syncthreads();

	for (size_t i = blockDim.x / 2; i > 0; i >>= 1) {
		if (tid < i) {
			sdata[tid] += sdata[tid + i];
		}
		__syncthreads();
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__ void reduce3(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx] + input[idx + blockDim.x];
	__syncthreads();

	for (size_t i = blockDim.x / 2; i > 0; i >>= 1) {
		if (tid < i) {
			sdata[tid] += sdata[tid + i];
		}
		__syncthreads();
	
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

template <typename T>
__device__ void warpReduce(volatile T* cache, int tid) {
	cache[tid] += cache[tid + 32];
	cache[tid] += cache[tid + 16];
	cache[tid] += cache[tid + 8];
	cache[tid] += cache[tid + 4];
	cache[tid] += cache[tid + 2];
	cache[tid] += cache[tid + 1];
}

template <typename T>
__global__ void reduce4(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx] + input[idx + blockDim.x];
	__syncthreads();

	for (size_t i = blockDim.x / 2; i > 32; i >>= 1) {
		if (tid < i) {
			sdata[tid] += sdata[tid + i];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid);
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__ void reduce5(T* input, T* output) {
	__shared__ T sdata[THREAD_PER_BLOCK];

	size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
	size_t tid = threadIdx.x;

	sdata[tid] = input[idx] + input[idx + blockDim.x];
	__syncthreads();

	#pragma unroll
	for (size_t i = blockDim.x / 2; i > 32; i >>= 1) {
		if (tid < i) {
			sdata[tid] += sdata[tid + i];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid);
	}

	if (tid == 0) output[blockIdx.x] = sdata[tid];	
}

__device__ __forceinline__ float warpReduceSum(float sum) {
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <typename T>
__global__ void reduce6(T *input, T *output){
    float sum = 0;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
	size_t tid = threadIdx.x;
	
	sum = input[idx] + input[idx + blockDim.x];

    // Shared mem for partial sums (one per warp in the block)
    __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

    // Final reduce using first warp
    if (warpId == 0) {
		sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    	sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    	sum += __shfl_down_sync(0xffffffff, sum, 1);
	}

    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sum;
}

template <typename T>
__global__ void reduce7(T* input, T* output, size_t n) {
	T sum = 0;
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
	size_t tid = threadIdx.x;

	sum += input[idx] + input[idx + blockDim.x] + input[idx + blockDim.x*2] + input[idx + blockDim.x*3];

	__shared__ T warpLevelSums[WARP_SIZE];
	const int laneId = threadIdx.x % WARP_SIZE;
	const int warpId = threadIdx.x / WARP_SIZE;
	
	sum = warpReduceSum(sum);
	if (laneId == 0) warpLevelSums[warpId] = sum;
	__syncthreads();
	
	sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

	// Final reduce using first warp
    if (warpId == 0) {
		sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    	sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    	sum += __shfl_down_sync(0xffffffff, sum, 1);
	}

	if (tid == 0) output[blockIdx.x] = sum;		
}

int main() {
	int n = 32 * 1024 * 1024;

	float* A_H = (float*) malloc(n * sizeof(float));
	for (int i = 0; i < n; i++) {
		A_H[i] = rand() / (float)RAND_MAX;;
	}

	float* A_D = nullptr;
	checkCudaErrors(cudaMalloc((void**)&A_D, n * sizeof(float)));
	checkCudaErrors(cudaMemcpy(A_D, A_H, n * sizeof(float), cudaMemcpyHostToDevice));

	float* res_D = nullptr;
	checkCudaErrors(cudaMalloc((void**)&res_D, n * sizeof(float)));
	
	int num_block = (n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	int iter = 1;
	for (int i = 0; i < iter; i++) {
		reduce0<<<num_block, THREAD_PER_BLOCK>>>(A_D, res_D);
		reduce1<<<num_block, THREAD_PER_BLOCK>>>(A_D, res_D);
		reduce2<<<num_block * 2, THREAD_PER_BLOCK / 2>>>(A_D, res_D);
	}

	reduce3<<<num_block / 2, THREAD_PER_BLOCK>>>(A_D, res_D);
	reduce4<<<num_block / 2, THREAD_PER_BLOCK>>>(A_D, res_D);	
	reduce5<<<num_block / 2, THREAD_PER_BLOCK>>>(A_D, res_D);
	reduce6<<<num_block / 2, THREAD_PER_BLOCK>>>(A_D, res_D);	
	reduce7<<<num_block / 4, THREAD_PER_BLOCK>>>(A_D, res_D, n);

	return 0;
}