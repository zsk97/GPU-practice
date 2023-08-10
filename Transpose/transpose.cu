#include <cuda_runtime.h>

#include "../helper_cuda.h"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

template <typename T>
__global__ void transpose0(T* input, T* output) {
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		output[(yIndex+j)*width + xIndex] = input[(xIndex+j)*width + yIndex];
}

template <typename T>
__global__ void transpose1(T* input, T* output) {
	__shared__ T sdata[TILE_DIM][TILE_DIM];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int width = gridDim.x * TILE_DIM;	

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		sdata[threadIdx.y + j][threadIdx.x] = input[(yIndex+j)*width + xIndex];

	__syncthreads();

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
		output[(yIndex+j)*width + xIndex] = sdata[threadIdx.x][threadIdx.y + j];
	}
}


template <typename T>
__global__ void transpose2(T* input, T* output) {
	__shared__ T sdata[TILE_DIM][TILE_DIM+1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int width = gridDim.x * TILE_DIM;	

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		sdata[threadIdx.y + j][threadIdx.x] = input[(yIndex+j)*width + xIndex];

	__syncthreads();

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
		output[(yIndex+j)*width + xIndex] = sdata[threadIdx.x][threadIdx.y + j];
	}
}

int main() {
	const int nx = 1024;
	const int ny = 1024;
	const int mem_size = nx*ny*sizeof(float);

	dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	float *h_idata = (float*)malloc(mem_size);
	float *h_cdata = (float*)malloc(mem_size);
	float *h_tdata = (float*)malloc(mem_size);
	float *gold    = (float*)malloc(mem_size);
	
	float *d_idata, *d_cdata, *d_tdata;
	checkCudaErrors( cudaMalloc(&d_idata, mem_size) );
	checkCudaErrors( cudaMalloc(&d_cdata, mem_size) );
	checkCudaErrors( cudaMalloc(&d_tdata, mem_size) );

	transpose0<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
	transpose1<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
	transpose2<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
}