#include <iostream>
#include <vector>

#include "interpolate.h"


// Kernel functions
__device__ double gaussian_device(const double x, const double mu, const double sigma) {
	return std::exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)) / (sigma * std::sqrt(2 * M_PI));
}


__global__ void sum_kernel(const float *array, float *result, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
		atomicAdd(result, array[index]);
}

__global__ void normalize_kernel(float *array, const float *sum, const float weight, const int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
		array[index] = (array[index] * weight) / *sum;
}

__global__ void coordi_interp_kernel(const float *coord, float *interpolated, const int *dims, const double cutoff, const double sigma){
	/*
	Interpolate the atomic density to the grid
	*/
	unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int grid_size = dims[0] * dims[1] * dims[2];

	if (task_index < grid_size){
		// Compute the grid coordinate from the grid index
    // TODO: Check why the grid coordinate is like this???
		float grid_coord[3] = {
      static_cast<float>(task_index / dims[0] / dims[1]),
      static_cast<float>(task_index / dims[0] % dims[1]),
      static_cast<float>(task_index % dims[0]),
		};
		float dist_square = 0.0f;
		for (int i = 0; i < 3; ++i) {
			float diff = coord[i] - grid_coord[i];
			dist_square += diff * diff;
		}
		// Process the interpolated array with the task_index (Should not be race condition)
		if (dist_square < cutoff * cutoff)
			interpolated[task_index] = gaussian_device(sqrt(dist_square), 0.0, sigma);
	}
}

__global__ void add_temparray_kernel(const float *temp_array, float *interpolated, const int grid_size){
	/*
	*/
	unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (task_index < grid_size){
		interpolated[task_index] += temp_array[task_index];
	}
}

// Host functions
float sum_host(std::vector<float> input_array) {
	/*
	C++ wrapper of the CUDA kernel function sum_host
	*/
  int N = input_array.size();
  float *arr, *result;
  cudaMallocManaged(&arr, N * sizeof(float));
  cudaMallocManaged(&result, sizeof(float));
  cudaMemcpy(arr, input_array.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(result, 0, sizeof(float));

  int threads_per_block = 256;
  int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  sum_kernel<<<threads_per_block, number_of_blocks>>>(arr, result, N);
  cudaDeviceSynchronize();

  float result_host;
  cudaMemcpy(&result_host, result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  cudaFree(result);

  return result_host;
}

float sum_host(const float *input_array, int N) {
	/*
	C++ wrapper of the CUDA kernel function sum_host
	*/
	float *arr, *result;
	cudaMallocManaged(&arr, N * sizeof(float));
	cudaMallocManaged(&result, sizeof(float));
	cudaMemcpy(arr, input_array, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(result, 0, sizeof(float));

	int threads_per_block = 256;
	int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
	sum_kernel<<<threads_per_block, number_of_blocks>>>(arr, result, N);
	cudaDeviceSynchronize();

	float result_host;
	cudaMemcpy(&result_host, result, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(arr);
	cudaFree(result);

	return result_host;
}


void interpolate_host(float *interpolated, const float *coord, const float *weight, const int atom_nr,
	const int *dims, const double cutoff, const double sigma){
	/*
	Interpolate the atomic density to the grid
	*/
	// Allocate the temporary array for the interpolation computuation
	unsigned int grid_size = dims[0] * dims[1] * dims[2];
	float *tmp_interp, *tmp_sum, *norm_chk, *interp_gpu, *coordi_gpu;
	int *dims_gpu;
	cudaMallocManaged(&tmp_interp, grid_size * sizeof(float));
	cudaMallocManaged(&interp_gpu, grid_size * sizeof(float));
	cudaMallocManaged(&coordi_gpu, 3 * sizeof(float));
	cudaMallocManaged(&dims_gpu, 3 * sizeof(int));
	cudaMallocManaged(&tmp_sum, sizeof(float));
	cudaMallocManaged(&norm_chk, sizeof(float));
	cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

	int gpu_block_size = 256; // TODO: Hard coded for now
	int gpu_grid_size = (grid_size + gpu_block_size - 1) / gpu_block_size;
	for (int atm_idx = 0; atm_idx < atom_nr; ++atm_idx) {
		float coordi[3] = {coord[atm_idx*3], coord[atm_idx*3+1], coord[atm_idx*3+2]};
	 	cudaMemcpy(coordi_gpu, coordi, 3 * sizeof(float), cudaMemcpyHostToDevice);

		// Kernel to compute the temporary array
		coordi_interp_kernel<<<gpu_block_size, gpu_grid_size>>>(coordi_gpu, tmp_interp, dims_gpu, cutoff, sigma);
		cudaDeviceSynchronize();

		// Compute the sum of the temporary array and normalize it
		cudaMemset(tmp_sum, 0, sizeof(float));
		cudaMemset(norm_chk, 0, sizeof(float));
		sum_kernel<<<gpu_block_size, gpu_grid_size>>>(tmp_interp, tmp_sum, grid_size);
		cudaDeviceSynchronize();
		normalize_kernel<<<gpu_block_size, gpu_grid_size>>>(tmp_interp, tmp_sum, weight[atm_idx], grid_size);
		cudaDeviceSynchronize();
		sum_kernel<<<gpu_block_size, gpu_grid_size>>>(tmp_interp, norm_chk, grid_size);
		cudaDeviceSynchronize();

		// Check the quality of the interpolation
		float tmp_sum_host = 0;
		float norm_chk_host = 0;
		cudaMemcpy(&tmp_sum_host, tmp_sum, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&norm_chk_host, norm_chk, sizeof(float), cudaMemcpyDeviceToHost);
		if (tmp_sum_host - 0 < 0.01) {
			/* The sum of the temporary array is used for normalization, skip if the sum is 0 */
			std::cerr << "Warning: The sum of the temporary array is 0" << std::endl;
			continue;
		} else if (std::abs(norm_chk_host - weight[atm_idx]) > 0.01) {
			std::cerr << "Warning: The sum of the normalized temporary array is not equal to the weight: " << norm_chk_host << "/" << weight[atm_idx] << std::endl;
		}

		// Finally add the normalized temporary array to the final array
		add_temparray_kernel<<<gpu_block_size, gpu_grid_size>>>(tmp_interp, interp_gpu, grid_size);
		cudaDeviceSynchronize();
	}

	float tmp_sum_host = 0;
	cudaMemset(tmp_sum, 0, sizeof(float));
	sum_kernel<<<gpu_block_size, gpu_grid_size>>>(interp_gpu, tmp_sum, grid_size);
	cudaDeviceSynchronize();
	cudaMemcpy(&tmp_sum_host, tmp_sum, sizeof(float), cudaMemcpyDeviceToHost);
	float tmp_weights_sum = sum_host(weight, atom_nr);
	if (std::abs(tmp_sum_host - tmp_weights_sum) > 0.01) {
		std::cerr << "Warning: The sum of the interpolated array is not equal to the sum of the weights: " << tmp_sum_host << "/" << tmp_weights_sum << std::endl;
	}

	// Copy the interpolated array to the host and free the GPU memory
	cudaMemcpy(interpolated, interp_gpu, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(tmp_interp);
	cudaFree(interp_gpu);
	cudaFree(coordi_gpu);
	cudaFree(dims_gpu);
	cudaFree(tmp_sum);
	cudaFree(norm_chk);
}


