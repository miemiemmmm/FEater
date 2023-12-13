#include <cmath>
#include <vector>
#include "cuda_runtime.h"

//__global__ void sum_kernel(const float *array, float *result, int N);
//__global__ void normalize_kernel(float *array, float *sum, float weight, int N);
//__global__ void coordi_interp_kernel(const float *coord, float *interpolated, const int *dims, const double cutoff, const double sigma);

float sum_host(std::vector<float> input_array);
float sum_host(const float *input_array, int N);

void interpolate_host(float *interpolated, const float *coord, const float *weight, const int atom_nr,
	const int *dims, const double cutoff, const double sigma);

