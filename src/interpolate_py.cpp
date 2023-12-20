#include <iostream>
#include <exception>

#include "cuda_runtime.h"
#include "interpolate.h"
#include "utils.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;


void dtype_to_float(py::array &arr, float *outarr, const int elem_number){
	py::dtype datatype = arr.dtype();
	if (datatype.equal(py::dtype::of<double>())) {
		double *arr_ptr = static_cast<double *>(arr.mutable_data());
		for (int i = 0; i < elem_number; i++)
			outarr[i] = static_cast<float>(arr_ptr[i]);
	} else if (datatype.equal(py::dtype::of<float>())) {
		float *arr_ptr = static_cast<float *>(arr.mutable_data());
		for (int i = 0; i < elem_number; i++)
			outarr[i] = arr_ptr[i];
	} else {
		throw std::runtime_error("dtype is not float or double");
	}
}


void translate_coord(float* coord, const int atom_nr, const int *dims, const double spacing) {
	// Align the center of the coords to the center of the grid
	float cog_coord[3] = {0};
	for (int i = 0; i < atom_nr; i++) {
		cog_coord[0] += coord[i*3];
		cog_coord[1] += coord[i*3+1];
		cog_coord[2] += coord[i*3+2];
	}
	cog_coord[0] /= atom_nr;
	cog_coord[1] /= atom_nr;
	cog_coord[2] /= atom_nr;
//	std::cout << "center of geom is: " << cog_coord[0] << " " << cog_coord[1] << " " << cog_coord[2] << "\n";

	for (int i = 0; i < atom_nr; i++) {
		coord[i*3]   = (coord[i*3] - cog_coord[0])/spacing + dims[0]/2;
		coord[i*3+1] = (coord[i*3+1] - cog_coord[1])/spacing + dims[1]/2;
		coord[i*3+2] = (coord[i*3+2] - cog_coord[2])/spacing + dims[2]/2;
	}
	// write_xyzr("test.xyzr", coord, atom_nr);
}


py::array_t<float> run_interpolate(py::array arr_target, py::array arr_weights, py::array_t<int> grid_dims,
																	 const double spacing, const double cutoff, const double sigma){
	py::buffer_info targets = arr_target.request();
	py::buffer_info weights = arr_weights.request();
	py::buffer_info dims = grid_dims.request();
	if (targets.shape[0] != weights.shape[0]) {
		throw std::runtime_error("Number of weights must be equal to number of target points");
	}
	//	std::cout << "spacing: " << spacing << "\n";
	int atom_nr = targets.shape[0];
	int* dims_ptr = static_cast<int*>(dims.ptr);
	unsigned int grid_coord_nr = dims_ptr[0] * dims_ptr[1] * dims_ptr[2];

	float *coord_ptr = new float[atom_nr*3];
	float *weight_ptr = new float[atom_nr];
	dtype_to_float(arr_target, coord_ptr, atom_nr*3);
	dtype_to_float(arr_weights, weight_ptr, atom_nr);

	// TODO: change the cutoff to fit with the spacing
	translate_coord(coord_ptr, atom_nr, dims_ptr, spacing);
	double new_cutoff = cutoff/spacing;
	double new_sigma = sigma/spacing;

	// Perform the point interpolation and copy to the result array
	float *interpolated = new float[grid_coord_nr];
	interpolate_host(interpolated, coord_ptr, weight_ptr, atom_nr, dims_ptr, new_cutoff, new_sigma);
	py::array_t<float> result({grid_coord_nr});
	std::memcpy(result.mutable_data(), interpolated, grid_coord_nr * sizeof(float));
	delete[] coord_ptr;
	delete[] weight_ptr;
	delete[] interpolated;
	return result;
}


float sum_array(py::array arr) {
	py::buffer_info buf = arr.request();
	int elem_number = static_cast<int>(arr.size());
	float *arr_float = new float[elem_number];
	dtype_to_float(arr, arr_float, elem_number);
	float sum = sum_host(arr_float, elem_number);
	return sum;
}


PYBIND11_MODULE(voxelize, m) {
  m.def("sum_array", &sum_array,
  	py::arg("arr"),
  	"A test function to calculate the sum of array"
  );
  m.def("interpolate", &run_interpolate,
    py::arg("arr_target"),
    py::arg("arr_weights"),
    py::arg("grid_dims"),
    py::arg("spacing") = 0.25,
    py::arg("cutoff") = 12.0,
    py::arg("sigma") = 1.5,
    "Compute the interpolated values of the grid points (CUDA)"
  );
}


