#include <iostream>
#include <string>
#include <fstream>


inline void write_xyzr(std::string filename, const float* xyz_array, const float* r_array, int num_particles){
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < num_particles; i++)
	{
		file << xyz_array[3*i] << " " << xyz_array[3*i+1] << " " << xyz_array[3*i+2] << " " << r_array[i] << "\n";
	}
	file.close();
}

inline void write_xyzr(std::string filename, const float* xyz_array, int num_particles){
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < num_particles; ++i){
		std::cout << xyz_array[3*i] << " " << xyz_array[3*i+1] << " " << xyz_array[3*i+2] << "\n";
		file << xyz_array[3*i] << " " << xyz_array[3*i+1] << " " << xyz_array[3*i+2] << " 1 \n";
	}
	file.close();
}










