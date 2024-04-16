# FEater
FEater (Flexibility and Elasticity Assessment Tools for Essential Residues) proposed a challenge for 3D flexible objects recognition based on fragments from molecular biology. 
<!-- This challenge focused on the performance evaluation of 
models to flexible and elastic objects.  -->

## Installation
### Compile from source
```
micromamba activate env   # Assume micromamba as Python package manager, and environment named env
git clone https://github.com/miemiemmmm/FEater.git && cd FEater && make install
```
To compile the module for voxel generation, NVCC (NVIDIA CUDA Compiler) is required. If CUDA runtime is not in the default path (/usr/local/cuda), for instance on a supercomputer, you might want to set the following environment variables to match the corresponding [GPU architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list). 
```
export CUDA_HOME=/usr/local/cuda
export CUDA_COMPUTE_CAPABILITY=sm_80
```


## Dataset availability 
FEater-Single and FEater-Dual datasets are available for access and download on [Zenodo](https://zenodo.org/records/10593541). 
Alternatively, you can use the command below to download the full FEater dataset along with its **Minisets**:
```
wget https://zenodo.org/api/records/10593541/files-archive -O FEater_Data.zip
```

Each source file, labeled with the suffix **_PDBSRC.tar.gz**, comprises tiny fragments sourced from PDB structures, either single-residue or dual-residue, with each file containing over 8 million entries.
**NOTE**: Conducting operations in a folder with such a vast number of tiny files may lead to significant performance degradation on the system. 

The coordinate feature, identified by the suffix **_PDB.tar.gz**, is stored in HDF (Hierarchical Data Format) alongside the topology for each label. View the [following chapter](#data-loading-interation-and-spliting) for further details of its usage. 

**Miniset** contains a tiny subset of the training data used in [FEater research](https://zenodo.org/records/10593541/files/FEater_paper.pdf), 

## Visual inspection of dataset
The following console tools, including **viewcoord**, **viewsurf**, **viewvoxel**, are provided for visual inspection in to the dataset. [Open3D](https://www.open3d.org/) is required to render different molecular representations. 

Some example commands are as follow: 
```
viewcoord -f TestSet_Dataset.h5 -i 15 -t 1 -m 0
viewsurf -f ValidationSet_Surface.h5 -i 15
viewvoxel -f TestSet_Voxel.h5 -i 15 -r XXXX_Dataset.h5
```

## Data loading, iteration and spliting




## Feature generation 
### Surface generation
Surface generation requires the installation of [SiESTA-Surf](https://github.com/miemiemmmm/SiESTA). 

### Voxel generation


### Example scripts
See [slurm examples](slurm_examples/) for batch featurization. 

<!-- ## Personalized featurization -->




## Cite this work
If you find FEater useful in your research, please consider citing it in the following formats:

Bibtex style
```
@dataset{zhang2024feater,
  author = {Zhang, Yang and Vitalis, Andreas},
  title = {FEater: A large-scale 3D molecular conformation dataset to measure the geometric of models},
  month = feb,
  year = 2024,
  publisher = {Zenodo},
  version = {0.1.0},
  doi = {10.5281/zenodo.10593541},
  url = {https://doi.org/10.5281/zenodo.10593541}
}
```
APA style:
```
Zhang, Y., & Vitalis, A. (2024). FEater: A large-scale 3D molecular conformation dataset to measure the geometric invariance of models (0.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10593541
```

## Useful URLs
[FEater paper](https://zenodo.org/records/10593541/files/FEater_paper.pdf)
