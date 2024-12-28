# FEater
FEater (Flexibility and Elasticity Assessment Tools for Essential Residues) is a molecular fragment dataset for 3D flexible objects recognition. 


[Download dataset](https://doi.org/10.5281/zenodo.14235911) | [Source code](https://github.com/miemiemmmm/FEater)

## Installation
Run the following commands to install the package. This assumes micromamba being your python package manager. Feel free to replace it with your preferred package manager. 
```bash
micromamba activate feater_env 
git clone https://github.com/miemiemmmm/FEater.git
cd FEater
make install
micromamba install ambertools
make install_dependencies 
```


The generation of 3D voxel is accelerated by CUDA, hence a NVIDIA CUDA compiler (``nvcc``) is required. 
If CUDA runtime is not in the default path (``/usr/local/cuda``), for instance on a supercomputer, or an older NVIDIA [GPU architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) is used, you might want to set the following environment variables to match the corresponding configuration. 
```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_COMPUTE_CAPABILITY=sm_80
```

## Dataset availability 
### Download FEater datasets (Coordinates only)
FEater-Single and FEater-Dual datasets are available on [Zenodo](https://doi.org/10.5281/zenodo.14235911). 
Due to the file size of 3D representations, including surface, voxel, and Hilbert curve, hosting them in public repository is neither feasible nor efficient. 
Consequently, only the coordinate representations (along with their respective molecular topologies) are provided for download. 
For instructions on generating 3D representations locally, please see the [featurization](#featurization) section. 

To specify the download directory for the coordinate dataset, set the environment variable ``FEATER_DATA``. 
If ``FEATER_DATA`` is not defined, the sub-directory ``./all_data`` will be created in the current working directory (``$PWD``). 
Use the following command to download the FEater dataset. 

```bash 
export FEATER_DATA=/tmp/FEater_Data 
# Download individual datasets on demand
make download_baseline
make download_feater_single    # ~27  MiB 
make download_feater_dual      # ~957 MiB
make download_minisets         # ~214 MiB
# Or download all datasets 
make download_all
```


### Raw fragments
The PDB source files (suffix **PDBSRC.tar.gz**) comprise the original fragments in PDB format and each contains *c.a.* 8 million entries. 
If you want to re-split the dataset or create your sub-set, the following command downloads the source files to the data folder (``FEATER_DATA``). 
```bash
make download_source
```

> [!NOTE]
> Operations in a folder with such a vast number of tiny files may lead to significant performance degradation on the host system. 

<!-- <div style="padding: 10px; border: 2px solid #FF6363; background-color: #FFAB76; color: #000; margin-bottom:25px; margin-top:25px;">
    <strong>
    NOTE:</strong> 
</div> -->

<!-- The coordinate feature, identified by the suffix **_PDB.tar.gz**, is stored in HDF (Hierarchical Data Format) alongside the topology for each label. View the [following chapter](#data-loading-interation-and-spliting) for further details of its usage.  -->


## Featurization 
### Surface generation
Surface generation requires the installation of [SiESTA-Surf](https://github.com/miemiemmmm/SiESTA) as shown in [Installation](#installation). 
The input requires the coordinate data as input. 
Compression might save up to 50% of the disk usage, with a trade-off of slower read/write speed. 
```bash
$ makesurface -h
usage: makesurface [-h] [-i INPUT] [-o OUTPUT] [-c COMPRESS_LEVEL] [-f FORCE] [-g GRID_SPACING]
                   [--processes PROCESSES]

Generate surface HDF file from the coordinate file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The absolute path of the input coordinate HDF files
  -o OUTPUT, --output OUTPUT
                        The absolute path of the output surface HDF file
  -c COMPRESS_LEVEL, --compress-level COMPRESS_LEVEL
                        The compression level for the HDF deposition. Default is 0 (no compression)
  -f FORCE, --force FORCE
                        Force overwrite the output file
  -g GRID_SPACING, --grid-spacing GRID_SPACING
                        The grid spacing for surface generation
  --processes PROCESSES
                        The number of processes for parallel processing
# Suppose the input file is ${FEATER_DATA}/FEater_Single/TestSet_coord.h5
$ makesurface -i ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 -o ${FEATER_DATA}/FEater_Single/TestSet_surface.h5 
```

### 3D Voxel generation
Similar to the surface generation, the 3D voxel requires the coordinate data as input. 
```bash
$ makevoxel -h
usage: makevoxel [-h] -i INPUT -o OUTPUT [-f FORCE] [-c COMPRESS_LEVEL] [-d DIM] [-b BOXSIZE] [--sigma SIGMA]
                 [--cutoff CUTOFF] [--processes PROCESSES] [--tag-name TAG_NAME] [--feature-type FEATURE_TYPE]
                 [--only-element ONLY_ELEMENT]

Generate voxelized coordinates from the input coordinate file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The absolute path of the input coordinate HDF files
  -o OUTPUT, --output OUTPUT
                        The absolute path of the output voxel HDF file
  -f FORCE, --force FORCE
                        Force overwrite the output file; Default: 0
  -c COMPRESS_LEVEL, --compress-level COMPRESS_LEVEL
                        The compression level of the output HDF file; Default: 0
  -d DIM, --dim DIM     The dimension of the voxel; Default: 32
  -b BOXSIZE, --boxsize BOXSIZE
                        The box size of the voxel; Default: 16.0
  --sigma SIGMA         The sigma of the Gaussian kernel; Default: 1.0
  --cutoff CUTOFF       The cutoff of the Gaussian kernel; Default: 12.0
  --processes PROCESSES
                        The number of processes; Default: 8
  --tag-name TAG_NAME   The tag name for the voxel data; Default: 'voxel'
  --feature-type FEATURE_TYPE
                        The type of weights (uniform or elem); Default: uniform
  --only-element ONLY_ELEMENT
                        Focus only on one element (number), if the type is 'elem'. Default: None (All elements
                        are processed)
# Suppose the input file is ${FEATER_DATA}/FEater_Single/TestSet_coord.h5
$ makevoxel -i ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 -o ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5
```

### Hilbert curve generation
Hilbert curve is converted from the 3D voxel. Currently it support conversion of 5th order 3D hilbert curve (32×32×32) to 7th order 2D hilbert curve (128×128). It requires the [3D voxel](#voxel-generation) data rather than the original coordinate data. 
```bash
$ makehilbert -h
usage: makehilbert [-h] -i INPUT -o OUTPUT [-m MODE] [-f FORCE] [-c COMPRESS_LEVEL]
                   [--input-tagname INPUT_TAGNAME] [--output-tagname OUTPUT_TAGNAME] [--processes PROCESSES]

Transform the voxel to hilbert curve

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The input 3D voxel HDF file
  -o OUTPUT, --output OUTPUT
                        The output 2D hilbert curve HDF file
  -m MODE, --mode MODE  The mode of pooling transformation (max, mean); Default: max
  -f FORCE, --force FORCE
                        Force overwrite the output file; Default: 0
  -c COMPRESS_LEVEL, --compress-level COMPRESS_LEVEL
                        The compression level of the output HDF file; Default: 0
  --input-tagname INPUT_TAGNAME
                        The input tag name; Default: voxel
  --output-tagname OUTPUT_TAGNAME
                        The output tag name; Default: voxel
  --processes PROCESSES
                        The number of processes; Default: 8
# Suppose the input file is ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5
$ makehilbert -i ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5 -o ${FEATER_DATA}/FEater_Single/TestSet_hilbert.h5
```

## Benchmark the data extraction speed
Input file and dataloader type are required to benchmark the data extraction speed. 
```bash 
$ benchmarkdataset -h
usage: benchmarkdataset [-h] -f INPUT_FILE -d DATALOADER_TYPE [-b BATCH_SIZE] [-p PROCESS_NR] [-e EXIT_POINT]
                        [-c TOCUDA] [-v VERBOSE] [--pointnr POINTNR]

Benchmarking the data extraction speed of the HDF dataset.

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input-file INPUT_FILE
                        The absolute path of the input HDF5 file.
  -d DATALOADER_TYPE, --dataloader-type DATALOADER_TYPE
                        Specify the dataloader type; [coord, surface, voxel, hilbert]
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size; Default: 128.
  -p PROCESS_NR, --process-nr PROCESS_NR
                        Number of processes; Default: 8.
  -e EXIT_POINT, --exit-point EXIT_POINT
                        Exit the benchmarking after N batches.
  -c TOCUDA, --tocuda TOCUDA
                        Transfer data to cuda or not; Default: 0.
  -v VERBOSE, --verbose VERBOSE
                        Verbose mode; Default: 0.
  --pointnr POINTNR     Number of points if the dataloader is point-based [coord or surface].

$ benchmarkdataset -f ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 -d coord -b 64 -p 16 --pointnr 20
$ benchmarkdataset -f ${FEATER_DATA}/FEater_Single/TestSet_surface.h5 -d surface -b 64 -p 16 --pointnr 1000
$ benchmarkdataset -f ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5 -d voxel -b 64 -p 16
$ benchmarkdataset -f ${FEATER_DATA}/FEater_Single/TestSet_hilbert.h5 -d hilbert -b 64 -p 16
```


## Data loading and iteration 
There are four built-in dataloaders for different molecular representations: ``CoordDataset``, ``VoxelDataset``, ``SurfDataset``, ``HilbertCurveDataset``. 
The child method **mini_batches** is implemented to iterate throughout the dataset. 
Multiple files can be passed as a list for the dataloader.

```python
import feater, os
traning_file = os.path.join(os.getenv('FEATER_DATA'), 'FEater_Single/TrainingSet_coord.h5')
dset = feater.dataloader.CoordDataset([traning_file], target_np=24) 
print(dset[1])
for data, label in dset.mini_batches(batch_size=64, process_nr=16, exit_point=10):
  print(data.shape, label.shape)
```


## Model training
The following code snippet demonstrates the use of the coordinate dataset for training a model. 
```python
from feater.models.pointnet import PointNetCls
classifier = PointNetCls(20)
for data, label in dset.mini_batches(batch_size=64, process_nr=16):
  data = data.transpose(2, 1)   # PointNet requires the shape to be (B, C, N)
  pred = classifier(data)
```

There is also a script for training a model on the dataset. Note that the usage of the train/test dataset needs a list of the HDF files rather than the HDF file itself. 
```bash
$ feater_train -h   # To check the usage of the script
# Suppose you downloaded the FEater-Single dataset into FEATER_DATA directory
$ realpath ${FEATER_DATA}/FEater_Single/TrainingSet_coord.h5 > ${FEATER_DATA}/FEater_Single/train.txt
$ realpath ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 > ${FEATER_DATA}/FEater_Single/test.txt
$ feater_train --model pointnet --optimizer adam --loss-function crossentropy \
--training-data ${FEATER_DATA}/FEater_Single/train.txt \
--test-data ${FEATER_DATA}/FEater_Single/test.txt \
--output_folder /tmp \
--data-type single --dataloader-type coord --pointnet-points 24 \
-e 5 -b 64 -w 16 --lr-init 0.001 
```


## Visual inspection of dataset
Three console tools including ``viewcoord``, ``viewsurf``, ``viewvoxel``, are provided for visual inspection into the dataset. 
[Open3D](https://www.open3d.org/) is required to render different molecular representations. 
Here are some examples commands. 

```bash
viewcoord -f ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 -i 15 -t 1 -m 0
viewsurf -f ${FEATER_DATA}/FEater_Single/TestSet_surface.h5 -i 15
viewvoxel -f ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5 -i 15 
```


## Dataset subsetting
Making smaller subsets for your own purposes. It either look for files in the ``FEATER_DATA`` directory or you can explicitly specify the input files. 
```bash 
# Automatic look-up
$ feater_miniset -d single -s 50 -o /tmp/ --whichset train  
# Explicitly specify the input files (one or more files)
$ feater_miniset -d single -s 50 -o /tmp/ \
--coordfile ${FEATER_DATA}/FEater_Single/TestSet_coord.h5 \
--surfacefile ${FEATER_DATA}/FEater_Single/TestSet_surface.h5 \
--voxelfile ${FEATER_DATA}/FEater_Single/TestSet_voxel.h5 \
--hilbertfile ${FEATER_DATA}/FEater_Single/TestSet_hilbert.h5
```


### Example scripts
See [slurm examples](slurm_examples/) for the example usage of this dataset. 

