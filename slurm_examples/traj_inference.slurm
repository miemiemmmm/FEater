#!/bin/bash -l
#SBATCH --job-name=benchmark_1000             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/benchmark_models/benchmarking_results_%a.out
#SBATCH --error=/Weiss/benchmark_models/benchmarking_results_%a.err
#SBATCH --array=0-7   # 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24


# TODO: Running 23 and there is problem with GPU. Do it later
export SLURM_ARRAY_TASK_ID=0
export SLURM_CPUS_PER_TASK=16

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


"""
/Weiss/FEater_trajs/dual_coord.txt
/Weiss/FEater_trajs/dual_hilbert.txt
/Weiss/FEater_trajs/dual_surface.txt
/Weiss/FEater_trajs/dual_voxel.txt
/Weiss/FEater_trajs/single_coord.txt
/Weiss/FEater_trajs/single_hilbert.txt
/Weiss/FEater_trajs/single_surface.txt
/Weiss/FEater_trajs/single_voxel.txt
"""


output_dir="/Weiss/benchmark_models/"

train_files_single="/Matter/feater_train_1000/single_coord.txt%/Matter/feater_train_1000/single_surf.txt%/Matter/feater_train_1000/single_vox.txt%/Matter/feater_train_1000/single_hilbert.txt%"
test_files_single="/Weiss/FEater_trajs/single_coord.txt%/Weiss/FEater_trajs/single_surface.txt%/Weiss/FEater_trajs/single_voxel.txt%/Weiss/FEater_trajs/single_hilbert.txt%"
data_types_single="single%single%single%single%"
loader_single="coord%surface%vox%hilb%"
model_types_single="pointnet%pointnet%voxnet%resnet%"


train_files_dual="/Matter/feater_train_1000/dual_coord.txt%/Matter/feater_train_1000/dual_surf.txt%/Matter/feater_train_1000/dual_vox.txt%/Matter/feater_train_1000/dual_hilbert.txt%"
test_files_dual="/Weiss/FEater_trajs/dual_coord.txt%/Weiss/FEater_trajs/dual_surface.txt%/Weiss/FEater_trajs/dual_voxel.txt%/Weiss/FEater_trajs/dual_hilbert.txt%"
data_types_dual="dual%dual%dual%dual%"
loader_dual="coord%surface%vox%hilb%"
model_types_dual="pointnet%pointnet%voxnet%resnet%"





#!/bin/bash -l
#SBATCH --job-name=data_abundance_inference_             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/data_scarce_test/inferences/data_abundance_inference_%a.out
#SBATCH --error=/Weiss/data_scarce_test/inferences/data_abundance_inference_%a.err
#SBATCH --array=0-27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

tasklist="/MieT5/MyRepos/FEater/data/data_scarcity_test.csv"

inputfiles=$(python3 -c """import pandas as pd
df = pd.read_csv('$tasklist', index_col=None)
print(df.iloc[${SLURM_ARRAY_TASK_ID}]['param_path'], df.iloc[${SLURM_ARRAY_TASK_ID}]['testfile'], df.iloc[${SLURM_ARRAY_TASK_ID}]['metadatafile'])
""")

param_path=$(echo ${inputfiles} | awk '{print $1}')
testfile=$(echo ${inputfiles} | awk '{print $2}')
metadatafile=$(echo ${inputfiles} | awk '{print $3}')

echo "--pretrained ${param_path} --test-data ${testfile} --meta-information ${metadatafile}"

python /MieT5/MyRepos/FEater/feater/scripts/test_model.py \
  --pretrained ${param_path} --test-data ${testfile} --meta-information ${metadatafile} \
  --output-file ${tasklist} --data-workers ${SLURM_CPUS_PER_TASK} 
  


