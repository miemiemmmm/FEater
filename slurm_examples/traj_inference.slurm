#!/bin/bash -l
#SBATCH --job-name=traj_inference_onconverge_model             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/inference_jobs/traj_inference_%a.out
#SBATCH --error=/Weiss/inference_jobs/traj_inference_%a.err
#SBATCH --array=648,656,678
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

tasklist="/MieT5/MyRepos/FEater/data/trajdataset_inference.csv"

inputfiles=$(python3 -c """import pandas as pd
df = pd.read_csv('$tasklist', index_col=None)
print(df.iloc[${SLURM_ARRAY_TASK_ID}]['param_path'], df.iloc[${SLURM_ARRAY_TASK_ID}]['testdata_path'], df.iloc[${SLURM_ARRAY_TASK_ID}]['metadata_path'])
""")

param_path=$(echo ${inputfiles} | awk '{print $1}')
testdata=$(echo ${inputfiles} | awk '{print $2}')
metadatafile=$(echo ${inputfiles} | awk '{print $3}')

echo "--pretrained ${param_path} --test-data ${testdata} --meta-information ${metadatafile}"

python /MieT5/MyRepos/FEater/feater/scripts/test_model.py \
  --pretrained ${param_path} --test-data ${testdata} --meta-information ${metadatafile} \
  --output-file ${tasklist} --data-workers ${SLURM_CPUS_PER_TASK} 
  
