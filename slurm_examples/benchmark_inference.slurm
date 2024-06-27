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
  