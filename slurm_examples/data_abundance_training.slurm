#!/bin/bash -l
#SBATCH --job-name=data_abundance             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/data_scarce_test/model_combination_%a.out
#SBATCH --error=/Weiss/data_scarce_test/model_combination_%a.err
#SBATCH --array=16-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


# export SLURM_ARRAY_TASK_ID=7   % TODO


source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


# Get the task-specific parameters according to the task id start from 0
index_file="/MieT5/MyRepos/FEater/data/datascarce_joblist.txt"
train_file=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $1}' ${index_file})
test_file=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $2}' ${index_file})
data_type=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $3}' ${index_file})
output_paths=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $4}' ${index_file})
model_type=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $5}' ${index_file})
loader_type=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $6}' ${index_file})

echo "Task info: ${train_file} ${test_file} ${data_type} ${output_paths} ${model_type} ${loader_type}"

mkdir -p ${output_paths}

########################## Training ########################## 

pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=64

lr_init=0.001
lr_decay_steps=30
lr_decay_rate=0.5

# NOTE: PointNet2 Requires the number of point at least 32 
# single coord -> 24, dual coord > 42, single surf -> 1500, dual surf -> 2000
if [[ "${data_type}" == "single" ]] && [[ "${loader_type}" == "coord" ]]; then 
  point_nr=24
elif [[ "${data_type}" == "dual" ]] && [[ "${loader_type}" == "coord" ]]; then 
  point_nr=42
elif [[ "${data_type}" == "single" ]] && [[ "${loader_type}" == "surface" ]]; then 
  point_nr=1500
elif [[ "${data_type}" == "dual" ]] && [[ "${loader_type}" == "surface" ]]; then 
  point_nr=1500
else
  point_nr=0
fi

extraparms=""

if [ ${point_nr} -gt 0 ]; then
  extraparms="${extraparms} --pointnet-points ${point_nr}"
fi
if [ ${start_epoch} -gt 0 ]; then
  extraparms="${extraparms} --start-epoch ${start_epoch}"
fi
if [ ${#pretrained_model} -gt 0 ]; then
  extraparms="${extraparms} --pretrained ${pretrained_model}"
fi

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)" 
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py \
  --model ${model_type} --optimizer adam --loss-function crossentropy \
  --training-data ${train_file} --test-data ${test_file} --output_folder ${output_paths} --test-number 4000 \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  --data-type ${data_type} --production 1 --cuda 1 \
	--dataloader-type ${loader_type} ${extraparms} \
  # --production 0

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"




