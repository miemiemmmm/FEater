#!/bin/bash -l
#SBATCH --job-name=fused_test             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/fused_models/fused_test_%a.out
#SBATCH --error=/Weiss/fused_models/fused_test_%a.err
#SBATCH --array=6-11%3        #Following tasks, 24-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


# TODO: Running 23 and there is problem with GPU. Do it later
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-10}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


output_dir="/Weiss/fused_models/"
index_file="/MieT5/MyRepos/FEater/data/fused_models_joblist.txt"

train_file1=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $1}' ${index_file})
train_file2=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $2}' ${index_file})
test_file1=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $3}' ${index_file})
test_file2=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $4}' ${index_file})
data_type=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $5}' ${index_file})
output_paths=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $6}' ${index_file})
model_type=$(awk -v line=${SLURM_ARRAY_TASK_ID} 'NR==line+1 {print $7}' ${index_file})



###############################################################################
###############################################################################
###############################################################################
# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file="${train_file1}%${train_file2}"
test_file="${test_file1}%${test_file2}"



# outdir=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${output_paths}" "${SLURM_ARRAY_TASK_ID}")
# loader_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${loader_types}" "${SLURM_ARRAY_TASK_ID}")


output_paths=${output_dir}/${model_type}_${data_type}/
mkdir -p ${output_paths}

pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=64

lr_init=0.0001    # TODO doing test on Gnina with 1e-4 
lr_decay_steps=30
lr_decay_rate=0.5

# single coord -> 24, dual coord > 42, single surf -> 1500, dual surf -> 2000
if [[ "${data_type}" == "single" ]]; then
  # NOTE: PointNet2 Requires the number of point at least 32 
  point_nr1=24
  point_nr2=1500
elif [[ "${data_type}" == "dual" ]]; then
  point_nr1=42
  point_nr2=2000
else
  echo "Invalid data type: ${data_type}"
  exit 1
fi 

extraparms="--target_np ${point_nr1} --target_np2 ${point_nr2}"

if [ ${start_epoch} -gt 0 ]; then
  extraparms="${extraparms} --start-epoch ${start_epoch}"
fi
if [ ${#pretrained_model} -gt 0 ]; then
  extraparms="${extraparms} --pretrained ${pretrained_model}"
fi

echo $extraparms

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)" 
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_fused_models.py \
  --model ${model_type} --optimizer adam --loss-function crossentropy \
  --training-data ${train_file} --test-data ${test_file} --output-folder ${output_paths} --test-number 4000 \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  ${extraparms} 

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"