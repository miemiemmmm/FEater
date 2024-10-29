#!/bin/bash -l
#SBATCH --job-name=modelnet_test             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/modelnet_results/modelnet_test_%a.out
#SBATCH --error=/Weiss/modelnet_results/modelnet_test_%a.err
#SBATCH --array=1%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}  
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


output_dir="/Weiss/modelnet_results"


train_files="/Weiss/modelnet_data/tr.txt%/Weiss/modelnet_data/tr.txt%/Weiss/modelnet_data/tr.txt%/Weiss/modelnet_data/tr.txt%"
test_files="/Weiss/modelnet_data/te.txt%/Weiss/modelnet_data/te.txt%/Weiss/modelnet_data/te.txt%/Weiss/modelnet_data/te.txt%"
data_types="modelnet%modelnet%modelnet%modelnet"
loader_types="surface%surface%surface%surface%"
model_types="pointnet%pointnet2%dgcnn%paconv%"


###############################################################################
# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${train_files}" "${SLURM_ARRAY_TASK_ID}")
data_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${data_types}" "${SLURM_ARRAY_TASK_ID}")
loader_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${loader_types}" "${SLURM_ARRAY_TASK_ID}")
test_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${test_files}" "${SLURM_ARRAY_TASK_ID}")
model_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${model_types}" "${SLURM_ARRAY_TASK_ID}")

output_path=${output_dir}/${model_type}_${data_type}/
mkdir -p ${output_path}

pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=64

lr_init=0.001    # TODO doing test on Gnina with 1e-4 
lr_decay_steps=30
lr_decay_rate=0.5

# single coord -> 24, dual coord > 42, single surf -> 1500, dual surf -> 2000
point_nr=1500

extraparms="--production 1 --cuda 1 --verbose 0"

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
  --training-data ${train_file} --test-data ${test_file} --output_folder ${output_path} --test-number 4000 \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  --data-type ${data_type} --dataloader-type ${loader_type} ${extraparms} 

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"