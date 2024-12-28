#!/bin/bash -l
#SBATCH --job-name=baseline             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/baseline_models/baseline_%a.out
#SBATCH --error=/Weiss/baseline_models/baseline_%a.err
#SBATCH --array=1%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


# TODO: Running 23 and there is problem with GPU. Do it later
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}  
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


output_dir="/Weiss/baseline_models"


# train_files="/Weiss/clustered_single_10perclass/tr.txt%/Weiss/clustered_single_10perclass/tr_surf.txt%/Weiss/clustered_single_10perclass/tr_vox.txt%/Weiss/clustered_single_10perclass/tr_hilb.txt%/Weiss/clustered_dual_10perclass/tr.txt%/Weiss/clustered_dual_10perclass/tr_surf.txt%/Weiss/clustered_dual_10perclass/tr_vox.txt%/Weiss/clustered_dual_10perclass/tr_hilb.txt%"
# test_files="/Weiss/clustered_single_10perclass/te.txt%/Weiss/clustered_single_10perclass/te_surf.txt%/Weiss/clustered_single_10perclass/te_vox.txt%/Weiss/clustered_single_10perclass/te_hilb.txt%/Weiss/clustered_dual_10perclass/te.txt%/Weiss/clustered_dual_10perclass/te_surf.txt%/Weiss/clustered_dual_10perclass/te_vox.txt%/Weiss/clustered_dual_10perclass/te_hilb.txt%"
# data_types="single%single%single%single%dual%dual%dual%dual"
# loader_types="coord%surface%vox%hilb%coord%surface%vox%hilb"
# model_types="pointnet%pointnet%voxnet%resnet%pointnet%pointnet%voxnet%resnet"



train_files="/Weiss/clustered_single_10perclass/tr_vox.txt%/Weiss/clustered_dual_10perclass/tr_vox.txt%"
test_files="/Weiss/clustered_single_10perclass/te_vox.txt%/Weiss/clustered_dual_10perclass/te_vox.txt%"
data_types="single%dual%"
loader_types="vox%vox%"
model_types="gnina%gnina%"



###############################################################################
# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${train_files}" "${SLURM_ARRAY_TASK_ID}")
data_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${data_types}" "${SLURM_ARRAY_TASK_ID}")
loader_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${loader_types}" "${SLURM_ARRAY_TASK_ID}")
test_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${test_files}" "${SLURM_ARRAY_TASK_ID}")
model_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${model_types}" "${SLURM_ARRAY_TASK_ID}")

output_path=${output_dir}/${model_type}_${data_type}_${loader_type}/
mkdir -p ${output_path}

pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=64

lr_init=0.0001    # TODO doing test on Gnina with 1e-4 
lr_decay_steps=30
lr_decay_rate=0.5

# single coord -> 24, dual coord > 42, single surf -> 1500, dual surf -> 2000
if [[ "${data_type}" == "single" ]] && [[ "${loader_type}" == "coord" ]]; then
  # NOTE: PointNet2 Requires the number of point at least 32 
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

extraparms="--production 1 --cuda 1"

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

