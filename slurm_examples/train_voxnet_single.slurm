#!/bin/bash -l
#SBATCH --job-name=VoxNet_Single
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_vox/VoxNet_Single.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_vox/VoxNet_Single.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

train_file="/diskssd/yzhang/FEater_data/FEater_Single_VOX/tr.txt"
valid_file="/diskssd/yzhang/FEater_data/FEater_Single_VOX/val.txt"
test_file="/diskssd/yzhang/FEater_data/FEater_Single_VOX/te.txt"
outdir="/diskssd/yzhang/FEater_data/results_single_vox"

pretrained_model=""
start_epoch=0

total_epoch=36								# NOTE IMPORTANT
batch_size=512								# NOTE IMPORTANT
worker_nr=32									# NOTE IMPORTANT

learning_rate=0.001
dataset="single"							# NOTE IMPORTANT
eval_interval=1000     				# NOTE IMPORTANT


echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

python /MieT5/MyRepos/FEater/feater/scripts/train_voxnet.py \
		-train ${train_file} \
		-valid ${valid_file} \
		-test  ${test_file} \
		-o 		 ${outdir} \
		\
    --pretrained "${pretrained_model}" \
    --start_epoch ${start_epoch} \
		\
		--data_workers ${worker_nr} \
		--epochs ${total_epoch} \
		--batch_size ${batch_size} \
		\
		-lr ${learning_rate} \
		--interval ${eval_interval} \
		--dataset ${dataset}

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"



