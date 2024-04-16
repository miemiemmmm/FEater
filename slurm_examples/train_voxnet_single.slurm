#!/bin/bash -l
#SBATCH --job-name=VoxNet_Single
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_vox_miniset/VoxNet_Single.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_vox_miniset/VoxNet_Single.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

train_file="/Weiss/FEater_Data/FEater_Minisets/tr_single_vox.txt"
test_file="/Weiss/FEater_Single_VOX/te.txt"
outdir="/diskssd/yzhang/FEater_data/results_single_vox_miniset"

pretrained_model=""
start_epoch=0

total_epoch=200								# NOTE IMPORTANT
batch_size=256								# NOTE IMPORTANT
worker_nr=12									# NOTE IMPORTANT

learning_rate=0.001
dataset="single"							# NOTE IMPORTANT
eval_interval=1000     				# NOTE IMPORTANT


echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

python3 /MieT5/MyRepos/FEater/feater/scripts/train_voxnet.py \
	-train ${train_file} \
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



