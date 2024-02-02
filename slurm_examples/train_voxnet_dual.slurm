#!/bin/bash -l
#SBATCH --job-name=VoxNet_Dual    # TODO check the name of the job and log dir 
#SBATCH --output=/diskssd/yzhang/FEater_data/results_dual_vox/VoxNet_Dual.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_dual_vox/VoxNet_Dual.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


train_file="/media/yzhang/Black/FEater_Dual_VOX/tr.txt"
valid_file="/media/yzhang/Black/FEater_Dual_VOX/val.txt"
test_file="/media/yzhang/Black/FEater_Dual_VOX/te.txt"
outdir="/diskssd/yzhang/FEater_data/results_dual_vox"

pretrained_model="/diskssd/yzhang/FEater_data/results_dual_vox/voxnet_14.pth"
start_epoch=15								# NOTE IMPORTANT

total_epoch=20								# NOTE IMPORTANT
batch_size=512								# NOTE IMPORTANT
worker_nr=32									# NOTE IMPORTANT

learning_rate=0.001
dataset="dual"								# NOTE IMPORTANT
eval_interval=1000  			   	# NOTE IMPORTANT


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

