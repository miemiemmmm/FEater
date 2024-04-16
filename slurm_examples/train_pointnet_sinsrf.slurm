#!/bin/bash -l
#SBATCH --job-name=PN_SRF_tp1500   # NOTE: Job name and output/error files
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_surf_miniset/PN_SRF_tp1500.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_surf_miniset/PN_SRF_tp1500.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file="/Weiss/FEater_Data/FEater_Minisets/tr_single_surf.txt"
test_file="/Weiss/FEater_Single_SURF/te.txt"
outdir="/diskssd/yzhang/FEater_data/results_single_surf_miniset"


pretrained_model=""
start_epoch=0

total_epoch=200
batch_size=256
worker_nr=12

target_point_nr=1500   				# NOTE IMPORTANT
data_type="surf"							# NOTE IMPORTANT
dataset="single"							# NOTE IMPORTANT
learning_rate=0.001

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_pointnet.py \
	-train  ${train_file} \
	-test   ${test_file} \
	-o ${outdir} \
	\
	--start_epoch ${start_epoch} \
	--pretrained 	"${pretrained_model}" \
	\
	--epochs ${total_epoch} \
	--batch_size ${batch_size} \
	--data_workers ${worker_nr} \
	\
	-lr ${learning_rate} \
	--n_points ${target_point_nr} \
	--interval 1000 \
	--date_type ${data_type} \
	--dataset ${dataset}

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"
