#!/bin/bash -l
#SBATCH --job-name=PN_CRD_tp42   # NOTE: Job name and output/error files
#SBATCH --output=/diskssd/yzhang/FEater_data/result_dual_coord_miniset/PN_CRD_tp42.out
#SBATCH --error=/diskssd/yzhang/FEater_data/result_dual_coord_miniset/PN_CRD_tp42.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch


# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
# Coord Dual
train_file="/Weiss/FEater_Data/FEater_Minisets/tr_dual_coord.txt"
test_file="/Weiss/FEater_Dual_PDBHDF/te.txt"
outdir="/diskssd/yzhang/FEater_data/result_dual_coord_miniset"

pretrained_model=""
start_epoch=0

total_epoch=200
batch_size=256
worker_nr=12

target_point_nr=42   				# NOTE IMPORTANT
data_type="coord"							# NOTE IMPORTANT
dataset="dual"							# NOTE IMPORTANT
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
	--interval 10000 \
	--date_type ${data_type} \
	--dataset ${dataset}

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"
