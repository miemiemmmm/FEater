#!/bin/bash -l
#SBATCH --job-name=PN_CRD_tp24   # NOTE: Job name and output/error files
#SBATCH --output=/home/yzhang/Documents/FEater_Dataset/results/pointnet_coord_tp24_single/PN_CRD_tp24.out
#SBATCH --error=/home/yzhang/Documents/FEater_Dataset/results/pointnet_coord_tp24_single/PN_CRD_tp24.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

source /home/yzhang/mamba/bin/loadmamba
micromamba activate mlenv


# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
# Coord Single
train_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_PDBHDF/train.txt"
valid_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_PDBHDF/valid.txt"
test_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_PDBHDF/test.txt"
outdir="/home/yzhang/Documents/FEater_Dataset/results/pointnet_coord_tp24_single"


# Surf single
# train_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_SURF/train.txt"
# valid_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_SURF/valid.txt"
# test_file="/home/yzhang/Documents/FEater_Dataset/FEater_Single_SURF/test.txt"
# outdir="/home/yzhang/Documents/FEater_Dataset/results/pointnet_surf_tp1500_single"


# Coord Dual
# train_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_PDBHDF/train.txt"
# valid_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_PDBHDF/valid.txt"
# test_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_PDBHDF/test.txt"
# outdir="/home/yzhang/Documents/FEater_Dataset/results/pointnet_coord_tp42_dual"


# Surf Dual
# train_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_SURF/train.txt"
# valid_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_SURF/valid.txt"
# test_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_SURF/test.txt"
# outdir="/home/yzhang/Documents/FEater_Dataset/results/pointnet_surf_tp2000_dual


mkdir -p ${outdir}


pretrained_model=""
start_epoch=0

total_epoch=36
batch_size=256
worker_nr=24

target_point_nr=24   				# NOTE IMPORTANT
data_type="coord"							# NOTE IMPORTANT
dataset="single"							# NOTE IMPORTANT
learning_rate=0.001

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_pointnet.py \
	-train  ${train_file} \
	-test   ${test_file} \
	-valid  ${valid_file} \
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
