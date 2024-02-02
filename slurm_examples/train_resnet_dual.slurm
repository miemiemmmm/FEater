#!/bin/bash -l
#SBATCH --job-name=ResNet18_Dual      # TODO: change job name and log folder 
#SBATCH --output=/home/yzhang/Documents/FEater_Dataset/results/resnet_hilbert_dual/ResNet18_Dual.out
#SBATCH --error=/home/yzhang/Documents/FEater_Dataset/results/resnet_hilbert_dual/ResNet18_Dual.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate mlenv
export PYTHONUNBUFFERED=1

# NOTE: make change to the path of datasets
train_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_HILB/tr.txt"
valid_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_HILB/te.txt"
test_file="/home/yzhang/Documents/FEater_Dataset/FEater_Dual_HILB/val.txt"
outdir="/home/yzhang/Documents/FEater_Dataset/results/resnet_hilbert_dual"

pretrained_model=""
start_epoch=0

total_epoch=24            # NOTE IMPORTANT
batch_size=256            # NOTE IMPORTANT
worker_nr=32              # NOTE IMPORTANT

network_type="resnet18"         # NOTE IMPORTANT
learning_rate=0.0005            # NOTE IMPORTANT
dataset="dual"							  	# NOTE IMPORTANT
eval_interval=1000  			   	    # NOTE IMPORTANT

# Valid settings: resnet18, 0.001, 2000 per epoch

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

# NOTE: change the script path if run on different machine
# python3 /MieT5/MyRepos/FEater/feater/scripts/train_resnet.py \
python3 train_resnet.py \
  -train ${train_file} \
  -valid ${valid_file} \
  -test  ${test_file}  \
  -o     ${outdir}     \
  --pretrained "${pretrained_model}" \
  --start_epoch ${start_epoch} \
  --data_workers ${worker_nr} \
  --epochs ${total_epoch} \
  --batch_size ${batch_size} \
  -lr ${learning_rate} \
  --resnet_type ${network_type} \
  --interval ${eval_interval} \
  --dataset ${dataset}

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"
