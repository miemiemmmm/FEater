#!/bin/bash -l
#SBATCH --job-name=ResNet18                 # TODO: change job name and log folder 
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_hilb_miniset/ResNet18.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_hilb_miniset/ResNet18.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


train_file="/Weiss/FEater_Data/FEater_Minisets/tr_single_hilb.txt"
test_file="/Weiss/FEater_Single_HILB/te.txt"
outdir="/diskssd/yzhang/FEater_data/results_single_hilb_miniset"

pretrained_model=""
start_epoch=0

total_epoch=200            # NOTE IMPORTANT
batch_size=256            # NOTE IMPORTANT
worker_nr=12              # NOTE IMPORTANT

network_type="resnet18"         # NOTE IMPORTANT
learning_rate=0.001            # NOTE IMPORTANT
dataset="single"								# NOTE IMPORTANT
eval_interval=500  			   	    # NOTE IMPORTANT

# Valid settings: resnet18, 0.001, 2000 per epoch

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

python3 /MieT5/MyRepos/FEater/feater/scripts/train_resnet.py \
  -train ${train_file} \
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
