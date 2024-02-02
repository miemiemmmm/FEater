#!/bin/bash -l
#SBATCH --job-name=ResNet50      # TODO: change job name and log folder 
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_hilbert/ResNet18.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_hilbert/ResNet18.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


train_file="/software/database/FEater_Single_HILB/tr.txt"
valid_file="/software/database/FEater_Single_HILB/te.txt"
test_file="/software/database/FEater_Single_HILB/val.txt"
outdir="/diskssd/yzhang/FEater_data/results_single_hilbert"

pretrained_model=""
start_epoch=0

total_epoch=24            # NOTE IMPORTANT
batch_size=256            # NOTE IMPORTANT
worker_nr=32              # NOTE IMPORTANT

network_type="resnet18"         # NOTE IMPORTANT
learning_rate=0.0005            # NOTE IMPORTANT
dataset="single"								# NOTE IMPORTANT
eval_interval=500  			   	    # NOTE IMPORTANT

# Valid settings: resnet18, 0.001, 2000 per epoch

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

python3 /MieT5/MyRepos/FEater/feater/scripts/train_resnet.py \
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
