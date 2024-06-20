#!/bin/bash -l
#SBATCH --job-name=hilbert_training_results             									  # NOTE: Job name and output/error files
#SBATCH --output=/Weiss/training_results_HILB/hilbert_training_results_%a.out
#SBATCH --error=/Weiss/training_results_HILB/hilbert_training_results_%a.err
#SBATCH --array=0-6%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

##### Obsolete #####

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

train_files_single="/diskssd/yzhang/FEater_Minisets/miniset_2000_single/te_hilb.txt%/diskssd/yzhang/FEater_Minisets/miniset_200_single/te_hilb.txt%/diskssd/yzhang/FEater_Minisets/miniset_400_single/te_hilb.txt%/diskssd/yzhang/FEater_Minisets/miniset_800_single/te_hilb.txt%"
test_files_single="/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%"
data_types_single="single%single%single%single%"
output_paths_single="/Weiss/training_results_HILB/miniset_2000_single_result%/Weiss/training_results_HILB/miniset_200_single_result%/Weiss/training_results_HILB/miniset_400_single_result%/Weiss/training_results_HILB/miniset_800_single_result%"


# for i in 2000 200 400 800; do mkdir miniset_${i}_single_result ;done


train_files_dual="/diskssd/yzhang/FEater_Minisets/miniset_200/te_hilb.txt%/diskssd/yzhang/FEater_Minisets/miniset_400/te_hilb.txt%/diskssd/yzhang/FEater_Minisets/miniset_800/te_hilb.txt%"
test_files_dual="/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%"
data_types_dual="dual%dual%dual%"
output_paths_dual="/Weiss/training_results_HILB/miniset_200_dual_result%/Weiss/training_results_HILB/miniset_400_dual_result%/Weiss/training_results_HILB/miniset_800_dual_result%"

# Prepare the output directories
# for i in 200 400 800; do mkdir miniset_${i}_dual_result ;done



# NOTE: make change to the path of datasets
train_files=${train_files_single}${train_files_dual}
data_types=${data_types_single}${data_types_dual}
output_paths=${output_paths_single}${output_paths_dual}
test_files=${test_files_single}${test_files_dual}

# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${train_files}" "${SLURM_ARRAY_TASK_ID}")
data_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${data_types}" "${SLURM_ARRAY_TASK_ID}")
outdir=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${output_paths}" "${SLURM_ARRAY_TASK_ID}")
test_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${test_files}" "${SLURM_ARRAY_TASK_ID}")


pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=64

# Improve the initial learning rate and accelerate the learning rate decay
lr_init=0.001
lr_decay_steps=30
lr_decay_rate=0.5


network_type="resnet"       # NOTE IMPORTANT

# Valid settings: resnet18, 0.001, 2000 per epoch

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py \
  --model resnet --optimizer adam --loss-function crossentropy \
  --training-data ${train_file} --test-data ${test_file} --output_folder ${outdir} --test-number 4000 \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  --data-type ${data_type} --production 1 --cuda 1 
	# --dataloader-type surface \
  # --start_epoch ${start_epoch} --pretrained "${pretrained_model}" \

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"