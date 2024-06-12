#!/bin/bash -l
#SBATCH --job-name=surface_training_results 									  # NOTE: Job name and output/error files
#SBATCH --output=/Weiss/training_results_SURF/surface_training_results_%a.out
#SBATCH --error=/Weiss/training_results_SURF/surface_training_results_%a.err
#SBATCH --array=6%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

train_files_single="/diskssd/yzhang/FEater_Minisets/miniset_2000_single/te_surf.txt%/diskssd/yzhang/FEater_Minisets/miniset_200_single/te_surf.txt%/diskssd/yzhang/FEater_Minisets/miniset_400_single/te_surf.txt%/diskssd/yzhang/FEater_Minisets/miniset_800_single/te_surf.txt%"
test_files_single="/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%"
data_types_single="single%single%single%single%"
output_paths_single="/Weiss/training_results_SURF/miniset_2000_single_result%/Weiss/training_results_SURF/miniset_200_single_result%/Weiss/training_results_SURF/miniset_400_single_result%/Weiss/training_results_SURF/miniset_800_single_result%"
point_number_single="1500%1500%1500%1500%"


train_files_dual="/diskssd/yzhang/FEater_Minisets/miniset_200/te_surf.txt%/diskssd/yzhang/FEater_Minisets/miniset_400/te_surf.txt%/diskssd/yzhang/FEater_Minisets/miniset_800/te_surf.txt%"
test_files_dual="/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%"
data_types_dual="dual%dual%dual%"
output_paths_dual="/Weiss/training_results_SURF/miniset_200_result%/Weiss/training_results_SURF/miniset_400_result%/Weiss/training_results_SURF/miniset_800_result%"
point_number_dual="2000%2000%2000%"



train_files=${train_files_single}${train_files_dual}
test_files=${test_files_single}${test_files_dual}
data_types=${data_types_single}${data_types_dual}
output_paths=${output_paths_single}${output_paths_dual}
point_number=${point_number_single}${point_number_dual}


# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${train_files}" "${SLURM_ARRAY_TASK_ID}")
data_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${data_types}" "${SLURM_ARRAY_TASK_ID}")
outdir=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${output_paths}" "${SLURM_ARRAY_TASK_ID}")
test_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${test_files}" "${SLURM_ARRAY_TASK_ID}")
target_point_nr=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${point_number}" "${SLURM_ARRAY_TASK_ID}")


pretrained_model=""
start_epoch=0

total_epoch=120
batch_size=256

lr_init=0.001
lr_decay_steps=30
lr_decay_rate=0.5



echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py \
  --model pointnet --optimizer adam --loss-function crossentropy \
  --training-data ${train_file} --test-data ${test_file} --output_folder ${outdir} --test-number 4000 \
  --dataloader-type surface \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  --pointnet_points ${target_point_nr} --production 1 --cuda 1 \
  --data-type ${data_type}  \
  # --start_epoch ${start_epoch} --pretrained "${pretrained_model}" \

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"
