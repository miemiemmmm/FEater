#!/bin/bash -l
#SBATCH --job-name=benchmark_1000             			# NOTE: Job name and output/error files
#SBATCH --output=/Weiss/benchmark_models/benchmarking_results_%a.out
#SBATCH --error=/Weiss/benchmark_models/benchmarking_results_%a.err
#SBATCH --array=21   #Following tasks, 24-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12


# TODO: Running 23 and there is problem with GPU. Do it later
# export SLURM_ARRAY_TASK_ID=21
# export SLURM_CPUS_PER_TASK=16

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


output_dir="/Weiss/benchmark_models/"

train_files_single="/Matter/feater_train_1000/single_coord.txt%/Matter/feater_train_1000/single_coord.txt%/Matter/feater_train_1000/single_coord.txt%/Matter/feater_train_1000/single_coord.txt%/Matter/feater_train_1000/single_surf.txt%/Matter/feater_train_1000/single_surf.txt%/Matter/feater_train_1000/single_surf.txt%/Matter/feater_train_1000/single_surf.txt%/Matter/feater_train_1000/single_vox.txt%/Matter/feater_train_1000/single_vox.txt%/Matter/feater_train_1000/single_vox.txt%/Matter/feater_train_1000/single_hilbert.txt%/Matter/feater_train_1000/single_hilbert.txt%/Matter/feater_train_1000/single_hilbert.txt%/Matter/feater_train_1000/single_hilbert.txt%/Matter/feater_train_1000/single_hilbert.txt%"
test_files_single="/Weiss/FEater_Single_PDBHDF/te.txt%/Weiss/FEater_Single_PDBHDF/te.txt%/Weiss/FEater_Single_PDBHDF/te.txt%/Weiss/FEater_Single_PDBHDF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_SURF/te.txt%/Weiss/FEater_Single_VOX/te.txt%/Weiss/FEater_Single_VOX/te.txt%/Weiss/FEater_Single_VOX/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%/Weiss/FEater_Single_HILB/te.txt%"
data_types_single="single%single%single%single%single%single%single%single%single%single%single%single%single%single%single%single%"
loader_single="coord%coord%coord%coord%surface%surface%surface%surface%vox%vox%vox%hilb%hilb%hilb%hilb%hilb%"
model_types_single="pointnet%pointnet2%dgcnn%paconv%pointnet%pointnet2%dgcnn%paconv%voxnet%deeprank%gnina%resnet%convnext%convnext_iso%swintrans%ViT%"


train_files_dual="/Matter/feater_train_1000/dual_coord.txt%/Matter/feater_train_1000/dual_coord.txt%/Matter/feater_train_1000/dual_coord.txt%/Matter/feater_train_1000/dual_coord.txt%/Matter/feater_train_1000/dual_surf.txt%/Matter/feater_train_1000/dual_surf.txt%/Matter/feater_train_1000/dual_surf.txt%/Matter/feater_train_1000/dual_surf.txt%/Matter/feater_train_1000/dual_vox.txt%/Matter/feater_train_1000/dual_vox.txt%/Matter/feater_train_1000/dual_vox.txt%/Matter/feater_train_1000/dual_hilbert.txt%/Matter/feater_train_1000/dual_hilbert.txt%/Matter/feater_train_1000/dual_hilbert.txt%/Matter/feater_train_1000/dual_hilbert.txt%/Matter/feater_train_1000/dual_hilbert.txt%"
test_files_dual="/Weiss/FEater_Dual_PDBHDF/te.txt%/Weiss/FEater_Dual_PDBHDF/te.txt%/Weiss/FEater_Dual_PDBHDF/te.txt%/Weiss/FEater_Dual_PDBHDF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_SURF/te.txt%/Weiss/FEater_Dual_VOX/te.txt%/Weiss/FEater_Dual_VOX/te.txt%/Weiss/FEater_Dual_VOX/te.txt%/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%/Weiss/FEater_Dual_HILB/te.txt%"
data_types_dual="dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%dual%"
loader_dual="coord%coord%coord%coord%surface%surface%surface%surface%vox%vox%vox%hilb%hilb%hilb%hilb%hilb%"
model_types_dual="pointnet%pointnet2%dgcnn%paconv%pointnet%pointnet2%dgcnn%paconv%voxnet%deeprank%gnina%resnet%convnext%convnext_iso%swintrans%ViT%"


# NOTE: make change to the path of datasets
train_files=${train_files_single}${train_files_dual}
data_types=${data_types_single}${data_types_dual}
loader_types=${loader_single}${loader_dual}
test_files=${test_files_single}${test_files_dual}
model_types=${model_types_single}${model_types_dual}

###############################################################################
###############################################################################
###############################################################################
# NOTE: Set the correct path to your train.txt, valid.txt and test.txt and Output directory
train_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${train_files}" "${SLURM_ARRAY_TASK_ID}")
data_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${data_types}" "${SLURM_ARRAY_TASK_ID}")
# outdir=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${output_paths}" "${SLURM_ARRAY_TASK_ID}")
loader_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${loader_types}" "${SLURM_ARRAY_TASK_ID}")
test_file=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${test_files}" "${SLURM_ARRAY_TASK_ID}")
model_type=$(python -c "import sys; filestr = sys.argv[1]; files=[i for i in filestr.strip().strip('%').split('%') if len(i) > 0]; idx = int(sys.argv[2]); print(files[idx])" "${model_types}" "${SLURM_ARRAY_TASK_ID}")

output_paths=${output_dir}/${model_type}_${data_type}_${loader_type}/
mkdir -p ${output_paths}

pretrained_model="/Weiss/benchmark_models/paconv_dual_surface/paconv_64.pth"
start_epoch=65

total_epoch=120
batch_size=64

lr_init=0.0001    # TODO doing test on Gnina with 1e-4 and 1e-5
lr_decay_steps=30
lr_decay_rate=0.5

# single coord -> 24, dual coord > 42, single surf -> 1500, dual surf -> 2000
if [[ "${data_type}" == "single" ]] && [[ "${loader_type}" == "coord" ]]; then
  # NOTE: PointNet2 Requires the number of point at least 32 
  point_nr=24
elif [[ "${data_type}" == "dual" ]] && [[ "${loader_type}" == "coord" ]]; then
  point_nr=42
elif [[ "${data_type}" == "single" ]] && [[ "${loader_type}" == "surface" ]]; then
  point_nr=1500
elif [[ "${data_type}" == "dual" ]] && [[ "${loader_type}" == "surface" ]]; then
  point_nr=1500
else
  point_nr=0
fi

extraparms=""

if [ ${point_nr} -gt 0 ]; then
  extraparms="${extraparms} --pointnet-points ${point_nr}"
fi
if [ ${start_epoch} -gt 0 ]; then
  extraparms="${extraparms} --start-epoch ${start_epoch}"
fi
if [ ${#pretrained_model} -gt 0 ]; then
  extraparms="${extraparms} --pretrained ${pretrained_model}"
fi

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)" 
# NOTE: change the python script path 
python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py \
  --model ${model_type} --optimizer adam --loss-function crossentropy \
  --training-data ${train_file} --test-data ${test_file} --output_folder ${output_paths} --test-number 4000 \
  -e ${total_epoch} -b ${batch_size} -w ${SLURM_CPUS_PER_TASK} --lr-init ${lr_init} --lr-decay-steps ${lr_decay_steps} --lr-decay-rate ${lr_decay_rate} \
  --data-type ${data_type} --production 1 --cuda 1 \
	--dataloader-type ${loader_type} ${extraparms} \
  # --production 0

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"