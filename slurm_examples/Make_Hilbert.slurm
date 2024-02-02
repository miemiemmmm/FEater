#!/bin/bash -l
#SBATCH --job-name=Vox_Hilbert
#SBATCH --output=/media/yzhang/MieT72/Data/Vox_Hilbert%a.out     # %a is the Array index
#SBATCH --error=/media/yzhang/MieT72/Data/Vox_Hilbert%a.err      # %a is the Array index
#SBATCH --array=4-5%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal

source /home/yzhang/mamba/bin/loadmamba
micromamba activate mlenv 

inputfiles="/media/yzhang/MieT72/Data/FEater_Single_VOX/TestSet_Voxel.h5%/media/yzhang/MieT72/Data/FEater_Single_VOX/TrainingSet_Voxel.h5%/media/yzhang/MieT72/Data/FEater_Single_VOX/ValidationSet_Voxel.h5%/media/yzhang/MieT72/Data/FEater_Dual_VOX/TestSet_Voxel.h5%/media/yzhang/MieT72/Data/FEater_Dual_VOX/TrainingSet_Voxel.h5%/media/yzhang/MieT72/Data/FEater_Dual_VOX/ValidationSet_Voxel.h5"
outfolders="/media/yzhang/MieT72/Data/FEater_Single_HILB%/media/yzhang/MieT72/Data/FEater_Single_HILB%/media/yzhang/MieT72/Data/FEater_Single_HILB%/disk2b/yzhang/testdata/FEater_Dual_HILB%/disk2b/yzhang/testdata/FEater_Dual_HILB%/disk2b/yzhang/testdata/FEater_Dual_HILB"

input_i=$(python3 -c "import sys; taskfile=sys.argv[1].strip('%').split('%')[int(sys.argv[2])]; print(taskfile)" ${inputfiles} ${SLURM_ARRAY_TASK_ID})
outputfolder_i=$(python3 -c "import sys; taskfile=sys.argv[1].strip('%').split('%')[int(sys.argv[2])]; print(taskfile)" ${outfolders} ${SLURM_ARRAY_TASK_ID})
outputfilename=$(basename ${input_i} | sed 's|Voxel|Hilbert|g')
echo "Processing the input file ${input_i} and generating the output file ${outputfolder_i}/${outputfilename}"

export PYTHONUNBUFFERED=1

python3 /MieT5/MyRepos/FEater/feater/scripts/hilbert_hdf.py \
  -i ${input_i} \
  -o ${outputfolder_i}/${outputfilename}  \
  --processes ${SLURM_CPUS_PER_TASK} 
