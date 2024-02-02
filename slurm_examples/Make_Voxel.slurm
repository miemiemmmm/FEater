#!/bin/bash -l
#SBATCH --job-name=Dual_VOX
#SBATCH --output=/home/yazhang/scratch/FEater_trajs/Dual_VOX%a.out    # %a is the Array index
#SBATCH --error=/home/yazhang/scratch/FEater_trajs/Dual_VOX%a.err    # %a is the Array index
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=standard
#SBATCH --mem=64000
#SBATCH --gres gpu:V100:1
#SBATCH --time=72:00:00


module load v100
module load cuda
module load mamba

source activate mlenv

echo "####################################"
echo "Check GPU"
nvidia-smi

echo "####################################"
echo "Check memory"
free -h 

echo "####################################"
echo "Check CPU"
lscpu 
echo "####################################"

# source /home/yzhang/mamba/bin/loadmamba
# micromamba activate mlenv
export PYTHONUNBUFFERED=1

files="/scratch/yazhang/FEater_trajs/sarscov2-15235444_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235444_s20_out_single_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235449_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235449_s20_out_single_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235455_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235455_s20_out_single_s1.h5"
job_file=$(python3 -c "import sys; print(sys.argv[1].strip('%').split('%')[${SLURM_ARRAY_TASK_ID}])" ${files})


# TODO: Check folders
cd /home/yazhang/scratch/FEater_trajs/
output=$(echo ${job_file} | sed "s/.h5/_vox.h5/g")

echo "The task ${SLURM_ARRAY_TASK_ID} focus on ${job_file}"
echo "Writing the output file to ${output}"

if [ ${#job_file} -gt 0 ] && [ -f ${job_file} ]; then 
  echo "The input file ${job_file} exists"
  [ -f ${output} ] && rm -f ${output} && echo "Removed the existing file ${output}"
  python3  /data/yazhang/FEater_data/coord_to_voxel.py -i ${job_file}  -o ${output} \
  --processes ${SLURM_CPUS_PER_TASK}

else
  echo "The input file ${job_file} does not exist"
  exit 1
fi


echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)";




