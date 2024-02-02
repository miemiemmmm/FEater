#!/bin/bash -l
#SBATCH --job-name=Surf_Gen
#SBATCH --output=/home/yazhang/scratch/FEater_trajs/Surf_Gen%a.out    # %a is the Array index
#SBATCH --error=/home/yazhang/scratch/FEater_trajs/Surf_Gen%a.err    # %a is the Array index
#SBATCH --array=0-5        # TODO: change this according to the number of jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=standard
#SBATCH --mem=512000
#SBATCH --gres gpu:V100:1
#SBATCH --time=36:00:00


module load v100
module load cuda
module load mamba
source activate mlenv

# source /home/yzhang/mamba/bin/loadmamba
# which micromamba
# micromamba activate mlenv

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


# Failed Job Array TODO: Change the following array Accordingly
files="/scratch/yazhang/FEater_trajs/sarscov2-15235444_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235444_s20_out_single_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235449_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235449_s20_out_single_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235455_s20_out_dual_s1.h5%/scratch/yazhang/FEater_trajs/sarscov2-15235455_s20_out_single_s1.h5"
job_file=$(python3 -c "import sys; print(sys.argv[1].strip('%').split('%')[${SLURM_ARRAY_TASK_ID}])" ${files})

export PYTHONUNBUFFERED=1

# TODO: Check folders
cd /home/yazhang/scratch/FEater_trajs/
output=$(echo ${job_file} | sed "s/.h5/_surf.h5/g")

echo "The task ${SLURM_ARRAY_TASK_ID} focus on ${job_file}"
echo "Writing the output file to ${output}"

if [ ${#job_file} -gt 0 ] && [ -f ${job_file} ]; then 
  echo "The input file ${job_file} exists"
  echo "Writing the output file to ${output}"
  # [ -f ${output} ] && rm -f ${output} && echo "Removed the existing file ${output}"
  # NOTE: make change to the start batch 
  # echo "python3  /home/yazhang/data/FEater_data/surf_HDF.py -i ${input}  -o ${output}"
  python3  /home/yazhang/data/FEater_data/surf_HDF.py -i ${job_file}  -o ${output} \
  --processes ${SLURM_CPUS_PER_TASK} --start_batch 0

else
  echo "The input file ${job_file} does not exist" >&2
  exit 1
fi


echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)";








