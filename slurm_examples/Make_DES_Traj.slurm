#!/bin/bash -l
#SBATCH --job-name=DES_traj_extract
#SBATCH --output=/storage006/yzhang/tests/FEater_dynamics/DES_traj_extract%a.out    # %a is the Array index
#SBATCH --error=/storage006/yzhang/tests/FEater_dynamics/DES_traj_extract%a.err    # %a is the Array index
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=stromboli

source /storage006/yzhang/mamba/bin/loadmamba
micromamba activate mlenv 

trajs="/storage006/yzhang/tests/DESRES-Trajectory_sarscov2-15235444-peptide-C-no-water-no-ion/sarscov2-15235444-peptide-C-no-water-no-ion/sarscov2-15235444_s20_out.pdb%/storage006/yzhang/tests/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449_s20_out.pdb%/storage006/yzhang/tests/DESRES-Trajectory_sarscov2-15235455-peptide-B-no-water-no-ion/sarscov2-15235455-peptide-B-no-water-no-ion/sarscov2-15235455_s20_out.pdb"
outfolder="/storage006/yzhang/tests/FEater_dynamics"

ret=$(python -c 'import sys; i=int(sys.argv[1]); trajs=sys.argv[2]; modes = sys.argv[3] ; print(trajs.split("%")[i%3], modes.split("%")[i%2])' ${SLURM_ARRAY_TASK_ID} "${trajs}" "single%dual")
trajfile=$(echo ${ret} | awk '{print $1}')
mode=$(echo ${ret} | awk '{print $2}')

echo "For task ${SLURM_ARRAY_TASK_ID}, the traj is ${trajfile} and the mode is ${mode}"

export PYTHONUNBUFFERED=1

python3 /storage006/yzhang/tests/FEater/feater/scripts/process_deshaw_traj.py \
  --traj ${trajfile} \
  --output ${outfolder} \
  --mode ${mode} \
  --stride 1 
  






# parser.add_argument("-i", "--traj", type=str, default="", help="The trajectory file.")
#   parser.add_argument("-t", "--top", type=str, default="", help="The topology file.")
#   parser.add_argument("-o", "--output", type=str, help="The output HDF file.")
#   parser.add_argument("-s", "--stride", type=int, default=1, help="The stride for the trajectory file.")
#   parser.add_argument("-m", "--mode"