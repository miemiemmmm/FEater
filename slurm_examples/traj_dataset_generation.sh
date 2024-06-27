#!/bin/bash -l

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1

trajfile="/Matter/misato_database/MD.hdf5"

python3 -c """from feater.scripts import process_trajectories; import pandas as pd; 

df = pd.read_csv('/MieT5/MyRepos/FEater/data/misato_trajs.csv');

[process_trajectories.programming_interface(traj='${trajfile}', top=df['topfile'][i], output='/tmp/misato_single.h5', 
stride=50, mode='single', trajtype='misato') for i in range(df.shape[0])];

[process_trajectories.programming_interface(traj='${trajfile}', top=df['topfile'][i], output='/tmp/misato_dual.h5', 
stride=50, mode='dual', trajtype='misato') for i in range(df.shape[0])];
"""


################################################################################
trajfile="/MieT5/DataSets/DESRES-Trajectory_sarscov2-15235449-peptide-A-no-water-no-ion/sarscov2-15235449-peptide-A-no-water-no-ion/s800_out.pdb"
echo Processing ${trajfile}
python3 -c """from feater.scripts import process_trajectories;

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='single', 
output='/tmp/DES_single.h5', stride=1, trajtype='normal')

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='dual', 
output='/tmp/DES_dual.h5', stride=1, trajtype='normal')
"""

################################################################################
trajfile="/MieT5/DataSets/DESRES-Trajectory_sarscov2-15235455-peptide-B-no-water-no-ion/sarscov2-15235455-peptide-B-no-water-no-ion/s800_out.pdb"
echo Processing ${trajfile}
python3 -c """from feater.scripts import process_trajectories;

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='single', 
output='/tmp/DES_single.h5', stride=1, trajtype='normal')

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='dual', 
output='/tmp/DES_dual.h5', stride=1, trajtype='normal')
"""

################################################################################
trajfile="/MieT5/DataSets/DESRES-Trajectory_sarscov2-15235444-peptide-C-no-water-no-ion/sarscov2-15235444-peptide-C-no-water-no-ion/s800_out.pdb"
echo Processing ${trajfile}
python3 -c """from feater.scripts import process_trajectories;

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='single', 
output='/tmp/DES_single.h5', stride=1, trajtype='normal')

process_trajectories.programming_interface(traj='${trajfile}', top='', mode='dual', 
output='/tmp/DES_dual.h5', stride=1, trajtype='normal')
"""

################################################################################
# Process the output files 

# cp /tmp/misato_single.h5  /Weiss/FEater_trajs/misato_coord_single.h5
# cp /tmp/misato_dual.h5    /Weiss/FEater_trajs/misato_coord_dual.h5
# cp /tmp/DES_single.h5     /Weiss/FEater_trajs/DES_coord_single.h5
# cp /tmp/DES_dual.h5       /Weiss/FEater_trajs/DES_coord_dual.h5


# python /MieT5/MyRepos/FEater/feater/scripts/make_surface.py  -i /Weiss/FEater_trajs/misato_coord_single.h5 -o /Weiss/FEater_trajs/misato_surface_single.h5   #
# python /MieT5/MyRepos/FEater/feater/scripts/make_surface.py  -i /Weiss/FEater_trajs/DES_coord_single.h5    -o /Weiss/FEater_trajs/DES_surface_single.h5      #
# python /MieT5/MyRepos/FEater/feater/scripts/make_surface.py  -i /Weiss/FEater_trajs/misato_coord_dual.h5   -o /Weiss/FEater_trajs/misato_surface_dual.h5     # 
# python /MieT5/MyRepos/FEater/feater/scripts/make_surface.py  -i /Weiss/FEater_trajs/DES_coord_dual.h5      -o /Weiss/FEater_trajs/DES_surface_dual.h5        #

# python /MieT5/MyRepos/FEater/feater/scripts/make_voxel.py    -i /Weiss/FEater_trajs/misato_coord_single.h5 -o /Weiss/FEater_trajs/misato_voxel_single.h5     #
# python /MieT5/MyRepos/FEater/feater/scripts/make_voxel.py    -i /Weiss/FEater_trajs/DES_coord_single.h5 -o /Weiss/FEater_trajs/DES_voxel_single.h5           # 
# python /MieT5/MyRepos/FEater/feater/scripts/make_voxel.py    -i /Weiss/FEater_trajs/misato_coord_dual.h5 -o /Weiss/FEater_trajs/misato_voxel_dual.h5         # 
# python /MieT5/MyRepos/FEater/feater/scripts/make_voxel.py    -i /Weiss/FEater_trajs/DES_coord_dual.h5 -o /Weiss/FEater_trajs/DES_voxel_dual.h5               #

# python /MieT5/MyRepos/FEater/feater/scripts/make_hilbert.py  -i /Weiss/FEater_trajs/misato_voxel_single.h5 -o /Weiss/FEater_trajs/misato_hilbert_single.h5   # 
# python /MieT5/MyRepos/FEater/feater/scripts/make_hilbert.py  -i /Weiss/FEater_trajs/DES_voxel_single.h5 -o /Weiss/FEater_trajs/DES_hilbert_single.h5         #
# python /MieT5/MyRepos/FEater/feater/scripts/make_hilbert.py  -i /Weiss/FEater_trajs/misato_voxel_dual.h5 -o /Weiss/FEater_trajs/misato_hilbert_dual.h5       # 
# python /MieT5/MyRepos/FEater/feater/scripts/make_hilbert.py  -i /Weiss/FEater_trajs/DES_voxel_dual.h5 -o /Weiss/FEater_trajs/DES_hilbert_dual.h5             #
