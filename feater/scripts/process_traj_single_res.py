import tempfile, time, os, sys, hashlib, argparse, json
import multiprocessing as mp

import numpy as np
import pytraj as pt

from feater import utils, constants, io, dataloader
from feater.scripts import fix_residue


def traj_to_pdb(resname, top, crd):
  with tempfile.TemporaryDirectory() as tempdir:
    tempf_pdb = os.path.join(tempdir, "tmp.pdb")
    tempf_key = os.path.join(tempdir, "tmp.key")
    tempf_seq = os.path.join(tempdir, "tmp.seq")
    tmp_traj = pt.Trajectory(top=top, xyz=np.array([crd]))
    tmp_traj.save(tempf_pdb, overwrite=True)

    with open(tempf_seq, "w") as tempf:
      if resname == "HIS":
        r1name = "HID_N_C"
      elif resname in ["HID", "HIE", "HIP"]:
        r1name = resname[:-1] + "_N_C"
      else:
        r1name = resname
      tempf.write(f"{r1name}_N_C\n")
      tempf.write("END\n")
    output_pdb = fix_residue.fix_residue(tempdir, seq_file=tempf_seq, pdb_file=tempf_pdb, key_file=tempf_key)
  return output_pdb.get("result_pdb", "")
  
def traj_to_surf(resname, top, crd):
  thepdb = traj_to_pdb(resname, top, crd)

  return ([], [])


# Needed settings:
# - output_file: str
# - stride: int
# Process all of the single reisdues in the trajectory file and save them to the output_file
def coord_from_traj(complex_file, topology_file, settings: dict):
  stride = settings.get("stride", 1)
  output_file = settings["output_file"]
  traj = pt.load(complex_file, top=topology_file, stride=stride)
  residues = [res for res in traj.top.residues]
  pool = mp.Pool(4)
  if os.path.exists(output_file):
    os.remove(output_file)

  for res_idx, res1 in enumerate(residues):
    resname = res1.name
    retop = traj.top[res1.first_atom_index:res1.last_atom_index]
    if resname not in (constants.RES + ["HID", "HIE", "HIP", "CYX"]):
      continue
    print(f"There are {traj.n_frames} frames in the trajectory.")

    pdb_tasks = [(resname, retop, traj.xyz[frame_idx, res1.first_atom_index:res1.last_atom_index, :]) for frame_idx in range(traj.n_frames)]
    output_pdb = pool.starmap(traj_to_pdb, pdb_tasks)
    output_pdb = [pdb_str for pdb_str in output_pdb if len(pdb_str) > 0]
    print(f"Final valid pdb is {len(output_pdb)}")

    with io.hdffile(output_file, "a") as h5f:
      for pdb_str in output_pdb:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp_pdb:
          tmp_pdb.write(pdb_str.encode())
          tmp_pdb.flush()
          # Load the fixed residue and save it to the HDF file
          res1_processed = pt.load(tmp_pdb.name)
          if "coord_ends" in h5f.keys():
            global_index = h5f["coord_ends"][-1]
          else:
            global_index = 0
          
          nr_atoms_buffer = np.array([res1_processed.n_atoms], dtype=np.int32)
          coord_buffer = np.array(res1_processed.xyz[0], dtype=np.float32)
          label_buffer = np.array([constants.RES2LAB[resname]], dtype=np.int32)
          start_idxs_buffer = np.array([global_index], dtype=np.uint64)
          end_idxs_buffer = np.array([global_index + res1_processed.n_atoms], dtype=np.uint64)
          
          # No buffering for coding simplicity
          utils.add_data_to_hdf(h5f, "coordinates", coord_buffer, dtype=np.float32, maxshape=[None, 3], chunks=True,
                                compression="gzip", compression_opts=1)
          utils.add_data_to_hdf(h5f, "atom_number", nr_atoms_buffer, dtype=np.int32, maxshape=[None], chunks=True)
          utils.add_data_to_hdf(h5f, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=True)
          utils.add_data_to_hdf(h5f, "coord_starts", start_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True)
          utils.add_data_to_hdf(h5f, "coord_ends", end_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True)
          if "entry_number" not in h5f.keys():
            h5f.create_dataset('entry_number', data=np.array([len(label_buffer)], dtype=np.int32), dtype=np.int32, maxshape=[1])
          else:
            h5f["entry_number"][0] += len(label_buffer)
    print(f"The {res_idx+1:4d}/{len(residues):4d} residue is processed.")
  print(f"The trajectory file {complex_file} is processed.")

def voxel_from_traj(traj, ):
  pass

def parser(): 
  parser = argparse.ArgumentParser(description="Process the trajectory file and save the single residues to the HDF file.")
  parser.add_argument("-i", "--traj", type=str, help="The trajectory file.")
  parser.add_argument("-t", "--top", type=str, help="The topology file.")
  parser.add_argument("-o", "--output", type=str, help="The output HDF file.")
  parser.add_argument("-s", "--stride", type=int, default=1, help="The stride for the trajectory file.")
  args = parser.parse_args()
  return args


def console_interface():
  args = parser()
  traj = args.traj
  top = args.top

  outsettings = {
    "output_file": args.output,
    "stride": args.stride, 
  }

  coord_from_traj(traj, top, outsettings)



if __name__ == "__main__":
  console_interface()
  # traj = "/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc"
  # top = "/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"

  # traj = "/home/yzhang/Downloads/rqo149P4_Extraction/Extraction_traj.pdb"
  # top = "/home/yzhang/Downloads/rqo149P4_Extraction/Extraction_END.pdb"

  # outsettings = {
  #   "output_file": "test.h5",
  #   "stride": 5, 
  # }

  # coord_from_traj(traj, top, outsettings)



