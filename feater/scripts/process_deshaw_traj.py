import os, sys, tempfile, hashlib, time, argparse, json
import numpy as np
import pytraj as pt 
import multiprocessing as mp
from feater import constants, io, utils
from feater.scripts import fix_residue
import h5py



def to_pdb_single(resname, top, crd):
  with tempfile.TemporaryDirectory() as tempdir:
    tempf_pdb = os.path.join(tempdir, "tmp.pdb")
    tempf_key = os.path.join(tempdir, "tmp.key")
    tempf_seq = os.path.join(tempdir, "tmp.seq")
    tmp_traj = pt.Trajectory(top=top, xyz=np.array([crd]))
    tmp_traj.save(tempf_pdb, overwrite=True, options='chainid 1')
    r1name = get_seq_name(resname)
    with open(tempf_seq, "w") as tempf:
      tempf.write(f"{r1name}_N_C\n")
      tempf.write("END\n")
    output_pdb = fix_residue.fix_residue(tempdir, seq_file=tempf_seq, pdb_file=tempf_pdb, key_file=tempf_key)
  return output_pdb.get("result_pdb", "")

def to_pdb_dual(r1name, r2name, top, crd):
  with tempfile.TemporaryDirectory() as tempdir:
    tempf_pdb = os.path.join(tempdir, "tmp.pdb")
    tempf_key = os.path.join(tempdir, "tmp.key")
    tempf_seq = os.path.join(tempdir, "tmp.seq")
    tmp_traj = pt.Trajectory(top=top, xyz=np.array([crd]))
    tmp_traj.save(tempf_pdb, overwrite=True, options='chainid 1')
    r1name = get_seq_name(r1name)
    r2name = get_seq_name(r2name)
    with open(tempf_seq, "w") as tempf: 
      tempf.write(f"{r1name}_N\n")
      tempf.write(f"{r2name}_C\n")
      tempf.write("END\n")
    output_pdb = fix_residue.fix_residue(tempdir, seq_file=tempf_seq, pdb_file=tempf_pdb, key_file=tempf_key)
    return output_pdb.get("result_pdb", "")

def get_seq_name(inputname): 
  if inputname == "HIS":
    outname = "HID"
  elif inputname in ["HID", "HIE", "HIP"]:
    outname = inputname
  elif inputname in ["CYS", "CYX"]:
    outname = "CYS"
  else:
    outname = inputname
  return outname

def get_lab_name(inputname): 
  if inputname in ["HID", "HIE", "HIP"]:
    outname = "HIS"
  elif inputname in ["CYS", "CYX"]:
    outname = "CYS"
  else: 
    outname = inputname
  return outname



def coord_from_traj(trajfile:str, topfile:str, settings:dict): 
  stride = settings.get("stride", 10)
  if len(topfile) == 0: 
    traj = pt.load(trajfile, stride=stride)
  else:
    traj = pt.load(trajfile, top=topfile, stride=stride)
  residues = [i for i in traj.top.residues]
  print(f"There are {traj.n_frames} frames and {len(residues)} residues in the trajectory.")
  pool = mp.Pool(32)

  MODE = settings.get("mode", "dual")
  OUTPUT_NCFILE = settings.get("output", "output.h5")
  RES_CANDIDATES = constants.RES + ["HID", "HIE", "HIP", "CYX"]
  PB_THRESHOLD = 1.32*1.25
  for res_idx, res1 in enumerate(residues[:-1]):
    st = time.perf_counter()
    print(f"Processing the {res_idx+1:4d}/{len(residues):4d} residue")
    res1_slice = np.s_[res1.first_atom_index:res1.last_atom_index]
    top_res1 = traj.top[res1_slice]
    crd_res1_0 = traj.xyz[0][res1_slice]

    if MODE == "dual":
      res2 = residues[res_idx+1]
      res2_slice = np.s_[res2.first_atom_index:res2.last_atom_index]
      top_res2 = traj.top[res2_slice]
      crd_res2_0 = traj.xyz[0][res2_slice]

      atomc = top_res1.select("@C")    # C from residue 1 for peptide bond
      atomn = top_res2.select("@N")    # N from residue 2 for peptide bond
      if (res1.name not in RES_CANDIDATES): 
        print(f"Warning: Skipping {res1.name} {res1.index:<4d}, not found in the candidate list", file=sys.stderr)
        continue
      elif (res2.name not in RES_CANDIDATES): 
        print(f"Warning: Skipping {res2.name} {res2.index:<4d}, not found in the candidate list", file=sys.stderr)
        continue
      
      if len(atomc) == 0 or len(atomn) == 0:
        continue
      elif len(atomc) > 1 or len(atomn) > 1:
        print(f"Warning: Found multiple atoms as C {len(atomc)}, N{len(atomn)}", file=sys.stderr)
        continue 

      pepbond_len = np.linalg.norm(crd_res1_0[atomc[0]] - crd_res2_0[atomn[0]])
      if pepbond_len > PB_THRESHOLD: 
        print(f"Skipping {res1.name}{res1.index:<4d} - {res2.name}{res2.index:<4d}: Not connected between ")
        continue

      sub_top = traj.top[res1.first_atom_index:res2.last_atom_index]
      tasks = [(res1.name, res2.name, sub_top, traj.xyz[frame_idx, res1.first_atom_index:res2.last_atom_index, :]) for frame_idx in range(traj.n_frames)]
      output_pdb = pool.starmap(to_pdb_dual, tasks) 
      label_map = constants.RES2LAB_DUAL
      top_name = get_lab_name(res1.name)+get_lab_name(res2.name)
    elif MODE == "single": 
      if (res1.name not in RES_CANDIDATES): 
        print(f"Warning: Skipping {res1.name} {res1.index:<4d}, not found in the candidate list", file=sys.stderr)
        continue
      tasks = [(res1.name, top_res1, traj.xyz[frame_idx, res1.first_atom_index:res1.last_atom_index, :]) for frame_idx in range(traj.n_frames)]
      output_pdb = pool.starmap(to_pdb_single, tasks)
      label_map = constants.RES2LAB
      top_name = get_lab_name(res1.name)
    
    output_pdb = [pdb_str for pdb_str in output_pdb if len(pdb_str) > 0]
    print(f"Found {len(output_pdb)} valid pdb entries for residue {res1.name}{res1.index:<4d}")
    trajs = []
    for pdb_str in output_pdb:
      with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp_pdb:
        tmp_pdb.write(pdb_str.encode())
        tmp_pdb.flush()
        # Load the fixed residue and save it to the HDF file
        res1_processed = pt.load(tmp_pdb.name)
        trajs.append(res1_processed)

        with io.hdffile(OUTPUT_NCFILE, "a") as f:
          if top_name not in f.keys():
            print(f"Dumping the topology file: {tmp_pdb.name} to the HDF file")
            f.dump_top(tmp_pdb.name, top_name)
    
    nr_atoms_buffer = np.array([t_.n_atoms for t_ in trajs], dtype=np.int32)
    nr_atoms_cum = [0] + np.cumsum(nr_atoms_buffer).tolist()
    coord_buffer = np.zeros((np.sum(nr_atoms_buffer),3), dtype=np.float32)
    for t_idx, t_ in enumerate(trajs): 
      coord_buffer[nr_atoms_cum[t_idx]:nr_atoms_cum[t_idx+1]] = t_.xyz[0]
    
    
    label_buffer = np.array([label_map[top_name] for i in range(len(trajs))], dtype=np.int32)
    atom_mass = [np.round(t_.top.mass) for t_ in trajs]
    elems_buffer = np.array(atom_mass, dtype=np.int32).reshape(-1)

    key_buffer = np.array([top_name for _ in range(len(label_buffer))])                # TODO NOTE: change this based on the residue type
    key_buffer = np.array(key_buffer, dtype=h5py.string_dtype())
    
    with io.hdffile(OUTPUT_NCFILE, "r") as h5f: 
      if "coord_ends" in h5f.keys():
        global_index = h5f["coord_ends"][-1]
      else:
        global_index = 0

    end_idxs_buffer = np.cumsum(nr_atoms_buffer, dtype=np.uint64) + global_index 
    start_idxs_buffer = end_idxs_buffer - np.array(nr_atoms_buffer, dtype=np.uint64)
      
    with io.hdffile(OUTPUT_NCFILE, "a") as f:
      utils.add_data_to_hdf(f, "coordinates", coord_buffer, dtype=np.float32, maxshape=[None, 3], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "elements", elems_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "atom_number", nr_atoms_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "coord_starts", start_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "coord_ends", end_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "topology_key", key_buffer, dtype=h5py.string_dtype(), maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      if "entry_number" not in f.keys(): 
        f.create_dataset('entry_number', data= np.array([len(label_buffer)], dtype=np.int32), dtype=np.int32, maxshape = [1])
      else:
        f["entry_number"][0] += len(label_buffer)
    print(f"The {res_idx+1:4d}/{len(residues):4d} residue took {time.perf_counter()-st:6.2f} seconds to process.")
    # exit(0)
  print(f"The trajectory file {trajfile} is processed.")    
  print(f"Finished ")



def parser(): 
  parser = argparse.ArgumentParser(description="Process the trajectory file and save the single residues to the HDF file.")
  parser.add_argument("-i", "--traj", type=str, default="", help="The trajectory file.")
  parser.add_argument("-t", "--top", type=str, default="", help="The topology file.")
  parser.add_argument("-o", "--output", type=str, help="The output HDF file.")
  parser.add_argument("-s", "--stride", type=int, default=1, help="The stride for the trajectory file.")
  parser.add_argument("-m", "--mode", type=str, default="single", help="The mode for processing the trajectory file.")
  parser.add_argument("--prefix", type=str, default="", help="The prefix for the output HDF file.")
  args = parser.parse_args()
  return args

def console_interface(): 
  args = parser()
  traj = args.traj
  top = args.top

  print(f"Trajectory file: {traj}")
  print(f"Topology file: {top}")
  outsettings = {
    "mode": args.mode,
    "output": os.path.join(args.output, f"{os.path.basename(traj).split('.')[0]}_{args.mode}_s{args.stride}.h5"),
    "stride": args.stride, 
  }
  print(json.dumps(outsettings, indent=2))

  coord_from_traj(traj, top, outsettings)



if __name__ == "__main__":
  console_interface()
  
  # data_folder = "/storage006/yzhang/tests/DESRES-Trajectory_sarscov2-15235455-peptide-B-no-water-no-ion/sarscov2-15235455-peptide-B-no-water-no-ion"
  # topfile = "s20_out.pdb"

  # setting = {
  #   "stride": 1,
  #   "output": "test.h5",
  # }
  # coord_from_traj(os.path.join(data_folder, topfile), "", setting)
  
  

