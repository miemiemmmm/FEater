###################################################################################################
################# This script converts a list of coordinate files to a HDF5 file ##################
###################################################################################################
import os, argparse, sys, time
import multiprocessing as mp
import h5py
import numpy as np
import pytraj as pt

from feater import RES2LAB, io, utils, constants


def read_coord(crdfile:str) -> np.ndarray:
  traj = pt.load(crdfile)
  coord = np.asarray(traj.xyz[0])
  elems = np.array(traj.top.mass).round().astype(np.int32)
  return coord, elems

def counter(arr): 
  uniqs = np.unique(arr, return_counts=True)
  for i in range(len(uniqs[0])):
    print(f"Rank {i:>6} --> {uniqs[0][i]:>6}: {uniqs[1][i]:6} entries")

def make_hdf(hdf_name:str, coord_files:list, kwargs):
  st = time.perf_counter()
  if kwargs.get("append", 0) == 1:
    write_flag = "a"
  else: 
    write_flag = "w"
  
  print(f"Processing {len(coord_files)} files")
  batches = np.array_split(coord_files, 10)
  pool = mp.Pool(processes=8)
  if kwargs["mode"] == "dual":
    RES_LAB_MAP = constants.RES2LAB_DUAL 
    KEY_LENGTH = 6
  elif kwargs["mode"] == "single":
    RES_LAB_MAP = constants.RES2LAB
    KEY_LENGTH = 3

  if os.path.exists(hdf_name):
    with io.hdffile(hdf_name, "r") as f:
      if "coord_ends" in f.keys():
        global_index = f["coord_ends"][-1]
  else:
    global_index = 0

  for idx, batch in enumerate(batches):
    print(f"Batch {idx+1}/{len(batches)}: {len(batch)} files")

    results = pool.starmap(read_coord, [(f,) for f in batch])
    coord_buffer = np.concatenate([r[0] for r in results], dtype=np.float32)
    elems_buffer = np.concatenate([r[1] for r in results], dtype=np.int32)

    nr_atoms_buffer = np.array([r[0].shape[0] for r in results], dtype=np.int32)
    label_buffer = np.array([RES_LAB_MAP[os.path.basename(f)[:KEY_LENGTH]] for f in batch], dtype=np.int32)
    key_buffer = np.array([os.path.basename(f)[:KEY_LENGTH] for f in batch])                # TODO NOTE: change this based on the residue type
    key_buffer = np.array(key_buffer, dtype=h5py.string_dtype())

    end_idxs_buffer = np.cumsum(nr_atoms_buffer, dtype=np.uint64) + global_index
    start_idxs_buffer = end_idxs_buffer - nr_atoms_buffer
    global_index = end_idxs_buffer[-1]

    if idx > 0: 
      # User only defines the first writing of the HDF file
      write_flag = "a"
    with io.hdffile(hdf_name, write_flag) as f:
      if key_buffer[0] not in f.keys():
        print(f"Dumping the topology file: {kwargs['topology']} to the HDF file")
        f.dump_top(kwargs["topology"], key_buffer[0])
      utils.add_data_to_hdf(f, "coordinates", coord_buffer, dtype=np.float32, maxshape=[None, 3], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "elements", elems_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "atom_number", nr_atoms_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "coord_starts", start_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "coord_ends", end_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "topology_key", key_buffer, dtype=h5py.string_dtype(), maxshape=[None], chunks=True, compression="gzip", compression_opts=4)
      if "entry_number" not in f.keys() or (write_flag == "w"): 
        f.create_dataset('entry_number', data= np.array([len(label_buffer)], dtype=np.int32), dtype=np.int32, maxshape = [1])
      else:
        f["entry_number"][0] += len(label_buffer)
  pool.close()
  pool.join()

  print("Doing final checking of the HDF file")
  with io.hdffile(hdf_name, "r") as f:
    f.draw_structure()
    final_entry_nr = f["entry_number"][0]
    print(f"Final entry number: {final_entry_nr}")
    print(f"Label Abundance: ")
    counter(f["label"][:final_entry_nr])
    print("Residue Atom number Abundance: ")
    counter(f["atom_number"][:final_entry_nr])


def parse_args():
  parser = argparse.ArgumentParser(description="Convert coordinate files to HDF5 format")
  parser.add_argument("-i", "--input", type=str, help="The input file containing a list of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
  parser.add_argument("-t", "--topology", type=str, default="", help="The topology file")
  parser.add_argument("-m", "--mode", type=str, default="dual", help="The mode of the topology name mappling, dual or single")
  parser.add_argument("--top_name", type=str, default="TOP", help="The topology file")
  parser.add_argument("--append", type=int, default=0, help="Append to the existing HDF5 file")
  args = parser.parse_args()

  if (args.input is None) or (not os.path.exists(args.input)):
    print("Fatal: Please specify the input file", file=sys.stderr)
    # print help
    parser.print_help()
    exit(1)

  if (args.output is None):
    print("Fatal: Please specify the output file", file=sys.stderr)
    parser.print_help()
    exit(1)

  if args.append == 1:
    print(f"Appending to the existing HDF file: {args.output}")
  elif (os.path.exists(args.output)) and (args.force == 0):
    print(f"Fatal: Output file '{args.output}' exists. Use -f to force overwrite", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and (args.force == 1):
    print(f"Removing the existing output file '{args.output}'")
    if os.path.exists(args.output):
      os.remove(args.output)
  return args


def console_interface():
  args = parse_args()
  files = utils.checkfiles(args.input) # [:888] #TODO: remove this
  print(f"Found {len(files)} files in the list")
  make_hdf(args.output, files, vars(args))


if __name__ == "__main__":
  console_interface()
  
  # hdf_output = "test.hdf5"
  # file_list = "/MieT5/tests/FEater/feater/scripts/tmpflist.txt"
  # files = checkfiles(file_list)
  # print(f"There are {len(files)} files in the list")
  # make_hdf(hdf_output, files)
  # print(f"Generation of HDF file is finished")


