import os, argparse, sys, time, copy

import numpy as np
import multiprocessing as mp

from feater import io, voxelize, utils


def to_voxel(hdffile, idx, settings): 
  dims = np.array(settings["dims"], dtype=int)
  boxsize = float(settings["boxsize"])
  spacing = float(boxsize / dims[0])
  cutoff = float(settings["cutoff"])
  sigma = float(settings["sigma"])

  with io.hdffile(hdffile, "r") as f:
    st_idx = f["coord_starts"][idx]
    end_idx = f["coord_ends"][idx]
    coord = f["coordinates"][st_idx:end_idx]
    coord = np.array(coord, dtype=np.float32)
    weights = np.ones(coord.shape[0], dtype=np.float32)     # Uniform weights as 1 
  ret_data = voxelize.interpolate(coord, weights, dims, spacing = spacing, cutoff = cutoff, sigma = sigma)
  ret_data = ret_data.reshape(dims)
  return ret_data


def make_hdf(inputhdf:str, outputhdf:str, interp_settings:dict):
  with io.hdffile(inputhdf, "r") as f: 
    entry_nr = f["label"].shape[0]
    print("Processing", entry_nr, "entries")

  NR_PROCESS = int(interp_settings.get("processes", 8))
  BATCH_SIZE = 500
  BIN_NR = (entry_nr + BATCH_SIZE - 1) // BATCH_SIZE

  # Input metadata
  with io.hdffile(outputhdf, "a") as f: 
    if "dimensions" not in f.keys():
      utils.add_data_to_hdf(f, "dimensions", interp_settings["dims"], dtype=np.int32, maxshape=[3])  
    if "cutoff" not in f.keys():
      utils.add_data_to_hdf(f, "cutoff", np.array([interp_settings["cutoff"]], dtype=np.float32), maxshape=[1])
    if "sigma" not in f.keys():
      utils.add_data_to_hdf(f, "sigma", np.array([interp_settings["sigma"]], dtype=np.float32), maxshape=[1])
    if "boxsize" not in f.keys():
      utils.add_data_to_hdf(f, "boxsize", np.array([interp_settings["boxsize"]], dtype=np.float32), maxshape=[1])

  batches = np.array_split(np.arange(entry_nr), BIN_NR)
  pool = mp.Pool(processes = NR_PROCESS)

  for idx, batch in enumerate(batches):
    st_batch = time.perf_counter()
    print(f"Processing batch {idx} / {len(batches)} containing {len(batch)} entries")
    tasks = [(inputhdf, _idx, interp_settings) for _idx in batch]
    # results = [to_voxel(*task) for task in tasks]
    results = pool.starmap(to_voxel, tasks)

    voxel_buffer = np.array(results, dtype=np.float32)
    results = []
    with io.hdffile(inputhdf, "r") as f: 
      label_buffer = [f["label"][i] for i in batch]
      label_buffer = np.array(label_buffer, dtype=np.int32)

    with io.hdffile(outputhdf, "a") as f:
      utils.add_data_to_hdf(f, "voxel", voxel_buffer, dtype=np.float32, chunks=True, maxshape=(None, 32, 32, 32), compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
    print(f"Batch {idx:4d} / {len(batches):4d} ({len(batch):4d} entries) done in {(time.perf_counter() - st_batch)*1000:6.2f} us, Average speed: {(time.perf_counter() - st_batch)*1000 / len(batch):6.2f} us per entry")
  
  pool.close()
  pool.join()
  print("Done")


def parse_args():
  parser = argparse.ArgumentParser(description="Make HDF5")
  parser.add_argument("-i", "--input", type=str, help="The input file containing a list of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes")
  args = parser.parse_args()

  if (args.input is None) or (not os.path.exists(args.input)):
    print("Fatal: Please specify the input file", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (args.output is None):
    print("Fatal: Please specify the output file", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and args.force:
    print(f"Fatal: Output file '{args.output}' exists. Use -f to force overwrite", file=sys.stderr)
    parser.print_help()
    exit(1)
  return args


def console_interface():
  args = parse_args()
  # NOTE: Temporary hard-coded voxelization settings
  voxel_settings = {
    "dims": np.array([32, 32, 32], dtype=int),     
    "boxsize": 16.0,
    "cutoff": 12.0,
    "sigma": 1.0,
    "processes": args.processes,
  }
  make_hdf(args.input, args.output, voxel_settings)

if __name__ == "__main__":
  console_interface()

