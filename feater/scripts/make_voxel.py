"""
Convert the coordinates to voxels
"""

import os, argparse, sys, time, json

import numpy as np
import multiprocessing as mp

from feater import io, voxelize, utils


def to_voxel(hdffile, idx, settings): 
  """
  Uniform weights for each coordinate
  """
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


def to_voxel_byelem(hdffile, idx, settings, only_elem = None):
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
    weights = f["elements"][st_idx:end_idx]
    weights = np.array(weights, dtype=np.int32)
    if only_elem is not None:
      weights = np.array(weights == only_elem, dtype=int)
  ret_data = voxelize.interpolate(coord, weights, dims, spacing = spacing, cutoff = cutoff, sigma = sigma)
  ret_data = ret_data.reshape(dims)
  return ret_data


def make_hdf(inputhdf:str, outputhdf:str, interp_settings:dict):
  with io.hdffile(inputhdf, "r") as f: 
    entry_nr = f["label"].shape[0]
    print("Processing", entry_nr, "entries")

  NR_PROCESS = int(interp_settings.get("processes", 8))
  BATCH_SIZE = 1000
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
    print(f"Processing batch {idx+1:4} / {len(batches):<6} containing {len(batch):6} entries")
    if interp_settings["type"] == "uniform":
      tasks = [(inputhdf, _idx, interp_settings) for _idx in batch]
      results = pool.starmap(to_voxel, tasks)
    elif interp_settings["type"] == "elem":
      tasks = [(inputhdf, _idx, interp_settings, interp_settings["only_element"]) for _idx in batch]
      results = pool.starmap(to_voxel_byelem, tasks)
    else: 
      raise ValueError(f"Unknown type {interp_settings['type']}")

    voxel_buffer = np.array(results, dtype=np.float32)
    results = []
    with io.hdffile(inputhdf, "r") as f: 
      label_buffer = [f["label"][i] for i in batch]
      label_buffer = np.array(label_buffer, dtype=np.int32)

    with io.hdffile(outputhdf, "a") as f:
      extra_config = {}
      if "compress_level" in interp_settings.keys() and interp_settings["compress_level"] > 0:
        extra_config["compression"] = "gzip"
        extra_config["compression_opts"] = interp_settings["compress_level"]
      utils.add_data_to_hdf(f, interp_settings["tag_name"], voxel_buffer, dtype=np.float32, maxshape=(None, 32, 32, 32), chunks=(1, 32, 32, 32), **extra_config)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=(1000), **extra_config)
    print(f"Batch {idx:4d} / {len(batches):4d} ({len(batch):4d} entries) done in {(time.perf_counter() - st_batch)*1000:6.2f} us, Average speed: {(time.perf_counter() - st_batch)*1000 / len(batch):6.2f} us per entry")
  
  pool.close()
  pool.join()
  print("Done")


def parse_args():
  parser = argparse.ArgumentParser(description="Generate voxelized coordinates from the input coordinate file")
  parser.add_argument("-i", "--input", type=str, required=True, help="The absolute path of the input coordinate HDF files")
  parser.add_argument("-o", "--output", type=str, required=True, help="The absolute path of the output voxel HDF file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file; Default: 0")
  parser.add_argument("-c", "--compress-level", type=int, default=0, help="The compression level of the output HDF file; Default: 0")
  parser.add_argument("-d", "--dim", type=int, default=32, help="The dimension of the voxel; Default: 32")
  parser.add_argument("-b", "--boxsize", type=float, default=16.0, help="The box size of the voxel; Default: 16.0")
  parser.add_argument("--sigma", type=float, default=1.0, help="The sigma of the Gaussian kernel; Default: 1.0")
  parser.add_argument("--cutoff", type=float, default=12.0, help="The cutoff of the Gaussian kernel; Default: 12.0")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes; Default: 8")
  parser.add_argument("--tag-name", type=str, default="voxel", help="The tag name for the voxel data; Default: 'voxel'")
  parser.add_argument("--feature-type", type=str, default="uniform", help="The type of weights (uniform or elem); Default: uniform")
  parser.add_argument("--only-element", type=int, default=None, help="Focus only on one element (number), if the type is 'elem'. Default: None (All elements are processed)")
  args = parser.parse_args()

  if (args.input is None) or (not os.path.exists(args.input)):
    print("Fatal: Please specify the input file", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (args.output is None):
    print("Fatal: Please specify the output file", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and (not args.force):
    print(f"Fatal: Output file '{args.output}' exists. Use -f to force overwrite", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and args.force:
    os.remove(args.output)
    print(f"Warning: Output file '{args.output}' exists. Overwriting...")
  return args


def console_interface():
  args = parse_args()
  print("Arguments: ")
  print(json.dumps(vars(args), indent=2))
  # NOTE: Temporary hard-coded voxelization settings
  voxel_settings = {
    "dims": np.array([args.dim, args.dim, args.dim], dtype=int), 
    "boxsize": args.boxsize,
    "cutoff": args.cutoff,
    "sigma": args.sigma,
    "compress_level": args.compress_level,
    "tag_name": args.tag_name,
    "processes": args.processes,
    "type": args.feature_type,
  }
  if args.feature_type == "elem":
    voxel_settings["only_element"] = args.only_element
  make_hdf(args.input, args.output, voxel_settings)

if __name__ == "__main__":
  console_interface()


