"""
Convert the voxel data to Hilbert curve representation
"""
import time, os, sys, argparse, json

import numpy as np
import multiprocessing as mp
from hilbertcurve.hilbertcurve import HilbertCurve

from feater import io, utils

INPUT_TAGNAME = "voxel"
OUTPUT_TAGNAME = "voxel"

def hilbert_map(iteration, dim):
  hbc = HilbertCurve(iteration, dim)
  ret = np.zeros((hbc.max_h + 1, dim), dtype=np.int32)
  for i in range(hbc.max_h + 1):
    ret[i] = hbc.point_from_distance(i)
  return ret

def transform_entry(hdffile, idx, map1, map2, transform_settings):
  if len(map1) % len(map2) != 0:
    raise ValueError(f"len(map1)={len(map1)} is not a multiple of len(map2)={len(map2)}")
  with io.hdffile(hdffile, "r") as f:
    if INPUT_TAGNAME not in f.keys():
      raise ValueError(f"Cannot find the input tag {INPUT_TAGNAME} in the input hdf file")
    voxel = f[INPUT_TAGNAME][idx]
    transform_1d = voxel[map1[:,0], map1[:,1], map1[:,2]]
    splited = transform_1d.reshape((len(map2), -1))
    if transform_settings["mode"] == "max": 
      converted = np.max(splited, axis=1)
    elif transform_settings["mode"] == "mean":
      converted = np.mean(splited, axis=1)
    transform_2d = np.zeros((int(np.sqrt(len(map2))), int(np.sqrt(len(map2)))), dtype=np.float32)
    transform_2d[map2[:,0], map2[:,1]] = converted
  return transform_2d


def make_hdf(inputhdf:str, outputhdf:str, interp_settings:dict):
  st = time.perf_counter()
  with io.hdffile(inputhdf, "r") as f:
    entry_nr = f["label"].shape[0]
    print(f"Processing {entry_nr} entries from {inputhdf} to {outputhdf}")
  
  NR_PROCESS = int(interp_settings.get("processes", 8))
  BATCH_SIZE = 1000
  BIN_NR = (entry_nr + BATCH_SIZE - 1) // BATCH_SIZE

  # Target 3D and 2D Hilbert coordinates 
  # NOTE: Hard coded dimensions
  coord_3d = hilbert_map(5, 3)
  coord_2d = hilbert_map(7, 2)
  len_2d = int(np.sqrt((len(coord_2d))))
  if os.path.exists(outputhdf):
    with io.hdffile(outputhdf, "w") as f:
      utils.add_data_to_hdf(f, "size", np.array([len_2d, len_2d], dtype=np.int32), dtype=np.int32, maxshape=[2])
  else:
    with io.hdffile(outputhdf, "a") as f:
      if "size" in f.keys():
        del f["size"]
      utils.add_data_to_hdf(f, "size", np.array([len_2d, len_2d], dtype=np.int32), dtype=np.int32, maxshape=[2])
  
  print(f"3D Hilbert curve has {len(coord_3d)} points, 2D Hilbert curve has {len(coord_2d)} points")

  # Make up the process pool
  batches = np.array_split(np.arange(entry_nr), BIN_NR)
  pool = mp.Pool(processes = NR_PROCESS)  
  st_batch = time.perf_counter()
  for idx, batch in enumerate(batches):
    results = pool.starmap(transform_entry, [(inputhdf, _idx, coord_3d, coord_2d, interp_settings) for _idx in batch])
    image_buffer = np.array(results, dtype=np.float32)
    
    # Read the label from the input hdf file
    with io.hdffile(inputhdf, "r") as f: 
      label_buffer = [f["label"][i] for i in batch]
      label_buffer = np.array(label_buffer, dtype=np.int32)
    
    # Write the batch to the output hdf file
    with io.hdffile(outputhdf, "a") as f:
      extra_config = {}
      if interp_settings.get("compress_level", 0) > 0:
        extra_config["compression"] = "gzip"
        extra_config["compression_opts"] = interp_settings["compress_level"]
        
      utils.add_data_to_hdf(f, OUTPUT_TAGNAME, image_buffer, dtype=np.float32, maxshape=(None, len_2d, len_2d), chunks=(1, len_2d, len_2d), **extra_config)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=True, **extra_config)
    
    # Compute the batch time consumption
    time_estimate = (time.perf_counter() - st_batch)
    st_batch = time.perf_counter()
    print(f"Batch {idx:4d} / {len(batches):4d} ({len(batch):4d} entries) done in {(time_estimate)*1000:6.2f} ms, Average speed: {time_estimate*1000 / len(batch):6.2f} ms per entry")
    print(f">>>> Estimated time left: {time_estimate * (len(batches) - idx):6.2f} s")
  pool.close()
  pool.join()
  print(f"Done, total time used {time.perf_counter()-st:6.2f} s")


def parser():
  parser = argparse.ArgumentParser(description="Transform the voxel to hilbert curve")
  parser.add_argument("-i", "--input", type=str, required=True, help="The input 3D voxel HDF file")
  parser.add_argument("-o", "--output", type=str, required=True, help="The output 2D hilbert curve HDF file") 
  parser.add_argument("-m", "--mode", type=str, default="max", help="The mode of pooling transformation (max, mean); Default: max")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file; Default: 0")
  parser.add_argument("-c", "--compress-level", type=int, default=0, help="The compression level of the output HDF file; Default: 0")
  parser.add_argument("--input-tagname", type=str, default="voxel", help="The input tag name; Default: voxel")
  parser.add_argument("--output-tagname", type=str, default="voxel", help="The output tag name; Default: voxel")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes; Default: 8")
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
    print(f"Fatal: Output file '{args.output}' exists. Use -f to force overwrite the output file. ", file=sys.stderr)
    parser.print_help()
    exit(1)
  elif (os.path.exists(args.output)) and args.force:
    os.remove(args.output)
    print(f"Warning: Output file '{args.output}' exists and the force overwrite flag is specified {args.force}. Overwriting...")
  return args


def console_interface():
  # Precompute the hilbert coordinates and splited array
  args = parser()
  print(json.dumps(vars(args), indent=2))
  global INPUT_TAGNAME, OUTPUT_TAGNAME
  INPUT_TAGNAME = args.input_tagname
  OUTPUT_TAGNAME = args.output_tagname
  make_hdf(args.input, args.output, vars(args))


if __name__ == "__main__":
  console_interface()
