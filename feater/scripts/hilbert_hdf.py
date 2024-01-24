import time, os, sys, argparse, json

import numpy as np
import multiprocessing as mp
from hilbertcurve.hilbertcurve import HilbertCurve

from feater import io, utils


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
    voxel = f["voxel"][idx]
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

  # entry_nr = 5000             # TODO: Remove this for production
  NR_PROCESS = int(interp_settings.get("processes", 8))
  BATCH_SIZE = 1000
  BIN_NR = (entry_nr + BATCH_SIZE - 1) // BATCH_SIZE

  # Target 3D and 2D Hilbert coordinates
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
    # results = [transform_entry(inputhdf, _idx, coord_3d, coord_2d, interp_settings) for _idx in batch]   # TODO: Remove this for production
    results = pool.starmap(transform_entry, [(inputhdf, _idx, coord_3d, coord_2d, interp_settings) for _idx in batch])
    image_buffer = np.array(results, dtype=np.float32)
    
    # Read the label from the input hdf file
    with io.hdffile(inputhdf, "r") as f: 
      label_buffer = [f["label"][i] for i in batch]
      label_buffer = np.array(label_buffer, dtype=np.int32)
    
    # Write the batch to the output hdf file
    with io.hdffile(outputhdf, "a") as f:
      utils.add_data_to_hdf(f, "voxel", image_buffer, dtype=np.float32, chunks=True, maxshape=(None, len_2d, len_2d), compression="gzip", compression_opts=4)
      utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
    
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
  parser.add_argument("-i", "--input_file", type=str, help="The input file")
  parser.add_argument("-o", "--output_folder", type=str, help="The output file")
  parser.add_argument("-m", "--mode", type=str, default="max", help="The mode of the transformation")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes")
  args = parser.parse_args()

  if not os.path.exists(args.input_file):
    parser.print_help()
    raise ValueError(f"Input file {args.input_file} does not exist")
  return args


def console_interface():
  # Precompute the hilbert coordinates and splited array
  args = parser()
  SETTINGS = json.dumps(vars(args))
  print(SETTINGS)
  make_hdf(args.input_file, args.output_folder, vars(args))


if __name__ == "__main__":
  console_interface()



# python /MieT5/MyRepos/FEater/feater/scripts/hilbert_hdf.py  -i /diskssd/yzhang/FEater_data/FEater_Single_VOX/TestSet_Voxel.h5  -o testhilb.h5  --processes 12
