import os, sys, argparse, time, json

import numpy as np
import multiprocessing as mp
from feater import io, utils

import siesta

mass_to_rad_map = {
  16: 1.52, 
  12: 1.70, 
  14: 1.55, 
  1:  1.20, 
  32: 1.80, 
  31: 1.80, 
  0:  1.40
}

# @profile
def coord_to_surf(hdffile, idx, settings):
  spacing = float(settings["grid_spacing"])
  smooth_step = int(settings["smooth_step"])
  slice_number = int(settings["slice_number"])

  with io.hdffile(hdffile, "r") as f:
    st_idx = f["coord_starts"][idx]
    end_idx = f["coord_ends"][idx]
    coord = f["coordinates"][st_idx:end_idx]
    coord = np.array(coord, dtype=np.float32)
    elems = f["elements"][st_idx:end_idx]
    rads = [mass_to_rad_map[ele] for ele in elems]
    thexyzr = np.zeros((coord.shape[0], 4), dtype=np.float32)
    thexyzr[:, :3] = coord
    thexyzr[:, 3] = rads
  
  verts, faces = siesta.xyzr_to_surf(thexyzr, grid_size=spacing, slice_number=slice_number, smooth_step=smooth_step)
  verts = np.array(verts, dtype=np.float32)
  faces = np.array(faces, dtype=np.int32)
  return verts, faces, thexyzr


# @profile
def make_hdf(inputhdf:str, outputhdf:str, interp_settings:dict):
  with io.hdffile(inputhdf, "r") as f: 
    entry_nr = f["label"].shape[0]
    print(f"Processing {entry_nr} entries in the input HDF file")
  BATCH_SIZE = 1000
  CHUNK_SIZE = 1000
  BIN_NR = (entry_nr + BATCH_SIZE - 1) // BATCH_SIZE
  NR_PROCESS = int(interp_settings.get("processes", 8))
  
  # Write the meta information 
  with io.hdffile(outputhdf, "a") as f: 
    if "grid_spacing" not in f.keys():
      utils.add_data_to_hdf(f, "grid_spacing", np.array([interp_settings["grid_spacing"]], dtype=np.float32), maxshape=[1])
    if "smooth_step" not in f.keys():
      utils.add_data_to_hdf(f, "smooth_step", np.array([interp_settings["smooth_step"]], dtype=np.int32), maxshape=[1])
    if "slice_number" not in f.keys():
      utils.add_data_to_hdf(f, "slice_number", np.array([interp_settings["slice_number"]], dtype=np.int32), maxshape=[1])

  batches = np.array_split(np.arange(entry_nr), BIN_NR)
  batches_lens = np.array([len(_b) for _b in batches])
  batches_cumsum = np.cumsum([0] + list(batches_lens))
  print(f"Splitted the job into {len(batches)} batches with ~{np.mean(batches_lens)} entries in each batch")

  pool = mp.Pool(processes=NR_PROCESS)
  for idx, batch in enumerate(batches):
    print(f"Processing batch {idx+1:4} / {len(batches):<6} containing {len(batch):6} entries")
    st_batch = time.perf_counter()
    tasks = [(inputhdf, i, interp_settings) for i in batch]
    # results = [coord_to_surf(*task) for task in tasks]
    results = pool.starmap(coord_to_surf, tasks)

    len_verts = np.cumsum([0] + [_r[0].shape[0] for _r in results])
    len_faces = np.cumsum([0] + [_r[1].shape[0] for _r in results])
    len_xyzr =  np.cumsum([0] + [_r[2].shape[0] for _r in results])

    vertex_buffer = np.zeros((len_verts[-1], 3), dtype=np.float32)
    face_buffer = np.zeros((len_faces[-1], 3), dtype=np.int32)
    xyzr_buffer = np.zeros((len_xyzr[-1], 4), dtype=np.float32)
    for res_i in range(len(results)):
      vertex_buffer[np.s_[len_verts[res_i]:len_verts[res_i+1]], :] = results[res_i][0]
      face_buffer[np.s_[len_faces[res_i]:len_faces[res_i+1]], :] = results[res_i][1]
      xyzr_buffer[np.s_[len_xyzr[res_i]:len_xyzr[res_i+1]], :] = results[res_i][2]

    with io.hdffile(inputhdf, "r") as f: 
      label_buffer = [f["label"][i] for i in batch]
      label_buffer = np.array(label_buffer, dtype=np.int32)
    
    if os.path.exists(outputhdf): 
      with io.hdffile(outputhdf, "r") as f: 
        if "vert_ends" in f.keys():
          idx_vert = f["vert_ends"][batches_cumsum[idx]-1]    # batch -> idx
          # idx_vert = f["vert_ends"][-1]
        else: 
          idx_vert = 0
        if "face_ends" in f.keys():
          idx_face = f["face_ends"][batches_cumsum[idx]-1]
          # idx_face = f["face_ends"][-1]
        else:
          idx_face = 0
        if "xyzr_ends" in f.keys():
          idx_xyzr = f["xyzr_ends"][batches_cumsum[idx]-1]
          # idx_xyzr = f["xyzr_ends"][-1]
        else: 
          idx_xyzr = 0
    else:
      idx_vert = 0
      idx_face = 0
      idx_xyzr = 0

    len_verts = np.array([_r[0].shape[0] for _r in results], dtype=np.uint64)
    len_faces = np.array([_r[1].shape[0] for _r in results], dtype=np.uint64)
    len_xyzr = np.array([_r[2].shape[0] for _r in results], dtype=np.uint64)

    vert_ed_buffer = np.cumsum([_r[0].shape[0] for _r in results], dtype=np.uint64) + idx_vert
    vert_st_buffer = vert_ed_buffer - len_verts
    face_ed_buffer = np.cumsum([_r[1].shape[0] for _r in results], dtype=np.uint64) + idx_face
    face_st_buffer = face_ed_buffer - len_faces
    xyzr_ed_buffer = np.cumsum([_r[2].shape[0] for _r in results], dtype=np.uint64) + idx_xyzr
    xyzr_st_buffer = xyzr_ed_buffer - len_xyzr
    
    # Prepare the slices for the HDF5 update 
    vert_slice = np.s_[idx_vert:idx_vert+np.uint64(len(vertex_buffer))]
    face_slice = np.s_[idx_face:idx_face+np.uint64(len(face_buffer))]
    xyzr_slice = np.s_[idx_xyzr:idx_xyzr+np.uint64(len(xyzr_buffer))]
    slice_labels = np.s_[batches_cumsum[idx]:batches_cumsum[idx]+len(label_buffer)]

    with io.hdffile(outputhdf, "a") as f:
      print(f"Dumping the batch {idx+1} / {len(batches)} into the output HDF file")
      extra_config = {}
      if "compress_level" in interp_settings.keys() and interp_settings["compress_level"] > 0:
        extra_config["compression"] = "gzip"
        extra_config["compression_opts"] = interp_settings["compress_level"]
      utils.update_hdf_by_slice(f, "xyzr", xyzr_buffer, xyzr_slice, dtype=np.float32, maxshape=[None, 4], chunks=(CHUNK_SIZE, 4), **extra_config)
      utils.update_hdf_by_slice(f, "vertices", vertex_buffer, vert_slice, dtype=np.float32, maxshape=[None, 3], chunks=(CHUNK_SIZE, 3), **extra_config)
      utils.update_hdf_by_slice(f, "faces", face_buffer, face_slice, dtype=np.int32, maxshape=[None, 3], chunks=(CHUNK_SIZE, 3), **extra_config)
      utils.update_hdf_by_slice(f, "label", label_buffer, slice_labels, dtype=np.int32, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "xyzr_starts", xyzr_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "xyzr_ends", xyzr_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "vert_starts", vert_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "vert_ends", vert_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "face_starts", face_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)
      utils.update_hdf_by_slice(f, "face_ends", face_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], chunks=(CHUNK_SIZE), **extra_config)

    print(f"Batch {idx+1:4d} / {len(batches):4d} ({len(batch):4d} entries) done in {(time.perf_counter() - st_batch)*1000:6.2f} us, Average speed: {(time.perf_counter() - st_batch)*1000 / len(batch):6.2f} us per entry")

  pool.close()
  pool.join()

def parse_args():
  parser = argparse.ArgumentParser(description="Generate surface HDF file from the coordinate file")
  parser.add_argument("-i", "--input", type=str, help="The absolute path of the input coordinate HDF files")
  parser.add_argument("-o", "--output", type=str, help="The absolute path of the output surface HDF file")
  parser.add_argument("-c", "--compress-level", type=int, default=0, help="The compression level for the HDF deposition. Default is 0 (no compression)")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
  parser.add_argument("-g", "--grid-spacing", type=float, default=0.35, help="The grid spacing for surface generation")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes for parallel processing")
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
  SurfConfig = {
    "grid_spacing": args.grid_spacing,
    "smooth_step": 1,
    "slice_number": 300,
    "processes": args.processes,
    "compress_level": args.compress_level,
  }

  make_hdf(args.input, args.output, SurfConfig)


if __name__ == "__main__":
  console_interface()

