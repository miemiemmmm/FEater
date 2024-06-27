import os, argparse, time

import numpy as np
import multiprocessing as mp
from feater import io, utils

import siesta


# std::map<char, float> radiusDic = {
#   {'O', 1.52f}, {'C', 1.70f}, {'N', 1.55f},
#   {'H', 1.20f}, {'S', 1.80f}, {'P', 1.80f},
#   {'X', 1.40f}
# };

mass_to_rad_map = {
  16: 1.52, 
  12: 1.70, 
  14: 1.55, 
  1: 1.20, 
  32: 1.80, 
  31: 1.80, 
  0: 1.40
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
    print("Processing", entry_nr, "entries")
  # entry_nr = 5000
  BATCH_SIZE = 500
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
  print(batches_cumsum, BIN_NR, "bins")
  pool = mp.Pool(processes=NR_PROCESS)
  
  for idx, batch in enumerate(batches):
    if idx < interp_settings["start_batch"]:
      print(f"Skiping batch {idx+1}/{len(batches)}: {len(batch)} files")
      continue

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
      print(f"Processing xyzr slice: {idx_xyzr} -> {idx_xyzr+np.uint64(len(xyzr_buffer))}")
      
      utils.update_hdf_by_slice(f, "xyzr", xyzr_buffer, xyzr_slice, dtype=np.float32, maxshape=[None, 4], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "vertices", vertex_buffer, vert_slice, dtype=np.float32, maxshape=[None, 3], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "faces", face_buffer, face_slice, dtype=np.int32, maxshape=[None, 3], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "label", label_buffer, slice_labels, dtype=np.int32, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "xyzr_starts", xyzr_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "xyzr_ends", xyzr_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "vert_starts", vert_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "vert_ends", vert_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "face_starts", face_st_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)
      utils.update_hdf_by_slice(f, "face_ends", face_ed_buffer, slice_labels, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=4)

    print(f"Batch {idx+1:4d} / {len(batches):4d} ({len(batch):4d} entries) done in {(time.perf_counter() - st_batch)*1000:6.2f} us, Average speed: {(time.perf_counter() - st_batch)*1000 / len(batch):6.2f} us per entry")

  pool.close()
  pool.join()

def parse_args():
  parser = argparse.ArgumentParser(description="Make HDF5")
  parser.add_argument("-i", "--input", type=str, help="The file writes all of the absolute path of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("--processes", type=int, default=8, help="The number of processes")
  parser.add_argument("--start_batch", type=int, default=0, help="The number of processes")
  args = parser.parse_args()
  return args


def console_interface():
  args = parse_args()
  print(vars(args))
  SurfConfig = {
    "grid_spacing": 0.35,
    "smooth_step": 1,
    "slice_number": 300,
    "processes": args.processes,
    "start_batch": args.start_batch,
  }

  make_hdf(args.input, args.output, SurfConfig)


if __name__ == "__main__":
  console_interface()
  
  # filelists = "/media/yzhang/MieT72/Data/feater_database/ValidationSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_VAL.txt"
  # listfiles = filelists.strip("%").split("%")
  # print(f"Processing {len(listfiles)} list files")
  #
  # basepath = "/media/yzhang/MieT72/Data/feater_database"
  # outputdir = "/media/yzhang/MieT72/Data/feater_database_surf"
  #
  # for listfile in listfiles[50:51]:
  #   resname = os.path.basename(listfile).split(".")[0].split("_")[1]
  #   _filename = os.path.basename(listfile).split(".")[0]
  #   outfile = os.path.join(outputdir, f"{_filename}.h5")
  #   files = utils.checkfiles(listfile, basepath=basepath)
  #   print(f"Found {len(files)} files in the {listfile}, will write to {outfile}")
  #   make_hdf(outfile, files)



