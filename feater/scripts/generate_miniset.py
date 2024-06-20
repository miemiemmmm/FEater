import json, os, time
import h5py
import numpy as np
import feater.io
import feater
from feater import dataloader, utils
import multiprocessing as mp

def get_surf(filename, index): 
  with feater.io.hdffile(filename, "r") as hdf:
    st_vert, end_vert = hdf["vert_starts"][index], hdf["vert_ends"][index]
    st_face, end_face = hdf["face_starts"][index], hdf["face_ends"][index]
    st_xyzr, end_xyzr = hdf["xyzr_starts"][index], hdf["xyzr_ends"][index]
    verts = np.array(hdf["vertices"][st_vert:end_vert], dtype=np.float32)
    faces = np.array(hdf["faces"][st_face:end_face], dtype=np.int32)
    xyzr = np.array(hdf["xyzr"][st_xyzr:end_xyzr], dtype=np.float32)
    label = np.array(hdf["label"][index], dtype=np.int32)
  return verts, faces, xyzr, label

def get_coord(filename, index): 
  with feater.io.hdffile(filename, "r") as hdf:
    st_coord, end_coord = hdf["coord_starts"][index], hdf["coord_ends"][index]
    coords = hdf["coordinates"][st_coord:end_coord]
    elements = hdf["elements"][st_coord:end_coord]
    label = hdf["label"][index]
  return coords, elements, label

def make_indices(sourcefile, indicefile, label_nr, target_nr):
  with feater.io.hdffile(sourcefile, "r") as hdf: 
    labels = np.array(hdf["label"])
    result_dict = {}
    for i in range(label_nr): 
      indices = np.where(labels == i)[0]
      # print(indices.shape, indices[:5])
      # in case the number of samples is less than target_nr
      if indices.shape[0] < target_nr:
        print(f"Warning: label {i} has less than {target_nr} samples")
        selected = np.array([i for i in indices], dtype=np.int32)
      else:
        selected = np.random.choice(indices, target_nr, replace=False)
      selected.sort()
      print(selected.shape, selected[:5])
      result_dict[i] = selected.tolist()

  with open(indicefile, "w") as f: 
    json.dump(result_dict, f, indent=2)

def generate_coord_miniset(sourcefile, minisetoutput, indicefile, label_nr, force=False, compression_level=4): 
  """

  """
  if os.path.exists(minisetoutput) and not force:
    raise ValueError(f"File {minisetoutput} already exists")
  elif os.path.exists(minisetoutput) and force:
    os.remove(minisetoutput)

  all_indices = []
  with open(indicefile, "r") as f:
    indices_dict = json.load(f)
    for i in range(label_nr): 
      all_indices += [i for i in indices_dict[str(i)]]

  with feater.io.hdffile(sourcefile, "r") as hdf:
    # copy the topology (group) to the new file
    with feater.io.hdffile(minisetoutput, "w") as new_hdf:
      if label_nr == 400: 
        for key in feater.constants.RES2LAB_DUAL.keys():
          topi = hdf.get_top(key)
          new_hdf.dump_top(topi, key)
      elif label_nr == 20:
        for key in feater.constants.RES2LAB.keys():
          topi = hdf.get_top(key)
          new_hdf.dump_top(topi, key)
  st = time.perf_counter()
  batches = np.array_split(all_indices, label_nr)  
  with mp.Pool(32) as pool:
    for idx, batch in enumerate(batches):
      print(f"Processing batch {idx}/{len(batches)} having {len(batch)} entries; Time remaining {(time.perf_counter()-st)/(idx+1)*(len(batches)-idx):.2f}")
      results = pool.starmap(get_coord, [(sourcefile, i) for i in batch])
      coord_buffer = np.concatenate([r[0] for r in results], dtype=np.float32)
      elems_buffer = np.concatenate([r[1] for r in results], dtype=np.int32)
      label_buffer = np.array([r[2] for r in results], dtype=np.int32)
      
      if os.path.exists(minisetoutput):
        with feater.io.hdffile(minisetoutput, "r") as f:
          if "coord_ends" in f.keys():
            global_index = f["coord_ends"][-1]
          else: 
            global_index = 0
      else:
        global_index = 0

      nr_atoms_buffer = np.array([r[0].shape[0] for r in results], dtype=np.uint64)
      end_idxs_buffer = np.cumsum(nr_atoms_buffer, dtype=np.uint64) + global_index
      start_idxs_buffer = end_idxs_buffer - nr_atoms_buffer

      with feater.io.hdffile(sourcefile, "r") as hdf:
        key_buffer = np.array([hdf["topology_key"][i] for i in batch], dtype=h5py.string_dtype())

      end_idxs_buffer = np.cumsum(nr_atoms_buffer, dtype=np.uint64) + global_index
      start_idxs_buffer = end_idxs_buffer - nr_atoms_buffer

      with feater.io.hdffile(minisetoutput, "a") as hdf:
        utils.add_data_to_hdf(hdf, "coordinates", coord_buffer, dtype=np.float32, maxshape=[None, 3], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "elements", elems_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "atom_number", nr_atoms_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "label", label_buffer, dtype=np.int32, maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "coord_starts", start_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "coord_ends", end_idxs_buffer, dtype=np.uint64, maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "topology_key", key_buffer, dtype=h5py.string_dtype(), maxshape=[None], chunks=True, compression="gzip", compression_opts=compression_level)
        if "entry_number" not in hdf.keys(): 
          print("creating the dataset")
          hdf.create_dataset('entry_number', data= np.array([len(label_buffer)], dtype=np.int32), dtype=np.int32, maxshape = [1])
        else:
          hdf["entry_number"][0] += len(label_buffer)
  print(f"Finished processing the dataset; Time elapsed: {time.perf_counter()-st:.2f} seconds")

def generate_surf_miniset(sourcefile, outputfile, indicefile, label_nr, force=False, compression_level=4): 
  """
  """
  if os.path.exists(outputfile) and not force:
    raise ValueError(f"File {outputfile} already exists")
  elif os.path.exists(outputfile) and force:
    os.remove(outputfile)
  
  all_indices = []
  with open(indicefile, "r") as f:
    indices_dict = json.load(f)
    for i in range(label_nr): 
      all_indices += [i for i in indices_dict[str(i)]]
  batches = np.array_split(all_indices, label_nr)

  st = time.perf_counter()
  with mp.Pool(32) as pool:
    for idx, batch in enumerate(batches):
      print(f"Processing batch {idx}/{len(batches)} having {len(batch)} entries; Time remaining {(time.perf_counter()-st)/(idx+1)*(len(batches)-idx):.2f}")
      results = pool.starmap(get_surf, [(sourcefile, i) for i in batch])
      vertex_buffer = np.concatenate([r[0] for r in results], dtype=np.float32)
      face_buffer = np.concatenate([r[1] for r in results], dtype=np.int32)
      xyzr_buffer = np.concatenate([r[2] for r in results], dtype=np.float32)
      label_buffer = np.array([r[3] for r in results], dtype=np.int32)

      if os.path.exists(outputfile):
        with feater.io.hdffile(outputfile, "r") as hdf:
          if "vert_ends" in hdf.keys():
            globidx_vert = hdf["vert_ends"][-1]
          else:
            globidx_vert = 0
          if "face_ends" in hdf.keys():
            globidx_face = hdf["face_ends"][-1]
          else:
            globidx_face = 0
          if "xyzr_ends" in hdf.keys():
            globidx_xyzr = hdf["xyzr_ends"][-1]
          else:
            globidx_xyzr = 0
      else:
        globidx_vert = 0
        globidx_face = 0
        globidx_xyzr = 0
      n_verts = np.array([r[0].shape[0] for r in results], dtype=np.uint64)
      n_faces = np.array([r[1].shape[0] for r in results], dtype=np.uint64)
      n_xyzr = np.array([r[2].shape[0] for r in results], dtype=np.uint64)
      
      vert_ed_buffer = np.cumsum(n_verts, dtype=np.uint64) + globidx_vert
      vert_st_buffer = vert_ed_buffer - n_verts
      face_ed_buffer = np.cumsum(n_faces, dtype=np.uint64) + globidx_face
      face_st_buffer = face_ed_buffer - n_faces
      xyzr_ed_buffer = np.cumsum(n_xyzr, dtype=np.uint64) + globidx_xyzr
      xyzr_st_buffer = xyzr_ed_buffer - n_xyzr

      with feater.io.hdffile(outputfile, "a") as f:
        utils.add_data_to_hdf(f, "xyzr", xyzr_buffer, dtype=np.float32, maxshape=[None, 4], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "vertices", vertex_buffer, dtype=np.float32, maxshape=[None, 3], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "faces", face_buffer, dtype=np.int32, maxshape=[None, 3], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "xyzr_starts", xyzr_st_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "xyzr_ends", xyzr_ed_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "vert_starts", vert_st_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "vert_ends", vert_ed_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "face_starts", face_st_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(f, "face_ends", face_ed_buffer, dtype=np.uint64, maxshape=[None], compression="gzip", compression_opts=compression_level)
  print(f"Finished processing the dataset; Time elapsed: {time.perf_counter()-st:.2f} seconds")

def generate_vox_miniset(sourcefile, outputfile, indicefile, label_nr, force=False, compression_level=4): 
  """
  """
  if os.path.exists(outputfile) and not force:
    raise ValueError(f"File {outputfile} already exists")
  elif os.path.exists(outputfile) and force:
    os.remove(outputfile)

  # Split job batches
  all_indices = []
  with open(indicefile, "r") as f:
    indices_dict = json.load(f)
    for i in range(label_nr): 
      all_indices += [i for i in indices_dict[str(i)]]

  # Put Meta information for the dataset
  with feater.io.hdffile(sourcefile, "r") as hdf_source:
    with feater.io.hdffile(outputfile, "w") as hdf:
      utils.add_data_to_hdf(hdf, "boxsize", hdf_source["boxsize"][:], dtype=np.float32, maxshape=[1])
      utils.add_data_to_hdf(hdf, "cutoff", hdf_source["cutoff"][:], dtype=np.float32, maxshape=[1])
      utils.add_data_to_hdf(hdf, "dimensions", hdf_source["dimensions"][:], dtype=np.int32, maxshape=[3])
      utils.add_data_to_hdf(hdf, "sigma", hdf_source["sigma"][:], dtype=np.float32, maxshape=[1])

  st = time.perf_counter()
  batches = np.array_split(all_indices, label_nr)
  with mp.Pool(32) as pool:
    dset = dataloader.VoxelDataset([sourcefile])
    for idx, batch in enumerate(batches):
      
      print(f"Processing batch {idx}/{len(batches)} having {len(batch)} entries; Time remaining {(time.perf_counter()-st)/(idx+1)*(len(batches)-idx):.2f}")
      tasks = [dset.mini_batch_task(i) for i in batch]
      results = pool.starmap(dataloader.readdata, tasks)
      
      # concatenate along the first axis
      voxel_buffer = np.concatenate(results, axis=0)
      with feater.io.hdffile(sourcefile, "r") as hdf:
        label_buffer = np.array([hdf["label"][i] for i in batch], dtype=np.int32)

      with feater.io.hdffile(outputfile, "a") as hdf:
        utils.add_data_to_hdf(hdf, "voxel", voxel_buffer, dtype=np.float32, chunks=True, maxshape=(None, 32, 32, 32), compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=compression_level)
  print(f"Finished processing the dataset; Time elapsed: {time.perf_counter()-st:.2f} seconds")

def generate_hilbert_miniset(sourcefile, outputfile, indicefile, label_nr, force=False, compression_level=4): 
  """
  """
  if os.path.exists(outputfile) and not force:
    raise ValueError(f"File {outputfile} already exists")
  elif os.path.exists(outputfile) and force:
    os.remove(outputfile)


  all_indices = []
  with open(indicefile, "r") as f:
    indices_dict = json.load(f)
    for i in range(label_nr): 
      all_indices += [i for i in indices_dict[str(i)]]
  batches = np.array_split(all_indices, label_nr)

  with feater.io.hdffile(outputfile, "w") as hdf:
    with feater.io.hdffile(sourcefile, "r") as hdf2:
      utils.add_data_to_hdf(hdf, "size", hdf2["size"][:], dtype=np.int32, maxshape=[2])

  st = time.perf_counter()
  with mp.Pool(32) as pool: 
    dset = dataloader.HilbertCurveDataset([sourcefile])
    for idx, batch in enumerate(batches):
      print(f"Processing batch {idx}/{len(batches)} having {len(batch)} entries; Time remaining {(time.perf_counter()-st)/(idx+1)*(len(batches)-idx):.2f}")
      tasks = [dset.mini_batch_task(i) for i in batch]
      results = pool.starmap(dataloader.readdata, tasks)
      
      voxel_buffer = np.asarray([r[0] for r in results], dtype=np.float32)
      with feater.io.hdffile(sourcefile, "r") as hdf:
        label_buffer = np.array([hdf["label"][i] for i in batch], dtype=np.int32)

      with feater.io.hdffile(outputfile, "a") as hdf:
        utils.add_data_to_hdf(hdf, "voxel", voxel_buffer, dtype=np.float32, chunks=True, maxshape=(None, 128, 128), compression="gzip", compression_opts=compression_level)
        utils.add_data_to_hdf(hdf, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=compression_level)
  print(f"Finished processing the dataset; Time elapsed: {time.perf_counter()-st:.2f} seconds")

if "__main__" == __name__: 
  label_nr = 400
  target_nr = 1000
  outputdir = "/Matter/feater_testset_1000/"
  compression_level = 0
  whichset = "test"

  #############################################################################
  ################# Finished the definition of needed parms ###################
  #############################################################################
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  if label_nr == 400:
    mark = "dual"
  elif label_nr == 20:
    mark = "single"
  else: 
    raise ValueError("label_nr must be either 400 or 20")

  indicefile = os.path.join(outputdir, f"Miniset_Indices_{target_nr}_{mark}.json")
  OutputFiles = [
    os.path.join(outputdir, f"Miniset_Coord_{target_nr}_{mark}.h5"),
    os.path.join(outputdir, f"Miniset_Surf_{target_nr}_{mark}.h5"),
    os.path.join(outputdir, f"Miniset_Vox_{target_nr}_{mark}.h5"),
    os.path.join(outputdir, f"Miniset_Hilbert_{target_nr}_{mark}.h5")
  ]

  if whichset == "train":
    print(f"Using the TRAIN dataset ")
    DualSourceFiles = [
      "/Weiss/FEater_Dual_PDBHDF/TrainingSet_Dataset.h5", 
      "/Weiss/FEater_Dual_SURF/TrainingSet_Surface.h5",
      "/Weiss/FEater_Dual_VOX/TrainingSet_Voxel.h5",
      "/Weiss/FEater_Dual_HILB/TrainingSet_Hilbert.h5"
    ]
    SingleSourceFiles = [
      "/Weiss/FEater_Single_PDBHDF/TrainingSet_Dataset.h5",
      "/Weiss/FEater_Single_SURF/TrainingSet_Surface.h5",
      "/Weiss/FEater_Single_VOX/TrainingSet_Voxel.h5",
      "/Weiss/FEater_Single_HILB/TrainingSet_Hilbert.h5",
    ]
  elif whichset == "test":
    print(f"Using the TEST dataset ")
    DualSourceFiles = [
    "/Weiss/FEater_Dual_PDBHDF/TestSet_Dataset.h5",
    "/Weiss/FEater_Dual_SURF/TestSet_Surface.h5",
    "/Weiss/FEater_Dual_VOX/TestSet_Voxel.h5",
    "/Weiss/FEater_Dual_HILB/TestSet_Hilbert.h5",
    ]
    
    SingleSourceFiles = [
      "/Weiss/FEater_Single_PDBHDF/TestSet_Dataset.h5",
      "/Weiss/FEater_Single_SURF/TestSet_Surface.h5",
      "/Weiss/FEater_Single_VOX/TestSet_Voxel.h5",
      "/Weiss/FEater_Single_HILB/TestSet_Hilbert.h5",
    ]
  else: 
    raise ValueError("whichset must be either train or test")

  if label_nr == 400:
    SourceFiles = DualSourceFiles
  elif label_nr == 20:
    SourceFiles = SingleSourceFiles

  # Firstly make the indices
  seed = label_nr
  np.random.seed(seed)
  make_indices(SourceFiles[0], indicefile, label_nr, target_nr)


  generate_coord_miniset(SourceFiles[0], OutputFiles[0], indicefile, label_nr, force=True, compression_level=compression_level)
  with open(os.path.join(outputdir, f"{mark}_coord.txt"), "w") as f: 
    f.write(os.path.abspath(OutputFiles[0]) + "\n")
  
  generate_surf_miniset(SourceFiles[1], OutputFiles[1], indicefile, label_nr, force=True, compression_level=compression_level)
  with open(os.path.join(outputdir, f"{mark}_surf.txt"), "w") as f: 
    f.write(os.path.abspath(OutputFiles[1]) + "\n")

  generate_vox_miniset(SourceFiles[2], OutputFiles[2], indicefile, label_nr, force=True, compression_level=compression_level)
  with open(os.path.join(outputdir, f"{mark}_vox.txt"), "w") as f: 
    f.write(os.path.abspath(OutputFiles[2]) + "\n")

  generate_hilbert_miniset(SourceFiles[3], OutputFiles[3], indicefile, label_nr, force=True, compression_level=compression_level)
  with open(os.path.join(outputdir, f"{mark}_hilbert.txt"), "w") as f: 
    f.write(os.path.abspath(OutputFiles[3]) + "\n")



  #############################################################################
  # Conclusion: 
  # Save 20% space but cause x4.7 increase in loading time for voxel representation
  # x3.2 increase in loading time for hilbert curve representation 
  #############################################################################
  # 50 samples per class with compression level 4
  # total 4.7G
  # -rw-r--r-- 1 yzhang users 254K Apr 16 14:20 indices_400.json
  # -rw-r--r-- 1 yzhang users  14M Apr 16 14:20 testpdb.h5
  # -rw-r--r-- 1 yzhang users 1.2G Apr 16 14:21 testsurf.h5
  # -rw-r--r-- 1 yzhang users 2.4G Apr 16 14:22 testvox.h5
  # -rw-r--r-- 1 yzhang users 1.2G Apr 16 14:22 testhilbert.h5
  #############################################################################
  # 50 samples per class without compression
  # total 5.9G
  # -rw-r--r-- 1 yzhang users 254K Apr 16 14:46 indices_400.json
  # -rw-r--r-- 1 yzhang users  18M Apr 16 14:46 testpdb.h5
  # -rw-r--r-- 1 yzhang users 2.2G Apr 16 14:46 testsurf.h5
  # -rw-r--r-- 1 yzhang users 2.5G Apr 16 14:46 testvox.h5
  # -rw-r--r-- 1 yzhang users 1.3G Apr 16 14:47 testhilbert.h5
  #############################################################################
  # No compression / 50 samples per class
  # Iteration of CoordDataset data: 1.76 seconds
  # SurfDataset: Average vertices per entry:  3098.46
  # Iteration of SurfDataset data: 59.16 seconds
  # Iteration of VoxelDataset data: 6.85 seconds
  # Iteration of HilbertCurveDataset data: 3.29 seconds
  #############################################################################
  # compression level 4 / 50 samples per class
  # Iteration of CoordDataset data: 1.89 seconds
  # Iteration of SurfDataset data: 59.89 seconds
  # Iteration of VoxelDataset data: 32.99 seconds
  # Iteration of HilbertCurveDataset data: 10.42 seconds
  #############################################################################
  # No compression / 200 samples per class 
  # Iteration of CoordDataset data: 7.07 seconds
  # Iteration of SurfDataset data: 234.27 seconds
  # Iteration of VoxelDataset data: 38.01 seconds
  # Iteration of HilbertCurveDataset data: 21.70 seconds
  #############################################################################
  # Iteration of CoordDataset data: 14.23 seconds
  # SurfDataset: Average vertices per entry:  3098.90
  # Iteration of SurfDataset data: 469.12 seconds
  # Iteration of VoxelDataset data: 123.14 seconds
  # Iteration of HilbertCurveDataset data: 67.88 seconds
  #############################################################################
  # No compression / 800 samples per class 
  # Iteration of CoordDataset data: 28.61 seconds
  # Iteration of SurfDataset data: 934.48 seconds
  # Iteration of VoxelDataset data: 439.55 seconds
  # Iteration of HilbertCurveDataset data: 137.56 seconds
  #############################################################################