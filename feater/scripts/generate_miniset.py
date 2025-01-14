"""
Generate a mini-set for the feater dataset 
"""

import json, os, time, argparse
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
        kwargs = {}
        if compression_level > 0:
          kwargs["compression"] = "gzip"
          kwargs["compression_opts"] = compression_level

        utils.add_data_to_hdf(hdf, "coordinates",   coord_buffer,       dtype=np.float32, maxshape=[None, 3], chunks=(32,3), **kwargs)
        utils.add_data_to_hdf(hdf, "elements",      elems_buffer,       dtype=np.int32,   maxshape=[None],    chunks=True,   **kwargs)
        utils.add_data_to_hdf(hdf, "atom_number",   nr_atoms_buffer,    dtype=np.int32,   maxshape=[None],    chunks=True,   **kwargs)
        utils.add_data_to_hdf(hdf, "label",         label_buffer,       dtype=np.int32,   maxshape=[None],    chunks=True,   **kwargs)
        utils.add_data_to_hdf(hdf, "coord_starts",  start_idxs_buffer,  dtype=np.uint64,  maxshape=[None],    chunks=True,   **kwargs)
        utils.add_data_to_hdf(hdf, "coord_ends",    end_idxs_buffer,    dtype=np.uint64,  maxshape=[None],    chunks=True,   **kwargs)
        utils.add_data_to_hdf(hdf, "topology_key",  key_buffer,         dtype=h5py.string_dtype(), maxshape=[None], chunks=True, **kwargs)
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
        kwargs = {}
        if compression_level > 0:
          kwargs["compression"] = "gzip"
          kwargs["compression_opts"] = compression_level

        utils.add_data_to_hdf(f, "xyzr",        xyzr_buffer,    dtype=np.float32, maxshape=[None, 4], chunks=(1000, 4), **kwargs)
        utils.add_data_to_hdf(f, "vertices",    vertex_buffer,  dtype=np.float32, maxshape=[None, 3], chunks=(1000, 3), **kwargs)
        utils.add_data_to_hdf(f, "faces",       face_buffer,    dtype=np.int32,   maxshape=[None, 3], chunks=(1000, 3), **kwargs)
        utils.add_data_to_hdf(f, "label",       label_buffer,   dtype=np.int32,   maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "xyzr_starts", xyzr_st_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "xyzr_ends",   xyzr_ed_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "vert_starts", vert_st_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "vert_ends",   vert_ed_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "face_starts", face_st_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
        utils.add_data_to_hdf(f, "face_ends",   face_ed_buffer, dtype=np.uint64,  maxshape=[None],    chunks=True, **kwargs)
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
        kwargs = {}
        if compression_level > 0:
          kwargs["compression"] = "gzip"
          kwargs["compression_opts"] = compression_level
        utils.add_data_to_hdf(hdf, "voxel", voxel_buffer, dtype=np.float32, chunks=(1, 32, 32, 32), maxshape=(None, 32, 32, 32), **kwargs)
        utils.add_data_to_hdf(hdf, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], **kwargs)
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
        kwargs = {}
        if compression_level > 0:
          kwargs["compression"] = "gzip"
          kwargs["compression_opts"] = compression_level
        utils.add_data_to_hdf(hdf, "voxel", voxel_buffer, dtype=np.float32, chunks=(1, 128, 128), maxshape=(None, 128, 128), **kwargs)
        utils.add_data_to_hdf(hdf, "label", label_buffer, dtype=np.int32,   chunks=True,          maxshape=[None],           **kwargs)
  print(f"Finished processing the dataset; Time elapsed: {time.perf_counter()-st:.2f} seconds")


def parse(): 
  parser = argparse.ArgumentParser(description="Generate a mini-set for the feater dataset")
  parser.add_argument("-d", "--dataset-type", type=str, required=True, help="The type of dataset to be generated, either dual or single")
  parser.add_argument("-s", "--subset-size", type=int, required=True, help="The size of the subset")
  parser.add_argument("-o", "--output-dir", type=str, required=True, help="The output directory")
  parser.add_argument("-c", "--compression-level", type=int, default=0, help="The compression level")
  parser.add_argument("--index-regeneration", type=bool, default=False, help="Whether to regenerate the indices")

  # Automatic perception of the source files
  parser.add_argument("--whichset", type=str, default="train", help="Try to find the source files in the FEATER_DATA directory")
  
  # Manually define the source files
  parser.add_argument("--coordfile", type=str, help="Manually define the coordinate file to be used")
  parser.add_argument("--surfacefile", type=str, help="Manually define the surface file to be used")
  parser.add_argument("--voxelfile", type=str, help="Manually define the voxel file to be used")
  parser.add_argument("--hilbertfile", type=str, help="Manually define the hilbert file to be used")
  
  return parser.parse_args()


def console_interface(): 
  args = parse()

  mark = args.dataset_type
  target_nr = args.subset_size
  outputdir = args.output_dir
  compression_level = args.compression_level
  whichset = args.whichset
  index_regeneration = args.index_regeneration

  #############################################################################
  ################# Finished the definition of needed parms ###################
  #############################################################################
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  if mark == "dual":
    label_nr = 400
  elif mark == "single":
    label_nr = 20
  else: 
    raise ValueError("Dataset type must be either dual or single")

  indicefile = os.path.join(outputdir, f"Miniset_Indices_{target_nr}_{mark}.json")
  OutputFiles = [
    # os.path.join(outputdir, f"Miniset_Coord_{target_nr}_{mark}.h5"),
    # os.path.join(outputdir, f"Miniset_Surf_{target_nr}_{mark}.h5"),
    # os.path.join(outputdir, f"Miniset_Vox_{target_nr}_{mark}.h5"),
    # os.path.join(outputdir, f"Miniset_Hilbert_{target_nr}_{mark}.h5")
    os.path.join(outputdir, f"Mini{target_nr}_{whichset}_coord.h5"),
    os.path.join(outputdir, f"Mini{target_nr}_{whichset}_surface.h5"),
    os.path.join(outputdir, f"Mini{target_nr}_{whichset}_voxel.h5"),
    os.path.join(outputdir, f"Mini{target_nr}_{whichset}_hilbert.h5")
  ]

  DATADIR = os.path.abspath(os.environ.get("FEATER_DATA", "/tmp/FEater_Data"))
  if whichset == "train":
    print(f"Using the TRAIN dataset ")
    DualSourceFiles = [
      f"{DATADIR}/FEater_Dual/TrainingSet_coord.h5", 
      f"{DATADIR}/FEater_Dual/TrainingSet_surface.h5",
      f"{DATADIR}/FEater_Dual/TrainingSet_voxel.h5",
      f"{DATADIR}/FEater_Dual/TrainingSet_hilbert.h5"
    ]
    SingleSourceFiles = [
      f"{DATADIR}/FEater_Single/TrainingSet_coord.h5", 
      f"{DATADIR}/FEater_Single/TrainingSet_surface.h5",
      f"{DATADIR}/FEater_Single/TrainingSet_voxel.h5",
      f"{DATADIR}/FEater_Single/TrainingSet_hilbert.h5"
    ]
  elif whichset == "test":
    print(f"Using the TEST dataset ")
    DualSourceFiles = [
      f"{DATADIR}/FEater_Dual/TestSet_coord.h5", 
      f"{DATADIR}/FEater_Dual/TestSet_surface.h5",
      f"{DATADIR}/FEater_Dual/TestSet_voxel.h5",
      f"{DATADIR}/FEater_Dual/TestSet_hilbert.h5"
    ]
    
    SingleSourceFiles = [
      f"{DATADIR}/FEater_Single/TestSet_coord.h5", 
      f"{DATADIR}/FEater_Single/TestSet_surface.h5",
      f"{DATADIR}/FEater_Single/TestSet_voxel.h5",
      f"{DATADIR}/FEater_Single/TestSet_hilbert.h5"
    ]
  elif whichset == "valid":
    print(f"Using the VALIDATION dataset ")
    DualSourceFiles = [
      f"{DATADIR}/FEater_Dual/ValidationSet_coord.h5",
      f"{DATADIR}/FEater_Dual/ValidationSet_surface.h5",
      f"{DATADIR}/FEater_Dual/ValidationSet_voxel.h5",
      f"{DATADIR}/FEater_Dual/ValidationSet_hilbert.h5",
    ]
    SingleSourceFiles = [
      f"{DATADIR}/FEater_Single/Validation_coord.h5",
      f"{DATADIR}/FEater_Single/Validation_surface.h5",
      f"{DATADIR}/FEater_Single/Validation_voxel.h5",
      f"{DATADIR}/FEater_Single/Validation_hilbert.h5",
    ]
  else: 
    DualSourceFiles = ["", "", "", ""]
    SingleSourceFiles = ["", "", "", ""]

  if label_nr == 400:
    SourceFiles = DualSourceFiles
  elif label_nr == 20:
    SourceFiles = SingleSourceFiles
  
  if args.coordfile is not None: 
    SourceFiles[0] = args.coordfile
  if args.surfacefile is not None:
    SourceFiles[1] = args.surfacefile
  if args.voxelfile is not None:
    SourceFiles[2] = args.voxelfile
  if args.hilbertfile is not None:
    SourceFiles[3] = args.hilbertfile

  if index_regeneration or (not os.path.exists(indicefile)):
    # Firstly make the indices
    seed = label_nr
    np.random.seed(seed)
    make_indices(SourceFiles[0], indicefile, label_nr, target_nr)
    print(f"Indices file {indicefile} has been created")
  else:
    if os.path.exists(indicefile):
      print(f"Using the existing indices file {indicefile}")
    else:
      raise FileNotFoundError(f"Indices file {indicefile} does not exist")

  for fidx, src_f in enumerate(SourceFiles):
    if len(src_f) == 0: 
      continue 
    elif not os.path.exists(src_f):
      print(f"Warning: File {src_f} does not exist; Skipping...")
      continue
    if fidx == 0:
      print(f"Processing the coord file {SourceFiles[0]}")
      generate_coord_miniset(SourceFiles[0], OutputFiles[0], indicefile, label_nr, force=True, compression_level=compression_level)
      with open(os.path.join(outputdir, f"{mark}_coord.txt"), "w") as f: 
        f.write(os.path.abspath(OutputFiles[0]) + "\n")
    elif fidx == 1:
      print(f"Processing the surface file {SourceFiles[1]}")
      generate_surf_miniset(SourceFiles[1], OutputFiles[1], indicefile, label_nr, force=True, compression_level=compression_level)
      with open(os.path.join(outputdir, f"{mark}_surf.txt"), "w") as f: 
        f.write(os.path.abspath(OutputFiles[1]) + "\n")
    elif fidx == 2:
      print(f"Processing the voxel file {SourceFiles[2]}")
      generate_vox_miniset(SourceFiles[2], OutputFiles[2], indicefile, label_nr, force=True, compression_level=compression_level)
      with open(os.path.join(outputdir, f"{mark}_vox.txt"), "w") as f: 
        f.write(os.path.abspath(OutputFiles[2]) + "\n")
    elif fidx == 3:
      print(f"Processing the hilbert file {SourceFiles[3]}")
      generate_hilbert_miniset(SourceFiles[3], OutputFiles[3], indicefile, label_nr, force=True, compression_level=compression_level)
      with open(os.path.join(outputdir, f"{mark}_hilbert.txt"), "w") as f: 
        f.write(os.path.abspath(OutputFiles[3]) + "\n")



if "__main__" == __name__: 
  console_interface()

  # mark = "dual"               # NOTE: 20 or 400
  # target_nr = 20
  # outputdir = "/Weiss/test_miniset/FEater_Mini200/FEater_Dual"
  # compression_level = 4
  # whichset = "train"     # NOTE: "train" or "test"; If other files are needed, please modify the code accordingly 
  # index_regeneration = False

