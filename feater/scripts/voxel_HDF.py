import os, argparse, sys, time, copy

import numpy as np
from pytraj import load as ptload

from feater import RES2LAB, io, voxelize, utils


def mol_to_voxel(mol:str, kwargs):
  the_mol = ptload(mol)
  coord = the_mol.xyz[0]
  weights = np.ones(coord.shape[0])
  elems = np.array(the_mol.top.mass).round().astype(np.int32)

  if "dims" in kwargs.keys():
    dims = np.array(kwargs["dims"], dtype=int)
  else:
    dims = np.array([32, 32, 32], dtype=int)
  boxsize = float(kwargs.get("boxsize", 10.0))
  spacing = float(boxsize / dims[0])
  cutoff = float(kwargs.get("cutoff", 12.0))
  sigma = float(kwargs.get("sigma", 0.2))
  _ret_data = voxelize.interpolate(coord, weights, dims, spacing = spacing, cutoff = cutoff, sigma = sigma)
  ret_data = _ret_data.reshape(dims)
  return ret_data, np.array(coord, dtype=np.float32), elems


def make_hdf(hdf_name:str, coord_files:list, **kwargs):
  st = time.perf_counter()
  
  c = 0
  global_start_idx = 0
  shape_voxel = np.array([32,32,32], dtype=np.int32)
  voxel_settings = {
    "dims": np.array(shape_voxel, dtype=int),
    "boxsize": 16.0,
    "cutoff": 12.0,
    "sigma": 0.2
  }
  utils.add_data_to_hdf(f, "shape", shape_voxel)  # TODO: temporarily hard-coded
  utils.add_data_to_hdf(f, "scale_factor", np.array([voxel_settings["boxsize"] / shape_voxel[0]], dtype=np.float32))
  utils.add_data_to_hdf(f, "cutoff", np.array([voxel_settings["cutoff"]], dtype=np.float32))
  utils.add_data_to_hdf(f, "sigma", np.array([voxel_settings["sigma"]], dtype=np.float32))
  
  output_interval = 1000
  print(f"Processing {len(coord_files)} files")
  batches = np.array_split(coord_files, 10)
  pool = mp.Pool(processes=8)

  if kwargs["mode"] == "dual":
    RES_LAB_MAP = constants.RES2LAB_DUAL 
    KEY_LENGTH = 6
  elif kwargs["mode"] == "single":
    RES_LAB_MAP = constants.RES2LAB         # TODO: check this 
    KEY_LENGTH = 3

  # The main loop. 
  for idx, batch in enumerate(batches):
    results = pool.starmap(mol_to_voxel, [(f, voxel_settings) for f in batch])
    voxel_buffer = np.concatenate([r[0] for r in results], dtype=np.float32)
    coord_buffer = np.concatenate([r[1] for r in results], dtype=np.float32)
    elems_buffer = np.concatenate([r[2] for r in results], dtype=np.int32)

    atom_nums = np.array([r[1].shape[0] for r in results], dtype=np.int32)

    label_buffer = np.array([RES_LAB_MAP[os.path.basename(f)[:KEY_LENGTH]] for f in batch], dtype=np.int32)
    key_buffer = np.array([os.path.basename(f)[:KEY_LENGTH] for f in batch])                     # TODO NOTE: change this based on the residue type
    key_buffer = np.array(key_buffer, dtype=h5py.string_dtype())





  with io.hdffile(hdf_name, 'w') as f:
    voxel_buffer = np.zeros((output_interval*shape_voxel[0], shape_voxel[1], shape_voxel[2]), dtype=np.float32)
    label_buffer = np.full((output_interval), -1, dtype=np.int32)
    start_buffer = np.full((output_interval), -1, dtype=np.uint64)
    end_buffer = np.full((output_interval), -1, dtype=np.uint64)

    pointer_xyz = 0
    estimate_coord_size = 1000
    coord_buffer = np.zeros((estimate_coord_size*output_interval, 3), dtype=np.float32)
    elems_buffer = np.zeros((estimate_coord_size*output_interval), dtype=np.int32)

    f.draw_structure()
    for fidx, file in enumerate(coord_files):
      file = os.path.abspath(file)
      # Get the voxelized data
      voxeli, coordi, elems = mol_to_voxel(file, **voxel_settings)
      coord_moli = np.array(coordi, dtype=np.float32)
      start_i = global_start_idx
      end_i = global_start_idx + coord_moli.shape[0]
      global_start_idx += coord_moli.shape[0]

      if coord_moli.shape[0] > estimate_coord_size:
        estimate_coord_size = int(coord_moli.shape[0] * 1.25)

      # Get the label of the residue
      restype = os.path.basename(file)[:3]
      if restype not in RES2LAB.keys():
        raise ValueError(f"Unknown residue type: {restype}")
      labeli = RES2LAB[restype]

      current_idx = fidx % output_interval
      label_buffer[current_idx] = labeli
      voxel_buffer[current_idx * shape_voxel[0]:((current_idx+1) * shape_voxel[0])] = voxeli
      start_buffer[current_idx] = start_i
      end_buffer[current_idx] = end_i

      # Add the data to the buffer
      if pointer_xyz + coord_moli.shape[0] > coord_buffer.shape[0]:
        _coord_buffer = copy.deepcopy(coord_buffer)
        _elems_buffer = copy.deepcopy(elems_buffer)
        coord_buffer = np.zeros((estimate_coord_size * output_interval, 3), dtype=np.float32)
        elems_buffer = np.zeros((estimate_coord_size * output_interval), dtype=np.int32)
        coord_buffer[:pointer_xyz] = _coord_buffer[:pointer_xyz]
        elems_buffer[:pointer_xyz] = _elems_buffer[:pointer_xyz]
        _coord_buffer = None
        _elems_buffer = None
      coord_buffer[pointer_xyz:pointer_xyz+coord_moli.shape[0]] = coord_moli
      elems_buffer[pointer_xyz:pointer_xyz+coord_moli.shape[0]] = elems
      pointer_xyz += coord_moli.shape[0]

      # Add the data to the HDF5 file
      c += 1
      if (c % output_interval == 0) or (c == len(coord_files)):
        print(f"Processed {c} files, Time elapsed: {time.perf_counter() - st:.2f} seconds")

        # Mask the unfilled entries
        label_count = np.count_nonzero(label_buffer>=0)
        if label_count != len(label_buffer):
          voxel_buffer = voxel_buffer[:label_count*shape_voxel[0]]
          label_buffer = label_buffer[:label_count]
          start_buffer = start_buffer[:label_count]
          end_buffer = end_buffer[:label_count]
        coord_buffer = coord_buffer[:pointer_xyz]
        elems_buffer = elems_buffer[:pointer_xyz]

        # Put the data to the HDF5 file  # compress 4500 entries: Uncompress 565 MB; GZip Level 4: 87 MB; GZip Level 9: 86 MB; Level 0: 566 MB; Level 1: 90 MB; Level 2: 89MB
        utils.add_data_to_hdf(f, "voxel", voxel_buffer, dtype=np.float32, chunks=True, maxshape=(None, shape_voxel[1], shape_voxel[2]), compression="gzip", compression_opts=4)
        utils.add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
        utils.add_data_to_hdf(f, "coord", coord_buffer, dtype=np.float32, chunks=True, maxshape=[None, 3], compression="gzip", compression_opts=4)
        utils.add_data_to_hdf(f, "coord_starts", start_buffer, dtype=np.uint64, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
        utils.add_data_to_hdf(f, "coord_ends", end_buffer, dtype=np.uint64, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)
        utils.add_data_to_hdf(f, "element_mass", elems_buffer, dtype=np.int32, chunks=True, maxshape=[None], compression="gzip", compression_opts=4)

        # Reset the buffers and the time counter
        voxel_buffer = np.zeros((output_interval * shape_voxel[0], shape_voxel[1], shape_voxel[2]), dtype=np.float32)
        label_buffer = np.full((output_interval), -1, dtype=np.int32)
        start_buffer = np.full((output_interval), -1, dtype=np.uint64)
        end_buffer = np.full((output_interval), -1, dtype=np.uint64)
        coord_buffer = np.zeros((estimate_coord_size * output_interval, 3), dtype=np.float32)
        elems_buffer = np.zeros((estimate_coord_size * output_interval), dtype=np.int32)
        pointer_xyz = 0
        st = time.perf_counter()
    f.draw_structure()
  print(f"Finished making HDF5 file {hdf_name}")


def parse_args():
  parser = argparse.ArgumentParser(description="Make HDF5")
  parser.add_argument("-i", "--input", type=str, help="The input file containing a list of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
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
  files = utils.checkfiles(args.input)
  print(f"Found {len(files)} files in the list")
  make_hdf(args.output, files)


if __name__ == "__main__":
  console_interface()
  # filelists = "/media/yzhang/MieT72/Data/feater_database/ValidationSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/ValidationSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TestSet_VAL.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ALA.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ARG.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ASP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_CYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLN.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_GLY.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_HIS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_ILE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LEU.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_LYS.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_MET.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PHE.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_PRO.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_SER.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_THR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TRP.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_TYR.txt%/media/yzhang/MieT72/Data/feater_database/TrainingSet_VAL.txt"
  # listfiles = filelists.strip("%").split("%")
  # print(f"Processing {len(listfiles)} list files")
  #
  # basepath = "/media/yzhang/MieT72/Data/feater_database"
  # outputdir = "/media/yzhang/MieT72/Data/feater_database_voxel"
  # for listfile in listfiles[40:]:
  #   resname = os.path.basename(listfile).split(".")[0].split("_")[1]
  #   _filename = os.path.basename(listfile).split(".")[0]
  #   outfile = os.path.join(outputdir, f"{_filename}.h5")
  #   files = utils.checkfiles(listfile, basepath=basepath)
  #   print(f"Found {len(files)} files in the {listfile}")
  #   make_hdf(outfile, files)

  # Expected size of the hdffile:
  # 1. One entry (Stored in np.float32): 32*32*32*4/1024/1024 = 0.125 MB = 128 KB
  # 2. 1W entries: Actual size: 1.225 GB | Expected size: 1.221 GB
  # 3. Correction factor: 1.0032
  # 4. 880W entries in total = 1077.7 GB
  # 5. Hence, voxelize separately and not all at once
  # 6. 1W takes 45 seconds, 880W takes 11 hours
