import os, argparse, sys, time

import h5py as h5
import numpy as np
import pytraj as pt
from feater import RES2LAB


class hdffile(h5.File):
  def __init__(self, filename, reading_flag):
    # print(f"Opening HDF5 file '{filename}' in mode '{reading_flag}'")
    super().__init__(filename, reading_flag)

  # __enter__ and __exit__ are used to enable the "with" statement
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def data(self, key):
    dset = self[key]
    return np.asarray(dset)

  def dtype(self,key):
    dset = self[key]
    return dset.dtype

  def draw_structure(self, depth=0):
    """
    Draw the structure of the HDF file. By default, only display the first level of the file (depth=0).
    Args:
      depth (int): The depth of the file structure to display.
    """
    print("############## HDF File Structure ##############")
    def print_structure(name, obj):
      current_depth = name.count('/')
      if current_depth > depth:  # If the current item's depth exceeds the limit, return early
        return
      if isinstance(obj, h5.Group):
        print(f"$ /{name:20s}/Heterogeneous Group")
      else:
        print(f"$ /{name:20s}: Shape-{obj.shape}")

    self.visititems(print_structure)
    print("############ END HDF File Structure ############")

  def append_entries(self, dataset_name:str, data:np.ndarray):
    """
    Append entries to an existing dataset.
    Args:
      dataset_name (str): The name of the dataset to append to.
      data (np.ndarray): The data to append to the dataset.
    NOTE:
      When creating the dataset, make sure to set maxshape to [None, 3]
    """
    dset = self[dataset_name]
    current_shape = dset.shape
    # Calculate the new shape after appending the new data
    new_shape = (current_shape[0] + data.shape[0], *current_shape[1:])
    # Resize the dataset to accommodate the new data
    dset.resize(new_shape)
    # Append the new data to the dataset
    dset[current_shape[0]:new_shape[0]] = data


def checkfiles(file_list:str) -> list:
  with open(file_list, 'r') as f:
    files = f.read().strip("\n").split('\n')
    for file in files:
      if not os.path.isfile(file):
        raise ValueError(f"File {file} does not exist.")
  return files


def read_coord(file:str) -> np.ndarray:
  traj = pt.load(file)
  coord = traj.xyz[-1]
  return coord


def add_data_to_hdf(hdffile, dataset_name:str, data:np.ndarray, **kwargs):
  if dataset_name not in hdffile.keys():
    hdffile.create_dataset(dataset_name, data=data, **kwargs)
  else:
    hdffile.append_entries(dataset_name, data)


def make_hdf(hdf_name:str, coord_files:list, **kwargs):
  st = time.perf_counter()
  coord_buffer = np.array([])
  label_buffer = np.array([])
  nr_atoms_buffer = np.array([])
  with hdffile(hdf_name, 'w') as f:
    c = 0
    for file in coord_files:
      file = os.path.abspath(file)
      coordi = read_coord(file)
      coord_f32 = np.array(coordi, dtype=np.float32)
      nr_atoms = coord_f32.shape[0]
      restype = os.path.basename(file)[:3]
      if restype not in RES2LAB.keys():
        raise ValueError(f"Unknown residue type: {restype}")
      labeli = RES2LAB[restype]

      if len(coord_buffer) > 0:
        coord_buffer = np.concatenate((coord_buffer, coord_f32), axis=0)
        label_buffer = np.concatenate((label_buffer, np.array([labeli], dtype=np.int32)), axis=0)
        nr_atoms_buffer = np.concatenate((nr_atoms_buffer, np.array([nr_atoms], dtype=np.int32)), axis=0)
      else:
        coord_buffer = np.array(coord_f32, dtype=np.float32)
        label_buffer = np.array([labeli], dtype=np.int32)
        nr_atoms_buffer = np.array([nr_atoms], dtype=np.int32)
      # print(len(label_buffer))
      c += 1

      if ((c) % 1000 == 0) or (c == len(coord_files)):
        print(f"Processing file {c:8d}/{len(coord_files):<8d}: This slice takes: {time.perf_counter()-st:6.3f} seconds")
        st = time.perf_counter()
        print(f"Writing {len(label_buffer)} entries to the HDF file: {hdf_name}")
        coord_buffer = np.array(coord_buffer, dtype=np.float32)
        label_buffer = np.array(label_buffer, dtype=np.int32)
        nr_atoms_buffer = np.array(nr_atoms_buffer, dtype=np.int32)

        add_data_to_hdf(f, "coordinates", coord_buffer, dtype=np.float32, maxshape=[None, 3], **kwargs)
        add_data_to_hdf(f, "nr_atoms", nr_atoms_buffer, dtype=np.int32, maxshape=[None], **kwargs)
        add_data_to_hdf(f, "label", label_buffer, dtype=np.int32, maxshape=[None], **kwargs)

        if  "entry_number" not in f.keys():
          f.create_dataset('entry_number', data= np.array([len(label_buffer)], dtype=np.int32), dtype = np.int32, maxshape = [1], **kwargs)
        else:
          f["entry_number"][0] += len(label_buffer)
        coord_buffer = np.array([])
        label_buffer = np.array([])
        nr_atoms_buffer = np.array([])
        # f.draw_structure()

    f.draw_structure()
    final_entry_nr = f["entry_number"][0]
    print(f"Final entry number: {final_entry_nr}")
    uniq_count = np.unique(f["label"][:final_entry_nr], return_counts=True)
    print(f"Residue Atom Number Abundance: ")
    for i in range(len(uniq_count[0])):
      print(f"Rank {i} --> {uniq_count[0][i]}: {uniq_count[1][i]} entries")
    if final_entry_nr != len(coord_files):
      print(f"Warning: The number of entries in the HDF file ({final_entry_nr}) does not match the number of input files ({len(coord_files)}).", file=sys.stderr)
    else:
      print(f"Successfully wrote {final_entry_nr} entries to the HDF file: {hdf_name}")



def parse_args():
  parser = argparse.ArgumentParser(description="Convert coordinate files to HDF5 format")
  parser.add_argument("-i", "--input", type=str, help="The input file containing a list of coordinate files")
  parser.add_argument("-o", "--output", type=str, help="The output HDF5 file")
  parser.add_argument("-f", "--force", type=int, default=0, help="Force overwrite the output file")
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

  if (os.path.exists(args.output)) and (args.force == 0):
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
  files = checkfiles(args.input)
  print(f"Found {len(files)} files in the list")
  make_hdf(args.output, files)


if __name__ == "__main__":
  console_interface()
  # hdf_output = "test.hdf5"
  # file_list = "/MieT5/tests/FEater/feater/scripts/tmpflist.txt"
  # files = checkfiles(file_list)
  # print(f"There are {len(files)} files in the list")
  # make_hdf(hdf_output, files)
  # print(f"Generation of HDF file is finished")


