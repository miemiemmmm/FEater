import numpy as np
import pytraj as pt
import h5py as h5

class hdffile(h5.File):
  """
  This class is a wrapper of h5py.File which enables the "with" statement
  Wrapped up functions to read/write point data to HDF file
  """
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
  
  def dump_top(self, top, keyword):
    if isinstance(top, pt.Topology):
      thedict = top.to_dict()
    elif isinstance(top, str):
      thedict = pt.load_topology(top).to_dict()
    elif isinstance(top, dict):
      thedict = top
    else: 
      raise ValueError("Unknown type of topology")
    
    if keyword in self.keys():
      # remove the existing group
      print(f"Warning: Found previous group {keyword} in hdf file, removing it...")
      del self[keyword]
    
    group = self.create_group(keyword)
    for key,val in thedict.items(): 
      data = np.array(val)
      if "name" in key or "type" in key:
        data = data.astype(h5.string_dtype())
        group.create_dataset(key, data=data, dtype=h5.string_dtype(), shape=data.shape)
      elif "_index" in key or  key == "resid" or key == "mol_number": 
        data = data.astype(np.int64)
        group.create_dataset(key, data=data, dtype=np.int64, shape=data.shape)
      else:
        data = data.astype(np.float64)
        group.create_dataset(key, data=data, dtype=np.float64, shape=data.shape)

  def get_top(self, keyword): 
    if keyword not in self.keys(): 
      raise ValueError("Keyword {} not found in hdf file!".format(keyword))
    group = self[keyword]
    topo_dict = {}
    for key in group.keys(): 
      data = np.array(group[key])
      if "name" in key or "type" in key:
        data = data.astype(np.str_)
      elif "_index" in key or  key == "resid" or key == "mol_number": 
        data = data.astype(np.int64)
      else:
        data = data.astype(np.float64)
      topo_dict[key] = data
    top = pt.Topology()
    top = top.from_dict(topo_dict)
    return top


def mol_to_surf(mol, output_file):
  import siesta
  xyzr_arr = np.zeros((mol.GetNumAtoms(), 4), dtype=np.float32)
  conf = mol.GetConformer()
  for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    xyzr_arr[i, 0] = pos.x
    xyzr_arr[i, 1] = pos.y
    xyzr_arr[i, 2] = pos.z
    xyzr_arr[i, 3] = 1
  siesta.xyzr_to_file(xyzr_arr, output_file, format="ply", grid_size=0.2, slice_number=300, smooth_step=0)
  return xyzr_arr


def dump_pdb_to_hdf(pdbfile, h5file, keyword):
  top = pt.load_topology(pdbfile)
  dump_top_to_hdf(top, h5file, keyword)


def dump_top_to_hdf(top, h5file, keyword):
  thedict = top.to_dict()
  with hdffile(h5file, "a") as hdf: 
    if keyword in hdf.keys():
      # remove the existing group
      print(f"Warning: Found previous group {keyword} in hdf file, removing it...")
      del hdf[keyword]
    group = hdf.create_group(keyword)
    for key,val in thedict.items(): 
      data = np.array(val)
      if "name" in key or "type" in key:
        data = data.astype(h5.string_dtype())
        group.create_dataset(key, data=data, dtype=h5.string_dtype(), shape=data.shape)
      elif "_index" in key or  key == "resid" or key == "mol_number": 
        data = data.astype(np.int64)
        group.create_dataset(key, data=data, dtype=np.int64, shape=data.shape)
      else:
        data = data.astype(np.float64)
        group.create_dataset(key, data=data, dtype=np.float64, shape=data.shape)


def load_top_from_hdf(h5file, keyword):
  """
  All of them could be present as np.array
    string type: atom_name, atom_type, resname
    int type: bond_index, dihedral_index, mol_number, resid
    float type: atom_charge, atom_mass, box
  """
  with hdffile(h5file, "r") as hdf:
    if keyword not in hdf.keys(): 
      raise ValueError("Keyword {} not found in hdf file!".format(keyword))
    group = hdf[keyword]
    topo_dict = {}
    for key in group.keys(): 
      data = np.array(group[key])
      if "name" in key or "type" in key:
        data = data.astype(np.str_)
      elif "_index" in key or  key == "resid" or key == "mol_number": 
        data = data.astype(np.int64)
      else:
        data = data.astype(np.float64)
      topo_dict[key] = data
  top = pt.Topology()
  top = top.from_dict(topo_dict)
  return top




