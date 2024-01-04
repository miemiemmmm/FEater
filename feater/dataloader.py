import os, time
import multiprocessing as mp

import numpy as np
import h5py as h5
import torch
import torch.utils.data as data

from . import io


__all__ = [
  "HDF5Dataset",
  "CoordDataset",
  "VoxelDataset",
  "SurfDataset",
]


def readdata(input_file, key, start, end):
  with io.hdffile(input_file, "r") as h5file:
    ret_data = h5file[key][start:end]
  return ret_data


def readlabel(input_file, position):
  with io.hdffile(input_file, "r") as h5file:
    ret_data = h5file["label"][position]
  return ret_data


def padding(points, target_np):
  # Mask point at 0,0,0
  mask = np.all(points == 0, axis=1)
  points = points[~mask]
  if points.shape[0] < target_np:
    # Choose random points to fill with the result points
    points_copy = points.copy()
    choices = np.random.choice(target_np, points.shape[0], replace=False)
    points = np.zeros((target_np, 3), dtype=np.float32)
    points[choices] = points_copy
  else:
    # Randomly select target_np points
    choices = np.random.choice(points.shape[0], target_np, replace=False)
    points = points[choices]
  return points



def split_array(input_array, batch_size): 
  # For the N-1 batches, the size is uniformed and only variable for the last batch
  bin_nr = (len(input_array) + batch_size - 1) // batch_size 
  if len(input_array) == batch_size * bin_nr: 
    return np.array_split(input_array, bin_nr)
  else:
    final_batch_size = len(input_array) % batch_size
    if bin_nr-1 == 0: 
      return [input_array[-final_batch_size:]]
    else:
      return np.array_split(input_array[:-final_batch_size], bin_nr-1) + [input_array[-final_batch_size:]]


class CoordDataset(data.Dataset):
  """
  Supports constant time random access to the dataset
  """
  def __init__(self, hdffiles:list, target_np=25, padding=True):
    """
    Open the HDF files and generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files
    Args:
      hdffiles: The list of HDF5 files
      target_np: The target number of points for each residue
    """
    # Open the HDF5 files
    self.hdffiles = []
    self.total_entries = 0
    self.target_np = target_np
    self.do_padding = padding

    for file in hdffiles:
      h5file = h5.File(file, "r")
      self.hdffiles.append(h5file)
      self.total_entries += h5file["label"].shape[0]

    # Generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files
    # Number of files should be less than 256 (uint8)
    self.idx_to_file = np.zeros(self.total_entries, dtype=np.uint8)
    # Number of total entries should be less than 2^32 (uint32)
    self.idx_to_position = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_start = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_end = np.zeros(self.total_entries, dtype=np.uint32)
    memsize = self.idx_to_file.nbytes + self.idx_to_position.nbytes + self.idx_to_slice_start.nbytes + self.idx_to_slice_end.nbytes
    print(f"The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB or {memsize / 1024 / 1024 / 1024:6.2f} GB.")
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      entry_nr_i = file["entry_number"][0]
      starts_i = file["coord_starts"]
      ends_i = file["coord_ends"]
      self.idx_to_position[global_ind: global_ind + entry_nr_i] = np.arange(entry_nr_i)
      self.idx_to_file[global_ind: global_ind + entry_nr_i] = fidx
      self.idx_to_slice_start[global_ind: global_ind+entry_nr_i] = starts_i
      self.idx_to_slice_end[global_ind: global_ind+entry_nr_i] = ends_i
      global_ind += entry_nr_i

  def padding(self, points):
    """
    Pad or subset the points to the target number of points
    Args:
      points: The coordinates of the 3D points
    Returns:
      points: The padded points
    TODO: Padding operation might slow down 60% of the data loading time
    """
    lb = np.min(points, axis=0)
    points -= lb                                                                  # TODO: Justify the translation of the points
    if points.shape[0] < self.target_np:
      # Choose random points to fill with the result points in the case that the NP < TP
      # (Number of points VS Target number of points)
      choices = np.random.choice(self.target_np, points.shape[0], replace=False)
      order = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
      print_copy = points.copy()
      # choices.sort()     # TODO: Randomize the specific sequence of the PDB convention to avoid the bias
      # print(f"Padding Choices: {choices.tolist()}")
      points = np.zeros((self.target_np, 3), dtype=np.float32)                   # TODO: Justify the default values for the point coordinates

      points[choices] = print_copy[order, :]
      # for i, choice in enumerate(choices):
      #   _points[choice] = points[i]
      # points = _points
    elif points.shape[0] > self.target_np:
      # Randomly select points in the case that the NP > TP
      choices = np.random.choice(points.shape[0], self.target_np, replace=False)  # TODO: Justify the choice of random points
      # choices.sort()                 # TODO: Randomize the specific sequence of the PDB convention to avoid the bias
      points = points[choices]
    return points


  def __del__(self):
    """
    Close and clean the memory and close the opened HDF5 files
    """
    for h5file in self.hdffiles:
      h5file.close()
    self.idx_to_file.resize(0)
    self.idx_to_position.resize(0)
    self.idx_to_slice_start.resize(0)
    self.idx_to_slice_end.resize(0)

  def __getitem__(self, index):
    """
    Return the requested datapoint from the dataset based on an index
    Args:
      index: The index of the requested datapoint
    Returns:
      points: The coordinates of the atoms
      label: The label of the residue
    """
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    file_index = self.idx_to_file[index]
    slice_start = self.idx_to_slice_start[index]
    slice_end = self.idx_to_slice_end[index]
    label_position = self.idx_to_position[index]

    points = np.array(self.hdffiles[file_index]["coordinates"][slice_start: slice_end], dtype=np.float32)
    label = self.hdffiles[file_index]["label"][label_position]
    if self.do_padding:
      points = self.padding(points)
    points = torch.from_numpy(points)
    label = np.array(label, dtype=np.int64)
    return points, label

  def __len__(self):
    """
    Return the total number of entries in the dataset
    Returns:
      self.total_entries: The total number of entries in the dataset
    """
    return self.total_entries


class VoxelDataset(data.Dataset):
  """
  Supports constant time random access to the dataset
  """
  def __init__(self, hdffiles:list):
    self.hdffiles = []
    self.total_entries = 0
    # Get the shape of the voxels and the total number of entries
    for file in hdffiles:
      self.hdffiles.append(file)
      with io.hdffile(file, "r") as h5file:
        h5file.draw_structure()
        # h5file = h5.File(file, "r")
        self.total_entries += h5file["label"].shape[0]
        self.shape = np.array(h5file["shape"], dtype=np.uint32)

    # Initialize the map for getting the correct location of the data in the HDF5 files
    self.idx_to_file = np.zeros(self.total_entries, dtype=np.uint8)
    self.idx_to_position = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_start = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_end = np.zeros(self.total_entries, dtype=np.uint32)
    memsize = self.idx_to_file.nbytes + self.idx_to_position.nbytes + self.idx_to_slice_start.nbytes + self.idx_to_slice_end.nbytes
    print(f"The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB or {memsize / 1024 / 1024 / 1024:6.2f} GB.")

    # Generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        entry_nr_i = h5file["label"].shape[0]
        size_0 = h5file["shape"][0]
        starts = np.arange(entry_nr_i) * size_0
        ends = starts + size_0
        if ends[-1] != h5file["voxel"].shape[0]:
          raise ValueError(f"Unmatched array end indices: {ends[-1]} vs {h5file['voxel'].shape[0]}")
      self.idx_to_position[global_ind:global_ind+entry_nr_i] = np.arange(entry_nr_i)
      self.idx_to_file[global_ind: global_ind + entry_nr_i] = fidx
      self.idx_to_slice_start[global_ind:global_ind+entry_nr_i] = starts
      self.idx_to_slice_end[global_ind:global_ind+entry_nr_i] = ends
      global_ind += entry_nr_i

  def __del__(self):
    # Set the arrays to size 0
    self.idx_to_position.resize(0)
    self.idx_to_file.resize(0)
    self.idx_to_slice_start.resize(0)
    self.idx_to_slice_end.resize(0)

  def __getitem__(self, index):
    # Get the file
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    file_idx = self.idx_to_file[index]
    slice_start = self.idx_to_slice_start[index]
    slice_end = self.idx_to_slice_end[index]
    position = self.idx_to_position[index]

    voxel = readdata(self.hdffiles[file_idx], "voxel", slice_start, slice_end)
    label = readlabel(self.hdffiles[file_idx], position)
    voxel = np.asarray([voxel], dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    voxel = torch.from_numpy(voxel)
    return voxel, label
  
  def getitem_info(self, index):
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    file_idx = self.idx_to_file[index]
    slice_start = self.idx_to_slice_start[index]
    slice_end = self.idx_to_slice_end[index]
    position = self.idx_to_position[index]
    h5filename = self.hdffiles[file_idx]
    return (h5filename, "voxel", slice_start, slice_end)

  def __len__(self):
    return self.total_entries

  def retrieve(self, entry_list):
    # Manual data retrieval
    data = np.zeros((len(entry_list), 1, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
    label = np.zeros(len(entry_list), dtype=np.int64)
    for idx, entry in enumerate(entry_list):
      voxel, l = self.__getitem__(entry)
      data[idx, ...] = voxel
      label[idx] = l
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24): 
    pool = mp.Pool(process_nr)
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    batches = split_array(indices, batch_size)
    bytes_total = 0                 # TODO
    st = time.perf_counter()        # TODO
    for batch_idx, batch in enumerate(batches): 
      tasks = [self.getitem_info(i) for i in batch]
      ret_data = pool.starmap(readdata, tasks)
      data_numpy = np.asarray(ret_data, dtype=np.float32)
      data_numpy = data_numpy[:, np.newaxis, ...]

      # Benchmark the retrival rate
      for ret in ret_data:
        bytes_total += ret.nbytes               # TODO
      print(f"Batch {batch_idx+1:5d}/{len(batches)}: Retrival speed: {bytes_total / 1024 / 1024 / (time.perf_counter() - st):6.2f} MB/s. Time: {(time.perf_counter() - st)/(batch_idx+1):6.2f} seconds")     #

      tasks = [(self.hdffiles[self.idx_to_file[i]], self.idx_to_position[i]) for i in batch]
      labels = pool.starmap(readlabel, tasks)
      label_numpy = np.array(labels, dtype=np.int64)

      data = torch.from_numpy(data_numpy)
      label = torch.from_numpy(label_numpy)
      yield data, label



class SurfDataset(data.Dataset):
  def __init__(self, hdffiles: list, target_np=1024, pool_size=8):
    self.hdffiles = []
    self.total_entries = 0
    self.target_np = target_np

    with io.hdffile(hdffiles[0], "r") as h5file:
      h5file.draw_structure()

    # Prepare the map size for the dataset
    for file in hdffiles:
      self.hdffiles.append(file)
      with io.hdffile(file, "r") as h5file:
        self.total_entries += h5file["label"].shape[0]

    self.idx_to_file = np.zeros(self.total_entries, dtype=np.uint8)
    self.idx_to_position = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_face_start = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_face_end = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_vert_start = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_vert_end = np.zeros(self.total_entries, dtype=np.uint64)

    memsize = self.idx_to_file.nbytes + self.idx_to_position.nbytes + self.idx_to_face_start.nbytes + self.idx_to_face_end.nbytes + self.idx_to_vert_start.nbytes + self.idx_to_vert_end.nbytes
    print(f"SurfDataset: The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB")
    global_ind = 0
    for fidx, file in enumerate(hdffiles):
      with io.hdffile(file, "r") as h5file:
        entry_nr_i = h5file["label"].shape[0]
        self.idx_to_file[global_ind:global_ind + entry_nr_i] = fidx
        self.idx_to_position[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i)
        self.idx_to_vert_start[global_ind:global_ind + entry_nr_i] = h5file["vert_starts"]
        self.idx_to_vert_end[global_ind:global_ind + entry_nr_i] = h5file["vert_ends"]
        global_ind += entry_nr_i
    print(f"SurfDataset: Average vertices per entry: {(self.idx_to_vert_end - self.idx_to_vert_start).mean():.2f}")


  def __del__(self):
    # for file in self.hdffiles:
    #   file.close()
    # Set the arrays to size 0
    self.idx_to_position.resize(0)
    self.idx_to_file.resize(0)
    self.idx_to_face_start.resize(0)
    self.idx_to_face_end.resize(0)
    self.idx_to_vert_start.resize(0)
    self.idx_to_vert_end.resize(0)

  def __getitem__(self, index):
    # Get the file
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    file_idx = self.idx_to_file[index]
    vert_start = self.idx_to_vert_start[index]
    vert_end = self.idx_to_vert_end[index]
    position = self.idx_to_position[index]
    h5filename = self.hdffiles[file_idx]

    with io.hdffile(h5filename, "r") as h5file:
      verts = h5file["vertices"][vert_start:vert_end]       # Major bottleneck
      label = h5file["label"][position]
    verts = self.padding(verts)

    verts = np.asarray(verts, dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    verts = torch.from_numpy(verts)
    return verts, label

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24):
    pool = mp.Pool(process_nr)
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    batches = split_array(indices, batch_size)
    bytes_total = 0                 # TODO
    st = time.perf_counter()        # TODO
    for batch_idx, batch in enumerate(batches): 
      tasks = [self.getitem_info(i) for i in batch]
      ret_data = pool.starmap(readdata, tasks)

      # Benchmark the retrival rate
      for ret in ret_data: 
        bytes_total += ret.nbytes               # TODO
      print(f"Batch {batch_idx+1:5d}/{len(batches)}: Retrival speed: {bytes_total / 1024 / 1024 / (time.perf_counter() - st):6.2f} MB/s. Time: {(time.perf_counter() - st)/(batch_idx+1):6.2f} seconds")     # 
      
      tasks = [(ret, self.target_np) for ret in ret_data]
      ret_data = pool.starmap(padding, tasks)

      data_numpy = np.zeros((len(ret_data), self.target_np, 3), dtype=np.float32)
      for i, ret in enumerate(ret_data):
        data_numpy[i, ...] = ret

      tasks = [(self.hdffiles[self.idx_to_file[i]], self.idx_to_position[i]) for i in batch]
      labels = pool.starmap(readlabel, tasks)
      label_numpy = np.array(labels, dtype=np.int64)

      data = torch.from_numpy(data_numpy)
      label = torch.from_numpy(label_numpy)
      yield data, label

  def getitem_info(self, index):
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    file_idx = self.idx_to_file[index]
    vert_start = self.idx_to_vert_start[index]
    vert_end = self.idx_to_vert_end[index]
    h5filename = self.hdffiles[file_idx]

    ret_info = (h5filename, "vertices", vert_start, vert_end)
    return ret_info

  def getitem_label(self, index):
    position = self.idx_to_position[index]
    with io.hdffile(self.hdffiles[0], "r") as h5file:
      label = h5file["label"][position]
    return label

  def padding(self, points):
    # Mask point at 0,0,0
    mask = np.all(points == 0, axis=1)
    points = points[~mask]
    if points.shape[0] < self.target_np:
      # Choose random points to fill with the result points
      points_copy = points.copy()
      choices = np.random.choice(self.target_np, points.shape[0], replace=False)
      points = np.zeros((self.target_np, 3), dtype=np.float32)
      points[choices] = points_copy
    else:
      # Randomly select self.target_np points
      choices = np.random.choice(points.shape[0], self.target_np, replace=False)
      points = points[choices]
    return points


  def viewvert(self, index):
    import open3d as o3d
    points, _ = self.__getitem__(index)
    points = points[0].numpy()
    # count number of points at 0,0,0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

  def __len__(self):
    return self.total_entries


class BaseDataset(data.Dataset): 
  def __init__(self, hdffiles: list): 
    # Initialize the hdffiles and general information
    self.hdffiles = []
    self.total_entries = 0
    # Prepare the map size for the dataset
    for file in hdffiles:
      self.hdffiles.append(file)
      with io.hdffile(file, "r") as h5file:
        self.total_entries += h5file["label"].shape[0]

    # Initial behaviors
    with io.hdffile(hdffiles[0], "r") as h5file:
      h5file.draw_structure()
  
  def __len__(self):
    return self.total_entries

    
class HilbertCurveDataset(BaseDataset):
  def __init__(self, hdffiles: list):
    super().__init__(hdffiles)
    # Generate the map for __getitem__ method to correctly slice the data from a set of HDF5 files
    self.idx_to_file = np.zeros(self.total_entries, dtype=np.uint8)
    self.idx_to_position = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_slice_start = np.zeros(self.total_entries, dtype=np.uint64)
    self.idx_to_slice_end = np.zeros(self.total_entries, dtype=np.uint64)
    memsize = self.idx_to_file.nbytes + self.idx_to_position.nbytes + self.idx_to_slice_start.nbytes + self.idx_to_slice_end.nbytes
    print(f"The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB or {memsize / 1024 / 1024 / 1024:6.2f} GB.")

    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        entry_nr_i = h5file["label"].shape[0]
        self.idx_to_file[global_ind:global_ind + entry_nr_i] = fidx
        self.idx_to_position[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i)
        self.idx_to_slice_start[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) 
        self.idx_to_slice_end[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) + 1
        global_ind += entry_nr_i

  def __getitem__(self, index):
    fileidx = self.idx_to_file[index]
    position = self.idx_to_position[index]
    start_pos = self.idx_to_slice_start[index]
    end_pos = self.idx_to_slice_end[index]
    fname = self.hdffiles[fileidx]
    with io.hdffile(fname, "r") as h5file:
      data = h5file["voxel"][start_pos:end_pos]
      label = h5file["label"][position]
      data = np.asarray(data, dtype=np.float32)
      label = np.array(label, dtype=np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label

  def __del__(self):
    self.idx_to_file.resize(0)
    self.idx_to_position.resize(0)

  def getitem_info(self, index):
    fileidx = self.idx_to_file[index]
    position = self.idx_to_position[index]
    start_pos = self.idx_to_slice_start[index]
    end_pos = self.idx_to_slice_end[index]
    fname = self.hdffiles[fileidx]
    return (fname, "voxel", start_pos, end_pos)

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24):
    pool = mp.Pool(process_nr)
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    batches = split_array(indices, batch_size)
    for batch_idx, batch in enumerate(batches): 
      tasks = [self.getitem_info(i) for i in batch]
      ret_data = pool.starmap(readdata, tasks)
      data_numpy = np.zeros((len(ret_data), 128, 128), dtype=np.float32)
      for i, ret in enumerate(ret_data):
        data_numpy[i, ...] = ret
      data_numpy = data_numpy[:, np.newaxis, ...]

      tasks = [(self.hdffiles[self.idx_to_file[i]], self.idx_to_position[i]) for i in batch]
      labels = pool.starmap(readlabel, tasks)
      label_numpy = np.array(labels, dtype=np.int64)

      data = torch.from_numpy(data_numpy)
      label = torch.from_numpy(label_numpy)
      yield data, label



# class SurfLoader(data.DataLoader):
#   def __init__(self, dataset, batch_size=256, shuffle=False, *args, **kwargs):
#     super().__init__(dataset, batch_size, shuffle, *args, **kwargs)
#     # self.dataset = dataset
#     import multiprocessing as mp
#     self.pool = mp.Pool(8)

#   def __iter__(self):
#     # Custom batch retrieval logic goes here
#     # For now, this just mimics the standard DataLoader's batch retrieval
#     for batch in super().__iter__():
#       # You can modify the batch here if needed
#       ret_data = self.pool.starmap(self.dataset.real_getitem, [(int(i),) for i in batch])
#       print(ret_data)
#       yield batch

