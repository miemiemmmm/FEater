import time
import multiprocessing as mp

import numpy as np
import torch
import torch.utils.data as data

from . import io


__all__ = [
  "BaseDataset",
  "CoordDataset",
  "VoxelDataset",
  "SurfDataset",
  "HilbertCurveDataset", 
  "readdata",
  "readlabel",
  "readtop",
  "split_array",
]


def readdata(input_file, keyword, theslice):
  with io.hdffile(input_file, "r") as h5file:
    ret_data = h5file[keyword][theslice]
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


class BaseDataset(data.Dataset): 
  def __init__(self, hdffiles: list): 
    # Initialize the hdffiles and general information
    self.hdffiles = []
    self.total_entries = 0
    self.do_padding = False
    self.do_scaling = False

    # Prepare the map size for the dataset
    for file in hdffiles:
      self.hdffiles.append(file)
      with io.hdffile(file, "r") as h5file:
        self.total_entries += h5file["label"].shape[0]

    # Initial behaviors
    self.file_map = np.zeros(self.total_entries, dtype=np.uint64)
    self.position_map = np.zeros(self.total_entries, dtype=np.uint64)
    self.start_map = np.zeros(self.total_entries, dtype=np.uint64)
    self.end_map = np.zeros(self.total_entries, dtype=np.uint64)

  def __len__(self):
    return self.total_entries
  
  def __iter__(self):
    for i in range(self.total_entries):
      yield self.__getitem__(i)

  def map_size(self):
    memsize = self.file_map.nbytes + self.position_map.nbytes + self.start_map.nbytes + self.end_map.nbytes
    print(f"The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB or {memsize / 1024 / 1024 / 1024:6.2f} GB.")
    return memsize

  def get_file(self, index):
    return self.hdffiles[self.file_map[index]]
  
  def get_start(self, index):
    return self.start_map[index]
  
  def get_end(self, index):
    return self.end_map[index]
  
  def get_position(self, index):
    return self.position_map[index]

  def get_slice(self, index):
    return np.s_[self.get_start(index):self.get_end(index)]

  def mini_batch_task(self, index): 
    raise NotImplementedError("The mini_batch_task method should be implemented in the subclass.")

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, v=0, exit_point=999999999, **kwargs):
    """
    Base class for the mini-batch iteration. 
    """
    pool = mp.Pool(process_nr)
    indices = np.arange(self.total_entries)
    with mp.Pool(process_nr) as pool:
      if shuffle:
        seed = kwargs.get('seed', None)
        if seed is not None:
          np.random.seed(seed)
        np.random.shuffle(indices)
      st = time.perf_counter()
      batches = split_array(indices, batch_size)
      for batch_idx, batch in enumerate(batches): 
        if batch_idx >= exit_point: 
          break
        if batch_idx < kwargs.get("start_batch", 0): 
          continue
        elif batch_idx > kwargs.get("end_batch", len(batches)):
          break
        tasks = [self.mini_batch_task(i) for i in batch]
        ret_data = pool.starmap(readdata, tasks)

        if v: 
          print(f"Processing {batch_idx} batch of {len(batches)} batches, time: {time.perf_counter() - st:8.2f}")
          st = time.perf_counter()
        
        if self.do_padding:
          # Padding the data for heterogeneous shape (coordinates, surface vertices)
          data_numpy  = np.zeros((len(ret_data), self.target_np, 3), dtype=np.float32)
          for i, ret in enumerate(ret_data):
            ret = ret[np.sum(ret, axis=1) != 0]   # Mask the 0,0,0 points
            ret -= np.min(ret, axis=0)            # Move to the origin
            ret_pointnr = ret.shape[0]
            # Shuffle the order of points (n_samples, n_points, 3)
            if ret_pointnr < self.target_np:
              # Point number less than target number, shuffle the source points
              shuffuled = np.random.choice(self.target_np, ret_pointnr, replace=False)
              data_numpy[i, shuffuled, :] = ret
            else:
              # Point number more than target number, randomly select target_np points
              shuffuled = np.random.choice(ret_pointnr, self.target_np, replace=False)
              data_numpy[i, ...] = np.array(ret[shuffuled, :], dtype=np.float32)
        else: 
          # Homogeneous shape 
          data_numpy = np.array(ret_data, dtype=np.float32)
          assert data_numpy.shape == (len(ret_data), 1, *self.shape), f"Shape mismatch: {data_numpy.shape} vs {len(ret_data), *self.shape}"
        
        if self.do_scaling:
          # print("Scaling the data")
          for i, ret in enumerate(data_numpy):
            p_max = np.max(ret, axis=0)
            dist = np.linalg.norm(p_max)
            data_numpy[i] = ret / dist
            # print(">> ", dist, np.min(data_numpy[i]), np.max(data_numpy[i]))

        tasks = [(self.get_file(i), self.get_position(i)) for i in batch]
        labels = pool.starmap(readlabel, tasks)
        label_numpy = np.array(labels, dtype=np.int64)

        data = torch.from_numpy(data_numpy)
        label = torch.from_numpy(label_numpy)
        yield data, label

  def get_batch_by_index(self, index, process_nr=24): 
    with mp.Pool(process_nr) as pool:
      tasks = [self.mini_batch_task(i) for i in index]
      ret_data = pool.starmap(readdata, tasks)
      if self.do_padding:
        data_numpy  = np.zeros((len(ret_data), self.target_np, 3), dtype=np.float32)
        for i, ret in enumerate(ret_data):
          ret = ret[np.sum(ret, axis=1) != 0]
          ret -= np.min(ret, axis=0)
          ret_pointnr = ret.shape[0]
          if ret_pointnr < self.target_np:
            shuffuled = np.random.choice(self.target_np, ret_pointnr, replace=False)
            data_numpy[i, shuffuled, :] = ret
          else:
            shuffuled = np.random.choice(ret_pointnr, self.target_np, replace=False)
            data_numpy[i, ...] = np.array(ret[shuffuled, :], dtype=np.float32)
      else: 
        data_numpy = np.array(ret_data, dtype=np.float32)
        assert data_numpy.shape == (len(ret_data), 1, *self.shape), f"Shape mismatch: {data_numpy.shape} vs {len(ret_data), *self.shape}"
      tasks = [(self.get_file(i), self.get_position(i)) for i in index]
      labels = pool.starmap(readlabel, tasks)
      label_numpy = np.array(labels, dtype=np.int64)
      data = torch.from_numpy(data_numpy)
      label = torch.from_numpy(label_numpy)

    return data, label

class CoordDataset(BaseDataset):
  def __init__(self, hdffiles:list, target_np=25, resmap=None,padding=True):
    """
    Open the HDF files and generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files
    Supports constant time random access to the dataset
    Args:
      hdffiles: The list of HDF5 files
      target_np: The target number of points for each residue
    """
    super().__init__(hdffiles)
    self.target_np = target_np
    self.do_padding = padding        # When serve as dataloader, do the padding, other wise return the original data
    
    # Build the maps for correct data retrieval according the the structure of the HDF5 files
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        entry_nr_i = h5file["label"].shape[0]
        self.position_map[global_ind: global_ind + entry_nr_i] = np.arange(entry_nr_i)
        self.file_map[global_ind: global_ind + entry_nr_i] = fidx
        self.start_map[global_ind: global_ind+entry_nr_i] = np.asarray(h5file["coord_starts"])
        self.end_map[global_ind: global_ind+entry_nr_i] = np.asarray(h5file["coord_ends"])
        global_ind += entry_nr_i
    if resmap is not None: 
      self.resmap = resmap

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
    info = self.mini_batch_task(index)
    ret_data = readdata(*info)
    label = readlabel(self.get_file(index), self.get_position(index))
    if self.do_padding:
      data_numpy  = np.zeros((self.target_np, 3), dtype=np.float32)
      ret_data = ret_data[np.sum(ret_data, axis=1) != 0]   # Mask the 0,0,0 points
      ret_data -= np.min(ret_data, axis=0)
      ret_pointnr = ret_data.shape[0]
      if ret_pointnr < self.target_np:
        shuffuled = np.random.choice(self.target_np, ret_pointnr, replace=False)
        data_numpy[shuffuled, :] = ret_data
      else:
        shuffuled = np.random.choice(ret_pointnr, self.target_np, replace=False)
        data_numpy = np.array(ret_data[shuffuled, :], dtype=np.float32)
    else:
      data_numpy = np.array(ret_data, dtype=np.float32)
    data = torch.from_numpy(data_numpy)
    return data, label
  
  def __len__(self):
    return self.total_entries
  
  def __str__(self):
    return f"<CoordDataset with {self.total_entries} entries>"
  
  def get_top(self, index=None, restype=None):
    if index is None and restype is None:
      raise ValueError("Either index or restype should be provided.")
    if index is not None:
      label = readlabel(self.get_file(index), self.get_position(index))
      with io.hdffile(self.hdffiles[0], "r") as h5file:
        if self.resmap is not None: 
          restype = self.resmap[label]
        else: 
          raise ValueError("The resmap is not provided when initializing the dataloader if you want the topology.")
        top = h5file.get_top(restype)
    else: 
      with io.hdffile(self.hdffiles[0], "r") as h5file:
        top = h5file.get_top(restype)
    return top

  def mini_batch_task(self, index): 
    return (self.get_file(index), "coordinates", self.get_slice(index))

  
  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs): 
      yield data, label

class VoxelDataset(BaseDataset):
  def __init__(self, hdffiles:list):
    super().__init__(hdffiles)
    # Generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files 
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        self.shape = h5file["dimensions"][:].astype(np.int32)
        entry_nr_i = h5file["label"].shape[0]
      self.position_map[global_ind:global_ind+entry_nr_i] = np.arange(entry_nr_i)
      self.file_map[global_ind: global_ind + entry_nr_i] = fidx
      self.start_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) 
      self.end_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) + 1
      global_ind += entry_nr_i

  def __getitem__(self, index):
    # Get the file
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    data = readdata(*self.mini_batch_task(index))
    label = readlabel(self.get_file(index), self.get_position(index))
    # Add the bracket to represent there is one channel
    data = np.array([data], dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label
  
  def __len__(self):
    return self.total_entries
  
  def __str__(self):
    return f"<VoxelDataset with {self.total_entries} entries>"
  
  def mini_batch_task(self, index):
    return (self.get_file(index), "voxel", np.s_[self.get_start(index):self.get_end(index)])

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs): 
      data = data[:, ...]
      yield data, label


class SurfDataset(BaseDataset):
  def __init__(self, hdffiles: list, target_np=1024, padding=True, scale=False):
    super().__init__(hdffiles)
    self.target_np = target_np
    self.do_padding = padding        # When serve as dataloader, do the padding, other wise return the original data
    self.do_scaling = scale

    global_ind = 0
    for fidx, file in enumerate(hdffiles):
      with io.hdffile(file, "r") as h5file:
        entry_nr_i = h5file["label"].shape[0]
        self.file_map[global_ind:global_ind + entry_nr_i] = fidx
        self.position_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i)
        self.start_map[global_ind:global_ind + entry_nr_i] = h5file["vert_starts"]
        self.end_map[global_ind:global_ind + entry_nr_i] = h5file["vert_ends"]
        global_ind += entry_nr_i
    print(f"SurfDataset: Average vertices per entry: {np.mean(self.end_map - self.start_map):8.2f}")

  def __getitem__(self, index):
    # Check the index validity
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    # Prepare the task and read the data
    task = self.mini_batch_task(index)
    ret_data = readdata(*task)
    label = readlabel(self.get_file(index), self.get_position(index))
    # Perform padding if necessary
    if self.do_padding:
      data_numpy = np.zeros((self.target_np, 3), dtype=np.float32)
      ret_data = ret_data[np.sum(ret_data, axis=1) != 0]
      ret_data -= np.min(ret_data, axis=0)
      ret_pointnr = ret_data.shape[0]
      if ret_pointnr < self.target_np:
        shuffuled = np.random.choice(self.target_np, ret_pointnr, replace=False)
        data_numpy[shuffuled, :] = ret_data
      else:
        shuffuled = np.random.choice(ret_pointnr, self.target_np, replace=False)
        data_numpy = np.array(ret_data[shuffuled, :], dtype=np.float32)
    else:
      data_numpy = np.array(ret_data, dtype=np.float32)
    
    # Convert the data to torch.Tensor
    data = torch.from_numpy(data_numpy)
    return data, label

  def get_label(self, index):
    return readlabel(self.get_file(index), self.get_position(index))

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs):
      yield data, label

  def mini_batch_task(self, index):
    """
    Prepare arguments for the mini-batch task
    """
    start_idx = self.get_start(index)
    end_idx = self.get_end(index)
    theslice = np.s_[start_idx:end_idx]
    return (self.get_file(index), "vertices", theslice)

  def padding(self, points):
    """
    Padding the points to the target number of points

    Parameters
    ----------
    points : np.ndarray
      The input points

    Returns
    -------
    np.ndarray
      The padded points

    Notes
    -----
    The padding is done by randomly selecting the points to fill the target number of points

    """
    # Placeholder coordinate is (0,0,0)
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

  def get_surf(self, index): 
    """
    Get an Open3D surface object of the surface via its index within the dataset 
    """
    import open3d as o3d
    st = time.perf_counter()
    filename = self.get_file(index)
    position = self.get_position(index)
    with io.hdffile(filename, "r") as hdf: 
      vert_sti = hdf["vert_starts"][position]
      vert_end = hdf["vert_ends"][position]
      face_sti = hdf["face_starts"][position]
      face_end = hdf["face_ends"][position]
      vert = hdf["vertices"][vert_sti:vert_end]
      face = hdf["faces"][face_sti:face_end]
    surf = o3d.geometry.TriangleMesh()
    surf.vertices = o3d.utility.Vector3dVector(vert)
    surf.triangles = o3d.utility.Vector3iVector(face)
    surf.compute_vertex_normals()
    surf.paint_uniform_color([0.5, 0.5, 0.5])
    print(f"The surface {index}/{len(self)} has {len(vert)} vertices and {len(face)} faces., time: {time.perf_counter() - st:8.2f}")
    return surf


class HilbertCurveDataset(BaseDataset):
  def __init__(self, hdffiles: list):
    super().__init__(hdffiles)
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        self.shape = h5file["size"][:].astype(np.int32)
        entry_nr_i = h5file["label"].shape[0]
        self.file_map[global_ind:global_ind + entry_nr_i] = fidx
        self.position_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i)
        self.start_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) 
        self.end_map[global_ind:global_ind + entry_nr_i] = np.arange(entry_nr_i) + 1
        global_ind += entry_nr_i

  def __getitem__(self, index):
    data = readdata(*self.mini_batch_task(index))
    label = readlabel(self.get_file(index), self.get_position(index))
    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label

  def mini_batch_task(self, index):
    return (self.get_file(index), "voxel", np.s_[self.get_start(index):self.get_end(index)])


# Coord Dataset
# T5  : Estimated Time:   466.46 seconds or     7.77 minutes
# SSD : Estimated Time:   467.22 seconds or     7.79 minutes
# No acceleration in SSD

# Hilbert Curve Dataset
# T5  : Estimated Time: 19104.34 seconds or   318.41 minutes
# SSD : Estimated Time:  2662.34 seconds or    44.37 minutes
# 7.1 fold acceleration in SSD 

# Surface Dataset
# T5  : Estimated Time: 16574.94 seconds or   276.25 minutes
# SSD : Estimated Time:  2656.86 seconds or    44.28 minutes
# 6.2 fold acceleration

# Voxel Dataset
# T5  : Estimated Time: 50537.85 seconds or   842.30 minutes
# SSD : Estimated Time:  4510.17 seconds or    75.17 minutes
# 11 fold acceleration in SSD
