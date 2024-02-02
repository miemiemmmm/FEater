import os, time
import multiprocessing as mp

import numpy as np
import open3d as o3d
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
    return ()

  # @profile
  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, v=0, **kwargs):
    pool = mp.Pool(process_nr)
    indices = np.arange(self.total_entries)
    if shuffle:
      np.random.shuffle(indices)
    st = time.perf_counter()
    batches = split_array(indices, batch_size)
    for batch_idx, batch in enumerate(batches): 
      print(kwargs, batch_idx, batch_idx < kwargs.get("end_batch"))
      if batch_idx < kwargs.get("start_batch", 0): 
        continue
      elif batch_idx > kwargs.get("end_batch", len(batches)):
        break
      tasks = [self.mini_batch_task(i) for i in batch]
      ret_data = pool.starmap(readdata, tasks)
      if v: 
        print(f"HERE processing {batch_idx} batch of {len(batches)} batches, time: {time.perf_counter() - st:8.2f}")
        st = time.perf_counter()
      _shape = [i.shape for i in ret_data]
      _shape = tuple(np.max(_shape, axis=0))
      data_numpy = np.zeros((len(ret_data), *_shape), dtype=np.float32)
      for i, ret in enumerate(ret_data):
        if ret.shape != _shape:
          # Do padding if the shape is heterogeneous
          theslice = np.s_[tuple(slice(0, i) for i in ret.shape)]
          data_numpy[i][theslice] = ret
        else:
          data_numpy[i, ...] = ret 

      tasks = [(self.get_file(i), self.get_position(i)) for i in batch]
      labels = pool.starmap(readlabel, tasks)
      label_numpy = np.array(labels, dtype=np.int64)

      data = torch.from_numpy(data_numpy)
      label = torch.from_numpy(label_numpy)
      yield data, label


class CoordDataset(BaseDataset):
  def __init__(self, hdffiles:list, target_np=25, padding=True):
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
    data = readdata(*info)
    label = readlabel(self.get_file(index), self.get_position(index))
    if self.do_padding:
      data = self.padding(data)

    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label
  
  def mini_batch_task(self, index): 
    return (self.get_file(index), "coordinates", self.get_slice(index))

  def mini_batches_(self, batch_size=512, shuffle=False, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs): 
      yield data, label
  
  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs): 
      data = self.padding_batch(data)
      yield data, label
  
  def padding_batch(self, batch):
    ret_points = np.zeros((batch.shape[0], self.target_np, batch.shape[2]), dtype=np.float32)
    batch = batch.numpy()
    randomization = np.arange(self.target_np)
    np.random.shuffle(randomization)

    for idx, entry in enumerate(batch):
      point_nr = np.count_nonzero(np.count_nonzero(entry, axis=1))
      batch[idx, :point_nr, :] -= np.min(entry[:point_nr, :], axis=0)
      _randomization = np.arange(min(self.target_np, point_nr))
      np.random.shuffle(_randomization)
      if point_nr <= self.target_np:
        # Choose random points to fill with the result points
        ret_points[idx, randomization[:point_nr], :] = batch[idx, _randomization, :]
      elif point_nr > self.target_np:
        # Randomly select target_np points
        ret_points[idx, randomization, :] = batch[idx, _randomization[:self.target_np], :]
    ret_points = torch.from_numpy(ret_points)
    return ret_points
    
  def padding(self, points):
    # Prerequisite: the points are not yet padded to a fixed length
    randomization = np.arange(self.target_np)
    np.random.shuffle(randomization)
    point_nr = points.shape[0]
    _randomization = np.arange(point_nr)
    ret_data = np.zeros((self.target_np, points.shape[1]), dtype=np.float32)
    if point_nr <= self.target_np:
      # Choose random points to fill with the result points
      ret_data[randomization[:point_nr], :] = points[_randomization, :]
    elif point_nr > self.target_np:
      # Randomly select target_np points
      ret_data[randomization, :] = points[_randomization[:self.target_np], :]
    return ret_data


class VoxelDataset(BaseDataset):
  def __init__(self, hdffiles:list):
    super().__init__(hdffiles)
    # Generate the map for __getitem__ method to correctly locate the data from a set of HDF5 files
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      with io.hdffile(file, "r") as h5file:
        self.shape = np.array(h5file["dimensions"], dtype=np.uint32)
        entry_nr_i = h5file["label"].shape[0]
      self.position_map[global_ind:global_ind+entry_nr_i] = np.arange(entry_nr_i)
      self.file_map[global_ind: global_ind + entry_nr_i] = fidx
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
  
  def mini_batch_task(self, index):
    return (self.get_file(index), "voxel", np.s_[index])

  def __len__(self):
    return self.total_entries

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24, **kwargs):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr, **kwargs): 
      data = data[:, np.newaxis, ...]
      yield data, label


class SurfDataset(BaseDataset):
  def __init__(self, hdffiles: list, target_np=1024, padding=True):
    super().__init__(hdffiles)
    self.target_np = target_np
    self.do_padding = padding        # When serve as dataloader, do the padding, other wise return the original data

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
    # Get the file
    if index >= self.total_entries:
      raise IndexError(f"Index {index} is out of range. The dataset has {self.total_entries} entries.")
    task = self.mini_batch_task(index)
    data = readdata(*task)
    label = readlabel(self.get_file(index), self.get_position(index))
    if self.do_padding:
      data = self.padding(data)

    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label

  def get_label(self, index):
    return readlabel(self.get_file(index), self.get_position(index))

  def mini_batches(self, batch_size=512, shuffle=True, process_nr=24):
    for data, label in super().mini_batches(batch_size=batch_size, shuffle=shuffle, process_nr=process_nr):
      data = self.padding_batch(data)
      yield data, label

  def mini_batch_task(self, index):
    start_idx = self.get_start(index)
    end_idx = self.get_end(index)
    if (end_idx - start_idx > self.target_np) and self.do_padding:
      # print("doing batch padding", self.do_padding)
      # Randomly select target_np points
      _rand = np.arange(start_idx, end_idx, dtype=np.uint64)
      np.random.shuffle(_rand)
      nr_query = int(min(self.target_np*1.05, end_idx - start_idx))
      _rand = _rand[:nr_query]  # NOTE: Increase a bit because the source vertices might have zero points
      _rand.sort()
      theslice = np.s_[_rand]
    else: 
      # No padding for visualization
      theslice = np.s_[start_idx:end_idx]
    return (self.get_file(index), "vertices", theslice)

  def padding_batch(self, batch):
    ret_points = np.zeros((batch.shape[0], self.target_np, batch.shape[2]), dtype=np.float32)
    batch = batch.numpy()
    ret_point_rand = np.arange(self.target_np)
    np.random.shuffle(ret_point_rand)
    for idx, entry in enumerate(batch):
      # Mask the points at origin point, move to the origin and then do the padding
      mask = np.count_nonzero(entry, axis=1).astype(bool)
      entry = entry[mask]
      lower_boundi = np.min(entry, axis=0)
      entry -= lower_boundi
      np.random.shuffle(entry)
      ret_points[idx][ret_point_rand[:min(self.target_np, entry.shape[0])], :] = entry[:min(self.target_np, entry.shape[0]), :]
    ret_points = torch.from_numpy(ret_points)
    # print(">>>> ", ret_points.shape, " size ", ret_points.nbytes/1024/1024/1024, " GB")
    return ret_points

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

  ################### ?????????? ###################
  def view_verts(self, index):
    import open3d as o3d
    points, _ = self.__getitem__(index)
    points = points.numpy()
    # Count number of points at 0,0,0
    zero_nr = np.count_nonzero(points, axis=1)
    print(points.shape)
    zero_point_nr = np.count_nonzero(~np.count_nonzero(points, axis=1).astype(bool))
    print(f"Number of points at 0,0,0: {zero_point_nr}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

  def view_verts2(self, index):
    import open3d as o3d
    points, _ = next(self.mini_batches(batch_size=1, shuffle=True, process_nr=1))
    points = points.numpy()[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

  def get_surf(self, index): 
    # make map for the faces
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
