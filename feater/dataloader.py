import os

import numpy as np
import h5py as h5
import torch
import torch.utils.data as data

from . import io

__all__ = [
  "HDF5Dataset",
  "VoxelDataset",
]


class CoordDataset(data.Dataset):
  """
  Supports constant time random access to the dataset
  """
  def __init__(self, hdffiles:list, target_np=25):
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

    for file in hdffiles:
      h5file = h5.File(file, "r")
      self.hdffiles.append(h5file)
      self.total_entries += h5file["nr_atoms"].shape[0]

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
      starts_i = file["start_indices"]
      ends_i = file["end_indices"]
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
      # Choose random points to fill with the result points
      choices = np.random.choice(self.target_np, points.shape[0], replace=False)
      choices.sort()
      # print(f"Padding Choices: {choices.tolist()}")
      _points = np.zeros((self.target_np, 3), dtype=np.float32)                   # TODO: Justify the default values for the point coordinates
      for i, choice in enumerate(choices):
        _points[choice] = points[i]
      points = _points
    elif points.shape[0] > self.target_np:
      # Randomly select self.target_np points
      choices = np.random.choice(points.shape[0], self.target_np, replace=False)  # TODO: Justify the choice of random points
      choices.sort()
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
    points = torch.from_numpy(self.padding(points))
    # label = torch.from_numpy(np.array([label], dtype=np.int32))
    # print("Processing the residue:", constants.LAB2RES[int(label)])
    # print(f"Item {index}: file {file_index}, slice {slice_start}:{slice_end}, label {label_position}, Number of points: {points.shape[0]}; ")
    return points, np.array(label, dtype=np.int64)

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

    with io.hdffile(hdffiles[0], "r") as h5file:
      h5file.draw_structure()
    for file in hdffiles:
      h5file = h5.File(file, "r")
      self.hdffiles.append(h5file)
      self.total_entries += h5file["label"].shape[0]
      self.shape = np.array(h5file["shape"], dtype=np.int32)

    self.idx_to_file = np.zeros(self.total_entries, dtype=np.uint8)
    self.idx_to_position = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_start = np.zeros(self.total_entries, dtype=np.uint32)
    self.idx_to_slice_end = np.zeros(self.total_entries, dtype=np.uint32)
    memsize = self.idx_to_file.nbytes + self.idx_to_position.nbytes + self.idx_to_slice_start.nbytes + self.idx_to_slice_end.nbytes
    print(f"The maps of the {self.total_entries} entries occupies {memsize / 1024 / 1024:6.2f} MB or {memsize / 1024 / 1024 / 1024:6.2f} GB.")
    global_ind = 0
    for fidx, file in enumerate(self.hdffiles):
      entry_nr_i = file["label"].shape[0]
      size_0 = file["shape"][0]
      starts = np.arange(entry_nr_i) * size_0
      ends = starts + size_0
      if ends[-1] != file["voxel"].shape[0]:
        raise ValueError(f"Unmatched array end indices: {ends[-1]} vs {file['voxel'].shape[0]}")
      self.idx_to_position[global_ind:global_ind+entry_nr_i] = np.arange(entry_nr_i)
      self.idx_to_file[global_ind: global_ind + entry_nr_i] = fidx
      self.idx_to_slice_start[global_ind:global_ind+entry_nr_i] = starts
      self.idx_to_slice_end[global_ind:global_ind+entry_nr_i] = ends
      global_ind += entry_nr_i

  def __del__(self):
    for file in self.hdffiles:
      file.close()
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

    voxel = self.hdffiles[file_idx]["voxel"][slice_start:slice_end]
    label = self.hdffiles[file_idx]["label"][position]
    voxel = np.asarray([voxel], dtype=np.float32)
    label = np.array(label, dtype=np.int64)
    voxel = torch.from_numpy(voxel)
    return voxel, label


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







