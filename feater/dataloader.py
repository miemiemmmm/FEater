import os



import numpy as np
import torch

import torch.utils.data as data

import h5py





class FEaterDataset(data.Dataset):
  def __init__(self, filelist:list):
    self.data_point_nr = 0
    self.filelist = filelist




  def __getitem__(self, index):
    pass

  def __len__(self):
    return self.data_point_nr






