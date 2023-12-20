import os
from numpy import ndarray

def checkfiles(file_list:str, basepath="") -> list:
  with open(file_list, 'r') as f:
    files = f.read().strip("\n").split('\n')
    if len(basepath) > 0:
      files = [os.path.join(basepath, file) for file in files]
    for file in files:
      if not os.path.isfile(file):
        raise ValueError(f"File {file} does not exist.")
  return files

def add_data_to_hdf(hdffile, dataset_name:str, data:ndarray, **kwargs):
  if dataset_name not in hdffile.keys():
    hdffile.create_dataset(dataset_name, data=data, **kwargs)
  else:
    hdffile.append_entries(dataset_name, data)

# def h5files_to_dataloader(filelist:list):

