import time
import multiprocessing as mp
import numpy as np 
import h5py as h5 
from feater import io, dataloader
from torch.utils.data import DataLoader


# @profile
def benchmark_dataset_method(dset, exit_point=None, **kwargs):
  st = time.perf_counter()
  st_total = time.perf_counter()
  print(f"Started benchmarking ...")
  data_size = 0
  batch_size = kwargs.get("batch_size", 256)
  process_nr = kwargs.get("process_nr", 12)
  batch_nr = (len(dset) + batch_size - 1) // batch_size
  print(f"Scheduled {batch_nr} batches with batch size {batch_size} and {process_nr} processes.")
  for idx, datai in enumerate(dset.mini_batches()): 
    data, label = datai
    print(f"batch {idx} used {time.perf_counter()-st:6.2f} s. {data.shape}")
    data_size += data.nbytes
    data.cuda()
    label.cuda()
    st = time.perf_counter()
    if exit_point is not None and idx == exit_point: 
      print(f"Estimated total extraction time: {(time.perf_counter()-st_total)/idx*batch_nr:6.2f} s. ")
      print(f"Data size: {data_size/1024/1024:6.2f} MB, retrieval rate: {data_size/(time.perf_counter()-st_total)/1024/1024:6.2f} MB/s ")
      break
  

def benchmark_dataloader(dset, exit_point=None, **kwargs):
  dataloader = DataLoader(dset, shuffle=True, batch_size=kwargs.get("batch_size", 100), num_workers=kwargs.get("process_nr", 4))
  st = time.perf_counter()
  last_st = None
  data_size = 0 
  for idx, datai in enumerate(dataloader): 
    data, label = datai
    data_size += data.nbytes
    data.cuda()
    label.cuda()
    if last_st is not None: 
      print(f"Batch {idx+1}: Retrieval rate: {data_size/(time.perf_counter()-last_st)/1024/1024:6.2f} MB/s. Retrieval Time: {time.perf_counter()-last_st:6.2f} s. ")
      last_st = time.perf_counter() 
    else: 
      print(f"Batch {idx+1}: Retrieval rate: {data_size/(time.perf_counter()-st)/1024/1024:6.2f} MB/s. Retrieval Time: {time.perf_counter()-st:6.2f} s. ")
      last_st = time.perf_counter() - st
    
    if exit_point is not None and idx == exit_point: 
      break
  print(f"Data size: {data_size/1024/1024:6.2f} MB, retrieval rate: {data_size/(time.perf_counter()-st)/1024/1024:6.2f} MB/s ")


if __name__ == "__main__": 
  # input_file = [
  #   "/diskssd/yzhang/Data_test/feater_database_surf/TrainingSet_LEU.h5", 
  #   "/diskssd/yzhang/Data_test/feater_database_surf/TrainingSet_TRP.h5", 
  # ]
  # dset = dataloader.SurfDataset(input_file, batch_size=100, process_nr=4)

  input_file = [
    # "/diskssd/yzhang/FEater_data/FEater_Single_VOX/TrainingSet_Voxel.h5"
    # "/disk2b/yzhang/testdata/FEater_Dual_VOX/TrainingSet_Voxel.h5"
    # "/media/yzhang/Black/FEater_Dual_VOX/TrainingSet_Voxel.h5"
    "/home/yzhang/Documents/FEater_Dataset/FEater_Dual_HILB/TrainingSet_Hilbert.h5",  
  ]

  pnr = 32
  batch_size=512
  dset = dataloader.VoxelDataset(input_file)

  benchmark_dataset_method(dset, batch_size=batch_size, process_nr=pnr, exit_point=100)

  # benchmark_dataloader(dset, batch_size=256, process_nr=pnr, exit_point=100)
