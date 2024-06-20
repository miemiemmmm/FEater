import time, os
import argparse
import multiprocessing as mp

from feater import dataloader
from torch.utils.data import DataLoader


# @profile
def benchmark_dataset_method(dset, exit_point=None, **kwargs):
  st = time.perf_counter()
  st_total = time.perf_counter()
  print(f"Started benchmarking ...")
  data_size = 0
  put_to_cuda = kwargs.get("tocuda", 0)
  batch_size = kwargs.get("batch_size", 256)
  process_nr = kwargs.get("process_nr", 12)
  verbose = kwargs.get("verbose", 0)
  
  batch_nr = (len(dset) + batch_size - 1) // batch_size
  if batch_nr * batch_size != len(dset):
    print(f"Data size is not aligned with batch size. {len(dset)} samples are extracted in total.")
    _exit_point = batch_nr - 1     # Skipt the last batch because the size is not aligned. 
  else:
    _exit_point = batch_nr
  exit_point = min(_exit_point, exit_point)  
  print(f"Benchmark Summary: batch_size {batch_size}, batch number {batch_nr}, process number {process_nr}, exit point {exit_point}.")

  for idx, datai in enumerate(dset.mini_batches(batch_size=batch_size, shuffle=True, process_nr=process_nr)): 
    data, label = datai
    if verbose:
      print(f"Batch {idx+1:5} used {time.perf_counter()-st:6.2f} s. {data.shape}")
    data_size += data.nbytes
    if put_to_cuda: 
      data.cuda()
      label.cuda()
    st = time.perf_counter()
    if (idx == exit_point-1): 
      print(f"Estimated total extraction time: {(time.perf_counter()-st_total)/idx*batch_nr:6.2f} s. ")
      time_elapsed = time.perf_counter()-st_total
      through_put = batch_size * (idx+1) / time_elapsed    # Unit: sample/s
      throughput_per_core = through_put / process_nr
      digit_thoughput = data_size / time_elapsed / 1024 / 1024    # Unit: MB/s
      total_size = data_size / 1024 / 1024    # Unit: MB
      
      print(f"Data_size: {total_size:6.3f} MB; Time_elapse {time_elapsed:6.3f} s; Retrieval_rate: {digit_thoughput:6.3f} MB/s; Throughput: {through_put:6.3f} samples/s; Throughput_per_core: {throughput_per_core:6.3f} samples/(core*s);")
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


def argument_parser():
  parser = argparse.ArgumentParser(description="Benchmarking the data extraction speed of the HDF5 dataset.")
  parser.add_argument("-f", '--input-file', type=str, help="Input file path.")
  parser.add_argument("-b", "--batch-size", default=128, type=int, help="Batch size.")
  parser.add_argument("-p", "--process-nr", default=8, type=int, help="Number of processes.")
  parser.add_argument("-e", "--exit-point", default=999999, type=int, help="Exit point.")
  parser.add_argument("-d", "--dataloader-type", type=str, help="Use dataloader.")
  parser.add_argument("-c", "--tocuda", default=0, type=int, help="Transfer data to cuda.") 
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose mode.")
  #### Point-cloud specific arguments
  parser.add_argument("--pointnr", type=int, default=None, help="Number of points.")

  args = parser.parse_args()
  if args.input_file is None: 
    raise ValueError("Input file is not specified.")
  if os.path.exists(args.input_file) is False: 
    raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
  
  return args

def console_interface(): 
  args = argument_parser()
  settings = vars(args)

  workernr = settings["process_nr"]
  batch_size=settings["batch_size"]
  input_file = settings["input_file"]
  exit_point = settings["exit_point"]

  if settings["dataloader_type"] == "coord":
    if settings["pointnr"] is not None: 
      dset = dataloader.CoordDataset([input_file], target_np=settings["pointnr"])
    else:
      raise ValueError("Point number is not specified for point-cloud-based representation.")
    
  elif settings["dataloader_type"] == "surface":
    if settings["pointnr"] is not None: 
      dset = dataloader.SurfDataset([input_file], target_np=settings["pointnr"])
    else: 
      raise ValueError("Point number is not specified for point-cloud-based representation.")
    
  elif settings["dataloader_type"] == "voxel":
    dset = dataloader.VoxelDataset([input_file])
  elif settings["dataloader_type"] == "hilbert":
    dset = dataloader.HilbertCurveDataset([input_file])
  else: 
    raise ValueError("Dataloader type is not specified.")
  

  benchmark_dataset_method(dset, batch_size=batch_size, process_nr=workernr, exit_point=exit_point, tocuda=settings["tocuda"], verbose=settings["verbose"])
  # benchmark_dataloader(dset, batch_size=256, process_nr=pnr, exit_point=100)




if __name__ == "__main__": 
  console_interface()
  # input_file = [
  #   "/diskssd/yzhang/Data_test/feater_database_surf/TrainingSet_LEU.h5", 
  #   "/diskssd/yzhang/Data_test/feater_database_surf/TrainingSet_TRP.h5", 
  # ]
  # dset = dataloader.SurfDataset(input_file, batch_size=100, process_nr=4)

  # input_file = [
    # "/diskssd/yzhang/FEater_data/FEater_Single_VOX/TrainingSet_Voxel.h5"
    # "/disk2b/yzhang/testdata/FEater_Dual_VOX/TrainingSet_Voxel.h5"
    # "/media/yzhang/Black/FEater_Dual_VOX/TrainingSet_Voxel.h5"
    # "/home/yzhang/Documents/FEater_Dataset/FEater_Dual_HILB/TrainingSet_Hilbert.h5",  
  # ]
  
