import os, sys, argparse, time, random

import numpy as np
import h5py as h5

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from feater.models.voxnet import VoxNet
from feater import io, dataloader


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






def parse_args():
  parser = argparse.ArgumentParser(description="Train VoxNet")

  parser.add_argument("--manualSeed", type=int, help="Manual seed")
  parser.add_argument("--load_model", type=str, default="", help="The model to load")
  args = parser.parse_args()
  # known_args, unknown_args = parser.parse_known_args()
  args.cuda = torch.cuda.is_available()

  return args

if __name__ == "__main__":
  args = parse_args()
  print(args)
  if args.manualSeed is None:
    print("Randomizing the Seed")
    args.manualSeed = random.randint(1, 10000)
  else:
    print(f"Using manual seed {args.manualSeed}")

  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  BATCH_SIZE = 256
  EPOCH_NR = 1
  verbose = False
  LOAD_MODEL = None
  OUTPUT_DIR = "/media/yzhang/MieT72/scripts_data/voxel_results"

  st = time.perf_counter()
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  # datafiles = "/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_ALA.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_ARG.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_ASN.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_ASP.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_CYS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_GLN.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_GLU.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_GLY.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_HIS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_ILE.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_LEU.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_LYS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_MET.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_PHE.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_PRO.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_SER.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_THR.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_TRP.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_TYR.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TrainingSet_VAL.h5"
  datafiles = "/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_ALA.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_ARG.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_ASN.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_ASP.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_CYS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_GLN.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_GLU.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_GLY.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_HIS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_ILE.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_LEU.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_LYS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_MET.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_PHE.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_PRO.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_SER.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_THR.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_TRP.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_TYR.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TrainingSet_VAL.h5"
  datafiles = datafiles.strip("%").split("%")
  training_data = dataloader.VoxelDataset(datafiles)
  loader_training = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

  # testdatafiles = "/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_ALA.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_ARG.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_ASN.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_ASP.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_CYS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_GLN.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_GLU.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_GLY.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_HIS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_ILE.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_LEU.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_LYS.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_MET.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_PHE.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_PRO.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_SER.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_THR.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_TRP.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_TYR.h5%/media/yzhang/MieT72/Data/feater_database_voxel/TestSet_VAL.h5%"
  testdatafiles ="/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_ALA.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_ARG.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_ASN.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_ASP.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_CYS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_GLN.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_GLU.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_GLY.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_HIS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_ILE.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_LEU.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_LYS.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_MET.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_PHE.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_PRO.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_SER.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_THR.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_TRP.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_TYR.h5%/diskssd/yzhang/Data_test/feater_database_voxel/TestSet_VAL.h5"
  testdatafiles = testdatafiles.strip("%").split("%")
  test_data = dataloader.VoxelDataset(testdatafiles)
  loader_test = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
  print(f"Dataloader creation took: {(time.perf_counter() - st)*1e3 :8.2f} ms")

  print(len(training_data), len(test_data))

  # for i in np.linspace(0, len(training_data)-1, 100, dtype=np.int32):
  #   st_time = time.perf_counter()
  #   voxel, label = training_data[i]
  #   print(f"Retrieval of {i:8} took: {(time.perf_counter() - st_time)*1e3:6.2f} ms; Number of points: {voxel.shape[0]}; Residue: {label}")
  # exit(1)

  # VoxNet model
  voxnet = VoxNet(n_classes=20)
  print(voxnet)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    voxnet.load_state_dict(torch.load(LOAD_MODEL))
  voxnet.cuda()

  optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
  criterion = nn.CrossEntropyLoss()

  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

  for epoch in range(EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    # mini_batches = np.array_split(np.arange(len(training_data)), (len(loader_training)+EPOCH_NR)//EPOCH_NR)
    # shapes = [subarray.shape for subarray in mini_batches]
    # print(np.unique(shapes, return_counts=True), len(training_data))
    # print(shapes)
    # exit(1)
    # for i, batch in enumerate(mini_batches):
    for i, data in enumerate(loader_training):
      if i > 0:
        t_dataloader = time.perf_counter() - time_finish
      st_load = time.perf_counter()
      inputs, labels = data
      # inputs, labels = training_data.retrieve(batch)
      inputs, labels = inputs.cuda(), labels.cuda()
      time_to_gpu = time.perf_counter() - st_load

      # print(f"Time to put data to GPU: {(time.perf_counter() - st_load)*1e3:8.3f} ms")

      # zero the parameter gradients
      optimizer.zero_grad()
      voxnet = voxnet.train()
      outputs = voxnet(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      time_train = time.perf_counter() - st_load

      # Print accuracy, loss, etc.
      preds = outputs.data.max(1)[1]
      correct = preds.eq(labels.data).cpu().sum()
      acc = correct * 100. / BATCH_SIZE
      print(f"Batch {i}/{len(loader_training)}: Accuracy: {acc:6.2f} %; Loss: {loss.item():8.4f};")
      # print(f"Time to load: {time_load*1e3:8.2f} ms; Time to train: {time_train*1e3:8.2f} ms")

      if (i+1) % 50 == 0:
        # Check the accuracy on the test set
        st = time.perf_counter()
        test_data, test_labels = next(loader_test.__iter__())
        test_data, test_labels = test_data.cuda(), test_labels.cuda()
        voxnet = voxnet.eval()
        test_outputs = voxnet(test_data)
        test_preds = test_outputs.data.max(1)[1]
        test_correct = test_preds.eq(test_labels.data).cpu().sum()
        test_acc = test_correct * 100. / BATCH_SIZE
        loss = criterion(test_outputs, test_labels)
        print(f"Test set Accuracy: {test_acc:6.2f} %; Loss: {loss.item():8.4f}; Time: {(time.perf_counter() - st)*1e3:8.2f} ms")

      # if i > 0:
      #   t_lastbatch = time.perf_counter() - time_finish
      #   ##### time_to_gpu # t_dataloader # time_train
      #   print(f"Batch {i:4d}/{len(loader_training)} takes {t_lastbatch*1e3:8.3f} ms; Dataloader: {t_dataloader*1e3:8.3f} ms; Data to GPU: {time_to_gpu*1e3:8.3f} ms; Train: {time_train*1e3:8.3f} ms")

      time_finish = time.perf_counter()
    # scheduler.step()
    # Save the model etc...
    print("^" * 80)
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_coord_{epoch}.pth")




  print('Finished Training')



