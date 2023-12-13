import os, time, random, argparse

from tqdm import tqdm
import numpy as np
import h5py as h5

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from pointnet.model import PointNetCls, feature_transform_regularizer

from feater import io, constants


class FEaterDataset(data.Dataset):
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



def parse_args():
  parser = argparse.ArgumentParser(description="Train PointNet on ModelNet40")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
  parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
  parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")

  parser.add_argument("--no_cuda", action="store_true", default=False, help="Disable CUDA")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)")
  parser.add_argument("--log_interval", type=int, default=10, metavar="N", help="How many batches to wait before logging training status")
  parser.add_argument("--dataset", type=str, default="modelnet40", help="Dataset to train on, [modelnet40, shapenet]")
  parser.add_argument("--dataset_root", type=str, default="./data", help="Dataset root dir")
  parser.add_argument("--outf", type=str, default="cls", help="Output folder")
  parser.add_argument("--feature_transform", action="store_true", default=False, help="Use feature transform")
  parser.add_argument("--n_points", type=int, default=1024, help="Num of points to use")
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--normal", action="store_true", default=False, help="Whether to use normal information")

  # TODO: Argument from my own definition
  parser.add_argument("--manualSeed", type=int, help="Manual seed")
  parser.add_argument("--load_model", type=str, default="", help="The model to load")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args


def print_label_dist(train_label):
  """
  Print the label distribution of the training data
  Args:
    train_label: The labels of the training data
  """
  uniq = np.unique([constants.LAB2RES[int(l)] for l in train_label], return_counts=True)
  for res in uniq[0]:
    print(f"{res:^4s}", end=" | ")
  print("\n", end="")
  for count in uniq[1]:
    print(f"{count:^4d}", end=" | ")
  print("\n", end="")


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
  batchsize = 5000

  # Load the dataset
  datafiles = [
    "/media/yzhang/MieT72/Data/feater_test4/ResALA.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResARG.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResASN.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResASP.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResCYS.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResGLN.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResGLU.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResGLY.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResHIS.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResILE.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResLEU.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResLYS.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResMET.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResPHE.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResPRO.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResSER.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResTHR.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResTRP.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResTYR.h5",
    "/media/yzhang/MieT72/Data/feater_test4/ResVAL.h5",
  ]

  st = time.perf_counter()
  feater_data = FEaterDataset(datafiles)

  print(f"Handled the {len(datafiles):3d} h5files: Time elapsed: {(time.perf_counter() - st)*1e3:8.2f} ms")
  print(f"The dataset has {len(feater_data)} entries")


  verbose = False

  if verbose:
    check_num = 15
    choices = np.random.choice(len(feater_data), check_num, replace=False)
    for i in choices:
      st = time.perf_counter()
      t, l = feater_data[i]
      print(f"Retrieval of {i} took: {(time.perf_counter() - st)*1e3:6.2f} ms; Number of points: {t.shape[0]}; Residue: {constants.LAB2RES[int(l)]}")

  # Load the dataset
  dataloader = data.DataLoader(feater_data, batch_size=batchsize, shuffle=True, num_workers=4)

  classifier = PointNetCls(k=20, feature_transform=False)
  if len(args.load_model) > 0:
    classifier.load_state_dict(torch.load(args.load_model))
  optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  classifier.cuda()

  # Check if the dataloader could successfully load the data
  for i, data in enumerate(dataloader):
    train_data, train_label = data
    train_data = train_data.transpose(2, 1)  # Move the coordinate 3 dim to channels of dimensions
    train_data, train_label = train_data.cuda(), train_label.cuda()

    if verbose:
      print(train_data.shape, train_label.shape)
      print(f"Retrieval of {i}/{len(dataloader)} took: {(time.perf_counter() - st) * 1e3:6.2f} ms; Batch size: {train_data.shape[0]}; Number of points: {train_data.shape};")
      st = time.perf_counter()

    optimizer.zero_grad()
    classifier = classifier.train()
    pred, trans, trans_feat = classifier(train_data)
    loss = F.nll_loss(pred, train_label)
    if False:
      loss += feature_transform_regularizer(trans_feat) * 0.001
    loss.backward()
    optimizer.step()
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(train_label.data).cpu().sum()
    print(f"Processing the block {i}/{len(dataloader)}; Loss: {loss.item():.4f}; Accuracy: {correct.item()/float(train_label.size()[0]):.4f}")

    if verbose:
      print_label_dist(train_label)

    # print(f"Batch {i}: shape train data: {train_data.shape}, shape train label: {train_label.shape}")


