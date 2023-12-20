import os, time, random, argparse

from tqdm import tqdm
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from pointnet.model import PointNetCls, feature_transform_regularizer

from feater import io, constants

def confusion_matrix(predictions, labels, output_file="confusion_matrix.png"):
  """
  Compute the confusion matrix for the predicted labels and the ground truth labels
  Args:
    pred: The predicted labels
    label: The ground truth labels
  Returns:
    Save to figure file
  """
  nr_classes = 20
  conf_mat = np.zeros((nr_classes, nr_classes), dtype=np.int32)
  for pred, label in zip(predictions, labels):
    conf_mat[pred, label] += 1

  # Compute the accuracies
  percent_acc = np.sum(np.array(predictions) == np.array(labels)) / len(predictions)
  accuracies = np.zeros(nr_classes, dtype=np.float32)
  for i in range(nr_classes):
    accuracies[i] = conf_mat[i, i] / np.sum(conf_mat[:, i])
  mean_acc = np.mean(accuracies)

  conf_mat = conf_mat / np.sum(conf_mat, axis=0, keepdims=True)
  conf_mat = conf_mat * 100

  # Plot the confusion matrix
  fig, ax  = plt.subplots(figsize=(16, 16))
  im = ax.imshow(conf_mat, cmap="inferno", vmin=0, vmax=100)
  ax.set_title(f"Confusion Matrix of the PointNet Classifier\nOvervall Accuracy: {percent_acc:.3f}, Mean Accuracy: {mean_acc:.3f}", fontsize=20)
  ax.set_xlabel("Ground Truth", fontsize=20)
  ax.set_ylabel("Prediction", fontsize=20)
  tick_labels = [f"{constants.LAB2RES[i]} ({i})" for i in range(nr_classes)]
  ax.set_xticks(np.arange(nr_classes))
  ax.set_yticks(np.arange(nr_classes))
  ax.set_xticklabels(tick_labels, rotation=-45, fontsize=20)
  ax.set_yticklabels(tick_labels, fontsize=20)

  # Mark text on the confusion matrix
  for i in range(nr_classes):
    for j in range(nr_classes):
      if conf_mat[i, j] > 10:
        ax.text(j,i, f"{int(conf_mat[i, j])}", ha="center", va="center", color="white" if conf_mat[i, j] < 50 else "black", fontsize=15, fontweight="bold")
  cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
  cbar.ax.tick_params(labelsize=20)
  plt.savefig(output_file, dpi=300, bbox_inches="tight")
  plt.close()


def evaluation(classifier, dataloader):
  """
  Evaluate the classifier on the validation set
  Args:
    classifier: The classifier
    dataloader: The dataloader for the validation set
  """
  classifier = classifier.eval()
  pred = []
  label = []
  for i, data in enumerate(dataloader):
    valid_data, valid_label = data
    valid_data = valid_data.transpose(2, 1)
    valid_data, valid_label = valid_data.cuda(), valid_label.cuda()
    pred_choice, _, _ = classifier(valid_data)
    pred_choice = pred_choice.data.max(1)[1]
    pred += pred_choice.cpu().tolist()
    label += valid_label.cpu().tolist()
  return pred, label


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

  BATCH_SIZE = 5000
  EPOCH_NR = 50
  verbose = False
  # LOAD_MODEL = "pointnet_coord_model.pth"
  OUTPUT_DIR = "/media/yzhang/MieT72/scripts_data/coord_results"
  LOAD_MODEL = None

  # Load the dataset TODO: Hardcoded training set
  st = time.perf_counter()
  # datafiles = "/media/yzhang/MieT72/Data/feater_database_coord/TestSet_ALA.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_ARG.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_ASN.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_ASP.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_CYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_GLN.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_GLU.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_GLY.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_HIS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_ILE.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_LEU.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_LYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_MET.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_PHE.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_PRO.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_SER.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_THR.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_TRP.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_TYR.h5%/media/yzhang/MieT72/Data/feater_database_coord/TestSet_VAL.h5"
  datafiles = "/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_ALA.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_ARG.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_ASN.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_ASP.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_CYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_GLN.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_GLU.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_GLY.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_HIS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_ILE.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_LEU.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_LYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_MET.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_PHE.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_PRO.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_SER.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_THR.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_TRP.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_TYR.h5%/media/yzhang/MieT72/Data/feater_database_coord/TrainingSet_VAL.h5"
  datafiles = datafiles.strip("%").split("%")
  training_data = CoordDataset(datafiles)
  dataloader_train = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  validfiles = "/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ALA.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ARG.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ASN.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ASP.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_CYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_GLN.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_GLU.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_GLY.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_HIS.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_ILE.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_LEU.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_LYS.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_MET.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_PHE.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_PRO.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_SER.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_THR.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_TRP.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_TYR.h5%/media/yzhang/MieT72/Data/feater_database_coord/ValidationSet_VAL.h5%"
  validfiles = validfiles.strip("%").split("%")
  valid_data = CoordDataset(validfiles)
  dataloader_valid = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

  print(f"Handled the {len(datafiles):3d} h5files: Time elapsed: {(time.perf_counter() - st)*1e3:8.2f} ms")
  print(f"The dataset has {len(training_data)} entries")

  classifier = PointNetCls(k=20, feature_transform=False)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    classifier.load_state_dict(torch.load(LOAD_MODEL))
  optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  classifier.cuda()

  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  # Check if the dataloader could successfully load the data
  for epoch in range(EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    for i, data in enumerate(dataloader_train):
      train_data, train_label = data
      train_data = train_data.transpose(2, 1)  # Move the coordinate 3 dim to channels of dimensions
      train_data, train_label = train_data.cuda(), train_label.cuda()

      if verbose:
        print(train_data.shape, train_label.shape)
        print(f"Retrieval of {i}/{len(dataloader_train)} took: {(time.perf_counter() - st) * 1e3:6.2f} ms; Batch size: {train_data.shape[0]}; Number of points: {train_data.shape};")
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
      print(f"Processing the block {i}/{len(dataloader_train)}; Loss: {loss.item():.4f}; Accuracy: {correct.item()/float(train_label.size()[0]):.4f}")

      if verbose:
        print_label_dist(train_label)

      if (i+1) % 50 == 0:
        valid_data, valid_label = next(dataloader_valid.__iter__())
        valid_data = valid_data.transpose(2, 1)
        valid_data, valid_label = valid_data.cuda(), valid_label.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(valid_data)
        loss = F.nll_loss(pred, valid_label)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(valid_label.data).cpu().sum()
        print(f"Validation: Loss: {loss.item():.4f}; Accuracy: {correct.item()/float(valid_label.size()[0]):.4f}")

    # Save the model
    print(f"Epock {epoch} takes {time.perf_counter() - st:.2f} seconds, saving the model and confusion matrix")
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_coord_{epoch}.pth")
    torch.save(classifier.state_dict(), modelfile_output)
    pred, label = evaluation(classifier, dataloader_valid)
    conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_confmatrix_{epoch}.png")
    confusion_matrix(pred, label, output_file=conf_mtx_output)


  conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), "pointnet_confmatrix_final.png")
  confusion_matrix(pred, label, output_file=conf_mtx_output)


