import os, argparse

import numpy as np
from numpy import ndarray, unique
from torch import Tensor
import torch.cuda
import torch.nn.functional as F
# import matplotlib
# Use Agg backend for matplotlib for non-GUI environment
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import constants


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

def update_hdf_by_slice(hdffile, dataset_name:str, data:ndarray, hdf_slice, **kwargs):
  if dataset_name not in hdffile.keys():
      hdffile.create_dataset(dataset_name, data=data, **kwargs)
  else:
    if hdf_slice.stop > hdffile[dataset_name].shape[0]: 
      hdffile[dataset_name].resize(hdf_slice.stop, axis=0)
    hdffile[dataset_name][hdf_slice] = data


def report_accuracy(pred, label, verbose=True):
  pred_choice = pred.data.max(1)[1]
  correct = pred_choice.eq(label.data).cpu().sum()
  accuracy = correct.item() / float(label.size()[0])
  if verbose:
    print(f"Accuracy: {accuracy}")
  return accuracy

def validation(classifier, valid_data, valid_label, usecuda=True, batch_size=50):
  print(">>> Start validation...")
  classifier.eval()
  valid_data_batches = torch.split(valid_data, batch_size, dim=0)
  valid_label_batches = torch.split(valid_label, batch_size, dim=0)
  ret_pred = []
  ret_label = []
  with torch.no_grad():
    for val_data, val_label in zip(valid_data_batches, valid_label_batches):
      if usecuda:
        val_data, val_label = val_data.cuda(), val_label.cuda()
      ret = classifier(val_data)

      if isinstance(ret, Tensor):
        # Default model which returns the predicted results
        pred_choice = ret
      elif isinstance(ret, tuple):
        # Customized model which returns a tuple rather than the predicted results
        pred_choice = ret[0]
      else:
        raise ValueError(f"Unexpected return type {type(ret)}")
      loss = F.cross_entropy(pred_choice, val_label)
      accuracy = report_accuracy(pred_choice, val_label, verbose=False)
      print(f">>> Validation loss: {loss.item():8.4f} | Validation accuracy: {accuracy:8.4f}")

      pred_choice = pred_choice.data.max(1)[1]
      pred = pred_choice.cpu().tolist()
      label = val_label.cpu().tolist()
      ret_pred += pred
      ret_label += label
  ret_pred = np.array(ret_pred)
  ret_label = np.array(ret_label)
  return ret_pred, ret_label


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


# def plot_matrix(conf_mat, output_file="confusion_matrix.png"):
#   # Plot the confusion matrix
#   fig, ax  = plt.subplots(figsize=(16, 16))
#   im = ax.imshow(conf_mat, cmap="inferno", vmin=0, vmax=100)
#   ax.set_title(f"Confusion Matrix of the PointNet Classifier\nOvervall Accuracy: {percent_acc:.3f}, Mean Accuracy: {mean_acc:.3f}", fontsize=20)
#   ax.set_xlabel("Ground Truth", fontsize=20)
#   ax.set_ylabel("Prediction", fontsize=20)
#   tick_labels = [f"{constants.LAB2RES[i]} ({i})" for i in range(nr_classes)]
#   ax.set_xticks(np.arange(nr_classes))
#   ax.set_yticks(np.arange(nr_classes))
#   ax.set_xticklabels(tick_labels, rotation=-45, fontsize=20)
#   ax.set_yticklabels(tick_labels, fontsize=20)

#   # Mark text on the confusion matrix
#   for i in range(nr_classes):
#     for j in range(nr_classes):
#       if conf_mat[i, j] > 10:
#         ax.text(j,i, f"{int(conf_mat[i, j])}", ha="center", va="center", color="white" if conf_mat[i, j] < 50 else "black", fontsize=15, fontweight="bold")
#   cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
#   cbar.ax.tick_params(labelsize=20)
#   plt.savefig(output_file, dpi=300, bbox_inches="tight")
#   plt.close()


def label_counts(train_label):
  """
  Print the label distribution of the training data
  Args:
    train_label: The labels of the training data
  """
  uniq = unique([constants.LAB2RES[int(l)] for l in train_label], return_counts=True)
  for res in constants.RES: 
    print(f"{res:^4s}", end=" | ")
  print("\n", end="")
  for res in constants.RES: 
    if res in uniq[0]:
      print(f"{uniq[1][uniq[0] == res][0]:^4d}", end=" | ")
    else:
      print(f"{0:^4d}", end=" | ")
  print("\n", end="")


def standard_parser(parser: argparse.ArgumentParser):
  # Data files
  parser.add_argument("-train", "--training_data", type=str, help="The file writes all of the absolute path of h5 files of training data set")
  # parser.add_argument("-valid", "--validation_data", type=str, help="The file writes all of the absolute path of h5 files of validation data set")
  parser.add_argument("-test", "--test_data", type=str, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, default="cls", help="Output folder")
  
  # Pretrained model and break point restart
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")

  # Training parameters
  parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("-itv", "--interval", type=int, default=50, help="How many batches to wait before logging training status")

  # Other parameters
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("--data_workers", type=int, default=4, help="Number of workers for data loading")
  parser.add_argument("--manualSeed", type=int, help="Manually set seed")
  return parser


def parser_sanity_check(parser: argparse.ArgumentParser):
  args = parser.parse_args()
  if (not args.training_data) or (not os.path.exists(args.training_data)):
    parser.print_help()
    raise ValueError(f"The training data file {args.training_data} does not exist.")
  # if (not args.validation_data) or (not os.path.exists(args.validation_data)):
  #   parser.print_help()
  #   raise ValueError(f"The validation data file {args.validation_data} does not exist.")
  if (not args.test_data) or (not os.path.exists(args.test_data)):
    parser.print_help()
    raise ValueError(f"The test data file {args.test_data} does not exist.")
  if (not args.output_folder) or (not os.path.exists(args.output_folder)):
    parser.print_help()
    raise ValueError(f"The output folder {args.output_folder} does not exist.")
  print(">>> Passed the sanity check of the parser.")
  return args
