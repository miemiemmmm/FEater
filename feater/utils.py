import os
from numpy import ndarray, unique
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


def h5files_to_dataloader(filelist:list):
  pass


def report_accuracy(pred, label, verbose=True):
  pred_choice = pred.data.max(1)[1]
  correct = pred_choice.eq(label.data).cpu().sum()
  accuracy = correct.item() / float(label.size()[0])
  if verbose:
    print(f"Accuracy: {accuracy}")
  return accuracy


def confusion_matrix(predictions, labels, output_file="confusion_matrix.png"):
  """
  Compute the confusion matrix for the predicted labels and the ground truth labels
  Args:
    pred: The predicted labels
    label: The ground truth labels
  Returns:
    Save to figure file
  """
  import numpy as np
  import matplotlib.pyplot as plt
  from . import constants

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






def label_counts(train_label):
  """
  Print the label distribution of the training data
  Args:
    train_label: The labels of the training data
  """
  uniq = unique([constants.LAB2RES[int(l)] for l in train_label], return_counts=True)
  for res in uniq[0]:
    print(f"{res:^4s}", end=" | ")
  print("\n", end="")
  for count in uniq[1]:
    print(f"{count:^4d}", end=" | ")
  print("\n", end="")


