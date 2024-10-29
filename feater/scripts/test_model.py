"""
Perform benchmarking on the pre-trained models. 

Example: 

python3 /MieT5/MyRepos/FEater/feater/scripts/train_models.py --model convnext --optimizer adam --loss-function crossentropy \
  --training-data /Matter/feater_train_1000/dual_hilbert.txt --test-data /Weiss/FEater_Dual_HILB/te.txt --output_folder /Weiss/benchmark_models/convnext_dual_hilb/ \
  --test-number 4000 -e 120 -b 64 -w 12 --lr-init 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 --data-type dual --cuda 1 --dataloader-type hilb --production 0
"""

import os, sys, time, io
import argparse, random, json 

import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import transformers

from torch.utils.tensorboard import SummaryWriter

# Import models 
from feater.models.pointnet import PointNetCls      
from feater import dataloader, utils
import feater


sys.path.append("/MieT5/MyRepos/FEater/feater/scripts/")
import train_models

tensorboard_writer = None 

# For point cloud type of data, the input is in the shape of (B, 3, N)
INPUT_POINTS = 0
def update_pointnr(pointnr):
  global INPUT_POINTS
  INPUT_POINTS = pointnr
DATALOADER_TYPE = ""
def update_dataloader_type(dataloader_type):
  global DATALOADER_TYPE
  DATALOADER_TYPE = dataloader_type

OPTIMIZERS = {
  "adam": optim.Adam, 
  "sgd": optim.SGD,
  "adamw": optim.AdamW,
}

LOSS_FUNCTIONS = {
  "crossentropy": nn.CrossEntropyLoss,
}

DATALOADER_TYPES = {
  "pointnet": dataloader.CoordDataset,
  "pointnet2": dataloader.CoordDataset,
  "dgcnn": dataloader.CoordDataset,
  "paconv": dataloader.CoordDataset,

  
  "voxnet": dataloader.VoxelDataset,
  "deeprank": dataloader.VoxelDataset,
  "gnina": dataloader.VoxelDataset,

  
  "resnet": dataloader.HilbertCurveDataset,
  "convnext": dataloader.HilbertCurveDataset,
  "convnext_iso": dataloader.HilbertCurveDataset,
  "swintrans": dataloader.HilbertCurveDataset,
  "ViT": dataloader.HilbertCurveDataset,

  "coord": dataloader.CoordDataset, 
  "surface": dataloader.SurfDataset, 
}


def match_data(pred, label):  
  predicted_labels = torch.argmax(pred, dim=1)
  plt.scatter(predicted_labels.cpu().detach().numpy(), label.cpu().detach().numpy(), s=4, c = np.arange(len(label))/len(label), cmap="inferno")
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Predicted vs. Actual")
  buf = io.BytesIO()
  plt.savefig(buf, format="png")
  buf.seek(0)
  Image.open(buf)
  image_tensor = torchvision.transforms.ToTensor()(Image.open(buf))
  plt.clf()
  buf.close()
  return image_tensor


def parse_args():
  parser = argparse.ArgumentParser(description="Train the models used in the FEater paper. ")
  # Input data files 
  parser.add_argument("-train", "--training-data", type=str, default=None, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test",  "--test-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-p",     "--pretrained", type=str, required=True, help="Pretrained model path")
  parser.add_argument("-m",     "--meta-information", type=str, required=True, help="The meta information file for the pretrained model")
  parser.add_argument("-o",     "--output-file", type=str, required=True, help="The output folder to store the model and performance data")
  parser.add_argument("-w",     "--data-workers", type=int, default=12, help="Number of workers for data loading")
  
  # Miscellanenous
  parser.add_argument("--test-number", type=int, default=100_000_000, help="Number of test samples")
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("-s", "--manualSeed", type=int, help="Manually set seed")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--production", type=int, default=0, help="Production mode")


  args = parser.parse_args()  
  # if not os.path.exists(args.training_data): 
  #   raise ValueError(f"The training data file {args.training_data} does not exist.")
  if not os.path.exists(args.test_data):
    raise ValueError(f"The test data file {args.test_data} does not exist.")
  
  if not os.path.exists(args.meta_information):
    raise ValueError(f"The meta data file {args.meta_information} does not exist.")

  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
  
  return args


def perform_testing(training_settings: dict): 
  USECUDA = training_settings["cuda"]
  MODEL_TYPE = training_settings["model"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  ###################
  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["dataloader_type"] in ("surface", "coord"):
    if training_settings["data_type"] == "modelnet": 
      print("Using the ModelNet dataset", trainingfiles, testfiles) 
      training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"], scale=True)
      test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"], scale=True)
    else: 
      training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"])
      test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"])
    
  elif MODEL_TYPE == "pointnet": 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles, target_np=training_settings["pointnet_points"])
  else: 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles)
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles)
  print(f"Training data size: {len(training_data)}; Test data size: {len(test_data)}; Batch size: {BATCH_SIZE}; Worker number: {WORKER_NR}")
  print(">>>>", training_data, training_data.do_scaling, test_data.do_scaling)

  ###################
  # Load the model
  if MODEL_TYPE == "vanillampnn": 
    import train_gnns
    classifier = train_gnns.get_model(MODEL_TYPE, training_settings["class_nr"])
  else: 
    classifier = train_models.get_model(MODEL_TYPE, training_settings["class_nr"], dataloader_type=training_settings["dataloader_type"], data_type=training_settings["data_type"], target_np=training_settings["pointnet_points"])
  print(f"Classifier: {classifier}")

  ###################
  # Load the pretrained model
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    print(f"Loading the pretrained model from {training_settings['pretrained']}")
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  else: 
    raise ValueError(f"Unexpected pretrained model {training_settings['pretrained']}")
  if USECUDA:
    classifier.cuda()

  # The loss function in the original training
  criterion = LOSS_FUNCTIONS.get(training_settings["loss_function"], nn.CrossEntropyLoss)()
  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  ####################
  test_number = training_settings["test_number"]
  if MODEL_TYPE == "vanillampnn":
    print(f"Testing {MODEL_TYPE} with training dataset containing {min(test_number, len(training_data))} samples ...")
    loss_on_train, accuracy_on_train = train_gnns.test_model(classifier, training_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR) 
    print(f"Testing {MODEL_TYPE} with test dataset containing {min(test_number, len(test_data))} samples ...")
    loss_on_test, accuracy_on_test = train_gnns.test_model(classifier, test_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR) 
  else: 
    print(f"Testing {MODEL_TYPE} with training dataset containing {min(test_number, len(training_data))} samples ...")
    loss_on_train, accuracy_on_train = train_models.test_model(classifier, training_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR) 
    print(f"Testing {MODEL_TYPE} with test dataset containing {min(test_number, len(test_data))} samples ...")
    loss_on_test, accuracy_on_test = train_models.test_model(classifier, test_data, criterion, test_number, BATCH_SIZE, USECUDA, WORKER_NR) 

  print(f"Checking the Performance on Loss: {loss_on_test}/{loss_on_train}; Accuracy: {accuracy_on_test}/{accuracy_on_train} at {time.ctime()}")
  return loss_on_train, accuracy_on_train, loss_on_test, accuracy_on_test

def get_predictions(training_settings: dict): 
  """
  Get the predictions from the pretrained model.

  Notes
  -----
  The prediction needs: 
  - **pretrained**(required): The path to the pretrained model
  - **test_data**(optional): The path to the test data to test the model

  The output will be automatically saved in the output folder with the name of the {pretrained_model_prefix}_result_array.h5 
  with keys including predicted_train, predicted_test, ground_truth_train, ground_truth_test.
  """
  USECUDA = training_settings["cuda"]
  MODEL_TYPE = training_settings["model"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  ###################
  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["dataloader_type"] in ("surface", "coord"):
    training_data = DATALOADER_TYPES[training_settings["dataloader_type"]](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[training_settings["dataloader_type"]](testfiles, target_np=training_settings["pointnet_points"])
  elif MODEL_TYPE == "pointnet": 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles, target_np=training_settings["pointnet_points"])
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles, target_np=training_settings["pointnet_points"])
  else: 
    training_data = DATALOADER_TYPES[MODEL_TYPE](trainingfiles)
    test_data = DATALOADER_TYPES[MODEL_TYPE](testfiles)
  print(f"Training data size: {len(training_data)}; Test data size: {len(test_data)}; Batch size: {BATCH_SIZE}; Worker number: {WORKER_NR}")

  ###################
  # Load the model
  classifier = train_models.get_model(MODEL_TYPE, training_settings["class_nr"])
  print(f"Classifier: {classifier}")

  ###################
  # Load the pretrained model
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  else: 
    raise ValueError(f"Unexpected pretrained model {training_settings['pretrained']}")
  if USECUDA:
    classifier.cuda()
  
  results_train = np.full(len(training_data), -1, dtype=np.int32)
  ground_truth_train = np.full(len(training_data), -1, dtype=np.int32)
  results_test = np.full(len(test_data), -1, dtype=np.int32)
  ground_truth_test = np.full(len(test_data), -1, dtype=np.int32)
  with torch.no_grad():
    classifier.eval()
    c = 0
    print("Getting the predictions from the training dataset")
    for data, target in training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if isinstance(training_data, dataloader.CoordDataset) or isinstance(training_data, dataloader.SurfDataset):
        data = data.transpose(2, 1)  
      if USECUDA:
        data, target = data.cuda(), target.cuda()
      pred = classifier(data)
      if isinstance(classifier, PointNetCls) or isinstance(pred, tuple):
        pred = pred[0]
      elif hasattr(pred, "logits"): 
        # Get the logit if the huggingface models is used
        pred = pred.logits
      pred_choice = torch.argmax(pred, dim=1)
      elem_nr = len(pred)
      results_train[c:c+elem_nr] = pred_choice.cpu().detach().numpy()
      ground_truth_train[c:c+elem_nr] = target.cpu().detach().numpy()
      c += elem_nr
    c = 0
    print("Getting the predictions from the test dataset")
    for data, target in test_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR):
      if isinstance(test_data, dataloader.CoordDataset) or isinstance(test_data, dataloader.SurfDataset):
        data = data.transpose(2, 1)  
      if USECUDA:
        data, target = data.cuda(), target.cuda()
      pred = classifier(data)
      if isinstance(classifier, PointNetCls) or isinstance(pred, tuple):
        pred = pred[0]
      elif hasattr(pred, "logits"): 
        # Get the logit if the huggingface models is used
        pred = pred.logits
      pred_choice = torch.argmax(pred, dim=1)
      elem_nr = len(pred_choice)
      results_test[c:c+elem_nr] = pred_choice.cpu().detach().numpy()
      ground_truth_test[c:c+elem_nr] = target.cpu().detach().numpy()

      c += elem_nr
  
  if -1 in results_train or -1 in results_test:
    raise ValueError("Unexpected error in the prediction")
  elif -1 in ground_truth_train or -1 in ground_truth_test:
    raise ValueError("Unexpected error in the ground truth")

  outfolder = training_settings["output_folder"]
  pretrained_model_file = training_settings["pretrained"]
  filename = os.path.basename(pretrained_model_file)
  if "result_predictions" in training_settings:
    outfilename = training_settings["result_predictions"]
  else:
    outfilename = filename.replace(".pth", "_results.h5")
    outfilename = os.path.join(outfolder, outfilename)
  print(f"Saving the results in {outfilename}")
  with h5.File(outfilename, "w") as f:
    f.create_dataset("predicted_train", data=results_train, dtype=np.int32, chunks=True)
    f.create_dataset("predicted_test", data=results_test, dtype=np.int32, chunks=True)
    f.create_dataset("ground_truth_train", data=ground_truth_train, dtype=np.int32, chunks=True)
    f.create_dataset("ground_truth_test", data=ground_truth_test, dtype=np.int32, chunks=True)
  

if __name__ == "__main__":
  """
  Test the model with 
  """

  args = parse_args()
  SETTINGS = vars(args)
  print("Settings of this training:")

  # Read the input meta-information 
  with open(SETTINGS["meta_information"], "r") as f:
    meta_information = json.load(f)
    del meta_information["test_data"]
    del meta_information["pretrained"]
    del meta_information["test_number"]
    if SETTINGS["training_data"] != None:
      del meta_information["training_data"]
    update_pointnr(meta_information["pointnet_points"])
    update_dataloader_type(meta_information["dataloader_type"])

  SETTINGS.update(meta_information)  # Update the settings with the requested meta-information 
  print(json.dumps(SETTINGS, indent=2))
  
  loss_on_train, accuracy_on_train, loss_on_test, accuracy_on_test = perform_testing(SETTINGS)

  # Find matched model file and set the corresponding output column
  print(f"Updating the performance data in {SETTINGS['output_file']}, with the pretrained model {SETTINGS['pretrained']}")
  df = pd.read_csv(SETTINGS["output_file"], index_col=None)
  df.loc[(df["param_path"] == SETTINGS["pretrained"]) * (df["testdata_path"] == SETTINGS["test_data"]), "acc_test"] = float(accuracy_on_test)
  df.loc[(df["param_path"] == SETTINGS["pretrained"]) * (df["testdata_path"] == SETTINGS["test_data"]), "loss_test"] = float(loss_on_test)
  df.loc[(df["param_path"] == SETTINGS["pretrained"]) * (df["testdata_path"] == SETTINGS["test_data"]), "acc_train"] = float(accuracy_on_train)
  df.loc[(df["param_path"] == SETTINGS["pretrained"]) * (df["testdata_path"] == SETTINGS["test_data"]), "loss_train"] = float(loss_on_train)
  df.to_csv(SETTINGS["output_file"], index=False) 


