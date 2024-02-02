import argparse, time, json, os, random, subprocess

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models
from feater import io, utils, dataloader


def get_resnet_model(resnettype: str, class_nr:int):
  FCFeatureNumberMap = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048,  "resnet152": 2048}
  resnettype = resnettype.lower()
  if resnettype == "resnet18":
    model = torchvision.models.resnet18()
  elif resnettype == "resnet34":
    model = torchvision.models.resnet34()
  elif resnettype == "resnet50":
    model = torchvision.models.resnet50()
  elif resnettype == "resnet101":
    model = torchvision.models.resnet101()
  elif resnettype == "resnet152":
    model = torchvision.models.resnet152()
  else: 
    raise ValueError(f"Unsupported ResNet type: {resnettype}")
  fc_number = FCFeatureNumberMap.get(resnettype, 2048)
  model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
  model.fc = nn.Linear(fc_number, class_nr)
  return model


def parse_args():
  parser = argparse.ArgumentParser(description="Train the ResNet model based on the 2D Hilbert map")
  parser = utils.standard_parser(parser)
  
  # Parameters specifically related to the network architecture, this training or script
  parser.add_argument("--resnet_type", type=str, default="resnet18", help="ResNet type")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--dataset", type=str, default="single", help="Dataset to use")

  # AFTER (ONLY AFTER) adding all arguments, do the sanity check
  utils.parser_sanity_check(parser)
  args = parser.parse_args()
  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
  return args


def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  st = time.perf_counter()
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  validfiles = utils.checkfiles(training_settings["validation_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  training_data = dataloader.HilbertCurveDataset(trainingfiles)
  valid_data = dataloader.HilbertCurveDataset(validfiles)
  test_data = dataloader.HilbertCurveDataset(testfiles)

  if training_settings["dataset"] == "single":
    class_nr = 20
  elif training_settings["dataset"] == "dual":
    class_nr = 400

  classifier = get_resnet_model(training_settings["resnet_type"], class_nr)
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    classifier.cuda()

  optimizer = optim.Adam(classifier.parameters(), lr=training_settings["learning_rate"], betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss(label_smoothing=0.5)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.4, verbose = True)

  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")
  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  INTERVAL = training_settings["interval"]
  WORKER_NR = training_settings["data_workers"]
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      # if (batch_idx+1) % 500 == 0:   # TODO: Remove this line when production
      #   break
      train_data, train_label = batch
      
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      
      pred = classifier(train_data)

      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

      if batch_idx % 50 == 0:
        accuracy = utils.report_accuracy(pred, train_label, verbose=False)
        print(f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {loss.item():8.4f}; Accuracy: {accuracy:8.4f}")

      if (batch_idx + 1) % INTERVAL == 0:
        if not os.path.exists(os.path.join(training_settings["output_folder"], "tmp_figs")):
          subprocess.call(["mkdir", "-p", os.path.join(training_settings["output_folder"], "tmp_figs")])
        vdata, vlabel = next(valid_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
        preds, labels = utils.validation(classifier, vdata, vlabel, usecuda=USECUDA)
        print(f"Estimated epoch time: {(time.perf_counter() - st) / (batch_idx + 1) * batch_nr:.2f} seconds")
        if training_settings["dataset"] == "single":
          print("#"*100)
          print("Preds:")
          utils.label_counts(preds)
          print("Labels:")
          utils.label_counts(labels)
          print("#"*100)
          utils.confusion_matrix(preds, labels, output_file=os.path.join(os.path.abspath(training_settings["output_folder"]), f"tmp_figs/idx_{epoch}_{batch_idx}.png"))
    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"resnet_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(training_settings["output_folder"]), f"resnet_confmatrix_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    print(f"Performing the prediction on the test set ...")
    tdata, tlabel = next(test_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
    pred, label = utils.validation(classifier, tdata, tlabel, usecuda=USECUDA)
    with io.hdffile(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      correct = np.count_nonzero(pred == label)
      accuracy = correct / float(label.shape[0])
      utils.update_hdf_by_slice(hdffile, "accuracy", np.array([accuracy], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
    
    if training_settings["dataset"] == "single":
      print(f"Saving the confusion matrix to {conf_mtx_output} ...")
      utils.confusion_matrix(pred, label, output_file=conf_mtx_output)
    else: 
      pass
    scheduler.step()

if __name__ == '__main__':
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(vars(args), indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)
  
  perform_training(SETTINGS)



