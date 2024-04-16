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
  testfiles = utils.checkfiles(training_settings["test_data"])
  training_data = dataloader.HilbertCurveDataset(trainingfiles)
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
      train_data, train_label = batch
      
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      
      pred = classifier(train_data)

      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

    _loss_test_cache = []
    _loss_train_cache = []
    _accuracy_test_cache = []
    _accuracy_train_cache = []
    with torch.no_grad(): 
      # Use 8000 samples for the training and test data
      for (trdata, trlabel) in training_data.mini_batches(batch_size=1000, process_nr=WORKER_NR):
        if USECUDA:
          trdata, trlabel = trdata.cuda(), trlabel.cuda()
        pred = classifier(trdata)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(trlabel.data).cpu().sum()
        tr_accuracy = correct.item() / float(trlabel.size()[0])
        tr_loss = criterion(pred, trlabel)
        _loss_train_cache.append(tr_loss.item())
        _accuracy_train_cache.append(tr_accuracy)
        if len(_loss_train_cache) == 8:
          break
        
      for (tedata, telabel) in test_data.mini_batches(batch_size=1000, process_nr=WORKER_NR):
        if USECUDA:
          tedata, telabel = tedata.cuda(), telabel.cuda()
        pred = classifier(tedata)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(telabel.data).cpu().sum()
        te_accuracy = correct.item() / float(telabel.size()[0])
        te_loss = criterion(pred, telabel)
        _loss_test_cache.append(te_loss.item())
        _accuracy_test_cache.append(te_accuracy)
        if len(_loss_test_cache) == 8:
          break

    scheduler.step()

    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"resnet_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    
    with io.hdffile(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      utils.update_hdf_by_slice(hdffile, "loss_train", np.array([np.mean(_loss_train_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "loss_test", np.array([np.mean(_loss_test_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_train", np.array([np.mean(_accuracy_train_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_test", np.array([np.mean(_accuracy_test_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))


if __name__ == '__main__':
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(vars(args), indent=2)
  print("Settings of this training:")
  print(_SETTINGS)

  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)
  
  perform_training(SETTINGS)



