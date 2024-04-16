import os, time, random, argparse, json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from feater.models.pointnet import PointNetCls
from feater import io, dataloader, utils


def parse_args():
  parser = argparse.ArgumentParser(description="Train PointNet Classifier with the FEater coordinate dataset")
  parser = utils.standard_parser(parser)
  
  # Parameters related to the network architecture
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--n_points", type=int, default=27, help="Num of points to use")
  parser.add_argument("--dataset", type=str, default="single", help="Dataset to use")
  parser.add_argument("--date_type", type=str, default="coord", help="Data type to use")
  # TODO: Arguments to check
  # parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
  # parser.add_argument("--normal", action="store_true", default=False, help="Whether to use normal information")

  # AFTER (ONLY AFTER) adding all arguments, do the sanity check
  utils.parser_sanity_check(parser)
  args = parser.parse_args()
  args.cuda = torch.cuda.is_available()
  if args.manualSeed is None:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  if args.dataset == "single":
    args.class_nr = 20
  elif args.dataset == "dual":
    args.class_nr = 400
  else:
    raise ValueError(f"Unexpected dataset {args.dataset}; Only single (FEater-Single) and dual (FEater-Dual) are supported")
  
  args.training_data = os.path.abspath(args.training_data)
  args.test_data = os.path.abspath(args.test_data)
  args.output_folder = os.path.abspath(args.output_folder)
  return args


def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])

  st = time.perf_counter()
  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["date_type"] == "coord":
    training_data = dataloader.CoordDataset(trainingfiles, target_np=training_settings["n_points"])
    test_data = dataloader.CoordDataset(testfiles, target_np=training_settings["n_points"])
  elif training_settings["date_type"] == "surf":
    training_data = dataloader.SurfDataset(trainingfiles, target_np=training_settings["n_points"])
    test_data = dataloader.SurfDataset(testfiles, target_np=training_settings["n_points"])
    

  # Initialize the classifier
  classifier = PointNetCls(k=training_settings["class_nr"], feature_transform=False)
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    classifier.cuda()

  optimizer = optim.Adam(classifier.parameters(), lr=training_settings["learning_rate"], betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  WORKER_NR = training_settings["data_workers"]
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")

    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      train_data, train_label = batch
      train_data = train_data.transpose(2, 1)  # Important: Move the coordinate's 3 dim as channels of the data
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()

      optimizer.zero_grad()
      classifier = classifier.train()
      pred, _, _ = classifier(train_data)
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

    _loss_test_cache = []
    _loss_train_cache = []
    _accuracy_test_cache = []
    _accuracy_train_cache = []
    with torch.no_grad(): 
      for (trdata, trlabel) in training_data.mini_batches(batch_size=1000, process_nr=WORKER_NR):
        trdata = trdata.transpose(2, 1)
        if USECUDA:
          trdata, trlabel = trdata.cuda(), trlabel.cuda()
        pred, _, _ = classifier(trdata)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(trlabel.data).cpu().sum()
        tr_accuracy = correct.item() / float(trlabel.size()[0])
        tr_loss = criterion(pred, trlabel)
        _loss_train_cache.append(tr_loss.item())
        _accuracy_train_cache.append(tr_accuracy)
        if len(_loss_train_cache) == 8:
          break

      for (tedata, telabel) in test_data.mini_batches(batch_size=1000, process_nr=WORKER_NR):
        tedata = tedata.transpose(2, 1)
        if USECUDA:
          tedata, telabel = tedata.cuda(), telabel.cuda()
        pred, _, _ = classifier(tedata)
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
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"pointnet_coord_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    message = f" Training of the epoch {epoch:>4d}/{EPOCH_NR:<4d} took {time.perf_counter() - st:6.2f} seconds "
    
    with io.hdffile(os.path.join(training_settings["output_folder"], "performance.h5") , "a") as hdffile:
      utils.update_hdf_by_slice(hdffile, "loss_train", np.array([np.mean(_loss_train_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "loss_test", np.array([np.mean(_loss_test_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_train", np.array([np.mean(_accuracy_train_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_test", np.array([np.mean(_accuracy_test_cache)], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))

if __name__ == "__main__":
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)

  print("Settings of this training:")
  print(_SETTINGS)
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  perform_training(SETTINGS)
