import os, sys, argparse, time, random, json

import numpy as np
import h5py as h5

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F


from feater.models.voxnet import VoxNet
from feater import io, dataloader, utils


def parse_args():
  parser = argparse.ArgumentParser(description="Train VoxNet")
  parser = utils.standard_parser(parser)

  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--dataset", type=str, default="single", help="Dataset to use")

  utils.parser_sanity_check(parser)
  args = parser.parse_args()  
  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  if args.dataset == "single":
    args.class_nr = 20
  elif args.dataset == "dual":
    args.class_nr = 400
  else:
    raise ValueError(f"Unexpected dataset {args.dataset}; Only single (FEater-Single) and dual (FEater-Dual) are supported")
  return args


def perform_training(training_settings: dict): 
  USECUDA = training_settings["cuda"]

  random.seed(training_settings["seed"])
  torch.manual_seed(training_settings["seed"])
  np.random.seed(training_settings["seed"])
  
  st = time.perf_counter()
  # Load the datasets
  trainingfiles = utils.checkfiles(training_settings["training_data"])
  # validfiles = utils.checkfiles(training_settings["validation_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  training_data = dataloader.VoxelDataset(trainingfiles)
  # valid_data = dataloader.VoxelDataset(validfiles)
  test_data = dataloader.VoxelDataset(testfiles)

  classifier = VoxNet(n_classes=training_settings["class_nr"])
  if training_settings["pretrained"] and len(training_settings["pretrained"]) > 0:
    classifier.load_state_dict(torch.load(training_settings["pretrained"]))
  if USECUDA:
    classifier.cuda()

  optimizer = optim.Adam(classifier.parameters(), lr=training_settings["learning_rate"], betas=(0.9, 0.999))
  # The loss function in the original training is tf.losses.softmax_cross_entropy
  # criterion = nn.BCEWithLogitsLoss()
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  
  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")
  START_EPOCH = training_settings["start_epoch"]
  EPOCH_NR = training_settings["epochs"]
  BATCH_SIZE = training_settings["batch_size"]
  INTERVAL = training_settings["interval"]
  WORKER_NR = training_settings["data_workers"]
  _loss_test_cache = []
  _loss_train_cache = []
  _accuracy_test_cache = []
  _accuracy_train_cache = []
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      train_data, train_label = batch
      # train_label = F.one_hot(train_label, num_classes=training_settings["class_nr"]).float()
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      pred = classifier(train_data)
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()

      # if (batch_idx + 1) % (batch_nr//5) == 0:
    _loss_test_cache = []
    _loss_train_cache = []
    _accuracy_test_cache = []
    _accuracy_train_cache = []
    with torch.no_grad(): 
      trdata, trlabel = next(training_data.mini_batches(batch_size=5000, process_nr=WORKER_NR))
      if USECUDA:
        trdata, trlabel = trdata.cuda(), trlabel.cuda()
      pred = classifier(trdata)
      # raise ValueError("Stop here")
      # _train_label = torch.argmax(train_label, dim=1)
      pred_choice = pred.data.max(1)[1]
      correct = pred_choice.eq(trlabel.data).cpu().sum()
      tr_accuracy = correct.item() / float(trlabel.size()[0])
      tr_loss = criterion(pred, trlabel)

      tedata, telabel = next(test_data.mini_batches(batch_size=5000, process_nr=WORKER_NR))
      if USECUDA:
        tedata, telabel = tedata.cuda(), telabel.cuda()
      pred = classifier(tedata)
      pred_choice = pred.data.max(1)[1]
      correct = pred_choice.eq(telabel.data).cpu().sum()
      te_accuracy = correct.item() / float(telabel.size()[0])
      te_loss = criterion(pred, telabel)
      print(f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {te_loss.item():8.4f}/{tr_loss.item():8.4f}; Accuracy: {te_accuracy:8.4f}/{tr_accuracy:8.4f}")
      _loss_train_cache.append(tr_loss.item())
      _loss_test_cache.append(te_loss.item())
      _accuracy_train_cache.append(tr_accuracy)
      _accuracy_test_cache.append(te_accuracy)
    
    scheduler.step()
    # Save the model  
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"VoxNet_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)

    # Save the performance to a HDF5 file
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


