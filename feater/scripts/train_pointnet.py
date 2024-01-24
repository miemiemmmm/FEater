import os, time, random, argparse, json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from feater.models.pointnet import PointNetCls
from feater import io, constants, dataloader, utils


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
  args.validation_data = os.path.abspath(args.validation_data)
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
  validfiles = utils.checkfiles(training_settings["validation_data"])
  testfiles = utils.checkfiles(training_settings["test_data"])
  if training_settings["date_type"] == "coord":
    training_data = dataloader.CoordDataset(trainingfiles, target_np=training_settings["n_points"])
    valid_data = dataloader.CoordDataset(validfiles, target_np=training_settings["n_points"])
    test_data = dataloader.CoordDataset(testfiles, target_np=training_settings["n_points"])
  elif training_settings["date_type"] == "surf":
    training_data = dataloader.SurfDataset(trainingfiles, target_np=training_settings["n_points"])
    valid_data = dataloader.SurfDataset(validfiles, target_np=training_settings["n_points"])
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
  INTERVAL = training_settings["interval"]
  WORKER_NR = training_settings["data_workers"]
  for epoch in range(START_EPOCH, EPOCH_NR):
    st = time.perf_counter()
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()} "
    print(f"{message:#^80}")
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      # if (batch_idx+1) % 300 == 0:   # TODO: Remove this line when production
      #   break
      train_data, train_label = batch
      train_data = train_data.transpose(2, 1)  # Important: Move the coordinate's 3 dim as channels of the data
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()

      optimizer.zero_grad()
      classifier = classifier.train()
      pred, trans, trans_feat = classifier(train_data)
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()
      accuracy = utils.report_accuracy(pred, train_label, verbose=False)
      print(f"Processing the block {batch_idx:>5d}/{batch_nr:<5d}; Loss: {loss.item():8.4f}; Accuracy: {accuracy:8.4f}")

      if (batch_idx + 1) % INTERVAL == 0:
        vdata, vlabel = next(valid_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
        vdata = vdata.transpose(2, 1)
        utils.validation(classifier, vdata, vlabel, usecuda=USECUDA)
        print(f"Estimated epoch time: {(time.perf_counter() - st) / (batch_idx + 1) * batch_nr:.2f} seconds")
    
    scheduler.step()
    
    # Save the model
    modelfile_output = os.path.join(os.path.abspath(training_settings["output_folder"]), f"pointnet_coord_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(training_settings["output_folder"]), f"pointnet_confmatrix_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    print(f"Performing the prediction on the test set ...")
    tdata, tlabel = next(test_data.mini_batches(batch_size=10000, process_nr=WORKER_NR))
    tdata = tdata.transpose(2, 1)
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
      # matrix = 
      # utils.plot_matrix()

    message = f" Training of the epoch {epoch:>4d}/{EPOCH_NR:<4d} took {time.perf_counter() - st:6.2f} seconds "
    print(f"{message:^^80}")


if __name__ == "__main__":
  args = parse_args()
  SETTINGS = vars(args)
  _SETTINGS = json.dumps(SETTINGS, indent=2)

  print("Settings of this training:")
  print(_SETTINGS)
  with open(os.path.join(SETTINGS["output_folder"], "settings.json"), "w") as f:
    f.write(_SETTINGS)

  perform_training(SETTINGS)


  # st = time.perf_counter()
  # trainingfiles = utils.checkfiles(TRAIN_DATA)
  # training_data = dataloader.CoordDataset(trainingfiles)
  # validfiles = utils.checkfiles(VALID_DATA)
  # valid_data = dataloader.CoordDataset(validfiles)
  # testfiles = utils.checkfiles(TEST_DATA)
  # test_data = dataloader.CoordDataset(testfiles)


  # classifier = PointNetCls(k=20, feature_transform=False)
  # if LOAD_MODEL and len(LOAD_MODEL) > 0:
  #   classifier.load_state_dict(torch.load(LOAD_MODEL))
  # optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  # criterion = nn.CrossEntropyLoss()
  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  # if USECUDA:
  #   classifier.cuda()

  # print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  # # Check if the dataloader could successfully load the data
  # for epoch in range(START_EPOCH, EPOCH_NR):
  #   print("#" * 80)
  #   print(f"Running the epoch {epoch}/{EPOCH_NR}")
  #   st = time.perf_counter()
  #   batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
  #   for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
  #     train_data, train_label = batch
  #     train_data = train_data.transpose(2, 1)  # Important: Move the coordinate's 3 dim as channels of the data
  #     if USECUDA:
  #       train_data, train_label = train_data.cuda(), train_label.cuda()

  #     optimizer.zero_grad()
  #     classifier = classifier.train()
  #     pred, trans, trans_feat = classifier(train_data)
  #     loss = criterion(pred, train_label)
  #     loss.backward()
  #     optimizer.step()
  #     accuracy = utils.report_accuracy(pred, train_label, verbose=False)
  #     print(f"Processing the block {batch_idx}/{batch_nr}; Loss: {loss.item():8.4f}; Accuracy: {accuracy:8.4f}")

  #     if VERBOSE:
  #       utils.label_counts(train_label)

  #     if ((batch_idx + 1) % INTERVAL) == 0:
  #       vdata, vlabel = next(valid_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR))
  #       vdata = vdata.transpose(2, 1)
  #       utils.validation(classifier, vdata, vlabel, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)

  #   scheduler.step()
  #   print(f"Epoch {epoch} takes {time.perf_counter() - st:.2f} seconds")
  #   print("^" * 80)
    
  #   # Save the model
  #   modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_coord_{epoch}.pth")
  #   conf_mtx_output  = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_confmatrix_{epoch}.png")
  #   print(f"Saving the model to {modelfile_output} ...")
  #   torch.save(classifier.state_dict(), modelfile_output)
  #   print(f"Performing the prediction on the test set ...")
  #   pred, label = evaluation(classifier, test_data)
  #   print(f"Saving the confusion matrix to {conf_mtx_output} ...")
  #   utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


