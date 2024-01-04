import os, sys, argparse, time, random, json

import numpy as np
import h5py as h5

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from feater.models.voxnet import VoxNet
from feater import io, dataloader, utils


def evaluation(classifier, dataset, usecuda=True, batch_size=256, process_nr=32):
  print("Start evaluation...")
  from torch import Tensor
  classifier = classifier.eval()
  pred = []
  label = []
  for i, data in enumerate(dataset.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
    valid_data, valid_label = data
    if usecuda:
      valid_data, valid_label = valid_data.cuda(), valid_label.cuda()
    ret = classifier(valid_data)
    if isinstance(ret, tuple):
      pred_choice = ret[0]
    elif isinstance(ret, Tensor):
      pred_choice = ret
    else:
      raise ValueError(f"Unexpected return type {type(ret)}")
    pred_choice = pred_choice.data.max(1)[1]
    pred += pred_choice.cpu().tolist()
    label += valid_label.cpu().tolist()
  return pred, label



def parse_args():
  parser = argparse.ArgumentParser(description="Train VoxNet")
  parser.add_argument("-train", "--training_data", type=str, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-valid", "--validation_data", type=str, help="The file writes all of the absolute path of h5 files of validation data set")
  parser.add_argument("-test", "--test_data", type=str, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, default="cls", help="Output folder")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("-itv", "--interval", type=int, default=5, help="How many batches to wait before logging training status")
  parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("--data_workers", type=int, default=4, help="Number of workers for data loading")
  parser.add_argument("--manualSeed", type=int, help="Manual seed")
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")

  args = parser.parse_args()

  args.cuda = torch.cuda.is_available()
  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
  # if (not args.training_data) or (not os.path.exists(args.training_data)):
  #   parser.print_help()
  #   raise ValueError(f"The training data file {args.training_data} does not exist.")
  # if (not args.validation_data) or (not os.path.exists(args.validation_data)):
  #   parser.print_help()
  #   raise ValueError(f"The validation data file {args.validation_data} does not exist.")
  # if (not args.test_data) or (not os.path.exists(args.test_data)):
  #   parser.print_help()
  #   raise ValueError(f"The test data file {args.test_data} does not exist.")
  # if (not args.output_folder) or (not os.path.exists(args.output_folder)):
  #   parser.print_help()
  #   raise ValueError(f"The output folder {args.output_folder} does not exist.")
  return args

if __name__ == "__main__":
  args = parse_args()
  SETTINGS = json.dumps(vars(args), indent=2)
  print("Settings of this training:")
  print(SETTINGS)

  random.seed(args.seed)
  torch.manual_seed(args.seed)
  BATCH_SIZE = args.batch_size
  START_EPOCH = args.start_epoch
  EPOCH_NR = args.epochs
  LEARNING_RATE = args.learning_rate

  TRAIN_DATA = os.path.abspath(args.training_data)
  VALID_DATA = os.path.abspath(args.validation_data)
  TEST_DATA = os.path.abspath(args.test_data)
  OUTPUT_DIR = os.path.abspath(args.output_folder)

  INTERVAL = args.interval
  WORKER_NR = args.data_workers
  # VERBOSE = True if args.verbose > 0 else False
  LOAD_MODEL = args.pretrained
  USECUDA = True if args.cuda else False

  st = time.perf_counter()
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  trainingfiles = utils.checkfiles(TRAIN_DATA)
  training_data = dataloader.VoxelDataset(trainingfiles)
  # loader_train = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)

  validfiles = utils.checkfiles(VALID_DATA)
  valid_data = dataloader.VoxelDataset(validfiles)
  loader_valid = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)


  print(f"Dataloader creation took: {(time.perf_counter() - st)*1e3 :8.2f} ms")

  # VoxNet model
  voxnet = VoxNet(n_classes=20)
  print(voxnet)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    voxnet.load_state_dict(torch.load(LOAD_MODEL))
  voxnet.cuda()

  optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
  criterion = nn.CrossEntropyLoss()

  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

  for epoch in range(START_EPOCH, EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    for i, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      if i > 0:
        t_dataloader = time.perf_counter() - time_finish
      st_load = time.perf_counter()
      inputs, labels = batch
      inputs, labels = inputs.cuda(), labels.cuda()
      time_to_gpu = time.perf_counter() - st_load

      # zero the parameter gradients
      optimizer.zero_grad()
      voxnet = voxnet.train()
      outputs = voxnet(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      time_train = time.perf_counter() - st_load

      # Print accuracy, loss, etc.
      preds = outputs.data.max(1)[1]
      correct = preds.eq(labels.data).cpu().sum()
      acc = correct * 100. / BATCH_SIZE
      print(f"Batch {i}/{len(training_data)//BATCH_SIZE}: Accuracy: {acc:6.2f} %; Loss: {loss.item():8.4f};")

      if (i+1) % 50 == 0:
        # Check the accuracy on the test set
        st = time.perf_counter()
        test_data, test_labels = next(loader_valid.__iter__())
        test_data, test_labels = test_data.cuda(), test_labels.cuda()
        voxnet = voxnet.eval()
        test_outputs = voxnet(test_data)
        test_preds = test_outputs.data.max(1)[1]
        test_correct = test_preds.eq(test_labels.data).cpu().sum()
        test_acc = test_correct * 100.0 / BATCH_SIZE
        loss = criterion(test_outputs, test_labels)
        print(f"Test set Accuracy: {test_acc:6.2f} %; Loss: {loss.item():8.4f}; Time: {(time.perf_counter() - st)*1e3:8.2f} ms")

      time_finish = time.perf_counter()
    scheduler.step()
    # Save the model etc...
    print("^" * 80)
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"voxnet_{epoch}.pth")
    torch.save(voxnet.state_dict(), modelfile_output)
    pred, label = evaluation(voxnet, valid_data)
    conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"voxnet_confmatrix_{epoch}.png")
    utils.confusion_matrix(pred, label, output_file=conf_mtx_output)

  print('Finished Training')
  test_data = utils.checkfiles(TEST_DATA)
  test_data = dataloader.VoxelDataset(test_data)
  dataloader_test = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)
  pred, label = evaluation(voxnet, valid_data)
  print(f"Final accuracy: {np.sum(np.array(pred) == np.array(label)) / len(pred):.4f}")
  conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), "voxnet_confmatrix_final.png")
  utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


