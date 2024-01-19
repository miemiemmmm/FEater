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
    # TODO: Remove this line
    if i == 50:
      break 
  return pred, label


def parse_args():
  parser = argparse.ArgumentParser(description="Train VoxNet")
  parser = utils.standard_parser(parser)

  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")

  utils.parser_sanity_check(parser)
  args = parser.parse_args()  
  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
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
  VERBOSE = True if args.verbose > 0 else False
  LOAD_MODEL = args.pretrained
  USECUDA = True if args.cuda else False

  with open(os.path.join(OUTPUT_DIR, "settings.json"), 'w') as f:
    f.write(SETTINGS)

  
  # Load the data.
  # NOTE: In Voxel based training, the bottleneck is the SSD data reading.
  # NOTE: Putting the dataset to SSD will significantly speed up the training.
  trainingfiles = utils.checkfiles(TRAIN_DATA)
  training_data = dataloader.VoxelDataset(trainingfiles)
  validfiles = utils.checkfiles(VALID_DATA)
  valid_data = dataloader.VoxelDataset(validfiles)
  testfiles = utils.checkfiles(TEST_DATA)
  test_data = dataloader.VoxelDataset(testfiles)

  # VoxNet model
  voxnet = VoxNet(n_classes=20)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    voxnet.load_state_dict(torch.load(LOAD_MODEL))
  voxnet.cuda()

  optimizer = optim.Adam(voxnet.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

  st = time.perf_counter()
  for epoch in range(START_EPOCH, EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      inputs, labels = batch
      if USECUDA:
        inputs, labels = inputs.cuda(), labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()
      voxnet = voxnet.train()
      outputs = voxnet(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      accuracy = utils.report_accuracy(outputs, labels, verbose=False)
      print(f"Processing Batch {batch_idx:>6d}/{batch_nr:<6d} | Loss: {loss.item():8.4f} | Accuracy: {accuracy:8.4f}")

      if (batch_idx + 1) % 50 == 0:
        # Check the accuracy on the validation set
        vdata, vlabel = next(valid_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR))
        utils.validation(voxnet, vdata, vlabel, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)

      # TODO: Remove this when production
      if (batch_idx + 1) == 2000:
        break
    scheduler.step()
    print(f"Epoch {epoch} takes {time.perf_counter() - st:.2f} seconds")
    print("^" * 80)

    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"voxnet_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(OUTPUT_DIR), f"voxnet_confmatrix_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(voxnet.state_dict(), modelfile_output)

    print(f"Performing the prediction on the test set ...")
    pred, label = evaluation(voxnet, test_data)
    print(f"Saving the confusion matrix to {conf_mtx_output} ...")
    utils.confusion_matrix(pred, label, output_file=conf_mtx_output)

