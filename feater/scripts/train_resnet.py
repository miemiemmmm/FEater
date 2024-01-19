import argparse, time, json, os, random

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models
from feater import io, utils, dataloader


def get_resnet_model(resnettype: str):
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
  model.fc = nn.Linear(fc_number, 20)
  return model


def evaluation(classifier, dataset, usecuda=True, batch_size=256, process_nr=32):
  print(">>> Start evaluation...")
  classifier = classifier.eval()
  pred = []
  label = []
  for i, data in enumerate(dataset.mini_batches(batch_size=batch_size, process_nr=process_nr)):
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
    
    # TODO: Breakpoint Only for debugging
    if i == 50:
      break 
  valid_accuracy = np.sum(np.array(pred) == np.array(label)) / len(pred)
  print(f">>> Test set accuracy: {valid_accuracy:8.4f}")
  return pred, label


def parse_args():
  parser = argparse.ArgumentParser(description="Train the ResNet model based on the 2D Hilbert map")
  parser = utils.standard_parser(parser)
  
  # Parameters specifically related to the network architecture, this training or script
  parser.add_argument("--type", type=str, default="resnet18", help="ResNet type")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")
  parser.add_argument("--train_batch_nr", type=int, default=4000, help="Breakpoint for training")   # TODO: Remove this line

  # AFTER (ONLY AFTER) adding all arguments, do the sanity check
  utils.parser_sanity_check(parser)
  args = parser.parse_args()
  if not args.manualSeed:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)
  return args


if __name__ == '__main__':
  args = parse_args()
  SETTINGS = json.dumps(vars(args), indent=2)
  print("Settings of this training:")
  print(SETTINGS)
  random.seed(args.seed)
  torch.manual_seed(args.seed)

  BATCH_SIZE = args.batch_size
  EPOCH_NR = args.epochs
  LEARNING_RATE = args.learning_rate

  TRAIN_DATA = os.path.abspath(args.training_data)
  VALID_DATA = os.path.abspath(args.validation_data)
  TEST_DATA = os.path.abspath(args.test_data)
  OUTPUT_DIR = os.path.abspath(args.output_folder)
  LOAD_MODEL = args.pretrained
  INTERVAL = args.interval
  WORKER_NR = args.data_workers
  VERBOSE = True if args.verbose > 0 else False
  USECUDA = True if args.cuda else False
  START_EPOCH = args.start_epoch
  RESNET_TYPE = args.type

  with open(os.path.join(OUTPUT_DIR, "settings.json"), "w") as f:
    f.write(SETTINGS)

  # Load necessary datasets 
  trainingfiles = utils.checkfiles(TRAIN_DATA)
  training_data = dataloader.HilbertCurveDataset(trainingfiles)
  validfiles = utils.checkfiles(VALID_DATA)
  valid_data = dataloader.HilbertCurveDataset(validfiles)
  testfiles = utils.checkfiles(TEST_DATA)
  test_data = dataloader.HilbertCurveDataset(testfiles)

  classifier = get_resnet_model(RESNET_TYPE)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    classifier.load_state_dict(torch.load(LOAD_MODEL))
  optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  if USECUDA:
    classifier = classifier.cuda()

  for epoch in range(START_EPOCH, EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    batch_nr = int((len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE)
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      inputs, labels = batch
      if USECUDA:
        inputs, labels = inputs.cuda(), labels.cuda()
      optimizer.zero_grad()
      classifier.train()
      outputs = classifier(inputs)         # Output is a Tensor(batch_size, 20)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      accuracy = utils.report_accuracy(outputs, labels, verbose=False)
      print(f"Processing Batch {batch_idx:>6d}/{batch_nr:<6d} | Loss: {loss.item():8.4f} | Accuracy: {accuracy:8.4f}")

      if ((batch_idx + 1) % INTERVAL) == 0:
        vdata, vlabel = next(valid_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR))
        utils.validation(classifier, vdata, vlabel, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)

      # TODO: Remove this when production
      if (batch_idx + 1) == args.train_batch_nr:
        break
    scheduler.step()
    print(f"Epoch {epoch} takes {time.perf_counter() - st:.2f} seconds")
    print("^" * 80)
    
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"{RESNET_TYPE}_statusdic_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(OUTPUT_DIR), f"{RESNET_TYPE}_confusion_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    print(f"Performing the prediction on the test set ...")
    pred, label = evaluation(classifier, test_data, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)
    print(f"Saving the confusion matrix to {conf_mtx_output} ...")
    utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


