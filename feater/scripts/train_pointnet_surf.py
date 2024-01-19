import os, time, random, argparse, json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from pointnet.model import PointNetCls
from feater import io, constants, dataloader, utils

def evaluation(classifier, dataset, usecuda=True, batch_size=256, process_nr=32):
  """
  Evaluate the classifier on the validation set
  Args:
    classifier: The classifier
    dataloader: The dataloader for the validation set
  """
  from torch import Tensor
  classifier = classifier.eval()
  pred = []
  label = []
  for i, data in enumerate(dataset.mini_batches(batch_size=batch_size, process_nr=process_nr)):
    valid_data, valid_label = data
    valid_data = valid_data.transpose(2, 1)
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
  parser = argparse.ArgumentParser(description="Train PointNet Classifier with the FEater coordinate dataset")
  parser = utils.standard_parser(parser)

  # Model related
  parser.add_argument("--n_points", type=int, default=27, help="Num of points to use")
  parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()), help="Use CUDA")

  utils.parser_sanity_check(parser)
  args = parser.parse_args()
  args.cuda = torch.cuda.is_available()
  if args.manualSeed is None:
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
  EPOCH_NR = args.epochs
  PADDING_NPOINTS = args.n_points
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

  with open(os.path.join(OUTPUT_DIR, "settings.json"), "w") as f:
    f.write(SETTINGS)

  st = time.perf_counter()
  trainingfiles = utils.checkfiles(TRAIN_DATA)
  training_data = dataloader.SurfDataset(trainingfiles, target_np=PADDING_NPOINTS)
  validfiles = utils.checkfiles(VALID_DATA)
  valid_data = dataloader.SurfDataset(validfiles, target_np=PADDING_NPOINTS)
  testfiles = utils.checkfiles(TEST_DATA)
  test_data = dataloader.SurfDataset(testfiles, target_np=PADDING_NPOINTS)

  classifier = PointNetCls(k=20, feature_transform=False)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    classifier.load_state_dict(torch.load(LOAD_MODEL))
  optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  if USECUDA:
    classifier.cuda()

  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  # Check if the dataloader could successfully load the data
  for epoch in range(START_EPOCH, EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    batch_nr = (len(training_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, batch in enumerate(training_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR)):
      train_data, train_label = batch
      train_data = train_data.transpose(2, 1)  # Move the coordinate 3 dim to channels of dimensions
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()

      optimizer.zero_grad()
      classifier = classifier.train()
      pred, trans, trans_feat = classifier(train_data)
      loss = criterion(pred, train_label)
      loss.backward()
      optimizer.step()
      accuracy = utils.report_accuracy(pred, train_label, verbose=False)
      print(f"Processing the block {batch_idx}/{batch_nr}; Loss: {loss.item():8.4f}; Accuracy: {accuracy:8.4f}")

      if VERBOSE:
        utils.label_counts(train_label)

      if ((batch_idx + 1) % INTERVAL) == 0:
        vdata, vlabel = next(valid_data.mini_batches(batch_size=BATCH_SIZE, process_nr=WORKER_NR))
        vdata = vdata.transpose(2, 1)
        utils.validation(classifier, vdata, vlabel, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)
      
      # TODO: Remove this when production
      if (batch_idx + 1) == 8000:
        break
    scheduler.step()
    print(f"Epoch {epoch} takes {time.perf_counter() - st:.2f} seconds")
    print("^" * 80)

    # Save the model
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_surf_{epoch}.pth")
    conf_mtx_output  = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_confmatrix_{epoch}.png")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(classifier.state_dict(), modelfile_output)
    print(f"Performing the prediction on the test set ...")
    pred, label = evaluation(classifier, test_data, usecuda=USECUDA, batch_size=BATCH_SIZE, process_nr=WORKER_NR)
    print(f"Saving the confusion matrix to {conf_mtx_output} ...")
    utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


  # # Final evaluation on test set
  # print("Final evaluation")
  # test_data = utils.checkfiles(TEST_DATA)
  # test_data = dataloader.SurfDataset(test_data, target_np=PADDING_NPOINTS)
  # dataloader_test = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)
  # pred, label = evaluation(classifier, dataloader_test)
  # print(f"Final accuracy: {np.sum(np.array(pred) == np.array(label)) / len(pred):.4f}")
  # conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), "pointnet_confmatrix_final.png")
  # utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


