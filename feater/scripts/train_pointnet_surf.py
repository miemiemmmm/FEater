import os, time, random, argparse, json

import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from pointnet.model import PointNetCls
from feater import io, constants, dataloader, utils

def evaluation(classifier, dataloader, usecuda=True):
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
  for i, data in enumerate(dataloader):
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
  return pred, label



def parse_args():
  parser = argparse.ArgumentParser(description="Train PointNet Classifier with the FEater coordinate dataset")

  parser.add_argument("-train", "--training_data", type=str, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-valid", "--validation_data", type=str, help="The file writes all of the absolute path of h5 files of validation data set")
  parser.add_argument("-test", "--test_data", type=str, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output_folder", type=str, default="cls", help="Output folder")

  parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
  parser.add_argument("-itv", "--interval", type=int, default=50, help="How many batches to wait before logging training status")
  parser.add_argument("-v", "--verbose", default=0, type=int, help="Verbose printing while training")
  parser.add_argument("--data_workers", type=int, default=4, help="Number of workers for data loading")
  parser.add_argument("--n_points", type=int, default=27, help="Num of points to use")
  parser.add_argument("--manualSeed", type=int, help="Manual seed")
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")

  # TODO: Arguments to check
  # parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
  # parser.add_argument("--normal", action="store_true", default=False, help="Whether to use normal information")

  args = parser.parse_args()
  args.cuda = torch.cuda.is_available()
  if args.manualSeed is None:
    args.seed = int(time.perf_counter().__str__().split(".")[-1])
  else:
    args.seed = int(args.manualSeed)

  if (not args.training_data) or (not os.path.exists(args.training_data)):
    parser.print_help()
    raise ValueError(f"The training data file {args.training_data} does not exist.")
  if (not args.validation_data) or (not os.path.exists(args.validation_data)):
    parser.print_help()
    raise ValueError(f"The validation data file {args.validation_data} does not exist.")
  if (not args.test_data) or (not os.path.exists(args.test_data)):
    parser.print_help()
    raise ValueError(f"The test data file {args.test_data} does not exist.")
  if (not args.output_folder) or (not os.path.exists(args.output_folder)):
    parser.print_help()
    raise ValueError(f"The output folder {args.output_folder} does not exist.")
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
  dataloader_train = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)

  validfiles = utils.checkfiles(VALID_DATA)
  valid_data = dataloader.SurfDataset(validfiles, target_np=PADDING_NPOINTS)
  dataloader_valid = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)

  print(f"The dataset has {len(training_data)} entries; Validation set has {len(valid_data)} entries.")

  # t_total = 0
  # n = 1024
  # for i in np.random.choice(len(training_data)-1, n).astype(np.int32):
  #   tmp_st = time.perf_counter()
  #   _, _ = training_data[i]
  #   t_diff = time.perf_counter() - tmp_st
  #   t_total += t_diff
  #   print(f"Entry {i}: {training_data[i][1]}; Time: {t_diff*1e3:8.3f} ms")
  # print(f"Total time: {t_total*1e3:8.3f} ms; Average time: {t_total*1e3/n:8.3f} ms")

  classifier = PointNetCls(k=20, feature_transform=False)
  if LOAD_MODEL and len(LOAD_MODEL) > 0:
    classifier.load_state_dict(torch.load(LOAD_MODEL))
  optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  if USECUDA:
    classifier.cuda()

  print(f"Number of parameters: {sum([p.numel() for p in classifier.parameters()])}")

  # Check if the dataloader could successfully load the data
  for epoch in range(START_EPOCH, EPOCH_NR):
    print("#" * 80)
    print(f"Running the epoch {epoch}/{EPOCH_NR}")
    st = time.perf_counter()
    for i, batch in enumerate(training_data.mini_batches()):
      train_data, train_label = batch
      train_data = train_data.transpose(2, 1)  # Move the coordinate 3 dim to channels of dimensions
      if USECUDA:
        train_data, train_label = train_data.cuda(), train_label.cuda()

      optimizer.zero_grad()
      classifier = classifier.train()
      pred, trans, trans_feat = classifier(train_data)
      loss = F.nll_loss(pred, train_label)
      loss.backward()
      optimizer.step()
      pred_choice = pred.data.max(1)[1]
      correct = pred_choice.eq(train_label.data).cpu().sum()
      print(f"Processing the block {i}/{len(dataloader_train)}; Loss: {loss.item():8.4f}; Accuracy: {correct.item()/float(train_label.size()[0]):8.4f}")

      if VERBOSE:
        utils.label_counts(train_label)

      if ((i+1) % INTERVAL) == 0:
        valid_data, valid_label = next(dataloader_valid.__iter__())
        valid_data = valid_data.transpose(2, 1)
        if USECUDA:
          valid_data, valid_label = valid_data.cuda(), valid_label.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(valid_data)
        loss = F.nll_loss(pred, valid_label)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(valid_label.data).cpu().sum()
        print(f"Validation: Loss: {loss.item():.4f}; Accuracy: {correct.item()/float(valid_label.size()[0]):.4f}")

    scheduler.step()
    # Save the model
    print(f"Epock {epoch} takes {time.perf_counter() - st:.2f} seconds, saving the model and confusion matrix")
    modelfile_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_coord_{epoch}.pth")
    torch.save(classifier.state_dict(), modelfile_output)
    pred, label = evaluation(classifier, dataloader_valid)
    conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), f"pointnet_confmatrix_{epoch}.png")
    utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


  # Final evaluation on test set
  print("Final evaluation")
  test_data = utils.checkfiles(TEST_DATA)
  test_data = dataloader.SurfDataset(test_data, target_np=PADDING_NPOINTS)
  dataloader_test = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NR)
  pred, label = evaluation(classifier, dataloader_test)
  print(f"Final accuracy: {np.sum(np.array(pred) == np.array(label)) / len(pred):.4f}")
  conf_mtx_output = os.path.join(os.path.abspath(OUTPUT_DIR), "pointnet_confmatrix_final.png")
  utils.confusion_matrix(pred, label, output_file=conf_mtx_output)


