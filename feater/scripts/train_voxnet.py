import os, sys, argparse, time, random

import torch
import torch.nn as nn
import torch.optim as optim
from feater.models.voxnet import VoxNet

def parse_args():
  parser = argparse.ArgumentParser(description="Train VoxNet")


  parser.add_argument("--manualSeed", type=int, help="Manual seed")
  parser.add_argument("--load_model", type=str, default="", help="The model to load")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  return args

if __name__ == "__main__":
  args = parse_args()
  print(args)
  if args.manualSeed is None:
    print("Randomizing the Seed")
    args.manualSeed = random.randint(1, 10000)
  else:
    print(f"Using manual seed {args.manualSeed}")

  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  batchsize = 5000

  # Load the data:  TODO - change this to load the HDF5 file
  train_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
  test_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')

  train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
  test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

  # VoxNet
  voxnet = VoxNet(n_classes=N_CLASSES)
  print(voxnet)
  if args.cuda:
    voxnet.cuda()

  optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
  criterion = nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

  for epoch in range(1, 5):
    scheduler.step()
    for i, data in enumerate(train_dataloader, 0):
      # get the inputs
      inputs, labels = data
      if args.cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = voxnet(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      print('[%d, %5d] loss: %.3f' % (epoch, i, loss.data[0]))
    voxnet.save_model(f'voxnet_epoch{epoch}.pth')
  print('Finished Training')