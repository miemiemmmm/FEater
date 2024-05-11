import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class VoxNet(nn.Module):
  """
  VoxNet model

  Original code is adopted from: https://github.com/MonteYang/VoxNet.pytorch/blob/master/voxnet.py
  
  Notes
  -----
  Compared with the original model, ReLU is chnaged to PReLU(1, 0.25)
  The higgen layer in the fully connected layer is changed to 1280
  Dropout rate changed from [0.2, 0.3, 0.4] to [0.1, 0.1, 0.1]
  """
  def __init__(self, 
               n_classes=10, 
               input_shape=(32, 32, 32)):
    super(VoxNet, self).__init__()
    dropout_rates = [0.1, 0.1, 0.1]
    self.n_classes = n_classes
    self.input_shape = input_shape
    self.feat = torch.nn.Sequential(OrderedDict([
      ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                   out_channels=32, kernel_size=5, stride=2)),
      # ('relu1', torch.nn.ReLU()),
      ('relu1', nn.PReLU()),
      ('drop1', torch.nn.Dropout(p=dropout_rates[0])),
      ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
      # ('relu2', torch.nn.ReLU()),
      ('relu2', nn.PReLU()),
      ('pool2', torch.nn.MaxPool3d(2)),
      ('drop2', torch.nn.Dropout(p=dropout_rates[1])),
    ]))
    x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
    dim_feat = 1
    for n in x.size()[1:]:
      dim_feat *= n

    self.mlp = torch.nn.Sequential(OrderedDict([
      ('fc1', torch.nn.Linear(dim_feat, 1280)),
      # ('relu1', torch.nn.ReLU()),
      ('relu1', nn.PReLU()),
      ('drop3', torch.nn.Dropout(p=dropout_rates[2])),
      ('fc2', torch.nn.Linear(1280, self.n_classes))
    ]))

  def forward(self, x):
    x = self.feat(x)
    x = x.view(x.size(0), -1)
    x = self.mlp(x)
    return x

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()



if __name__ == "__main__":
  voxnet = VoxNet()
  data = torch.rand([256, 1, 32, 32, 32])
  voxnet(data)
