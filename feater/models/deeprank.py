import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DeepRankNetwork(torch.nn.Module): 
  """
  Code adopted from: 
  DeepRank2: https://github.com/DeepRank/deeprank2/blob/main/deeprank2/neuralnets/cnn/model3d.py
  
  """
  def __init__(self, input_channel_number: int, output_dimension: int,
               box_shape:int, conv=[4, 5], fc = [84]):
    super(DeepRankNetwork, self).__init__()
    if isinstance(box_shape, int):
      box_shape = (box_shape, box_shape, box_shape)
    elif isinstance(box_shape, (tuple, list, np.ndarray)):
      box_shape = tuple(box_shape)[:3]

    conv_layers = OrderedDict()
    for i in range(len(conv)):
      if i == 0:
        conv_layers[f'conv_{i:d}'] = torch.nn.Conv3d(input_channel_number, conv[i], kernel_size=2)
      else: 
        conv_layers[f'conv_{i:d}'] = torch.nn.Conv3d(conv[i-1], conv[i], kernel_size=2)
      conv_layers[f'relu_{i:d}'] = torch.nn.ReLU()  
      conv_layers[f'pool_{i:d}'] = torch.nn.MaxPool3d((2, 2, 2)) 
    
    self.conv_blocks = torch.nn.Sequential(conv_layers)

    dummpy_out = self.conv_blocks(torch.rand(1, input_channel_number, *box_shape))
    size = dummpy_out.flatten().size()[0]

    fc_layers = OrderedDict()
    for i in range(len(fc)):
      if i == 0:
        fc_layers[f'fc_{i:d}'] = torch.nn.Linear(size, fc[i])
      else:
        fc_layers[f'fc_{i:d}'] = torch.nn.Linear(fc[i-1], fc[i])
      fc_layers[f'relu_{i:d}'] = torch.nn.ReLU()
    self.fc_layers = torch.nn.Sequential(fc_layers)
    self.output_layer = torch.nn.Linear(fc[-1], output_dimension)

  def forward(self, data):
    data = self.conv_blocks(data)
    data = data.view(data.size(0), -1)
    data = self.fc_layers(data)
    data = self.output_layer(data)
    return data 