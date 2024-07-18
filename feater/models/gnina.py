"""
PyTorch implementation of Gnina default2018: https://doi.org/10.1021/acs.jcim.0c00411

Code is adopted from: 
https://github.com/gnina/models/blob/master/pytorch/default2018_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GninaNetwork(nn.Module):
    def __init__(self, out_dims):
        super(GninaNetwork, self).__init__()
        input_channels = 1
        grid_dims = [32, 32, 32]
        self.features = nn.Sequential(
            nn.AvgPool3d(2,stride=2),                                           # avgpool_0
            nn.Conv3d(input_channels, out_channels=32,padding=1,kernel_size=3,stride=1),    # unit1_conv
            nn.ReLU(),
            nn.Conv3d(32, out_channels=32,padding=0,kernel_size=1,stride=1),    # unit2_conv
            nn.ReLU(),
            nn.AvgPool3d(2,stride=2),                                           # avgpool_1
            nn.Conv3d(32, out_channels=64,padding=1,kernel_size=3,stride=1),    # unit3_conv
            nn.ReLU(),
            nn.Conv3d(64, out_channels=64,padding=0,kernel_size=1,stride=1),    # unit4_conv
            nn.ReLU(),
            nn.AvgPool3d(2,stride=2),                                           # avgpool_2
            nn.Conv3d(64, out_channels=128,padding=1,kernel_size=3,stride=1),   # unit5_conv
            nn.ReLU(),
        )
        dummy_output = self.features(torch.zeros(1, input_channels, grid_dims[0], grid_dims[1], grid_dims[2]))
        flattened_feature_size = dummy_output.flatten().size()[0]
        self.affinity_output = nn.Linear(flattened_feature_size, out_dims)

    def forward(self, x): 
        x = self.features(x)
        x = nn.Flatten()(x)
        affinity = self.affinity_output(x)
        return affinity
            


class View(nn.Module):
    def __init__(self,shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class _GninaNetwork(nn.Module):
    def __init__(self, out_dims):
        super(GninaNetwork, self).__init__()
        self.modules_ = []
        dims = [1, 32, 32, 32]
        nchannels = dims[0]

        self.func = F.relu

        avgpool1 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_0', avgpool1)
        self.modules_.append(avgpool1)
        conv1 = nn.Conv3d(nchannels,out_channels=32,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit1_conv',conv1)
        self.modules_.append(conv1)
        conv2 = nn.Conv3d(32,out_channels=32,padding=0,kernel_size=1,stride=1) 
        self.add_module('unit2_conv',conv2)
        self.modules_.append(conv2)
        avgpool2 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_1', avgpool2)
        self.modules_.append(avgpool2)
        conv3 = nn.Conv3d(32,out_channels=64,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit3_conv',conv3)
        self.modules_.append(conv3)
        conv4 = nn.Conv3d(64,out_channels=64,padding=0,kernel_size=1,stride=1) 
        self.add_module('unit4_conv',conv4)
        self.modules_.append(conv4)
        avgpool3 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_2', avgpool3)
        self.modules_.append(avgpool3)
        conv5 = nn.Conv3d(64,out_channels=128,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit5_conv',conv5)
        self.modules_.append(conv5)
        div = 2*2*2
        last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 128)
        print(last_size)
        flattener = View((-1,last_size))
        self.add_module('flatten',flattener)
        self.modules_.append(flattener)
        self.affinity_output = nn.Linear(last_size, out_dims)
        # self.pose_output = nn.Linear(last_size,2)

    def forward(self, x): #should approximate the affinity of the receptor/ligand pair
        for layer in self.modules_:
            x = layer(x)
            if isinstance(layer,nn.Conv3d):
                x=self.func(x)
        affinity = self.affinity_output(x)
        # pose = F.softmax(self.pose_output(x),dim=1)[:,1]
        return affinity