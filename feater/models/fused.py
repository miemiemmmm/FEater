import time
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

import feater.models
import feater.models.resnet
import feater.models.gnina
import feater.models.pointnet


def fusion_checker(x1, x2, x_fused): 
  x1_ = x1.detach().cpu().numpy()
  x2_ = x2.detach().cpu().numpy()
  print(np.sum(x1_), np.sum(x2_))
  ##########################
  x_ = x_fused.detach().cpu().numpy()
  plt.pcolormesh(x_, vmax = 6, cmap="inferno")
  plt.colorbar(label="Activation")
  plt.title(f'{np.sum(x1_):.2f} vs. {np.sum(x2_):.2f}')
  plt.savefig(f"/tmp/diff_{time.perf_counter()}.png")
  plt.clf()


class gnina_pointnet(nn.Module): 
  def __init__(self, channel_in: int, output_dim: int, **kwargs): 
    """

    """
    super(gnina_pointnet, self).__init__()
    self.model1 = feater.models.gnina.GninaNetwork(output_dim)
    self.model2 = feater.models.pointnet.PointNetCls(output_dim)

    dummysample1 = torch.zeros(2, channel_in, 32, 32, 32)
    dim1 = torch.flatten(self.model1.featurize(dummysample1), 1).size()[1]

    dummysample2 = torch.zeros(2, 64, 3)
    dummysample2 = dummysample2.transpose(2, 1)
    dim2 = torch.flatten(self.model2.featurize(dummysample2), 1).size()[1]
    print(f"Flattening Gnina with dimension: {dim1}; PointNet: {dim2}; ")
    self.proj1 = nn.Linear(dim1, 256)
    self.proj2 = nn.Linear(dim2, 256)
    self.bn1 = nn.BatchNorm1d(256) 
    self.bn2 = nn.BatchNorm1d(256)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(512, output_dim)


  def forward(self, x1, x2, check_fusion=False):
    """
    Args:
      x1: GNINA 3D voxel input (batch_size, channel_in, 32, 32, 32)
      x2: PointNet input (batch_size, target_np, 3)
    """
    # Flatten each input
    x1 = self.model1.featurize(x1)
    x1 = torch.flatten(x1, 1)
    x1 = self.relu(self.bn1(self.proj1(x1)))

    if x2.size()[1] != 3:
      # transpose if necessary 
      x2 = x2.transpose(2, 1)
    x2 = self.model2.featurize(x2)
    x2 = torch.flatten(x2, 1)
    x2 = self.relu(self.bn2(self.proj2(x2)))
    
    # fuse the two outputs
    x = torch.cat((x1, x2), 1)
    if check_fusion: 
      fusion_checker(x1, x2, x)
    x = self.fc(x)
    return x



class gnina_resnet(nn.Module): 
  def __init__(self, channel_in: int, output_dim: int, **kwargs): 
    """
    """
    super(gnina_resnet, self).__init__()
    self.model1 = feater.models.gnina.GninaNetwork(output_dim)
    self.model2 = feater.models.resnet.ResNet(channel_in, output_dim, "resnet18")

    dummy1 = torch.zeros(2, channel_in, 32, 32, 32)
    dim1 = torch.flatten(self.model1.featurize(dummy1), 1).size()[1]

    dummy2 = torch.zeros(2, channel_in, 128, 128)
    dim2 = torch.flatten(self.model2.featurize(dummy2), 1).size()[1]
    print(f"Flattening Gnina with dimension: {dim1}; ResNet: {dim2}; ")

    self.proj1 = nn.Linear(dim1, 256) 
    self.proj2 = nn.Linear(dim2, 256) 
    self.bn1 = nn.BatchNorm1d(256) 
    self.bn2 = nn.BatchNorm1d(256) 
    self.relu = nn.ReLU() 
    self.fc = nn.Linear(512, output_dim) 

  def forward(self, x1, x2, check_fusion=False):
    """
    Args:
      x1: GNINA 3D voxel input (batch_size, channel_in, 32, 32, 32)
      x2: ResNet input (batch_size, channel_in, 128, 128)
    """
    x1 = self.model1.featurize(x1)
    x1 = torch.flatten(x1, 1)
    x1 = self.relu(self.bn1(self.proj1(x1)))

    x2 = self.model2.featurize(x2)
    x2 = torch.flatten(x2, 1)
    x2 = self.relu(self.bn2(self.proj2(x2)))

    x = torch.cat((x1, x2), 1)
    if check_fusion: 
      fusion_checker(x1, x2, x)
    x = self.fc(x)
    return x
  
class pointnet_pointnet(nn.Module): 
  def __init__(self, channel_in: int, output_dim: int, **kwargs): 
    """
    """
    super(pointnet_pointnet, self).__init__()
    self.model1 = feater.models.pointnet.PointNetCls(output_dim)
    self.model2 = feater.models.pointnet.PointNetCls(output_dim)
    dummy1 = torch.zeros(2, 64, 3).transpose(2, 1)
    dummy2 = torch.zeros(2, 64, 3).transpose(2, 1)

    dim1 = torch.flatten(self.model1.featurize(dummy1), 1).size()[1] 
    dim2 = torch.flatten(self.model2.featurize(dummy2), 1).size()[1] 
    fused_dim = dim1 + dim2
    print(f"Flattening PointNet1 with dimension: {dim1}; PointNet2: {dim2}; ")
    self.proj1 = nn.Linear(dim1, 256) 
    self.proj2 = nn.Linear(dim2, 256) 
    self.bn1 = nn.BatchNorm1d(256) 
    self.bn2 = nn.BatchNorm1d(256) 
    self.relu = nn.ReLU()
    self.fc = nn.Linear(fused_dim, output_dim)
  
  def forward(self, x1, x2, check_fusion=False):
    """
    Args:
      x1: PointNet input (batch_size, target_np, 3)
      x2: PointNet input (batch_size, target_np, 3)
    """
    if x1.size()[1] != 3:
      x1 = x1.transpose(2, 1)
    if x2.size()[1] != 3:
      x2 = x2.transpose(2, 1)
    x1 = self.model1.featurize(x1)
    x1 = torch.flatten(x1, 1)
    x1 = self.relu(self.bn1(self.proj1(x1)))
    x2 = self.model2.featurize(x2)
    x2 = torch.flatten(x2, 1)
    x2 = self.relu(self.bn2(self.proj2(x2)))

    x = torch.cat((x1, x2), 1)
    if check_fusion: 
      fusion_checker(x1, x2, x)

    x = self.fc(x)
    return x

class pointnet_resnet(nn.Module):
  def __init__(self, channel_in: int, output_dim: int, **kwargs): 
    super(pointnet_resnet, self).__init__()
    self.model1 = feater.models.pointnet.PointNetCls(output_dim)
    self.model2 = feater.models.resnet.ResNet(channel_in, output_dim, "resnet18")
    dummy1 = torch.zeros(2, 64, 3).transpose(2, 1)
    dummy2 = torch.zeros(2, channel_in, 128, 128)
    dim1 = torch.flatten(self.model1.featurize(dummy1), 1).size()[1]
    dim2 = torch.flatten(self.model2.featurize(dummy2), 1).size()[1]
    print(f"Flattening PointNet with dimension: {dim1}; ResNet: {dim2}; ")
    self.proj1 = nn.Linear(dim1, 256) 
    self.proj2 = nn.Linear(dim2, 256) 
    self.bn1 = nn.BatchNorm1d(256) 
    self.bn2 = nn.BatchNorm1d(256) 
    self.relu = nn.ReLU()
    self.fc = nn.Linear(512, output_dim)

  def forward(self, x1, x2, check_fusion=False):
    """
    Args:
      x1: PointNet input (batch_size, target_np, 3)
      x2: ResNet input (batch_size, channel_in, 128, 128)
    """
    if x1.size()[1] != 3:
      x1 = x1.transpose(2, 1)
    x1 = self.model1.featurize(x1)
    x1 = torch.flatten(x1, 1)
    x1 = self.relu(self.bn1(self.proj1(x1)))
    x2 = self.model2.featurize(x2)
    x2 = torch.flatten(x2, 1)
    x2 = self.relu(self.bn2(self.proj2(x2)))
    x = torch.cat((x1, x2), 1)
    if check_fusion: 
      fusion_checker(x1, x2, x)
    x = self.fc(x)
    return x


