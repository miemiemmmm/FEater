import torch
import torch.nn as nn
import torchvision.models


FCFeatureNumberMap = {
  "resnet18": 512,     "resnet34": 512,    "resnet50": 2048, 
  "resnet101": 2048,   "resnet152": 2048
}


def get_resnet_model(channel_in, resnettype: str, class_nr:int, ):
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
  
  # Initialize the first convolutional layer
  model.conv1 = nn.Conv2d(channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False) 
  nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

  fc_number = FCFeatureNumberMap.get(resnettype, 2048)
  model.fc = nn.Linear(fc_number, class_nr)
  return model


class ResNet(nn.Module):
  def __init__(self, 
               channel_in: int, 
               output_dim: int, 
               resnet_type: str):
    # Use get_resnet_model to initialize the model
    super(ResNet, self).__init__()
    self.model = get_resnet_model(channel_in, resnet_type, output_dim)

  def forward(self, x):
    return self.model(x)
  
  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()

