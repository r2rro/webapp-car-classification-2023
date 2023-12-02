from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, hidden_1: int, hidden_2: int, num_target_classes: int):
      super().__init__()
      self.hidden_1 = hidden_1
      self.hidden_2 = hidden_2

      backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
      _res_layers = list(backbone.children())[:-1]
      self.feature_extractor = nn.Sequential(*_res_layers)
      self.fc = nn.Sequential(nn.Linear(2048, num_target_classes),
                              #nn.Linear(hidden_1, hidden_2),
                              #nn.Linear(hidden_2, num_target_classes)
                             )

    def forward (self, x):
      x = self.feature_extractor(x)
      x = x.squeeze(-1).squeeze(-1) # check this
      x = self.fc(x)
      return x
    
class ResNet101(nn.Module):
    def __init__(self, hidden_1: int, hidden_2: int, num_target_classes: int):
      super().__init__()
      self.hidden_1 = hidden_1
      self.hidden_2 = hidden_2

      backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
      _res_layers = list(backbone.children())[:-1]
      self.feature_extractor = nn.Sequential(*_res_layers)
      self.fc = nn.Sequential(nn.Linear(2048, num_target_classes),
                              #nn.Linear(hidden_1, hidden_2),
                              #nn.Linear(hidden_2, num_target_classes)
                             )

    def forward (self, x):
      x = self.feature_extractor(x)
      x = x.squeeze(-1).squeeze(-1) # check this
      x = self.fc(x)
      return x