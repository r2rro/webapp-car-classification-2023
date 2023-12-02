import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def load_data(path_train: str,
              path_test: str,
              data_transforms: v2.Compose,
              batch_size: int,
              num_workers: int = 2 * torch.cuda.device_count()):

  train = ImageFolder(path_train, data_transforms)
  test = ImageFolder(path_test, data_transforms)

  class_count = len(train.classes)
  class_to_idx = train.class_to_idx

  train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)  
  test_dl = DataLoader(test, batch_size=64, shuffle=False) 

  return train_dl, test_dl, class_count, class_to_idx