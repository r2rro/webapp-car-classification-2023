import os
import random
import numpy as np

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from utils.seed_everything import seed_everything
from model_trainer import ModelTrainer
from classifier import ResNet50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_epochs = 45
batch_size = 16
image_dir = '/content/drive/MyDrive/scrapping/split/'
load_dir = '/content/drive/MyDrive/saved_models/model_checkpoint.pth'
datadir = '/content/drive/MyDrive/scrapping/Exterior Classifier/'
path_train = os.path.join(datadir, 'train')
path_test = os.path.join(datadir, 'test')

seed_everything()

def transform_image():
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((400,400), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std, inplace=True)
    ])

  return transforms

def load_data(path_train: str,
              path_test: str,
              data_transforms: v2.Compose,
              batch_size: int,
              num_workers: int = 2 * torch.cuda.device_count()):

  train = ImageFolder(path_train, data_transforms)
  test = ImageFolder(path_test, data_transforms)

  class_count = len(train.classes)
  class_to_idx = train.class_to_idx

  train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  test_dl = DataLoader(test, shuffle=False, num_workers=num_workers, pin_memory=True)

  return train_dl, test_dl, class_count, class_to_idx

def split_data(source: str, model: nn.module, train_ratio: float=0.8):
  """  
    Split the data into training and testing sets, and filter out the interiot car images based on model predictions.

    Parameters:
    - source (str): Path to the source directory containing the 'data' folder.
    - train_ratio (float): Ratio of data to be used for training (default is 0.8).

    Note:
    - Assumes the 'data' folder in the 'source' directory contains subdirectories for each label.
    - Assumes images are named in the format '<label>_<index>.<extension>'.
  """
  train_dir = os.path.join(source, 'train')
  test_dir = os.path.join(source, 'test')
  data_dir = os.path.join(source, 'data')

  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)

  for label in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, label)):

      label_path = os.path.join(data_dir, label)
      images = [(Image.open(os.path.join(label_path,name)).convert('RGB'), name)
                for name in sorted(os.listdir(label_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

      images_exterior = []
      for image, name in images[:-6]:

        trnsfrm = transform_image().to(device)
        in_trns = trnsfrm(image).to(device)

        batch_img_tensor = torch.unsqueeze(in_trns, 0)
        prediction = torch.argmax(torch.softmax(model(batch_img_tensor.to(device)), dim=1), dim=1).item()

        if prediction == 0:
          images_exterior.append((image, name))

      random.shuffle(images_exterior)

      train_size = int(len(images_exterior) * train_ratio)
      train_images = images_exterior[:train_size]
      test_images = images_exterior[train_size:]

      for img, name in train_images:

        dest_path = os.path.join(train_dir, label+'/', name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        img.save(dest_path)

      for img, name in test_images:

        dest_path = os.path.join(test_dir, label+'/', name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        img.save(dest_path)

      print(label)

if __name__ == "__main__":

    train_dl, test_dl, num_classes, class_to_idx = load_data(path_train, path_test,
                                            transform_image(), batch_size)

    model = ResNet50(hidden_1=64, hidden_2=32, num_target_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()

    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    model_v0 = ModelTrainer(model, train_dl, test_dl, loss_fn, optimizer,n_epochs,
                            save_interval=45, save_path='save_path', device=device)
    model_v0.load_model(load_dir)
    model_v0.inference()

    split_data(image_dir, model)