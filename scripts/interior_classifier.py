import os
import random
import numpy as np
from PIL import Image
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from classifier import ResNet50
from model_trainer import ModelTrainer
from utils.seed_everything import seed_everything

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything()

N_EPOCHS = 45
BATCH_SIZE = 16
SAVE_INTERVAL = 45
DATA_PATH = '/content/drive/MyDrive/Exterior Classifier/'
SAVE_PATH = os.path.join(DATA_PATH,'saved_models')
PATH_TRAIN = os.path.join(DATA_PATH, 'Train')
PATH_TEST = os.path.join(DATA_PATH, 'Test')

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

    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, test_dl, class_count, class_to_idx

# Load data
train_dl, test_dl, num_classes, class_to_idx = load_data(PATH_TRAIN, PATH_TEST,
                                                        transform_image(), BATCH_SIZE)

# Initialize the model
model = ResNet50(hidden_1=64, hidden_2=32, num_target_classes=num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()

# Freeze feature extractor parameters
for p in model.feature_extractor.parameters():
  p.requires_grad = False

# Initialize the optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Initialize the model trainer
model_v0 = ModelTrainer(model, train_dl, test_dl, loss_fn, optimizer, N_EPOCHS,
                        SAVE_INTERVAL, SAVE_PATH, device)

# Train the model
results = model_v0.train()