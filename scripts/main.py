import os
import random
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.optim import lr_scheduler

from classifier import ResNet50
from model_trainer import ModelTrainer
from utils.seed_everything import seed_everything
from utils.load_data import load_data
from utils.transform_image import transform_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything()

n_epochs = 100
batch_size = 128
hidden_1 = 1024
hidden_2 = 512
step_size = 30
gamma = 0.5
save_interval = 10

path_data = '/content/drive/MyDrive/car_data'
path_train = os.path.join(path_data, 'train_2023')
path_test = os.path.join(path_data, 'test_2023')
path_save = os.path.join(path_data,'saved_models_2023/')
if not os.path.exists(path_save): os.mkdir(path_save)

train_dl, test_dl, num_classes, class_to_idx = load_data(path_train, path_test,
                                               transform_image(), batch_size)

model = ResNet50(hidden_1=hidden_1, hidden_2=hidden_2, num_target_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()

for p in model.feature_extractor.parameters():
  p.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

model_v0 = ModelTrainer(model, train_dl, test_dl, loss_fn,
                        optimizer, scheduler, n_epochs,
                        save_interval, path_save, device)

results = model_v0.train()

train_loss = results["train_loss"]
test_loss = results["test_loss"]
train_acc = results["train_acc"]
test_acc =  results["test_acc"]

fig, axs = plt.subplots(2, 2, layout='constrained')
axs[0, 0].plot(train_loss)
axs[0, 0].set_title('train loss')
axs[0, 1].plot(test_loss)
axs[0, 1].set_title('test loss')
axs[1, 0].plot(train_acc)
axs[1, 0].set_title('train acc')
axs[1, 1].plot(test_acc)
axs[1, 1].set_title('test acc')