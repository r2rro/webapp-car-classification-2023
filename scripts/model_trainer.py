import random
import os
from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class ModelTrainer():
  """
  A class for training and evaluating a PyTorch model.
  
  Attributes:
    CRNT_EPOCH (int): Class variable to keep track of the current epoch.
    
  Parameters:
    model (nn.Module): The PyTorch model to be trained.
    train_dl (DataLoader): DataLoader for the training dataset.
    test_dl (DataLoader): DataLoader for the testing dataset
    loss_fn (nn.modules.loss._Loss): The loss function for training.
    optimizer (optim.Optimizer): The optimizer for training.
    scheduler (optim.lr_scheduler): Learning rate scheduler.
    num_epochs (int): Number of training epochs.
    save_interval (int): Interval for saving model checkpoints.
    save_path (str): Directory to save model checkpoints.
    device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
    """
  
  CRNT_EPOCH = 0

  def __init__(self, model: nn.Module, train_dl: DataLoader, test_dl: DataLoader,
               loss_fn: nn.modules.loss._Loss, optimizer: optim.Optimizer,
               scheduler: optim.lr_scheduler,  num_epochs: int,
               save_interval: int, save_path: str, device: torch.device):
     
    self.model = model
    self.train_dl = train_dl
    self.test_dl = test_dl
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.num_epochs = num_epochs
    self.device = device
    self.save_interval = save_interval
    self.save_path = save_path

  def train_step(self):

    self.model.train()
    train_acc, running_loss = 0, 0

    for i, (images, labels) in enumerate(self.train_dl):

      images, labels = images.to(self.device), labels.to(self.device)

      y_pred = self.model(images)
      criterion = self.loss_fn(y_pred, labels)
      running_loss += criterion.item()

      self.optimizer.zero_grad()
      criterion.backward()
      self.optimizer.step()

      with torch.inference_mode():

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / y_pred.shape[0]

    running_loss /= len(self.train_dl)
    train_acc /= len(self.train_dl)

    return running_loss, train_acc

  def test_step(self):
    self.model.eval()
    test_acc, test_loss = 0, 0

    with torch.inference_mode():
      for i, (images, labels) in enumerate(self.test_dl):

        images, labels = images.to(self.device), labels.to(self.device)
        test_pred_logits = self.model(images)
        criterion = self.loss_fn(test_pred_logits, labels)
        test_loss += criterion.item()

        test_pred_class = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
        test_acc += (test_pred_class == labels).sum().item() / test_pred_logits.shape[0]

      test_loss /= len(self.test_dl)
      test_acc /= len(self.test_dl)

    return test_loss, test_acc

  def train(self):

    results = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in tqdm(range(ModelTrainer.CRNT_EPOCH, self.num_epochs)):
      ModelTrainer.CRNT_EPOCH = epoch
      running_loss, train_acc = self.train_step()
      test_loss, test_acc = self.test_step()

      self.scheduler.step()

      print(
          f"Epoch: {epoch + 1} |",
          f"train loss: {running_loss:4f} |",
          f"train accuracy: {train_acc:4f} |",
          f"test loss: {test_loss:4f} |",
          f"test accuracy: {test_acc:4f} |"
      )

      results["train_loss"].append(running_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

      if epoch % self.save_interval == 0:
        self.save_model()

    return results

  def save_model(self):

    states = {
            'epoch': ModelTrainer.CRNT_EPOCH,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.loss_fn
            }

    filename = f'model_checkpoint_{ModelTrainer.CRNT_EPOCH}.pth'
    file_path = os.path.join(self.save_path, filename)
    torch.save(states, file_path)
    print('Model successfully saved to: ', file_path)

  def load_model(self, path: str):

    map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    ModelTrainer.CRNT_EPOCH = checkpoint['epoch'] + 1
    self.loss_fn.load_state_dict(checkpoint['loss'])
    print('Model successfully loaded from: ', path)