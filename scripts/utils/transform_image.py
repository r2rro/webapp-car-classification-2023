import torch
from torchvision.transforms import v2

def transform_image():
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((400,400), antialias=True),
    v2.RandomCrop(350),
    v2.RandomHorizontalFlip(0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean, std, inplace=True)
    ])

  return transforms