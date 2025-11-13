import numpy as np
from torch import nn
import torch
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

np.random.seed(123)

class SimCLRDataTransform:
    """Generates two augmented views from each image"""
    
    def __init__(self, size=32):
        """Pipeline"""
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8,0.8,0.8,0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # No Gaussian Blur
            # transforms.RandomApply([
            #     transforms.GaussianBlur(kernel_size=23)
            # ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.455,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    def __call__(self,x):
        return self.transform(x), self.transform(x)
    
def get_cifar10_dataloaders(batch_size=256, num_workers=4):
    """Get CIFAR-10 dataloaders for SimCLR training"""

    # Training data with SimCLR augmentations
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=SimCLRDataTransform(size=32)  # CIFAR-10 is 32x32
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for SimCLR!
    )
    
    return train_loader

