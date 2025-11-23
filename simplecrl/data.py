import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from PIL import Image

np.random.seed(123)

class SimCLRDataTransform:
    """Generates two augmented views from each image"""
    
    def __init__(self, size=96):
        """Pipeline"""
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8,0.8,0.8,0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9)
            ], p=0.5),
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
        drop_last=True  
    )
    
    return train_loader

class SimCLRImageFolder(ImageFolder):
    """ImageFolder that returns 2 augmented views per sample."""

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = Image.open(path).convert("RGB")

        view1, view2 = self.transform(image)
        return (view1, view2), 0
    
# class CompetitionDataset(Dataset):
#     """Custom dataset for your competition's 500k images"""
    
#     def __init__(self, cache_dir=None, transform=None):
#         """
#         Args:
#             cache_dir: Where to cache downloaded data (e.g., '/scratch/ap9283/data/hf_cache')
#             transform: SimCLRDataTransform instance
#         """
#         print(f"Loading dataset (will cache to: {cache_dir})...")
        
#         # This will download AND cache automatically
#         self.dataset = load_dataset(
#             "tsbpp/fall2025_deeplearning",
#             split='train',
#             cache_dir=cache_dir
#         )
        
#         self.transform = transform
#         print(f"Dataset ready: {len(self.dataset)} images")
#         print(f"First epoch will download data, subsequent epochs use cache")
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         sample = self.dataset[idx]
#         image = sample['image']
        
#         if self.transform:
#             view1, view2 = self.transform(image)
#             return (view1, view2), 0
        
#         return image, 0
    
    
def get_competition_dataloaders(data_path, batch_size=256, num_workers=4, image_size=96):
    """
    Get dataloaders for competition dataset
    
    Args:
        data_path: Path where dataset was saved (e.g., '/scratch/ap9283/data/competition')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image size (96x96 for competition)
    """
    transform = SimCLRDataTransform(size=image_size)

    train_dataset = SimCLRImageFolder(
        root=data_path, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader