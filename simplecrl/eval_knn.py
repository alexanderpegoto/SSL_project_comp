import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm

from model import ResNet


def extract_features(model, dataloader, device):
    """Extracting features from the encoder"""
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images,labels in tqdm(dataloader, desc='Extracting Features'):
            images = images.to(device)
            
            # get encoder features
            features, _ =  model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate all batches
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels

def knn_evaluate(train_features, train_labels, test_features, test_labels, k=200):
    """
    Train k-NN classifier and evaluate
    """
    print(f'\nTraining k-NN classifier with k={k}...')
    
    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = knn.score(train_features, train_labels)
    test_acc = knn.score(test_features, test_labels)
    
    return train_acc, test_acc

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = ResNet(base_model='resnet50', out_dim=128).to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]} with loss {checkpoint["loss"]:.4f}')
    
    # Data transforms (NO augmentation for evaluation!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 train and test sets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Extract features
    print('\nExtracting training features...')
    train_features, train_labels = extract_features(model, train_loader, device)
    
    print('Extracting test features...')
    test_features, test_labels = extract_features(model, test_loader, device)
    
    print(f'\nTrain features shape: {train_features.shape}')
    print(f'Test features shape: {test_features.shape}')
    
    # k-NN evaluation
    train_acc, test_acc = knn_evaluate(train_features, train_labels, test_features, test_labels, k=200)
    
    print(f'\n{"="*50}')
    print(f'k-NN Results (k=200):')
    print(f'Train Accuracy: {train_acc*100:.2f}%')
    print(f'Test Accuracy: {test_acc*100:.2f}%')
    print(f'{"="*50}')
    
    # Try different k values
    print('\nTrying different k values...')
    for k in [1, 10, 20, 50, 100, 200]:
        _, test_acc = knn_evaluate(train_features, train_labels, test_features, test_labels, k=k)
        print(f'k={k:3d}: Test Accuracy = {test_acc*100:.2f}%')


if __name__ == '__main__':
    main()