import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import argparse

from model import ResNet
from data import CSVImageDataset


def extract_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for images, lbls in tqdm(dataloader):
            images = images.to(device)
            f, _ = model(images)
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(feats), np.concatenate(labels)

def extract_features_test(model, dataloader, device):
    model.eval()
    feats, names = [], []

    with torch.no_grad():
        for images, fnames in tqdm(dataloader):
            images = images.to(device)
            f, _ = model(images)
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
            names.extend(fnames)

    return np.concatenate(feats), names


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

def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = ResNet(base_model='resnet50', out_dim=128).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]} with loss {checkpoint["loss"]:.4f}')
    
    # Data transforms
    transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
# ----- Build datasets -----
    # For tuning k (train + val mode)
    if args.mode == "val":
        train_dataset = CSVImageDataset(
            root_dir=args.data_root,
            img_subdir="train",
            csv_path=os.path.join(args.data_root, "train_labels.csv"),
            transform=transform,
            has_labels=True,
        )
        val_dataset = CSVImageDataset(
            root_dir=args.data_root,
            img_subdir="val",
            csv_path=os.path.join(args.data_root, "val_labels.csv"),
            transform=transform,
            has_labels=True,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)
        print("\nExtracting train features...")
        train_features, train_labels = extract_features(model, train_loader, device)

        print("Extracting val features...")
        val_features, val_labels = extract_features(model, val_loader, device)

        print(f"\nTrain features shape: {train_features.shape}")
        print(f"Val features shape:   {val_features.shape}")

        best_k = None
        best_val = -1.0

        # k-NN sweep
        for k in [1, 5, 10, 20, 50, 100, 200]:
            train_acc, val_acc = knn_evaluate(train_features, train_labels,
                                            val_features,   val_labels, k=k)
            print(f"k={k:3d}: Train {train_acc*100:.2f}%, Val {val_acc*100:.2f}%")

            if val_acc > best_val:
                best_val = val_acc
                best_k = k
                
        print("\nBest k on val:", best_k, f"(Val Acc = {best_val*100:.2f}%)")
        print("⇨ For test submissions, rerun with:  --mode test  --k", best_k)

        
    elif args.mode == "test":
        train_dataset = CSVImageDataset(
            root_dir=args.data_root,
            img_subdir="train",
            csv_path=os.path.join(args.data_root, "train_labels.csv"),
            transform=transform,
            has_labels=True,
        )
        test_dataset = CSVImageDataset(
            root_dir=args.data_root,
            img_subdir="test",
            csv_path=os.path.join(args.data_root, "test_images.csv"),
            has_labels=False,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        print("\nExtracting train features...")
        train_features, train_labels = extract_features(model, train_loader, device)

        print("Extracting test features...")
        test_features = extract_features_test(model, test_loader, device)

        print(f"\nTrain features shape: {train_features.shape}")
        print(f"Test features shape:  {test_features.shape}")
        
        k = args.k
        print(f"\nFitting final k-NN with k={k} and generating predictions...")
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(train_features, train_labels)
        preds = knn.predict(test_features)

        # Load filenames from test_images.csv and fill sample_submission
        test_csv = os.path.join(args.data_root, "test_images.csv")
        test_df = pd.read_csv(test_csv)  # has 'filename' column

        sample_path = os.path.join(args.data_root, "sample_submission.csv")
        sample_df = pd.read_csv(sample_path)  # has 'filename' and 'label'

        # Ensure alignment: filenames in sample_submission and test_images.csv
        # are in the same order (they should be by construction).
        sample_df["label"] = preds
        sample_df.to_csv(args.output, index=False)
        print(f"\nSaved submission to {args.output}")

    else:
        raise ValueError("mode must be 'val' or 'test'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str,
                        default="/scratch/ap9283/deep_learning/SSL_project_comp/simplecrl/checkpoints/best_model.pth")
    parser.add_argument("--data_root", type=str,
                        default="/scratch/ap9283/deep_learning/data/test1")

    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--mode", type=str, choices=["val", "test"], default="val",
                        help="'val' = train vs val, sweep k; 'test' = train vs test, make CSV")

    parser.add_argument("--k", type=int, default=5,
                        help="k used in TEST mode")
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[1, 5, 10, 20, 50, 100],
                        help="list of k values to try in VAL mode")

    parser.add_argument("--output", type=str, default="submission_knn.csv",
                        help="output CSV path in TEST mode")

    args = parser.parse_args()
    main(args)