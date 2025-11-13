import torch
import torch.nn.functional as F
import torch.nn as nn

def info_nce_loss(features, temperature, device):
    batch_size = features.shape[0] // 2
    
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Create mask for positive pairs
    positives_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
    for i in range(batch_size):
        positives_mask[i, batch_size + i] = True
        positives_mask[batch_size + i, i] = True
    
    # Remove diagonal (self-similarity)
    diag_mask = torch.eye(2*batch_size, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix[~diag_mask].view(2*batch_size, -1)
    positives_mask = positives_mask[~diag_mask].view(2*batch_size, -1)
    
    # Extract positives and negatives
    positives = similarity_matrix[positives_mask].view(2*batch_size, -1)
    negatives = similarity_matrix[~positives_mask].view(2*batch_size, -1)
    
    # Create logits
    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    
    # Labels: first column is always the positive
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=device)
    
    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    
    return loss

def train_epoch(model, train_loader, optimizer, temperature, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # images is a tuple of (view1, view2)
        view1, view2 = images
        images = torch.cat([view1, view2], dim=0).to(device)
        
        # Forward pass - use projection features (z)
        _, features = model(images)
        
        # Compute loss
        loss = info_nce_loss(features, temperature, device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)