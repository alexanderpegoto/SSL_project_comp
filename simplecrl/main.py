import torch
import os
from train import train_epoch
from data import get_cifar10_dataloaders
from model import ResNet


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = ResNet(base_model='resnet50', out_dim=128).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # Data
    train_loader = get_cifar10_dataloaders(batch_size=512)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_loader))
    
    # Training parameters
    num_epochs = 1000
    temperature = 0.5
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best loss for saving
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, temperature, device)
        
        # Step scheduler
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_path)
            print(f'New best model saved! Loss: {avg_loss:.4f}')
    
    print('\nTraining complete!')


if __name__ == '__main__':
    main()