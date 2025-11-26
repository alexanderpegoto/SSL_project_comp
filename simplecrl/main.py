import torch
import os
from train import train_epoch
from data import get_competition_dataloaders
from model import ResNet
import glob

def get_latest_checkpoint(save_dir):
    """Find the most recent checkpoint."""
    checkpoints = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]


def cleanup_old_checkpoints(save_dir, keep_last=3):
    """Keep only the last N checkpoints, delete older ones."""
    checkpoints = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if len(checkpoints) <= keep_last:
        return
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Delete all but last N
    for old_checkpoint in checkpoints[:-keep_last]:
        os.remove(old_checkpoint)
        print(f'Deleted old checkpoint: {os.path.basename(old_checkpoint)}')


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.5
    weight_decay = 1e-4
    temperature = 0.5
    
    # Cache directory for downloaded data
    data_dir = '/scratch/ap9283/deep_learning/data/extracted'
    os.makedirs(data_dir, exist_ok=True)
    
    # Cache directory for downloaded data
    save_dir = '/scratch/ap9283/deep_learning/SSL_project_comp/simplecrl/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Model
    model = ResNet(base_model='resnet50', out_dim=128).to(device)
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    # Data
    train_loader = get_competition_dataloaders(
        data_path=data_dir,
        batch_size=batch_size,
        num_workers=4,
        image_size=96
    )
    
    print(f'Training batches per epoch: {len(train_loader)}')
        
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )
    
    # Track best loss for saving
    best_loss = float('inf')
    start_epoch = 0
    

    # Resuming in case connection breaks
    latest_checkpoint = get_latest_checkpoint(save_dir)
    if latest_checkpoint:
        print(f'\n*** RESUMING from checkpoint: {latest_checkpoint} ***')
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Restore scheduler state
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
        
        print(f'Resumed from epoch {checkpoint["epoch"] + 1}')
        print(f'Best loss so far: {best_loss:.4f}\n')
    else:
        print('\n*** Starting training from scratch ***\n')
    
    # Training loop - FIXED: use 'epoch' variable!
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, temperature, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print with learning rate
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
            
            # Clean up old checkpoints, keep last 3
            cleanup_old_checkpoints(save_dir, keep_last=3)
        
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
            print(f'✓ New best model saved! Loss: {avg_loss:.4f}')
    
    print('\nTraining complete!')


if __name__ == '__main__':
    main()