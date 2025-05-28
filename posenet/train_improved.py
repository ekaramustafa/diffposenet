import sys
import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import logging
import json
from datetime import datetime

from model_improved import ImprovedPoseNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.tartanair import TartanAirDataset

def setup_logging(accelerator):
    """Setup logging that works with multi-GPU training"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if accelerator.is_local_main_process:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    return logger

def save_config(config, save_path):
    """Save training configuration"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    try:
        wandb.login(key="fb69f02bed97fefd3f9a152ab12abb8b32896b3d")
    except Exception as e:
        print(f"Warning: wandb login failed: {e}")
        print("You can set WANDB_API_KEY environment variable or run 'wandb login' manually")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",  # Use mixed precision for faster training
        log_with="wandb",
    )
    
    # Setup logging
    logger = setup_logging(accelerator)
    
    # Enhanced configuration
    config = {
        # Model parameters
        "backbone": "resnet50",  # resnet50, resnet34, or vgg16
        "hidden_dim": 256,
        "num_layers": 2,
        "use_attention": True,
        
        # Training parameters
        "learning_rate": 1e-5,
        "batch_size": 6,  # Per GPU batch size (reduced for ResNet50)
        "epochs": 50,
        "seq_len": 6,
        "image_size": (224, 224),
        "num_workers": 4,
        "weight_decay": 1e-4,
        
        # Loss parameters
        "lambda_q": 1.0,
        "lambda_smooth": 0.05,  # Reduced smoothness weight
        
        # Scheduler parameters
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "min_lr_ratio": 0.01,
        
        # Optimization parameters
        "gradient_clip_norm": 0.5, 
        "gradient_accumulation_steps": 1,
        
        # Checkpoint parameters
        "save_every": 5,
        "eval_every": 1,
        
        # Data parameters
        "train_subset_ratio": 0.5,  # Use 20% of training data for faster iteration
        "val_subset_ratio": 0.5,   # Use 50% of validation data
    }
    
    # Initialize wandb only on main process
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ImprovedPoseNet-{config['backbone']}-{accelerator.num_processes}GPUs-{timestamp}"
        
        wandb.init(
            project="diffposenet-improved",
            name=run_name,
            config=config
        )
        
        # Save config
        os.makedirs("configs", exist_ok=True)
        save_config(config, f"configs/config_{timestamp}.json")
    
    # Set random seed for reproducibility across all processes
    accelerate_set_seed(42)
    
    # Dataset loading
    logger.info("============= Loading Datasets =============")
    train_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", 
        size=config["image_size"], 
        seq_len=config["seq_len"]
    )
    val_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/", 
        size=config["image_size"], 
        seq_len=config["seq_len"]
    )
    
    # Create subsets for faster training
    train_size = int(len(train_dataset) * config["train_subset_ratio"])
    val_size = int(len(val_dataset) * config["val_subset_ratio"])
    
    train_indices = list(range(0, len(train_dataset), len(train_dataset) // train_size))[:train_size]
    val_indices = list(range(0, len(val_dataset), len(val_dataset) // val_size))[:val_size]
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"], 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=True,
        drop_last=False
    )
    
    # Model initialization
    pose_net = ImprovedPoseNet(
        backbone=config["backbone"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        use_attention=config["use_attention"]
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in pose_net.parameters() if p.requires_grad):,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        pose_net.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]
    
    if config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - warmup_steps,
            eta_min=config["learning_rate"] * config["min_lr_ratio"]
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Prepare everything with accelerator
    pose_net, optimizer, train_loader, val_loader = accelerator.prepare(
        pose_net, optimizer, train_loader, val_loader
    )
    
    logger.info(f"Training on {accelerator.num_processes} GPUs")
    logger.info(f"Effective batch size: {config['batch_size'] * accelerator.num_processes}")
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    global_step = 0
    
    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        pose_net.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'translation_loss': 0.0,
            'rotation_loss': 0.0,
            'smoothness_t': 0.0,
            'smoothness_q': 0.0
        }
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{config['epochs']} Training",
            disable=True  # Disable progress bar to reduce logging
        )
        
        for batch_idx, (images, translations, rotations) in enumerate(progress_bar):
            with accelerator.accumulate(pose_net):
                # Forward pass
                t_pred, q_pred = pose_net(images)
                loss, loss_dict = pose_net.module.pose_loss(
                    t_pred, q_pred, translations, rotations,
                    lambda_q=config["lambda_q"],
                    lambda_smooth=config["lambda_smooth"]
                )
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                    continue
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pose_net.module.parameters(), config["gradient_clip_norm"])
                
                optimizer.step()
                
                # Learning rate scheduling
                if global_step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            # Gather metrics from all processes
            for key in epoch_metrics:
                if key in loss_dict:
                    metric_tensor = torch.tensor(loss_dict[key], device=accelerator.device)
                    gathered_metric = accelerator.gather(metric_tensor)
                    epoch_metrics[key] += gathered_metric.mean().item()
            
            num_batches += 1
            
            # Update progress bar
            if accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        train_losses.append(epoch_metrics['total_loss'])
        
        # Validation phase
        if (epoch + 1) % config["eval_every"] == 0:
            pose_net.eval()
            val_metrics = {
                'total_loss': 0.0,
                'translation_loss': 0.0,
                'rotation_loss': 0.0,
                'smoothness_t': 0.0,
                'smoothness_q': 0.0
            }
            num_val_batches = 0
            
            progress_bar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{config['epochs']} Validation",
                disable=True  # Disable progress bar to reduce logging
            )
            
            with torch.no_grad():
                for images, translations, rotations in progress_bar:
                    t_pred, q_pred = pose_net(images)
                    loss, loss_dict = pose_net.module.pose_loss(
                        t_pred, q_pred, translations, rotations,
                        lambda_q=config["lambda_q"],
                        lambda_smooth=config["lambda_smooth"]
                    )
                    
                    # Gather metrics from all processes
                    for key in val_metrics:
                        if key in loss_dict:
                            metric_tensor = torch.tensor(loss_dict[key], device=accelerator.device)
                            gathered_metric = accelerator.gather(metric_tensor)
                            val_metrics[key] += gathered_metric.mean().item()
                    
                    num_val_batches += 1
                    
                    if accelerator.is_local_main_process:
                        progress_bar.set_postfix({'val_loss': f'{loss.item():.6f}'})
            
            # Average validation metrics
            for key in val_metrics:
                val_metrics[key] /= num_val_batches
            
            val_losses.append(val_metrics['total_loss'])
        else:
            val_metrics = {'total_loss': val_losses[-1] if val_losses else float('inf')}
            val_losses.append(val_metrics['total_loss'])
        
        # Logging (only on main process)
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train/total_loss": epoch_metrics['total_loss'],
                "train/translation_loss": epoch_metrics['translation_loss'],
                "train/rotation_loss": epoch_metrics['rotation_loss'],
                "train/smoothness_t": epoch_metrics['smoothness_t'],
                "train/smoothness_q": epoch_metrics['smoothness_q'],
            }
            
            if (epoch + 1) % config["eval_every"] == 0:
                log_dict.update({
                    "val/total_loss": val_metrics['total_loss'],
                    "val/translation_loss": val_metrics['translation_loss'],
                    "val/rotation_loss": val_metrics['rotation_loss'],
                    "val/smoothness_t": val_metrics['smoothness_t'],
                    "val/smoothness_q": val_metrics['smoothness_q'],
                })
            
            wandb.log(log_dict)
            
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {epoch_metrics['total_loss']:.6f} - "
                f"Val Loss: {val_metrics['total_loss']:.6f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Save checkpoint
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            if accelerator.is_main_process:
                accelerator.save_state(f"ebu/improved_posenet/checkpoints/best_model_epoch_{epoch+1}")
                logger.info(f"New best model saved at epoch {epoch+1}")
        
        # Regular checkpoint saving
        if (epoch + 1) % config["save_every"] == 0:
            if accelerator.is_main_process:
                accelerator.save_state(f"ebu/improved_posenet/checkpoints/checkpoint_epoch_{epoch+1}")
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    # Final model saving
    if accelerator.is_main_process:
        # Unwrap model for saving
        unwrapped_model = accelerator.unwrap_model(pose_net)
        torch.save(unwrapped_model.state_dict(), "pose_net_improved_final.pth")
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.title("Training and Validation Loss (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        # Plot learning rate schedule
        lr_history = []
        unwrapped_model = accelerator.unwrap_model(pose_net)
        temp_optimizer = optim.AdamW(unwrapped_model.parameters(), lr=config["learning_rate"])
        temp_scheduler = optim.lr_scheduler.CosineAnnealingLR(temp_optimizer, T_max=num_training_steps)
        for _ in range(num_training_steps):
            lr_history.append(temp_optimizer.param_groups[0]['lr'])
            temp_scheduler.step()
        
        plt.plot(lr_history)
        plt.xlabel("Training Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = "training_curves_improved.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log plot to wandb
        wandb.log({"training_curves": wandb.Image(plot_path)})
        
        logger.info("Training completed successfully!")
        wandb.finish()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    return accelerator.unwrap_model(pose_net)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("ebu/improved_posenet/checkpoints", exist_ok=True)
    os.makedirs("ebu/improved_posenet/configs", exist_ok=True)
    
    model = main() 