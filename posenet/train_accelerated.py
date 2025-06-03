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

from model import PoseNet

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

def main():
    # Initialize accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",  # Use mixed precision for faster training
        log_with="wandb" if wandb.api.api_key else None,
    )
    
    # Setup logging
    logger = setup_logging(accelerator)
    
    # Configuration
    config = {
        "learning_rate": 1e-5,
        "batch_size": 8,  # Per GPU batch size
        "epochs": 30,
        "train_seq_len": 6,
        "val_seq_len": 2,
        "image_size": (224, 224),
        "num_workers": 4,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_steps": 100,
        "gradient_clip_norm": 1.0,
        "save_every": 5,  # Save checkpoint every N epochs
    }
    
    # Initialize wandb only on main process
    if accelerator.is_main_process:
        wandb.init(
            project="diffposenet-accelerated",
            name=f"PoseNet-MultiGPU-{accelerator.num_processes}GPUs",
            config=config
        )
    
    accelerate_set_seed(42)
    
    # Dataset loading
    logger.info("============= Loading Datasets =============")
    train_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", 
        size=config["image_size"], 
        seq_len=config["train_seq_len"]
    )
    val_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/", 
        size=config["image_size"], 
        seq_len=config["val_seq_len"]
    )
    
    # Subset for faster training (adjust as needed)
    train_dataset = Subset(train_dataset, list(range(0, len(train_dataset), 5)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), 8)))
    
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
    
    pose_net = PoseNet()
    
    optimizer = optim.AdamW(
        pose_net.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    num_training_steps = len(train_loader) * config["epochs"]
    if config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps,
            eta_min=config["learning_rate"] * 0.01
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    pose_net, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        pose_net, optimizer, train_loader, val_loader, scheduler
    )
    
    logger.info(f"Training on {accelerator.num_processes} GPUs")
    logger.info(f"Effective batch size: {config['batch_size'] * accelerator.num_processes}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        pose_net.train()
        total_train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{config['epochs']} Training",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, (images, translations, rotations) in enumerate(progress_bar):
            with accelerator.accumulate(pose_net):
                t_pred, q_pred = pose_net(images)
                loss = pose_net.module.pose_loss(t_pred, q_pred, translations, rotations)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pose_net.parameters(), config["gradient_clip_norm"])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loss_gathered = accelerator.gather(loss.detach())
            total_train_loss += loss_gathered.mean().item()
            num_batches += 1
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        pose_net.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        progress_bar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{config['epochs']} Validation",
            disable=not accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for images, translations, rotations in progress_bar:
                t_pred, q_pred = pose_net(images)
                loss = pose_net.module.pose_loss(t_pred, q_pred, translations, rotations)
                
                loss_gathered = accelerator.gather(loss.detach())
                total_val_loss += loss_gathered.mean().item()
                num_val_batches += 1
                
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix({'val_loss': f'{loss.item():.6f}'})
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {avg_train_loss:.6f} - "
                f"Val Loss: {avg_val_loss:.6f} - "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if accelerator.is_main_process:
                # accelerator.save_state(f"ebu/checkpoints/best_model_epoch_{epoch+1}")
                logger.info(f"New best model saved at epoch {epoch+1}")
        
        # Regular checkpoint saving
        if (epoch + 1) % config["save_every"] == 0:
            if accelerator.is_main_process:
                # accelerator.save_state(f"ebu/checkpoints/checkpoint_epoch_{epoch+1}")
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    # Final model saving
    if accelerator.is_main_process:
        # Unwrap model for saving
        unwrapped_model = accelerator.unwrap_model(pose_net)
        torch.save(unwrapped_model.state_dict(), "pose_net_final.pth")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, config["epochs"] + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, config["epochs"] + 1), val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, config["epochs"] + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, config["epochs"] + 1), val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.title("Training and Validation Loss (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = "training_validation_loss_accelerated.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log plot to wandb
        wandb.log({"loss_plot": wandb.Image(plot_path)})
        
        logger.info("Training completed successfully!")
        wandb.finish()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    return accelerator.unwrap_model(pose_net)

if __name__ == "__main__":
    # Create checkpoints directory
    # os.makedirs("ebu/checkpoints", exist_ok=True)
    
    model = main() 