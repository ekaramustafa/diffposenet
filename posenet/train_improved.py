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
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_improved import ImprovedPoseNet
from dataset.tartanair import TartanAirDataset
from evaluation_metrics import TrajectoryEvaluator, convert_relative_to_absolute_poses

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
        "learning_rate": 1e-4,
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
        "train_subset_ratio": 0.5,
        "val_subset_ratio": 1, 
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
        os.makedirs("ebu/improved_posenet/configs", exist_ok=True)
        save_config(config, f"ebu/improved_posenet/configs/config_{timestamp}.json")
    
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
                
                # Check for NaN or infinite loss and terminate if found
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss detected at epoch {epoch+1}, batch {batch_idx}")
                    logger.error(f"Loss value: {loss.item()}")
                    logger.error(f"Loss components: {loss_dict}")
                    logger.error("Training terminated due to invalid loss. Check your data, model, or hyperparameters.")
                    
                    if accelerator.is_main_process:
                        wandb.log({
                            "error/invalid_loss_epoch": epoch + 1,
                            "error/invalid_loss_batch": batch_idx,
                            "error/loss_value": loss.item() if not torch.isnan(loss) else "NaN",
                            "error/loss_components": loss_dict
                        })
                        wandb.finish()
                    
                    # Wait for all processes and exit
                    accelerator.wait_for_everyone()
                    sys.exit(1)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pose_net.module.parameters(), config["gradient_clip_norm"])
                
                optimizer.step()
                
                # Learning rate scheduling
                # Temporarily disabled for testing - uncomment to re-enable
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
            
            # Initialize trajectory evaluator
            trajectory_evaluator = TrajectoryEvaluator(align_trajectories=True)
            
            # Collect all predictions and ground truth for trajectory evaluation
            all_pred_translations = []
            all_pred_quaternions = []
            all_gt_translations = []
            all_gt_quaternions = []
            
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
                    
                    # Collect predictions and ground truth for trajectory evaluation
                    # Convert relative poses to absolute poses for evaluation
                    batch_size, seq_len, _ = t_pred.shape
                    
                    for b in range(batch_size):
                        # Get relative poses for this sequence
                        rel_t = t_pred[b]  # [seq_len-1, 3]
                        rel_q = q_pred[b]  # [seq_len-1, 4]
                        gt_t = translations[b]  # [seq_len-1, 3]
                        gt_q = rotations[b]  # [seq_len-1, 4]
                        
                        # Convert to absolute poses
                        try:
                            abs_t_pred, abs_q_pred = convert_relative_to_absolute_poses(rel_t, rel_q)
                            abs_t_gt, abs_q_gt = convert_relative_to_absolute_poses(gt_t, gt_q)
                            
                            # Collect for trajectory evaluation
                            all_pred_translations.append(abs_t_pred)
                            all_pred_quaternions.append(abs_q_pred)
                            all_gt_translations.append(abs_t_gt)
                            all_gt_quaternions.append(abs_q_gt)
                        except Exception as e:
                            logger.warning(f"Error converting poses to absolute: {e}")
                            continue
                    
                    num_val_batches += 1
                    
                    if accelerator.is_local_main_process:
                        progress_bar.set_postfix({'val_loss': f'{loss.item():.6f}'})
            
            # Average validation metrics
            for key in val_metrics:
                val_metrics[key] /= num_val_batches
            
            # Compute trajectory evaluation metrics
            trajectory_metrics = {}
            if all_pred_translations and accelerator.is_main_process:
                try:
                    # Concatenate all trajectories
                    pred_translations = torch.cat(all_pred_translations, dim=0)
                    pred_quaternions = torch.cat(all_pred_quaternions, dim=0)
                    gt_translations = torch.cat(all_gt_translations, dim=0)
                    gt_quaternions = torch.cat(all_gt_quaternions, dim=0)
                    
                    logger.info(f"Evaluating trajectory with {len(pred_translations)} poses")
                    
                    # Compute ATE and RPE metrics
                    trajectory_metrics = trajectory_evaluator.evaluate_trajectory(
                        pred_translations, pred_quaternions,
                        gt_translations, gt_quaternions,
                        delta_t_list=[1, 5, 10]
                    )
                    
                    logger.info(f"Trajectory evaluation completed:")
                    logger.info(f"  ATE RMSE: {trajectory_metrics.get('ATE_RMSE', 'N/A'):.4f}")
                    logger.info(f"  RPE Trans (Δt=1): {trajectory_metrics.get('RPE_trans_RMSE_dt1', 'N/A'):.4f}")
                    logger.info(f"  RPE Rot (Δt=1): {trajectory_metrics.get('RPE_rot_mean_deg_dt1', 'N/A'):.2f}°")
                    
                except Exception as e:
                    logger.error(f"Error computing trajectory metrics: {e}")
                    trajectory_metrics = {
                        'ATE_RMSE': float('inf'),
                        'RPE_trans_RMSE_dt1': float('inf'),
                        'RPE_rot_mean_deg_dt1': float('inf')
                    }
            
            val_losses.append(val_metrics['total_loss'])
        else:
            val_metrics = {'total_loss': val_losses[-1] if val_losses else float('inf')}
            trajectory_metrics = {}
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
                
                # Add trajectory metrics to logging
                if trajectory_metrics:
                    # ATE metrics
                    if 'ATE_RMSE' in trajectory_metrics:
                        log_dict.update({
                            "val/ATE_RMSE": trajectory_metrics['ATE_RMSE'],
                            "val/ATE_mean": trajectory_metrics.get('ATE_mean', 0),
                            "val/ATE_std": trajectory_metrics.get('ATE_std', 0),
                            "val/ATE_median": trajectory_metrics.get('ATE_median', 0),
                        })
                    
                    # RPE metrics for different delta_t values
                    for dt in [1, 5, 10]:
                        if f'RPE_trans_RMSE_dt{dt}' in trajectory_metrics:
                            log_dict.update({
                                f"val/RPE_trans_RMSE_dt{dt}": trajectory_metrics[f'RPE_trans_RMSE_dt{dt}'],
                                f"val/RPE_rot_mean_deg_dt{dt}": trajectory_metrics.get(f'RPE_rot_mean_deg_dt{dt}', 0),
                                f"val/RPE_trans_mean_dt{dt}": trajectory_metrics.get(f'RPE_trans_mean_dt{dt}', 0),
                                f"val/RPE_rot_std_dt{dt}": trajectory_metrics.get(f'RPE_rot_std_dt{dt}', 0),
                            })
            
            wandb.log(log_dict)
            
            # Enhanced console logging with trajectory metrics
            log_message = (
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {epoch_metrics['total_loss']:.6f} - "
                f"Val Loss: {val_metrics['total_loss']:.6f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            if trajectory_metrics and (epoch + 1) % config["eval_every"] == 0:
                ate_rmse = trajectory_metrics.get('ATE_RMSE', float('inf'))
                rpe_trans = trajectory_metrics.get('RPE_trans_RMSE_dt1', float('inf'))
                rpe_rot = trajectory_metrics.get('RPE_rot_mean_deg_dt1', float('inf'))
                
                log_message += (
                    f" - ATE: {ate_rmse:.4f} - "
                    f"RPE_t: {rpe_trans:.4f} - "
                    f"RPE_r: {rpe_rot:.2f}°"
                )
            
            logger.info(log_message)
        
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