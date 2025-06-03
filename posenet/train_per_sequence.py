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
from collections import defaultdict
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PoseNet, PoseNetDino
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

def get_experiment_configs():
    base_config = {
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        
        "learning_rate": 1e-5,
        "batch_size": 8,
        "epochs": 30,
        "train_seq_len": 6,
        "val_seq_len": 6,
        "image_size": (224, 224),
        "num_workers": 4,
        "weight_decay": 1e-4,
        "skip": 1,
        
        "lambda_q": 1.0,
        
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "min_lr_ratio": 0.01,
        
        "gradient_clip_norm": 0.5, 
        "gradient_accumulation_steps": 1,
        
        "save_every": 5,
        "eval_every": 1,
        
        "train_subset_ratio": 1,
        "val_subset_ratio": 1,
        
        "evaluate_per_sequence": True,
    }
    
    configs = [
        {
            **base_config,
            "backbone": "dino",
            "freeze": True,
            "experiment_name": "dino",
            "model_size": "base"
        },
        {
            **base_config,
            "backbone": "vgg16",
            "freeze": True,
            "experiment_name": "vgg16_frozen",
            "model_size": "base"
        },
        {
            **base_config,
            "backbone": "vgg16",
            "freeze": False,
            "experiment_name": "vgg16_unfrozen"
        }
    ]
    
    return configs

def setup_datasets(config, logger):
    logger.info("============= Loading Datasets =============")
    train_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", 
        size=config["image_size"], 
        seq_len=config["train_seq_len"],
        track_sequences=False,
        skip=config["skip"]
    )
    
    val_dataset = TartanAirDataset(
        root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/cvpr_data/", 
        size=config["image_size"], 
        seq_len=config["val_seq_len"],
        track_sequences=True
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
    
    original_val_dataset = val_dataset.dataset
    if hasattr(original_val_dataset, 'get_sequence_names') and original_val_dataset.track_sequences:
        sequence_names = original_val_dataset.get_sequence_names()
        logger.info(f"Sequences in validation set: {sequence_names}")
    
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
    
    return train_loader, val_loader

def setup_model_and_optimizer(config, logger):
    if config["backbone"] == "dino":
        pose_net = PoseNetDino(
            model_size=config["model_size"],
            freeze_dino=config["freeze"]
        )
    else:
        pose_net = PoseNet(
            backbone=config["backbone"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            freeze_cnn=config["freeze"]
        )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in pose_net.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(
        pose_net.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    return pose_net, optimizer

def compute_sequence_ate_metrics(pred_translations, pred_quaternions, gt_translations, gt_quaternions, sequence_names, evaluator):
    sequence_data = defaultdict(lambda: {
        'pred_translations': [],
        'pred_quaternions': [],
        'gt_translations': [],
        'gt_quaternions': []
    })
    
    for i, seq_name in enumerate(sequence_names):
        if seq_name and seq_name != 'unknown':
            sequence_data[seq_name]['pred_translations'].append(pred_translations[i])
            sequence_data[seq_name]['pred_quaternions'].append(pred_quaternions[i])
            sequence_data[seq_name]['gt_translations'].append(gt_translations[i])
            sequence_data[seq_name]['gt_quaternions'].append(gt_quaternions[i])
    
    per_sequence_metrics = {}
    all_ate_values = []
    
    for seq_name, data in sequence_data.items():
        if len(data['pred_translations']) > 0:
            try:
                seq_pred_t = torch.stack(data['pred_translations'])
                seq_pred_q = torch.stack(data['pred_quaternions'])
                seq_gt_t = torch.stack(data['gt_translations'])
                seq_gt_q = torch.stack(data['gt_quaternions'])
                
                ate_metrics = evaluator.compute_ate(seq_pred_t, seq_pred_q, seq_gt_t, seq_gt_q)
                per_sequence_metrics[seq_name] = ate_metrics
                all_ate_values.append(ate_metrics['ATE_RMSE'])
                
            except Exception as e:
                logging.warning(f"Error computing ATE for sequence {seq_name}: {e}")
                per_sequence_metrics[seq_name] = {'ATE_RMSE': float('inf')}
                all_ate_values.append(float('inf'))
    
    if all_ate_values:
        valid_ate_values = [ate for ate in all_ate_values if not np.isinf(ate)]
        if valid_ate_values:
            aggregated_metrics = {
                'ATE_mean_across_sequences': np.mean(valid_ate_values),
                'ATE_std_across_sequences': np.std(valid_ate_values),
                'ATE_median_across_sequences': np.median(valid_ate_values),
                'ATE_min_across_sequences': np.min(valid_ate_values),
                'ATE_max_across_sequences': np.max(valid_ate_values),
                'num_valid_sequences': len(valid_ate_values),
                'num_total_sequences': len(all_ate_values)
            }
        else:
            aggregated_metrics = {
                'ATE_mean_across_sequences': float('inf'),
                'ATE_std_across_sequences': float('inf'),
                'ATE_median_across_sequences': float('inf'),
                'ATE_min_across_sequences': float('inf'),
                'ATE_max_across_sequences': float('inf'),
                'num_valid_sequences': 0,
                'num_total_sequences': len(all_ate_values)
            }
    else:
        aggregated_metrics = {
            'ATE_mean_across_sequences': float('inf'),
            'num_valid_sequences': 0,
            'num_total_sequences': 0
        }
    
    return {
        'per_sequence': per_sequence_metrics,
        'aggregated': aggregated_metrics
    }

def train_epoch(pose_net, train_loader, optimizer, accelerator, config, epoch, logger):
    """Train for one epoch"""
    pose_net.train()
    epoch_metrics = {
        'total_loss': 0.0,
        'translation_loss': 0.0,
        'quaternion_loss': 0.0,
    }
    
    if accelerator.is_local_main_process:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    else:
        progress_bar = train_loader
    
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(progress_bar):
        if len(batch_data) == 4:
            images, translations, rotations, _ = batch_data
        else:
            images, translations, rotations = batch_data
        
        batch_size, seq_len = images.shape[:2]
        
        optimizer.zero_grad()
        
        with accelerator.accumulate(pose_net):
            pred_translations, pred_rotations = pose_net(images)
            
            total_loss, loss_dict = pose_net.module.pose_loss(
                pred_translations, pred_rotations, 
                translations, rotations,
                lambda_q=config["lambda_q"],
            )
            
            accelerator.backward(total_loss)
            
            if config["gradient_clip_norm"] > 0:
                accelerator.clip_grad_norm_(pose_net.module.parameters(), config["gradient_clip_norm"])
            
            optimizer.step()
        
        epoch_metrics['total_loss'] += loss_dict['total_loss']
        epoch_metrics['translation_loss'] += loss_dict['translation_loss']
        epoch_metrics['quaternion_loss'] += loss_dict['quaternion_loss']
        num_batches += 1
        
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({'loss': f'{total_loss.item():.6f}'})
    
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_metrics

def validate_epoch(pose_net, val_loader, accelerator, config, epoch, trajectory_evaluator, logger):
    """Validate for one epoch"""
    pose_net.eval()
    val_metrics = {
        'total_loss': 0.0,
        'translation_loss': 0.0,
        'quaternion_loss': 0.0,
    }
    
    all_pred_translations = []
    all_pred_quaternions = []
    all_gt_translations = []
    all_gt_quaternions = []
    all_sequence_names = []
    
    if accelerator.is_local_main_process:
        progress_bar = tqdm(val_loader, desc=f"Validation {epoch+1}")
    else:
        progress_bar = val_loader
    
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_data in progress_bar:
            if len(batch_data) == 4:
                images, translations, rotations, sequence_names = batch_data
            else:
                images, translations, rotations = batch_data
                sequence_names = None
            
            batch_size, seq_len = images.shape[:2]
            
            pred_translations, pred_rotations = pose_net(images)
            
            loss, loss_dict = pose_net.module.pose_loss(
                pred_translations, pred_rotations,
                translations, rotations,
                lambda_q=config["lambda_q"],
            )
            
            val_metrics['total_loss'] += loss_dict['total_loss']
            val_metrics['translation_loss'] += loss_dict['translation_loss']
            val_metrics['quaternion_loss'] += loss_dict['quaternion_loss']
            
            for b in range(batch_size):
                try:
                    abs_t_pred, abs_q_pred = convert_relative_to_absolute_poses(
                        pred_translations[b], pred_rotations[b]
                    )
                    abs_t_gt, abs_q_gt = convert_relative_to_absolute_poses(
                        translations[b], rotations[b]
                    )
                    
                    all_pred_translations.append(abs_t_pred)
                    all_pred_quaternions.append(abs_q_pred)
                    all_gt_translations.append(abs_t_gt)
                    all_gt_quaternions.append(abs_q_gt)
                    
                    if sequence_names is not None:
                        if isinstance(sequence_names[b], str):
                            all_sequence_names.extend([sequence_names[b]] * len(abs_t_pred))
                        else:
                            all_sequence_names.extend(['unknown'] * len(abs_t_pred))
                    else:
                        all_sequence_names.extend(['unknown'] * len(abs_t_pred))
                        
                except Exception as e:
                    logger.warning(f"Error converting poses to absolute: {e}")
                    continue
            
            num_val_batches += 1
            
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({'val_loss': f'{loss.item():.6f}'})
    
    for key in val_metrics:
        val_metrics[key] /= num_val_batches
    
    sequence_metrics = {}
    if all_pred_translations and accelerator.is_main_process and config["evaluate_per_sequence"]:
        try:
            pred_translations_tensor = torch.cat(all_pred_translations, dim=0)
            pred_quaternions_tensor = torch.cat(all_pred_quaternions, dim=0)
            gt_translations_tensor = torch.cat(all_gt_translations, dim=0)
            gt_quaternions_tensor = torch.cat(all_gt_quaternions, dim=0)
            
            logger.info(f"Evaluating trajectory with {len(pred_translations_tensor)} poses across sequences")
            
            sequence_metrics = compute_sequence_ate_metrics(
                pred_translations_tensor, pred_quaternions_tensor,
                gt_translations_tensor, gt_quaternions_tensor,
                all_sequence_names, trajectory_evaluator
            )
            
            logger.info(f"Per-sequence ATE evaluation completed:")
            for seq_name, metrics in sequence_metrics['per_sequence'].items():
                ate_rmse = metrics.get('ATE_RMSE', float('inf'))
                logger.info(f"  {seq_name}: ATE_RMSE = {ate_rmse:.4f}")
            
            agg_metrics = sequence_metrics['aggregated']
            logger.info(f"Aggregated ATE metrics:")
            logger.info(f"  Mean ATE across sequences: {agg_metrics.get('ATE_mean_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Std ATE across sequences: {agg_metrics.get('ATE_std_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Valid sequences: {agg_metrics.get('num_valid_sequences', 0)}/{agg_metrics.get('num_total_sequences', 0)}")
            
        except Exception as e:
            logger.error(f"Error computing per-sequence trajectory metrics: {e}")
            sequence_metrics = {
                'per_sequence': {},
                'aggregated': {
                    'ATE_mean_across_sequences': float('inf'),
                    'num_valid_sequences': 0
                }
            }
    
    return val_metrics, sequence_metrics

def save_and_log_results(pose_net, accelerator, config, final_sequence_metrics, train_losses, val_losses, logger, timestamp):
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(pose_net)
        model_path = f"pose_net_improved_per_sequence_{config['experiment_name']}_final.pth"
        torch.save(unwrapped_model.state_dict(), model_path)
        
        if final_sequence_metrics and config["evaluate_per_sequence"]:
            logger.info("============= FINAL PER-SEQUENCE SUMMARY =============")
            
            per_seq_metrics = final_sequence_metrics.get('per_sequence', {})
            agg_metrics = final_sequence_metrics.get('aggregated', {})
            
            for seq_name, metrics in per_seq_metrics.items():
                ate_rmse = metrics.get('ATE_RMSE', float('inf'))
                logger.info(f"Final {seq_name}: ATE_RMSE = {ate_rmse:.4f}")
            
            logger.info(f"Final Aggregated Metrics:")
            logger.info(f"  Mean ATE across sequences: {agg_metrics.get('ATE_mean_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Std ATE across sequences: {agg_metrics.get('ATE_std_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Median ATE across sequences: {agg_metrics.get('ATE_median_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Min ATE: {agg_metrics.get('ATE_min_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Max ATE: {agg_metrics.get('ATE_max_across_sequences', 'N/A'):.4f}")
            logger.info(f"  Valid sequences: {agg_metrics.get('num_valid_sequences', 0)}/{agg_metrics.get('num_total_sequences', 0)}")
            
            final_metrics_path = f"ebu/improved_posenet_per_seq/final_metrics_{config['experiment_name']}_{timestamp}.json"
            with open(final_metrics_path, 'w') as f:
                json.dump(final_sequence_metrics, f, indent=2, default=str)
            logger.info(f"Final metrics saved to: {final_metrics_path}")
            
            final_log_dict = {}
            for seq_name, metrics in per_seq_metrics.items():
                if 'ATE_RMSE' in metrics:
                    final_log_dict[f"final/ATE_{seq_name}"] = metrics['ATE_RMSE']
            
            final_log_dict.update({
                "final/ATE_mean_across_sequences": agg_metrics.get('ATE_mean_across_sequences', float('inf')),
                "final/ATE_std_across_sequences": agg_metrics.get('ATE_std_across_sequences', float('inf')),
                "final/ATE_median_across_sequences": agg_metrics.get('ATE_median_across_sequences', float('inf')),
                "final/num_valid_sequences": agg_metrics.get('num_valid_sequences', 0),
            })
            
            wandb.log(final_log_dict)
        
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
        if final_sequence_metrics:
            per_seq_metrics = final_sequence_metrics.get('per_sequence', {})
            seq_names = list(per_seq_metrics.keys())
            ate_values = [per_seq_metrics[seq].get('ATE_RMSE', 0) for seq in seq_names]
            
            if seq_names and ate_values:
                plt.bar(seq_names, ate_values)
                plt.xlabel("Sequences")
                plt.ylabel("Final ATE RMSE")
                plt.title("Final ATE per Sequence")
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No sequence data available', ha='center', va='center')
                plt.title("Final ATE per Sequence")
        else:
            plt.text(0.5, 0.5, 'No sequence data available', ha='center', va='center')
            plt.title("Final ATE per Sequence")
        
        plt.tight_layout()
        plot_path = f"training_curves_per_sequence_{config['experiment_name']}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        wandb.log({"training_curves": wandb.Image(plot_path)})

def run_experiment(config):
    """Run a single experiment with the given configuration"""
    try:
        wandb.login(key="fb69f02bed97fefd3f9a152ab12abb8b32896b3d")
    except Exception as e:
        print(f"Warning: wandb login failed: {e}")
        print("You can set WANDB_API_KEY environment variable or run 'wandb login' manually")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="wandb",
    )
    
    logger = setup_logging(accelerator)
    
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"PoseNet-PerSeq-{config['experiment_name']}-{accelerator.num_processes}GPUs-{timestamp}"
        
        wandb.init(
            project="diffposenet-per-sequence",
            name=run_name,
            config=config
        )
        
        os.makedirs("ebu/improved_posenet_per_seq/configs", exist_ok=True)
        save_config(config, f"ebu/improved_posenet_per_seq/configs/config_{config['experiment_name']}_{timestamp}.json")
    
    accelerate_set_seed(42)
    
    train_loader, val_loader = setup_datasets(config, logger)
    pose_net, optimizer = setup_model_and_optimizer(config, logger)
    
    trajectory_evaluator = TrajectoryEvaluator(align_trajectories=True)
    
    pose_net, optimizer, train_loader, val_loader = accelerator.prepare(
        pose_net, optimizer, train_loader, val_loader
    )   
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    final_sequence_metrics = {}
    
    logger.info("============= Starting Training =============")
    
    for epoch in range(config["epochs"]):
        epoch_metrics = train_epoch(pose_net, train_loader, optimizer, accelerator, config, epoch, logger)
        train_losses.append(epoch_metrics['total_loss'])
        
        if (epoch + 1) % config["eval_every"] == 0:
            val_metrics, sequence_metrics = validate_epoch(
                pose_net, val_loader, accelerator, config, epoch, trajectory_evaluator, logger
            )
            
            if epoch == config["epochs"] - 1:
                final_sequence_metrics = sequence_metrics
        else:
            val_metrics = {'total_loss': val_losses[-1] if val_losses else float('inf')}
            sequence_metrics = {}
        
        val_losses.append(val_metrics['total_loss'])
        
        if accelerator.is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train/total_loss": epoch_metrics['total_loss'],
                "train/translation_loss": epoch_metrics['translation_loss'],
                "train/quaternion_loss": epoch_metrics['quaternion_loss'],
            }
            
            if (epoch + 1) % config["eval_every"] == 0:
                log_dict.update({
                    "val/total_loss": val_metrics['total_loss'],
                    "val/translation_loss": val_metrics['translation_loss'],
                    "val/quaternion_loss": val_metrics['quaternion_loss'],
                })
                
                if sequence_metrics and config["evaluate_per_sequence"]:
                    agg_metrics = sequence_metrics.get('aggregated', {})
                    if 'ATE_mean_across_sequences' in agg_metrics:
                        log_dict.update({
                            "val/ATE_mean_across_sequences": agg_metrics['ATE_mean_across_sequences'],
                            "val/ATE_std_across_sequences": agg_metrics.get('ATE_std_across_sequences', 0),
                            "val/ATE_median_across_sequences": agg_metrics.get('ATE_median_across_sequences', 0),
                            "val/ATE_min_across_sequences": agg_metrics.get('ATE_min_across_sequences', 0),
                            "val/ATE_max_across_sequences": agg_metrics.get('ATE_max_across_sequences', 0),
                            "val/num_valid_sequences": agg_metrics.get('num_valid_sequences', 0),
                        })
                    
                    per_seq_metrics = sequence_metrics.get('per_sequence', {})
                    for seq_name, metrics in per_seq_metrics.items():
                        if 'ATE_RMSE' in metrics:
                            log_dict[f"val/ATE_{seq_name}"] = metrics['ATE_RMSE']
            
            wandb.log(log_dict)
            
            log_message = (
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {epoch_metrics['total_loss']:.6f} - "
                f"Val Loss: {val_metrics['total_loss']:.6f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            if sequence_metrics and (epoch + 1) % config["eval_every"] == 0 and config["evaluate_per_sequence"]:
                agg_metrics = sequence_metrics.get('aggregated', {})
                mean_ate = agg_metrics.get('ATE_mean_across_sequences', float('inf'))
                num_valid = agg_metrics.get('num_valid_sequences', 0)
                
                log_message += f" - Mean ATE: {mean_ate:.4f} ({num_valid} seqs)"
            
            logger.info(log_message)
        
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            if accelerator.is_main_process:
                checkpoint_dir = f"ebu/improved_posenet_per_seq/checkpoints/{config['experiment_name']}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.save_state(f"{checkpoint_dir}/best_model_epoch_{epoch+1}")
                logger.info(f"New best model saved at epoch {epoch+1}")
        
        if (epoch + 1) % config["save_every"] == 0:
            if accelerator.is_main_process:
                checkpoint_dir = f"ebu/improved_posenet_per_seq/checkpoints/{config['experiment_name']}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.save_state(f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}")
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_and_log_results(pose_net, accelerator, config, final_sequence_metrics, train_losses, val_losses, logger, timestamp)
        
        logger.info(f"Training {config['experiment_name']} completed successfully!")
        wandb.finish()
    
    accelerator.wait_for_everyone()
    
    return accelerator.unwrap_model(pose_net)

def main():
    parser = argparse.ArgumentParser(description='Train PoseNet with multiple configurations')
    parser.add_argument('--config_idx', type=int, default=None, 
                        help='Index of configuration to run (0: dino, 1: vgg16_frozen, 2: vgg16_unfrozen). If not specified, runs all configs.')
    args = parser.parse_args()
    
    os.makedirs("ebu/improved_posenet_per_seq/checkpoints", exist_ok=True)
    os.makedirs("ebu/improved_posenet_per_seq/configs", exist_ok=True)
    
    configs = get_experiment_configs()
    
    if args.config_idx is not None:
        if 0 <= args.config_idx < len(configs):
            print(f"Running single experiment: {configs[args.config_idx]['experiment_name']}")
            model = run_experiment(configs[args.config_idx])
        else:
            print(f"Invalid config index {args.config_idx}. Available indices: 0-{len(configs)-1}")
            return
    else:
        print(f"Running all {len(configs)} experiments...")
        for i, config in enumerate(configs):
            print(f"\n{'='*50}")
            print(f"Starting experiment {i+1}/{len(configs)}: {config['experiment_name']}")
            print(f"{'='*50}")
            model = run_experiment(config)
            print(f"Completed experiment: {config['experiment_name']}")

if __name__ == "__main__":
    main() 