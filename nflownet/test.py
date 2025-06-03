import os
import sys
import time
import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import flow_to_image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dataset.nflownet_test_dataloader2 import nflownet_test_dataloader
from nflownet.model import NFlowNet

def ProjectionEndpointError(normal_pred: torch.Tensor, opt_gt: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    """
    Computes the Projected Endpoint Error (PEE) between optical flow u and predicted normal flow nhat.

    Args:
        u (Tensor): ground truth optical flow of shape (B, 2, H, W)
        nhat (Tensor): predicted normal flow of shape (B, 2, H, W)
        eps (float): small value to avoid division by zero

    Returns:
        Tensor: mean PEE per sample (B,)
    """    
    normal_norm = torch.norm(normal_pred, dim=1, keepdim=True)    # (B, 1, H, W)
    dot_prod = (opt_gt * normal_pred).sum(dim=1, keepdim=True)    # (B, 1, H, W)  
    projected = torch.abs(dot_prod / (normal_norm + eps))         # (B, 1, H, W)
    pee_map = projected - normal_norm                             # (B, 1, H, W)
    return pee_map.abs().mean(dim=(2, 3)).squeeze(1)

def save_flow_image(tensor, path):
    """Convert flow tensor to RGB image and save."""
    image = flow_to_image(tensor)  # expects (B, 2, H, W)
    image = TF.to_pil_image(image[0].cpu())
    image.save(path)

def crop_border(tensor: torch.Tensor, border: int = 1) -> torch.Tensor:
    return tensor[..., border:-border, border:-border]  

def test(test_root_dir, checkpoint_path):
    
    accelerator = Accelerator()
    batch_size = 8

    if accelerator.is_local_main_process:
        print("==== test_root_dir ====")

    # if accelerator.is_local_main_process:
    #     print("\n============= Accelerator Info =============")
    #     print(f"Device: {accelerator.device}")
    #     print(f"Number of GPUs used: {accelerator.num_processes}")
    #     print(f"Process rank: {accelerator.process_index}")
    #     print(f"Total processes: {accelerator.state.num_processes}")
    #     print(f"Mixed precision: {accelerator.mixed_precision}")
    #     if torch.cuda.is_available():
    #         print(f"CUDA device count: {torch.cuda.device_count()}")
    #         for i in range(torch.cuda.device_count()):
    #             print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        print("\n============= Setting Up Wandb =============")
        wandb.login(key="66820f29cb45c85261f7dfd317c43275e8d82562")
        wandb.init(
            project="diffposenet",
            name="Nflownet-Testing-HPC",
            config={
                "learning_rate": 0.0001,
                "epochs": 400,
                "batch_size": 8,
                "optimizer": "Adam",
            }
        )
    
    test_dataset = nflownet_test_dataloader(test_root_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=20, prefetch_factor=4, 
        drop_last=True, persistent_workers=False)

    model = NFlowNet(base_channels=32)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    model, test_loader = accelerator.prepare(model, test_loader)

    if accelerator.is_local_main_process:
        pbar = tqdm(test_loader, desc=f"Processing the Test Dataset")
    else:
        pbar = test_loader

    running_pee_loss = 0.0
    plot_id = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            paired_batch, normal_flow_batch = batch
            outputs = model(paired_batch)
            cropped_outputs = crop_border(outputs)
            pee_loss = ProjectionEndpointError(cropped_outputs, normal_flow_batch)

            if torch.isnan(pee_loss).any() or (pee_loss > 1e4).any():
                print("⚠️ NaN or large value detected, skipping batch")
                continue
                
            all_losses = accelerator.gather(pee_loss).mean()
            
            if accelerator.is_local_main_process:
                running_pee_loss += pee_loss.mean().item()
                pbar.set_postfix({'Batch PEE Loss': all_losses.item()})

                for i, loss_val in enumerate(pee_loss):
                    if loss_val.item() > 10:
                        # Convert predicted and GT flow to RGB images (3, H, W)
                        pred_flow_tensor = flow_to_image(cropped_outputs[i].unsqueeze(0).detach().cpu())[0]
                        gt_flow_tensor = flow_to_image(normal_flow_batch[i].unsqueeze(0).detach().cpu())[0]
                
                        # Log to WandB as side-by-side
                        wandb.log({
                            f"High PEE Sample {plot_id}": [
                                wandb.Image(TF.to_pil_image(gt_flow_tensor), caption=f"GT Flow"),
                                wandb.Image(TF.to_pil_image(pred_flow_tensor), caption=f"Predicted Flow (PEE={loss_val.item():.2f})")
                            ]
                        })
                        plot_id += 1

                        
    if accelerator.is_local_main_process:
        avg_pee_loss = running_pee_loss / len(test_loader)
        print("Average PEE: ", avg_pee_loss)
    


if __name__ == "__main__":
    base_dir = "comp447_project/tartanair_dataset/test_data/abandonedfactory/Easy"
    checkpoint_path = "comp447_project/nflownet_checkpoints/nflownet_final.pth"

    exclude = {"P000", "P008"}
    p_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("P") and d not in exclude])
    
    for p_id in p_dirs:
        test_root_dir = os.path.join(base_dir, p_id)
        print(f"\n\n========== Processing {p_id} ==========\n")
        wandb.finish()  # close previous run if any
        test(test_root_dir, checkpoint_path)

    



    

        