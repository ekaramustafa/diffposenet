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


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dataset.nflownet_dataloader2 import nflownet_dataloader
from nflownet.model import NFlowNet
import logging

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(num_epochs, batch_size, train_root_dir, test_root_dir):
    set_seed()
    accelerator = Accelerator(split_batches=False, gradient_accumulation_steps=2)

    if accelerator.is_local_main_process:
        print("\n============= Accelerator Info =============")
        print(f"Device: {accelerator.device}")
        print(f"Number of GPUs used: {accelerator.num_processes}")
        print(f"Process rank: {accelerator.process_index}")
        print(f"Total processes: {accelerator.state.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    
        print("\n============= Setting Up Wandb =============")
        wandb.login(key="66820f29cb45c85261f7dfd317c43275e8d82562")
        wandb.init(
            project="diffposenet",
            name="Nflownet-Training-HPC",
            config={
                "learning_rate": 0.001,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "optimizer": "Adam",
            }
        )

    train_dataset = nflownet_dataloader(root_dir_path=train_root_dir)
    test_dataset = nflownet_dataloader(root_dir_path=test_root_dir)

    # Take random 1/5 subset of test
    test_indices = torch.randperm(len(test_dataset)).tolist()[:len(test_dataset) // 5]
    test_dataset = Subset(test_dataset, test_indices)

    # Take 2/100 subset of train
    #indices = torch.randperm(len(train_dataset)).tolist()[:len(train_dataset) // 50]
    #train_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=25, prefetch_factor=4, 
        drop_last=True, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=25, prefetch_factor=4, 
        drop_last=True, persistent_workers=True)
    test_log_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=8, persistent_workers=False)
    
    model = NFlowNet(base_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model, optimizer, train_loader, test_loader, test_log_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, test_log_loader)

    train_losses = []
    test_losses = [] 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        if accelerator.is_local_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        else:
            pbar = train_loader

        for batch in pbar:
            with accelerator.accumulate(model):
                paired_batch, normal_flow_batch = batch 
                outputs = model(paired_batch)
                loss = criterion(outputs, normal_flow_batch)
                accelerator.backward(loss)
                #accelerator.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if accelerator.is_local_main_process:
                pbar.set_postfix({"Batch Loss": loss.item()})
                wandb.log({"Loss": loss.item()})                

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch + 1}] Average Loss: {avg_train_loss:.4f}")

        # ------------------- Validation -------------------
        accelerator.wait_for_everyone()
        model.eval()
        running_test_loss = 0.0
        if accelerator.is_local_main_process:
            pbar = tqdm(test_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        else:
            pbar = test_loader
            
        with torch.no_grad():
            for batch in pbar:
                paired_batch, normal_flow_batch = batch
                outputs = model(paired_batch)
                loss = criterion(outputs, normal_flow_batch)
                
                if torch.isnan(loss) or loss.item() > 1e5:
                    print(f"⚠️ Batch {i} in test_loader caused loss={loss.item()}")
                    loss = 10
                all_losses = accelerator.gather(loss).mean()
                
                if accelerator.is_local_main_process:
                    running_test_loss += all_losses.item()
                    pbar.set_postfix({'Batch Loss': all_losses.item()})
        
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_test_loss:.6f}")
        

        # ----- Log Sample Predictions to wandb -----
        accelerator.wait_for_everyone()
        images_to_log = {}
        max_samples = 8
        with torch.no_grad():
            for paired_batch, normal_flow_batch in test_log_loader:
                pred_flow = model(paired_batch)
                pred_flow = flow_to_image(pred_flow)
                gt_flow = flow_to_image(normal_flow_batch)
    
                for j in range(min(batch_size, max_samples)):
                    img1 = TF.to_pil_image(paired_batch[j][:3].cpu())
                    img2 = TF.to_pil_image(paired_batch[j][3:].cpu())
                    flow_pred = TF.to_pil_image(pred_flow[j].cpu())
                    flow_gt = TF.to_pil_image(gt_flow[j].cpu())
                    
                    images_to_log[f"Sample {j+1} - Image 1"] = wandb.Image(img1, caption="Input Image 1")
                    images_to_log[f"Sample {j+1} - Image 2"] = wandb.Image(img2, caption="Input Image 2")
                    images_to_log[f"Sample {j+1} - Predicted Normal Flow"] = wandb.Image(flow_pred, caption="Predicted Normal Flow")
                    images_to_log[f"Sample {j+1} - Ground Truth Normal Flow"] = wandb.Image(flow_gt, caption="Ground Truth Normal Flow")
                break
        if accelerator.is_local_main_process:     
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "validation_loss": avg_test_loss,
                **images_to_log
            })

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            if epoch % 10 == 0:
                model_to_save = accelerator.unwrap_model(model)
                torch.save(model_to_save.state_dict(), f"nflownet_epoch_{epoch}.pth")
                print(f"Model saved to nflownet_epoch_{epoch}.pth")
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    model_final = accelerator.unwrap_model(model)
    torch.save(model_final.state_dict(), "nflownet_final.pth")
    print("Model saved to nflownet_final.pth")

    if accelerator.is_local_main_process:
        wandb.log({"train_loss_curve": train_losses, "validation_loss_curve": test_losses})
        wandb.finish()
    
        print(f"🟢 Sync point after epoch {epoch + 1} | Process {accelerator.process_index}")

    if accelerator.is_local_main_process:
       print("Training complete.")



if __name__ == "__main__":
    num_epochs=400
    batch_size=8
    train_root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/"
    test_root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/"
    train(num_epochs, batch_size, train_root_dir, test_root_dir)
