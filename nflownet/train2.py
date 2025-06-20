import sys
import os
import wandb
import random
import torch
import time
import numpy as np
import torch.nn as nn
from torchvision.utils import flow_to_image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Subset

from accelerate import Accelerator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.nflownet_dataloader2 import nflownet_dataloader
from nflownet.model import NFlowNet

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
    accelerator = Accelerator()

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
                "batch_size": batch_size*4,
                "optimizer": "Adam",
            }
        )

    if accelerator.is_local_main_process:
        print("\n============= Loading Datasets =============")
    start = time.time()
    train_dataset = nflownet_dataloader(root_dir_path=train_root_dir)
    if accelerator.is_local_main_process:
        print(f"Train dataset loaded in {time.time() - start:.2f} seconds.")
    test_dataset = nflownet_dataloader(root_dir_path=test_root_dir)
    if accelerator.is_local_main_process:
        print(f"Test dataset loaded in {time.time() - start:.2f} seconds.")
    torch.cuda.empty_cache()

    # Take 2/100 subset of train
    # indices = torch.randperm(len(train_dataset)).tolist()[:len(train_dataset)]
    # train_dataset = Subset(train_dataset, indices)
    
    # Take random 1/3 subset of test
    # test_indices = torch.randperm(len(test_dataset)).tolist()[:len(test_dataset)]
    # test_dataset = Subset(test_dataset, test_indices)

    batch_size = batch_size * accelerator.num_processes

    
    if accelerator.is_local_main_process:
        print("\n============= Dataloaders =============")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=False, persistent_workers=False)
    test_loader_log = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False, persistent_workers=False)

    if accelerator.is_local_main_process:
        print(f"Training set contains {len(train_dataset)} samples.")
        print(f"Validation set contains {len(test_dataset)} samples.")

    if accelerator.is_local_main_process:
        print("\n============= Initializing the Model =============") 
    model = NFlowNet(base_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if accelerator.is_local_main_process:
        print(f"Number of parameters in NFlowNet: {sum(p.numel() for p in model.parameters())}")

    model, optimizer, train_loader, test_loader, test_loader_log = accelerator.prepare(
        model, optimizer, train_loader, test_loader, test_loader_log
    )

    train_losses = []
    test_losses = []  

    if accelerator.is_local_main_process:
        print("\n============= Training Loop =============")
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = torch.tensor([0.0], device=accelerator.device)
        print("Running train loss shape: ", running_train_loss.shape)
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", disable=not accelerator.is_local_main_process)

        for paired_batch, normal_flow_batch in pbar:
            optimizer.zero_grad()
            outputs = model(paired_batch)
            loss = criterion(outputs, normal_flow_batch)
            print("Loss item shape: ", loss.item().shape)
            accelerator.backward(loss)
            optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix({'Batch Loss': loss.item()})

            if accelerator.is_local_main_process:
                wandb.log({"Loss": loss.item()})

        #avg_train_loss = accelerator.gather(running_train_loss).sum()
        #avg_train_loss = avg_train_loss / len(train_loader)
        total_loss = accelerator.reduce(running_train_loss, reduction="sum")
        avg_train_loss = total_loss.item() / accelerator.num_processes / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f" Average Train Loss: {avg_train_loss:.6f}")

        
        # ------------------- Validation -------------------
        accelerator.wait_for_everyone()
        
        model.eval()
        running_test_loss = 0.0
        pbar = tqdm(test_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for i, (paired_batch, normal_flow_batch) in enumerate(pbar):
                outputs = model(paired_batch)
                loss = criterion(outputs, normal_flow_batch)
                
                if torch.isnan(loss) or loss.item() > 1e5:
                    print(f"⚠️ Batch {i} in test_loader caused loss={loss.item()}")
                    loss = torch.tensor([1e5], device=accelerator.device) 
                all_losses = accelerator.gather(loss).sum()
                
                if accelerator.is_local_main_process:
                    running_test_loss += all_losses.item()
                pbar.set_postfix({'Batch Loss': all_losses.item()})
        
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_test_loss:.6f}")

        accelerator.wait_for_everyone()

        
        
        # ----- Log Sample Predictions to wandb -----
        images_to_log = {}
        max_samples = 8
        with torch.no_grad():
            for paired_batch, normal_flow_batch in test_loader_log:
                pred_flow = model(paired_batch)
                pred_flow = flow_to_image(pred_flow)
                gt_flow = flow_to_image(normal_flow_batch)

                batch_size = paired_batch.size(0)
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
            if epoch % 20 == 0:
                model_to_save = accelerator.unwrap_model(model)
                torch.save(model_to_save.state_dict(), f"nflownet_epoch_{epoch}.pth")
                print(f"Model saved to nflownet_epoch_{epoch}.pth")
        accelerator.wait_for_everyone()

    # Save model after training
    accelerator.wait_for_everyone()
    model_final = accelerator.unwrap_model(model)
    torch.save(model_final.state_dict(), "nflownet_final.pth")
    print("Model saved to nflownet_final.pth")

    if accelerator.is_local_main_process:
        wandb.log({"train_loss_curve": train_losses, "validation_loss_curve": test_losses})
        wandb.finish()

    return train_losses, test_losses

if __name__ == "__main__":
    num_epochs = 400
    batch_size = 8
    train_root_dir = "/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/"
    test_root_dir = "/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/"
    train_losses, test_losses = train(num_epochs, batch_size, train_root_dir, test_root_dir)
