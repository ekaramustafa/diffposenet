import sys
import os
import wandb
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.nflownet_dataloader import nflownet_dataloader
from nflownet.utils import compute_normal_flow
from nflownet.model import NFlowNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(num_epochs, train_root_dir, test_root_dir):  
    set_seed(42)
    
    wandb.login(key="66820f29cb45c85261f7dfd317c43275e8d82562")
    wandb.init(
        project="diffposenet",
        name="Nflownet-Training",
        config={
            "learning_rate": 0.001,
            "epochs": num_epochs,
            "batch_size": 8,
            "optimizer": "Adam",
        }
    )
    
    print("============= Loading the Train Dataset =============")
    train_dataset = nflownet_dataloader(root_dir_path=train_root_dir)
    print("Success")
    print("============= Loading the Validation Dataset =============")
    test_dataset = nflownet_dataloader(root_dir_path=test_root_dir)
    torch.cuda.empty_cache()
    print("Success")

    print("============= Dataloaders =============")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(test_dataset)}")
    
    print("============= Initializing the Model =============")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nflow_net = NFlowNet(base_channels=64).to(device)
    optimizer = torch.optim.Adam(nflow_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []


    print("============= Training Loop =============")
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        nflow_net.train()
        running_train_loss = 0.0
        batch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for paired_batch, normal_flow_batch in pbar:
            paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)

            optimizer.zero_grad()
            outputs = nflow_net(paired_batch)

            loss = criterion(outputs, normal_flow_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            batch_losses.append(loss.item())

            pbar.set_postfix({'Batch Loss': loss.item()})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f" Avg Train Loss: {avg_train_loss:.6f}")

        # ----- Evaluation on Validation Data -----
        nflow_net.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for paired_batch, normal_flow_batch in test_loader:
                paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)
                outputs = nflow_net(paired_batch)
                loss = criterion(outputs, normal_flow_batch)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_test_loss:.6f}")

        # ----- Log Sample Predictions to wandb -----
        nflow_net.eval()
        sample_images = []
        logged = 0
        max_samples = 8
        with torch.no_grad():
            for paired_batch, normal_flow_batch in test_loader:
                paired_batch = paired_batch.to(device)
                normal_flow_batch = normal_flow_batch.to(device)
                pred_flow = nflow_net(paired_batch).cpu()
                input_images = paired_batch.cpu()

                batch_size = paired_batch.size(0)
                for j in range(batch_size):
                    if logged >= max_samples:
                        break

                    img1 = input_images[j][:3]
                    img2 = input_images[j][3:]
                    flow_pred = pred_flow[j]
                    flow_gt = normal_flow_batch[j].cpu()

                    flow_pred_mag = flow_pred.norm(dim=0)
                    flow_gt_mag = flow_gt.norm(dim=0)

                    flow_pred_x = flow_pred[0].numpy()  # x-direction
                    flow_pred_y = flow_pred[1].numpy()  # y-direction
                    flow_gt_x = flow_gt[0].numpy()
                    flow_gt_y = flow_gt[1].numpy()

                    # Convert images and flow magnitudes to numpy arrays
                    img1_np = img1.numpy()
                    img2_np = img2.numpy()
                    flow_pred_mag_np = flow_pred_mag.numpy()
                    flow_gt_mag_np = flow_gt_mag.numpy()

                    # Convert images to PIL for visualization in wandb
                    img1_pil = TF.to_pil_image(img1)
                    img2_pil = TF.to_pil_image(img2)

                    # Visualize predictions and ground truth
                    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
                    # Row 1: Inputs + Predicted Flow
                    axes[0, 0].imshow(img1_pil)
                    axes[0, 0].set_title("Image 1")
                    axes[0, 1].imshow(img2_pil)
                    axes[0, 1].set_title("Image 2")
                    axes[0, 2].imshow(flow_pred_mag, cmap='viridis')
                    axes[0, 2].set_title("Predicted Flow Mag")
                    axes[0, 3].imshow(flow_gt_mag, cmap='viridis')
                    axes[0, 3].set_title("Ground Truth Flow Mag")

                    # Row 2: x & y components
                    axes[1, 0].imshow(flow_pred_x, cmap='coolwarm')
                    axes[1, 0].set_title("Pred Flow X")
                    axes[1, 1].imshow(flow_gt_x, cmap='coolwarm')
                    axes[1, 1].set_title("GT Flow X")
                    axes[1, 2].imshow(flow_pred_y, cmap='coolwarm')
                    axes[1, 2].set_title("Pred Flow Y")
                    axes[1, 3].imshow(flow_gt_y, cmap='coolwarm')
                    axes[1, 3].set_title("GT Flow Y")

                    for ax_row in axes:
                        for ax in ax_row:
                            ax.axis("off")

                    fig.tight_layout()
                    sample_images.append(wandb.Image(fig, caption=f"Sample {logged + 1}"))
                    plt.close(fig)

                    logged += 1

                if logged >= max_samples:
                    break

        # Log images and scalar losses to wandb
        wandb.log({
        f"sample_predictions_epoch_{epoch+1}": sample_images,
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_test_loss,
        f"raw_samples_epoch_{epoch+1}": {
            "image1": [img1_np for img1_np in sample_images],
            "image2": [img2_np for img2_np in sample_images],
            "pred_flow_mag": [flow_pred_mag_np for flow_pred_mag_np in sample_images],
            "gt_flow_mag": [flow_gt_mag_np for flow_gt_mag_np in sample_images],
            "pred_flow_x": [flow_pred_x_np for flow_pred_x_np in sample_images],
            "gt_flow_x": [flow_gt_x_np for flow_gt_x_np in sample_images],
            "pred_flow_y": [flow_pred_y_np for flow_pred_y_np in sample_images],
            "gt_flow_y": [flow_gt_y_np for flow_gt_y_np in sample_images],
        }
    })

        if epoch % 20 == 0:
            torch.save(nflow_net.state_dict(), f"nflownet_epoch_{epoch}.pth")
            print(f"Model saved to nflownet_epoch_{epoch}.pth")

    # Save model after training
    torch.save(nflow_net.state_dict(), "nflownet.pth")
    print("Model saved to nflownet.pth")

    wandb.log({"train_loss": train_losses, "validation_loss": test_losses})
    wandb.finish()

    return train_losses, test_losses

if __name__ == "__main__":
    num_epochs = 400
    train_root_dir = "/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/"
    test_root_dir = "/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/"
    train_losses, test_losses = train(num_epochs, train_root_dir, test_root_dir)
