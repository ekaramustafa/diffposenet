import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataset.nflownet_dataloader import nflownet_dataloader
from nflownet.utils import compute_normal_flow
from nflownet.model import NFlowNet
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch

def train():
    wandb.login()
    wandb.init(
        project="nflownet_train",
        name="experiment-001",
        config={
            "learning_rate": 0.001,
            "epochs": 400,
            "batch_size": 8,
            "optimizer": "Adam",
        }
    )

    dataset = nflownet_dataloader2(root_dir_path="/content/drive/My Drive/tartanair/")
    torch.cuda.empty_cache()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nflow_net = NFlowNet().to(device)
    optimizer = torch.optim.Adam(nflow_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    sample = dataset[0]
    image_pair, flow = sample  
    image1 = image_pair[:3]   # Channels 0–2 → image1
    image2 = image_pair[3:]   # Channels 3–5 → image2

    img1 = TF.to_pil_image(image1)
    img2 = TF.to_pil_image(image2)

    # Compute flow magnitude (Euclidean norm across u and v)
    flow_magnitude = flow.norm(dim=0)  # (H, W)

    # Plot images and flow magnitude
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(flow_magnitude.cpu(), cmap='viridis')
    plt.title("Flow Magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    train_losses = []
    test_losses = []

    # ----- Compute Initial Loss (Before any training) -----
    pbar = tqdm(test_loader, desc=f"Initialization")
    nflow_net.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for paired_batch, normal_flow_batch in pbar:
            paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)
            outputs = nflow_net(paired_batch)

            loss = criterion(outputs, normal_flow_batch)
            initial_test_loss += loss.item()

    initial_test_loss /= len(test_loader)
    print(f"Initial Test Loss (Before Training): {initial_test_loss:.4f}")

    # ----- Training Loop -----
    num_epochs = 3
    for epoch in range(num_epochs):
        nflow_net.train()
        running_train_loss = 0.0
        batch_losses = []

        # Wrap train_loader with tqdm
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

            # Update progress bar with current batch loss
            pbar.set_postfix({'Batch Loss': loss.item()})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}")

        # ----- Evaluate on Test Data -----
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

        # ----- Log Sample Predictions to wandb -----
        nflow_net.eval()
        sample_images = []
        logged = 0
        max_samples = 16
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

                    # Convert images and flow magnitudes to numpy arrays
                    img1_np = img1.numpy()
                    img2_np = img2.numpy()
                    flow_pred_mag_np = flow_pred_mag.numpy()
                    flow_gt_mag_np = flow_gt_mag.numpy()

                    # Convert images to PIL for visualization in wandb
                    img1_pil = TF.to_pil_image(img1)
                    img2_pil = TF.to_pil_image(img2)

                    # Visualize predictions and ground truth
                    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                    axes[0].imshow(img1_pil)
                    axes[0].set_title("Image 1")
                    axes[1].imshow(img2_pil)
                    axes[1].set_title("Image 2")
                    axes[2].imshow(flow_pred_mag, cmap='magma')
                    axes[2].set_title("Predicted Flow Mag")
                    axes[3].imshow(flow_gt_mag, cmap='viridis')
                    axes[3].set_title("Ground Truth Flow Mag")

                    for ax in axes:
                        ax.axis("off")

                    fig.tight_layout()
                    sample_images.append(wandb.Image(fig, caption=f"Sample {logged + 1}"))

                    logged += 1

                if logged >= max_samples:
                    break

        # Log images and scalar losses to wandb
        wandb.log({
            f"sample_predictions_epoch_{epoch+1}": sample_images,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            f"raw_samples_epoch_{epoch+1}": {
                "image1": [img1_np for img1_np in sample_images],
                "image2": [img2_np for img2_np in sample_images],
                "pred_flow_mag": [flow_pred_mag_np for flow_pred_mag_np in sample_images],
                "gt_flow_mag": [flow_gt_mag_np for flow_gt_mag_np in sample_images],
            }
        })

    # Save model after training
    torch.save(nflow_net.state_dict(), "nflownet.pth")
    print("Model saved to nflownet.pth")

    wandb.log({"train_loss": train_losses, "test_loss": test_losses})
    wandb.finish()

    return train_losses, test_losses


if __name__ == "__main__":
    train()
