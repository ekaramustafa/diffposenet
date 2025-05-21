import sys
import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from model import PoseNet
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.tartanair import TartanAirDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import AutoImageProcessor
from torchvision.transforms.functional import to_pil_image

def main():
    # Initialize wandb
    # wandb.login(key="66820f29cb45c85261f7dfd317c43275e8d82562")
    # wandb.init(
    #     project="diffposenet",
    #     name="PoseNet-Training",
    #     config={
    #         "learning_rate": 1e-4,
    #         "batch_size": 8,
    #         "epochs": 30,
    #         "optimizer": "Adam"
    #     }
    # )
    # config = wandb.config
    config={
            "learning_rate": 1e-5,
            "batch_size": 8,
            "epochs": 30,
            "optimizer": "Adam"
        }

    # Set random seed for reproducibility
    set_seed()

    # Dataset and splitting
    print("============= Loading the Train Dataset =============")
    train_dataset = TartanAirDataset(root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/", size=(224, 224))
    print("============= Loading the Validation Dataset =============")
    val_dataset = TartanAirDataset(root_dir="/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/test_data/", size=(224, 224))
    
    train_dataset = Subset(train_dataset, list(range(0, len(train_dataset), 5)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), 8)))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, processor and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_net = PoseNet().to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    optimizer = optim.Adam(pose_net.parameters(), lr=config.learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        pose_net.train()
        total_train_loss = 0.0

        for images, translations, rotations in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            # Preprocess images using DINOv2 processor
            batch = []
            for img in images:  # images: [B, T, C, H, W]
                frames = [to_pil_image(frame) for frame in img]  # Convert to PIL for processor
                inputs = processor(images=frames, return_tensors="pt")["pixel_values"]  # [T, 3, 224, 224]
                batch.append(inputs)

            batch = torch.stack(batch).to(device)  # shape: [B, T, 3, 224, 224]
            translations = translations.to(device)
            rotations = rotations.to(device)

            optimizer.zero_grad()
            t_pred, q_pred = pose_net(batch)
            loss = pose_net.pose_loss(t_pred, q_pred, translations, rotations)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        pose_net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, translations, rotations in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                batch = []
                for img in images:
                    frames = [to_pil_image(frame) for frame in img]
                    inputs = processor(images=frames, return_tensors="pt")["pixel_values"]
                    batch.append(inputs)

                batch = torch.stack(batch).to(device)
                translations = translations.to(device)
                rotations = rotations.to(device)

                t_pred, q_pred = pose_net(batch)
                loss = pose_net.pose_loss(t_pred, q_pred, translations, rotations)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    # Plot + save
    plt.figure()
    plt.plot(range(1, config.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config.epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plot_path = "training_validation_loss.png"
    plt.savefig(plot_path)
    plt.show()
    wandb.log({"loss_plot": wandb.Image(plot_path)})

    print("Training completed.")
    # wandb.finish()
    return pose_net

if __name__ == "__main__":
    model = main()
    torch.save(pose_net.state_dict(), "pose_net_progress.pth")
