import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt

from model import PoseNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.tartanair import TartanAirDataset

def main():
    dataset = TartanAirDataset(root_dir="data/image_left", size=(224, 224))

    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_net = PoseNet().to(device)
    optimizer = optim.Adam(pose_net.parameters(), lr=1e-4)

    num_epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        pose_net.train()
        total_train_loss = 0.0
        for images, translations, rotations in train_loader:
            images = images.to(device, non_blocking=True)
            translations = translations.to(device, non_blocking=True)
            rotations = rotations.to(device, non_blocking=True)

            optimizer.zero_grad()
            t_pred, q_pred = pose_net(images)
            loss = pose_net.pose_loss(t_pred, q_pred, translations, rotations)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validation step
        pose_net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, translations, rotations in val_loader:
                images = images.to(device, non_blocking=True)
                translations = translations.to(device, non_blocking=True)
                rotations = rotations.to(device, non_blocking=True)
                t_pred, q_pred = pose_net(images)
                val_loss = pose_net.pose_loss(t_pred, q_pred, translations, rotations)
                total_val_loss += val_loss.item()
        val_losses.append(total_val_loss / len(val_loader))

        print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PoseNet Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Finished training and validation.")

if __name__ == "__main__":
    main()
