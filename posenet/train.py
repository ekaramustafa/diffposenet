import sys
import os
from torch.utils.data import DataLoader, Subset
import torch
import torch.optim as optim
from model import PoseNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.tartanair import TartanAirDataset

def main():
    dataset = TartanAirDataset(root_dir="diffposenet/data/image_left")
    torch.cuda.empty_cache()
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda")
    pose_net = PoseNet().to(device)
    images, translations, rotations = next(iter(train_loader))
    images = images.to(device)
    translations = translations.to(device)
    rotations = rotations.to(device)

    optimizer = optim.Adam(pose_net.parameters(), lr=1e-4)
    for epoch in range(10):  
        total_epoch_loss = 0.0
        for images, translations, rotations in train_loader:
            images = images.to(device, non_blocking=True)
            translations = translations.to(device, non_blocking=True)
            rotations = rotations.to(device, non_blocking=True)

            optimizer.zero_grad()
            t_pred, q_pred = pose_net(images)

            loss = pose_net.pose_loss(t_pred, q_pred, translations, rotations)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_epoch_loss / len(train_loader):.6f}")

    print("Finished overfitting entire dataset.")
        
        
    return
    


if __name__ == "__main__":
    main()

