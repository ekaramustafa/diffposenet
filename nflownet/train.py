import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import compute_normal_flow
from model import NFlowNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.tartanair import TartanAirDataset


def main():
  
    dataset = TartanAirDataset(root_dir="diffposenet/data/image_left")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda')
    nflownet = NFlowNet().to(device)
    optimizer = torch.optim.Adam(nflownet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    images, translations, rotations = next(iter(train_loader))
    images = images.to(device)
    translations = translations.to(device)
    rotations = rotations.to(device)

    images, _, _ = next(iter(train_loader))
    images = images.to(device)
    concatted_images = images.view(images.shape[0],images.shape[1] * images.shape[2], images.shape[3], images.shape[4])

    for i in range(10):
        optimizer.zero_grad()
        pred = nflownet(concatted_images)
        truth = compute_normal_flow(images[:,:1].squeeze(1), images[:,1:].squeeze(1))
        truth.to(device)
        loss = nflownet.loss(pred,truth)
        loss.backward()
        print(f"Epoch {i}: Avg Loss = {loss.item()}")

    # optimizer = optim.Adam(nflownet.parameters(), lr=1e-4)
    # for epoch in range(10):  
    #     total_epoch_loss = 0.0
    #     for images, translations, rotations in train_loader:
    #         images = images.to(device, non_blocking=True)
    #         translations = translations.to(device, non_blocking=True)
    #         rotations = rotations.to(device, non_blocking=True)

    #         optimizer.zero_grad()
    #         pred = nflownet(images)
    #         truth = compute_normal_flow(images[:,:1], images[:,1:])

    #         loss = nflownet.loss(pred, truth)

    #         loss.backward()
    #         optimizer.step()

    #         total_epoch_loss += loss.item()

    #     if epoch % 1 == 0:
    #         print(f"Epoch {epoch}: Avg Loss = {total_epoch_loss / len(train_loader):.6f}")

    print("Finished overfitting entire dataset.")

if __name__ == "__main__":
    main()
