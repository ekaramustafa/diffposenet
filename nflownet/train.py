import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from utils import compute_normal_flow
from model import NFlowNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.tartanair import TartanAirDataset



def train_epochs(model, optimizer, criterion, train_loader, test_loader, num_epochs, device='cuda'):
    train_losses = []
    test_losses = []

    device = torch.device(device)
    
    # ----- Compute Initial Loss (Before training) -----
    model.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for paired_batch, normal_flow_batch in test_loader:
            paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)
            outputs = model(paired_batch)

            loss = criterion(outputs, normal_flow_batch)
            initial_test_loss += loss.item()

    initial_test_loss /= len(test_loader)
    print(f"Initial Test Loss (Before Training): {initial_test_loss:.4f}")

    # ----- Training Loop -----
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        batch_losses = []

        # Wrap train_loader with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for paired_batch, normal_flow_batch in pbar:
            paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)

            optimizer.zero_grad()
            outputs = model(paired_batch)

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
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for paired_batch, normal_flow_batch in test_loader:
                paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)
                outputs = model(paired_batch)
                loss = criterion(outputs, normal_flow_batch)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

    return train_losses, test_losses


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
