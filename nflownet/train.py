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

from dataset.tartanair2 import PairedImageDataset     # <- will be changed


def main():
    dataset = PairedImageDataset(img_file_path="diffposenet/data/image_left", opt_flow_file_path="diffposenet/data/flow")
    torch.cuda.empty_cache()
    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nflow_net = NFlowNet().to(device)
    optimizer = torch.optim.Adam(nflow_net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    # ----- Compute Initial Loss (Before any training) -----
    nflow_net.eval()
    initial_test_loss = 0.0
    with torch.no_grad():
        for paired_batch, normal_flow_batch in test_loader:
            paired_batch, normal_flow_batch = paired_batch.to(device), normal_flow_batch.to(device)
            outputs = nflow_net(paired_batch)
    
            loss = criterion(outputs, normal_flow_batch)
            initial_test_loss += loss.item()

    initial_test_loss /= len(test_loader)
    print(f"Initial Test Loss (Before Training): {initial_test_loss:.4f}")

    # ----- Training Loop -----
    num_epochs = 10
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

    # Save model after training
    torch.save(nflow_net.state_dict(), "nflownet.pth")
    print("Model saved to nflownet.pth")
    
    return train_losses, test_losses


if __name__ == "__main__":
    main()
