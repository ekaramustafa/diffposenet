import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import NFlowNet
from dataset.tartanair import TartanAirDataset

def main():
  
    dataset = TartanAirDataset(root_dir="data/image_left")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    nflownet = NFlowNet()
    optimizer = torch.optim.Adam(nflownet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # TO BE CONTINUED

    # TO DO: PROPERLY IMPLEMENT THE DATASET
    #        CHECK IF WE NEED TO SET UP CHECKPOINTS


if __name__ == "__main__":
    main()
