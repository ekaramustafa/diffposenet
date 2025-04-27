import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from nflownet.utils import compute_normal_flow


class PairedImageDataset(Dataset):
    def __init__(self, img_folder_path, opt_flow_folder_path):
        self.image_paths = sorted(glob.glob(img_folder_path + '/*.png'))
        self.opt_flow_paths = sorted(glob.glob(opt_flow_folder_path + '/*_flow.npy'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        # Load two consecutive images
        img1 = Image.open(self.image_paths[idx]).convert('RGB')
        img2 = Image.open(self.image_paths[idx + 1]).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        paired = torch.cat((img1, img2), dim=0)

        opt_flow = np.load(self.opt_flow_paths[idx])
        opt_flow = torch.from_numpy(opt_flow).permute(2, 0, 1).float()

        normal_flow = compute_normal_flow(opt_flow, paired)
        return paired, normal_flow


