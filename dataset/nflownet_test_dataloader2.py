import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import warnings
from nflownet.utils import compute_normal_flow


class nflownet_test_dataloader(Dataset):
    def __init__(self, traj_dir_path, img_transform=None):
        self.traj_dir_path = traj_dir_path  # now pointing to Pxxx
        self.data_paths = []
        self.img_transform = img_transform or transforms.ToTensor()

        self._load_paths()

    def _load_paths(self):
        image_dir = os.path.join(self.traj_dir_path, 'image_left')
        normal_flow_dir = os.path.join(self.traj_dir_path, 'normal_flow')

        if os.path.exists(image_dir) and os.path.exists(normal_flow_dir):
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            normal_flow_files = sorted(glob.glob(os.path.join(normal_flow_dir, '*.pt')))
            #flow_files = sorted(glob.glob(os.path.join(flow_dir, '*_flow.npy')))
            #flow_mask_files = sorted(glob.glob(os.path.join(flow_dir, '*_mask.npy')))

            if len(image_files) == len(normal_flow_files) + 1:
                for i in range(len(normal_flow_files)):
                    img1 = image_files[i]
                    img2 = image_files[i + 1]
                    #flow = flow_files[i]
                    #flow_mask = flow_mask_files[i]
                    #self.data_paths.append((img1, img2, flow, flow_mask))
                    normal_flow = normal_flow_files[i]
                    self.data_paths.append((img1, img2, normal_flow))
            else:
                warnings.warn(f"Length mismatch in {self.traj_dir_path}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, normal_flow_path = self.data_paths[idx]

        img1 = self._read_image(img1_path)
        img2 = self._read_image(img2_path)
        paired_images = torch.cat([img1, img2], dim=0)    
        normal_flow = torch.load(normal_flow_path).float()
        return paired_images, self._crop_border(normal_flow)

    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)
        image = self._crop_to_divisible_by_16(image)
        return image

    def _crop_to_divisible_by_16(self, img: torch.Tensor):
        C, H, W = img.shape
        new_H = H - (H % 16)
        new_W = W - (W % 16)
        return img[:, :new_H, :new_W]

    def _crop_border(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Crops a fixed number of pixels from all four sides of a (C, H, W) tensor.
        """
        C, H, W = tensor.shape
        return tensor[:, 1:H-1, 1:W-1]