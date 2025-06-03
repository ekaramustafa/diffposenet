import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import warnings
from nflownet.utils import compute_normal_flow


class nflownet_dataloader(Dataset):
    def __init__(self, root_dir_path, img_transform=None):
        self.root_dir_path = root_dir_path
        self.data_paths = []
        self.img_transform = img_transform
        self.env_count = 0
        if img_transform is None:
            self.img_transform = transforms.ToTensor()

        self._load_paths()
        

    def _load_paths(self):
        for env_dir in os.listdir(self.root_dir_path):
            env_path = os.path.join(self.root_dir_path, env_dir)
            if os.path.isdir(env_path): 
                for difficulty in os.listdir(env_path):
                    difficulty_path = os.path.join(env_path, difficulty)
                    if difficulty == "Easy": 
                        for traj_dir in os.listdir(difficulty_path):
                            traj_path = os.path.join(difficulty_path, traj_dir)
                            if os.path.isdir(traj_path):
                                image_dir = os.path.join(traj_path, 'image_left')
                                normal_flow_dir = os.path.join(traj_path, 'normal_flow')
                                #normal_flow_dir = os.path.join(traj_path, 'unmasked_normal_flow')

                                if os.path.exists(image_dir) and os.path.exists(normal_flow_dir):
                                    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                                    normal_flow_files = sorted(glob.glob(os.path.join(normal_flow_dir, '*.pt')))
                                    
                                    if (len(image_files) == len(normal_flow_files) + 1):
                                        for i in range(len(normal_flow_files)):
                                            img1 = image_files[i]
                                            img2 = image_files[i+1]
                                            normal_flow = normal_flow_files[i]
                                            self.data_paths.append((img1, img2, normal_flow))
                                    else:
                                        warnings.warn(f"Length mismatch in {traj_path}")
                                    

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, normal_flow_path = self.data_paths[idx]

        img1 = self._read_image(img1_path)
        img2 = self._read_image(img2_path)
        normal_flow = torch.load(normal_flow_path).float()
        paired_images = torch.cat([img1, img2], dim=0)     
        return paired_images, normal_flow
    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)
        image = self._crop_to_divisible_by_16(image)
        return image

    def _crop_to_divisible_by_16(self, img: torch.Tensor):
        C, H, W = img.shape
        new_H = H - (H % 16)
        new_W = W - (W % 16)
        return img[..., :new_H, :new_W]
