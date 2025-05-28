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
                                flow_dir = os.path.join(traj_path, 'flow')

                                if os.path.exists(image_dir) and os.path.exists(flow_dir):
                                    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                                    flow_files = sorted(glob.glob(os.path.join(flow_dir, '*_flow.npy')))
                                    flow_mask_files = sorted(glob.glob(os.path.join(flow_dir, '*_mask.npy')))
                                    
                                    if (len(image_files) == len(flow_files) + 1) and (len(flow_files) == len(flow_mask_files)):
                                        for i in range(len(flow_files)):
                                            img1 = image_files[i]
                                            img2 = image_files[i+1]
                                            flow = flow_files[i]
                                            flow_mask = flow_mask_files[i]
                                            self.data_paths.append((img1, img2, flow, flow_mask))
                                    else:
                                        warnings.warn(f"Length mismatch in {traj_entry.path}")
                                    

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, flow_path, flow_mask_path = self.data_paths[idx]

        img1 = self._read_image(img1_path)
        img2 = self._read_image(img2_path)
        flow = self._read_opt_flow(flow_path)
        flow_mask = self._read_opt_mask(flow_mask_path)
        weights = 1.0 - (flow_mask / 100.0)
        weights = weights.unsqueeze(0)
        masked_flow = flow * weights
        paired_images = torch.cat([img1, img2], dim=0)  # shape (6, H, W)
        normal_flow = compute_normal_flow(masked_flow, paired_images)        
        return paired_images, normal_flow

    def _read_opt_mask(self, opt_mask_path):
        opt_mask = np.load(opt_mask_path)
        opt_mask = torch.from_numpy(opt_mask).float()
        opt_mask = self._crop_to_divisible_by_16(opt_mask)
        return opt_mask
        
    def _read_opt_flow(self, opt_flow_path):
        opt_flow = np.load(opt_flow_path)
        opt_flow = torch.from_numpy(opt_flow).permute(2, 0, 1).float()
        opt_flow = self._crop_to_divisible_by_16(opt_flow)
        return opt_flow
    
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
