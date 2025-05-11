import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
from nflownet.utils import compute_normal_flow


class nflownet_dataloader(Dataset):
    def __init__(self, root_dir_path, img_transform=None, flow_transform=None):
        self.root_dir_path = root_dir_path
        self.data_paths = []
        self.img_transform = img_transform
        self.flow_transform = flow_transform
        self.env_count = 0
        if img_transform is None:
            self.img_transform = transforms.Compose([
                #transforms.Resize((480, 640)) if size is None else transforms.Resize(size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        #if flow_transform is None:   
        #   self.flow_transform = transforms.Compose([])

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
                                    
                                    if len(image_files) == len(flow_files) + 1:
                                        for i in range(len(flow_files)):
                                            img1 = image_files[i]
                                            img2 = image_files[i+1]
                                            flow = flow_files[i]
                                            self.data_paths.append((img1, img2, flow))
                                    else:
                                        print(f"Length mismatch in {traj_path}")
                                    

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, flow_path = self.data_paths[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        flow = np.load(flow_path)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.img_transform:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)

        if self.flow_transform:
            flow = self.flow_transform(flow)
        else:
            flow = torch.tensor(flow, dtype=torch.float32)
        paired_images = torch.cat([img1, img2], dim=0)  # shape (6, H, W)
        normal_flow = compute_normal_flow(flow, paired_images)
        return paired_images, flow


    def _read_opt_flow(self, opt_flow_path):
        opt_flow = np.load(opt_flow_path)
        opt_flow = torch.from_numpy(opt_flow).permute(2, 0, 1).float()
        return opt_flow
    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)
        return image
