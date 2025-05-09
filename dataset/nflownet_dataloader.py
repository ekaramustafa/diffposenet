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
        self.image_paths = []
        self.opt_flow_paths = []
        self.img_transform = img_transform
        self.flow_transform = flow_transform
        if img_transform is None:
            self.img_transform = transforms.Compose([
                #transforms.Resize((480, 640)) if size is None else transforms.Resize(size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        #if flow_transform is None:   
        #   self.flow_transform = transforms.Compose([])

        self._load_paths()

        print("img size: ", len(self.image_paths))
        print("opt_flow size: ", len(self.opt_flow_paths))
        

    def _load_paths(self):
        for env_dir in os.listdir(self.root_dir_path):
            env_path = os.path.join(self.root_dir_path, env_dir)
            if os.path.isdir(env_path): 
                for difficulty in os.listdir(env_path):
                    difficulty_path = os.path.join(env_path, difficulty)
                    if difficulty == "Easy": 
                        for traj_dir in os.listdir(difficulty_path):
                            print(traj_dir)
                            traj_path = os.path.join(difficulty_path, traj_dir)
                            if os.path.isdir(traj_path):
                                image_dir = os.path.join(traj_path, 'image_left')
                                flow_dir = os.path.join(traj_path, 'flow')

                                if os.path.exists(image_dir) and os.path.exists(flow_dir):
                                    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                                    flow_files = sorted(glob.glob(os.path.join(flow_dir, '*_flow.npy')))
                                    
                                    if len(image_files) == len(flow_files)+1:
                                        self.image_paths.extend(image_files)
                                        self.opt_flow_paths.extend(flow_files)
                                    else:
                                        print(f"The lengths did not match in {traj_path}")
                                    

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        img1 = self._read_image(self.image_paths[idx])
        img2 = self._read_image(self.image_paths[idx + 1])
        paired = torch.cat((img1, img2), dim=0)

        opt_flow = self._read_opt_flow(self.opt_flow_paths[idx])

        normal_flow = compute_normal_flow(opt_flow, paired, magnitude=False)
        return paired, normal_flow


    def _read_opt_flow(self, opt_flow_path):
        opt_flow = np.load(opt_flow_path)
        opt_flow = torch.from_numpy(opt_flow).permute(2, 0, 1).float()
        return opt_flow
    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)
        return image
