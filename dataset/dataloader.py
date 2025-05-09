import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 

class TartanAirDataset(Dataset):
    def __init__(self, root_dir, seq_len=2, transform=None, size=None, nflownet=False):
        """
        root_dir: path to the root directory of the TartanAir dataset
        seq_length: number of consecutive video frames 
        transform: optional transform to be applied to the images
        size: reference size of the images
        nflownet: whether if the loaded data is for NFlowNet

        Note that the images and the pose text information must be in the same directory.
        Pose txt file must include [tx, ty, tz, qx, qy, qz, qw] in its every line.

        For example, the directory structure should be:
        tartanair_dataset/
        └── amusement/                             # Environment
            └── Easy/                              # Difficulty level (always Easy)
                └── P002/                          # Trajectory folder 
                    ├── pose_left.txt              # Left camera pose
                    └── flow/                      # Optical flow files (npy)
                        ├── 000000_000001_flow.npy
                        ├── 000001_000002_flow.npy
                        ├── ...
                    └── image_left/                # Folder for left camera images 
                        ├── 000000_left.png     
                        ├── 000001_left.png     
                        ├── ...       
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.ground_truth_normal_flow = []
        self.ground_truth_pose = []
        self.seq_len = seq_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((480, 640)) if size is None else transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self._load_data()

  
    def _load_data(self):
        temp_images = []
        temp_flows = []
        temp_poses = []
      
        img_files = []
        opt_flow_files = []
        pose_file = None
      
        for env_dir in os.listdir(self.root_dir):
            env_path = os.path.join(self.root_dir, env_dir)
            if os.path.isdir(env_path):
                for difficulty in os.listdir(env_path):
                    difficulty_path = os.path.join(env_path, difficulty)
                    if difficulty == "Easy":
                        for traj_dir in os.listdir(difficulty_path):                          
                            traj_path = os.path.join(difficulty_path, traj_dir)
                            if os.path.isdir(traj_path):
                                
                                if nflownet is False:
                                    image_dir = os.path.join(traj_path, 'image_left')
                                    pose_file = os.path.join(traj_path, 'pose_left.txt')
                                    if os.path.exists(pose_file) and os.path.exists(image_dir):
                                        poses = self._read_ground_truth(pose_file)
                                        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                                        if len(img_files) == len(poses):
                                            temp_images.extend(img_files)
                                            temp_poses.extend(poses)

                                        loaded_images = [self._load_image(img_path) for img_path in temp_images]
                                        for i in range(len(loaded_images) - self.seq_len):
                                            sequence = loaded_images[i:i+self.seq_len]
                                            poses = [self._compute_relative_pose(temp_poses[j], temp_poses[j+1]) for j in range(i, i + self.seq_len - 1)]
                                            self.images.append(sequence)
                                            self.ground_truth_pose.append(poses)

                                else:
                                    image_dir = os.path.join(traj_path, 'image_left')
                                    flow_dir = os.path.join(traj_path, 'flow')
                                    if os.path.exists(image_dir) and os.path.exists(flow_dir):
                                        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                                        flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])
                                        if len(img_files) == len(flow_files)+1:
                                            temp_images.extend(img_files)
                                            temp_flows.extend(flow_files)

                                        loaded_images = [self._load_image(img_path) for img_path in temp_images]
                                        loaded_flows = [self._load_image(img_path) for img_path in temp_images]
                                        for i in range(len(loaded_images) - self.seq_len):
                                            sequence = loaded_images[i:i+self.seq_len]
                                            poses = [self._compute_relative_pose(temp_poses[j], temp_poses[j+1]) for j in range(i, i + self.seq_len - 1)]
                                            self.images.append(sequence)
                                            self.ground_truth_pose.append(poses)
                                      
                                    
                                    
                                    

                                    for i, image_name in enumerate(sorted(os.listdir(image_dir))):
                                        if image_name.endswith('.png'):
                                            image_path = os.path.join(image_dir, image_name)
                                            image = Image.open(image_path)
                                            if self.transform:
                                                image = self.transform(image)
    
                                            if nflownet: 
                                                flow_name = image_name.replace('_left.png', '_flow.npy')       # this is most likely incorrect
                                                flow_path = os.path.join(flow_dir, flow_name)
                                                if os.path.exists(flow_path):
                                                    optical_flow = np.load(flow_path)
                                                    ...
                                            else:
                                                continue



          
            for file in files:
                if file.endswith(".png"):
                    if "left" not in file:
                        continue  # consider image_left only
                    img_files.append(os.path.join(root, file))
                elif file.endswith(".txt"):
                    pose_file = os.path.join(root, file)
            
            if pose_file and img_files:
                img_files.sort()
                poses = self._read_ground_truth(pose_file)
                
                if len(img_files) == len(poses):
                    temp_images.extend(img_files)
                    temp_poses.extend(poses)
        
        loaded_images = [self._load_image(img_path) for img_path in temp_images]
        
        for i in range(len(loaded_images) - self.seq_len):
            sequence = loaded_images[i:i+self.seq_len]
            poses = [self._compute_relative_pose(temp_poses[j], temp_poses[j+1]) for j in range(i, i + self.seq_len - 1)]
            self.images.append(sequence)
            self.ground_truth_pose.append(poses)
  

    def __len__(self):
        return len(self.images)

  
    def __getitem__(self, idx):
        sequence = self.images[idx]  
        poses = self.ground_truth_pose[idx]  # List of (t, q)

        images = torch.stack(sequence, dim=0)  # Shape: [seq_len, C, H, W]
        translations = torch.stack([p[0] for p in poses], dim=0)  # [seq_len-1, 3]
        rotations = torch.stack([p[1] for p in poses], dim=0)     # [seq_len-1, 4]

        return images, translations, rotations

  
    def _load_image(self, image_path):
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def _load_flow(self, flow_path):
    flow = np.load(flow_path)
    if self.transform:
        flow = self.transform(flow)
    
    return flow

    def _read_ground_truth(self, file_path):
        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            return [list(map(float, line.split())) for line in lines]


    def _compute_relative_pose(self, pose1, pose2):
        t1 = np.array(pose1[:3])
        q1 = np.array(pose1[3:])
        t2 = np.array(pose2[:3])
        q2 = np.array(pose2[3:])

        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        q1_inv = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
        q_rel = self._quaternion_multiply(q2, q1_inv)
        q_rel = q_rel / np.linalg.norm(q_rel)
        assert np.allclose(np.linalg.norm(q1), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(q2), 1.0, atol=1e-6)

        R1 = self._quaternion_to_rotation_matrix(q1)
        t_rel = np.dot(R1.T, (t2 - t1))

        return torch.from_numpy(t_rel).float(), torch.from_numpy(q_rel).float()

  
    def _quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])

  
    def _quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q  
        q_norm = np.sqrt(w*w + x*x + y*y + z*z)
        x /= q_norm
        y /= q_norm
        z /= q_norm
        w /= q_norm 
        R = np.zeros((3, 3))
        
        R[0, 0] = 1 - 2*y*y - 2*z*z
        R[0, 1] = 2*x*y - 2*w*z
        R[0, 2] = 2*x*z + 2*w*y
        
        R[1, 0] = 2*x*y + 2*w*z
        R[1, 1] = 1 - 2*x*x - 2*z*z
        R[1, 2] = 2*y*z - 2*w*x
        
        R[2, 0] = 2*x*z - 2*w*y
        R[2, 1] = 2*y*z + 2*w*x
        R[2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
