import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TartanAirDataset(Dataset):
    def __init__(self, root_dir, seq_len=2, transform=None, size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((480, 640)) if size is None else transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.image_files = []
        self.pose_files = []

        # ONLY store paths here
        for envs_dir in os.listdir(root_dir):
            env_path = os.path.join(root_dir, envs_dir)
            for difficulty in os.listdir(env_path):
                difficulty_path = os.path.join(env_path, difficulty)
                if difficulty == "Easy":
                    for traj_dir in os.listdir(difficulty_path):
                        traj_path = os.path.join(difficulty_path, traj_dir)
                        for traj in os.listdir(traj_path):
                            file_path = os.path.join(traj_path, traj)
                            if os.path.isdir(file_path):
                                for image in os.listdir(file_path):
                                    if image.endswith(".png"):
                                        self.image_files.append(os.path.join(file_path, image))
                            if file_path.endswith("left.txt"):
                                self.pose_files.append(file_path)

        self.image_files.sort()
        self.pose_files.sort()

        # load poses only once (small memory footprint)
        self.poses = self._read_ground_truth(self.pose_files)

    def __len__(self):
        return len(self.image_files) - self.seq_len

    def __getitem__(self, idx):
        # Load image sequence
        image_seq = []
        for i in range(idx, idx + self.seq_len):
            img = Image.open(self.image_files[i])
            if self.transform:
                img = self.transform(img)
            image_seq.append(img)
        image_seq = torch.stack(image_seq, dim=0)

        # Load pose pairs for relative pose
        poses = []
        for i in range(idx, idx + self.seq_len - 1):
            pose1 = self.poses[i]
            pose2 = self.poses[i + 1]
            rel_pose = self._compute_relative_pose(pose1, pose2)
            poses.append(rel_pose)

        translations = torch.stack([p[0] for p in poses], dim=0)
        rotations = torch.stack([p[1] for p in poses], dim=0)

        return image_seq, translations, rotations

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
    
    def _load_image(self, image_path):
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _read_ground_truth(self, file_paths):
        results = []
        for file_path in file_paths:
            lines = []
            with open(file_path, "r") as f:
                lines = f.readlines()
                arr = [list(map(float, line.split())) for line in lines]
                results.extend(arr)
        return results
