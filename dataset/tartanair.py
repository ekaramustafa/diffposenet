import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TartanAirDataset(Dataset):
    def __init__(self, root_dir, seq_len=2, transform=None, size=None, skip=1, track_sequences=False):
        logger.info(f"Initializing TartanAirDataset with root_dir: {root_dir}, seq_len: {seq_len}, skip: {skip}, track_sequences: {track_sequences}")
        
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.skip = skip
        self.track_sequences = track_sequences

        if transform is None:
            logger.debug("No transform provided, using default transform")
            self.transform = transforms.Compose([
                transforms.Resize((480, 640)) if size is None else transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.image_files = []
        self.pose_files = []
        
        self.sequence_mapping = []  # Maps each data point to its sequence
        self.sequence_names = []    # Unique sequence names
        self.sequence_boundaries = {}  # Start and end indices for each sequence

        # Collect all image and pose files
        self._collect_files()
        logger.debug(f"Last sorted image file: {self.image_files[-1] if self.image_files else 'None'}")

        logger.info("Sorting image and pose files")
        self.image_files.sort()
        logger.debug(f"Total image files found: {len(self.image_files)}")
        logger.debug(f"Last sorted image file: {self.image_files[-1] if self.image_files else 'None'}")
        
        self.pose_files.sort()
        logger.info(f"Total pose files found: {len(self.pose_files)}")

        # If tracking sequences, create mapping and boundaries after sorting
        if self.track_sequences:
            logger.info("Creating sequence mapping and boundaries")
            self._create_sequence_mapping_and_boundaries()
            logger.info(f"Tracking {len(self.sequence_names)} sequences: {self.sequence_names}")

        # load poses only once (small memory footprint)
        self.poses = self._read_ground_truth(self.pose_files)

    def _collect_files(self):
        """Collect all image and pose files from the directory structure"""
        for envs_dir in sorted(os.listdir(self.root_dir)):
            if envs_dir == "abandonedfactory":
                continue
            env_path = os.path.join(self.root_dir, envs_dir)
            logger.debug(f"Processing environment: {envs_dir}")
            
            for difficulty in sorted(os.listdir(env_path)):
                difficulty_path = os.path.join(env_path, difficulty)
                if difficulty == "Easy":
                    logger.debug(f"Processing difficulty level: {difficulty}")
                    
                    for traj_dir in sorted(os.listdir(difficulty_path)):
                        if traj_dir.startswith("ME"): # do not include the ME part from challenge dataset
                            logger.debug(f"Skipping ME trajectory: {traj_dir}")
                            continue
                            
                        traj_path = os.path.join(difficulty_path, traj_dir)
                        logger.debug(f"Processing trajectory: {traj_dir}")
                        
                        for traj in sorted(os.listdir(traj_path)):
                            file_path = os.path.join(traj_path, traj)
                            if os.path.isdir(file_path):
                                for image in sorted(os.listdir(file_path)):
                                    if image.endswith(".png"):
                                        self.image_files.append(os.path.join(file_path, image))
                            if file_path.endswith("left.txt"):
                                self.pose_files.append(file_path)

    def _create_sequence_mapping_and_boundaries(self):
        """Create mapping from data indices to sequence names and calculate boundaries after sorting"""
        self.sequence_mapping = ['unknown'] * len(self.image_files)
        self.sequence_names = []
        self.sequence_boundaries = {}
        
        # Group images by sequence name based on file paths
        sequence_groups = {}
        for i, img_path in enumerate(self.image_files):
            # Extract sequence name from path
            # Assuming path structure like: .../environment/Easy/trajectory_name/...
            path_parts = img_path.split(os.sep)
            sequence_name = None
            
            # Find the trajectory directory name (comes after "Easy")
            for j, part in enumerate(path_parts):
                if part == "Easy" and j + 1 < len(path_parts):
                    sequence_name = path_parts[j + 1]
                    break
            
            if sequence_name and not sequence_name.startswith("ME"):
                if sequence_name not in sequence_groups:
                    sequence_groups[sequence_name] = []
                sequence_groups[sequence_name].append(i)
                self.sequence_mapping[i] = sequence_name
        
        # Create sequence names list and boundaries
        for seq_name, indices in sequence_groups.items():
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)
            
            if indices:  # Only if there are images in this sequence
                self.sequence_boundaries[seq_name] = {
                    'start': min(indices),
                    'end': max(indices),
                    'length': len(indices),
                    'indices': sorted(indices)
                }
        
        # Sort sequence names for consistency
        self.sequence_names.sort()
        
        logger.debug(f"Created sequence boundaries: {self.sequence_boundaries}")

    def get_sequence_for_index(self, idx):
        """Get sequence name for a given data index"""
        if not self.track_sequences:
            return None
        if idx < len(self.sequence_mapping):
            return self.sequence_mapping[idx]
        return None

    def get_sequence_boundaries(self):
        """Get sequence boundaries for per-sequence evaluation"""
        if not self.track_sequences:
            return None
        return self.sequence_boundaries.copy()

    def get_sequence_names(self):
        """Get list of all sequence names"""
        if not self.track_sequences:
            return None
        return self.sequence_names.copy()

    def __len__(self):
        return len(self.poses) - (self.seq_len - 1) * self.skip

    def __getitem__(self, idx):
        image_seq = []
        for i in range(self.seq_len):
            img_idx = idx + i * self.skip
            img = Image.open(self.image_files[img_idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            image_seq.append(img)
        image_seq = torch.stack(image_seq, dim=0)

        poses = []
        for i in range(self.seq_len - 1):
            pose1_idx = idx + i * self.skip
            pose2_idx = idx + (i + 1) * self.skip
            pose1 = self.poses[pose1_idx]
            pose2 = self.poses[pose2_idx]
            rel_pose = self._compute_relative_pose(pose1, pose2)
            poses.append(rel_pose)

        translations = torch.stack([p[0] for p in poses], dim=0)
        rotations = torch.stack([p[1] for p in poses], dim=0)

        # Return sequence info if tracking
        if self.track_sequences:
            sequence_name = self.get_sequence_for_index(idx)
            return image_seq, translations, rotations, sequence_name
        else:
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
        t_rel = t2 - t1 

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
        image = Image.open(image_path).convert("RGB")
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
