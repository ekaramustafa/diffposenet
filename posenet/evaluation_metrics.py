import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TrajectoryEvaluator:
    """
    - Absolute Trajectory Error (ATE): Equation 16-17 from the paper
    - Relative Pose Error (RPE): Equations 18-19 from the paper
    """
    
    def __init__(self, align_trajectories: bool = True):
        """
        Args:
            align_trajectories: Whether to align predicted and ground truth trajectories
                               using rigid body transformation before computing ATE
        """
        self.align_trajectories = align_trajectories
    
    def poses_to_transformation_matrices(self, translations: torch.Tensor, 
                                       R: torch.Tensor) -> torch.Tensor:
        batch_size = translations.shape[0]
        device = translations.device
                
        T = torch.zeros(batch_size, 4, 4, device=device)
        T[:, :3, :3] = R
        T[:, :3, 3] = translations
        T[:, 3, 3] = 1.0
        
        return T
    
    def align_trajectories_umeyama(self, pred_poses: torch.Tensor, 
                                  gt_poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_trans = pred_poses[:, :3, 3]  
        gt_trans = gt_poses[:, :3, 3]      
        
        pred_np = pred_trans.detach().cpu().numpy()
        gt_np = gt_trans.detach().cpu().numpy()
        
        pred_centroid = np.mean(pred_np, axis=0)
        gt_centroid = np.mean(gt_np, axis=0)
        
        pred_centered = pred_np - pred_centroid
        gt_centered = gt_np - gt_centroid
        
        H = pred_centered.T @ gt_centered
        
        U, S, Vt = np.linalg.svd(H)
        
        R_align = Vt.T @ U.T
        
        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T
        
        scale = 1.0
        
        t_align = gt_centroid - scale * R_align @ pred_centroid
        
        S = np.eye(4)
        S[:3, :3] = scale * R_align
        S[:3, 3] = t_align
        
        S_torch = torch.from_numpy(S).float().to(pred_poses.device)
        
        aligned_pred_poses = torch.matmul(S_torch.unsqueeze(0), pred_poses)
        
        return aligned_pred_poses, S_torch
    
    def compute_ate(self, pred_translations: torch.Tensor, pred_rotations: torch.Tensor,
                   gt_translations: torch.Tensor, gt_rotations: torch.Tensor) -> Dict[str, float]:
        pred_poses = self.poses_to_transformation_matrices(pred_translations, pred_rotations)
        gt_poses = self.poses_to_transformation_matrices(gt_translations, gt_rotations)
        
        if self.align_trajectories:
            aligned_pred_poses, alignment_transform = self.align_trajectories_umeyama(pred_poses, gt_poses)
        else:
            aligned_pred_poses = pred_poses
            alignment_transform = torch.eye(4, device=pred_poses.device)
        
        error_matrices = torch.matmul(torch.inverse(gt_poses), aligned_pred_poses)
        
        error_translations = error_matrices[:, :3, 3]  # trans(E_t)
    
        translation_errors = torch.norm(error_translations, dim=1)  # ||trans(E_t)||
        ate = torch.sqrt(torch.mean(translation_errors ** 2)).item()
        
        ate_mean = torch.mean(translation_errors).item()
        ate_std = torch.std(translation_errors).item()
        ate_median = torch.median(translation_errors).item()
        ate_max = torch.max(translation_errors).item()
        ate_min = torch.min(translation_errors).item()
        
        logger.info(f"DEBUG ATE - Final ATE: {ate:.6f}, mean: {ate_mean:.6f}, std: {ate_std:.6f}")
        
        return {
            'ATE_RMSE': ate,
            'ATE_mean': ate_mean,
            'ATE_std': ate_std,
            'ATE_median': ate_median,
            'ATE_max': ate_max,
            'ATE_min': ate_min,
            'num_poses': len(translation_errors)
        }
    
    def compute_rpe(self, pred_translations: torch.Tensor, pred_rotations: torch.Tensor,
                   gt_translations: torch.Tensor, gt_rotations: torch.Tensor,
                   delta_t: int = 1) -> Dict[str, float]:
        n = len(pred_translations)
        m = n - delta_t
        
        if m <= 0:
            logger.warning(f"Not enough poses for RPE computation with delta_t={delta_t}")
            return {'RPE_trans': 0.0, 'RPE_rot': 0.0, 'num_relative_poses': 0}
        
        pred_poses = self.poses_to_transformation_matrices(pred_translations, pred_rotations)
        gt_poses = self.poses_to_transformation_matrices(gt_translations, gt_rotations)
        
        relative_errors = []
        
        for i in range(m):
            gt_relative = torch.matmul(torch.inverse(gt_poses[i + delta_t]), gt_poses[i])
            pred_relative = torch.matmul(torch.inverse(pred_poses[i + delta_t]), pred_poses[i])
            
            relative_error = torch.matmul(torch.inverse(gt_relative), pred_relative)
            relative_errors.append(relative_error)
        
        relative_errors = torch.stack(relative_errors)  # [m, 4, 4]
        
        trans_errors = relative_errors[:, :3, 3]  # [m, 3]
        rot_matrices = relative_errors[:, :3, :3]  # [m, 3, 3]
        
        trans_error_norms = torch.norm(trans_errors, dim=1)  # [m]
        rpe_trans = torch.sqrt(torch.mean(trans_error_norms ** 2)).item()
        
        traces = torch.diagonal(rot_matrices, dim1=-2, dim2=-1).sum(dim=-1)  # [m]
        cos_angle = torch.clamp((traces - 1) / 2, -1.0, 1.0)
        rot_angles = torch.acos(cos_angle)  # [m]
        rpe_rot = torch.mean(rot_angles).item()
        
        rpe_trans_mean = torch.mean(trans_error_norms).item()
        rpe_trans_std = torch.std(trans_error_norms).item()
        rpe_trans_median = torch.median(trans_error_norms).item()
        
        rpe_rot_mean = torch.mean(rot_angles).item()
        rpe_rot_std = torch.std(rot_angles).item()
        rpe_rot_median = torch.median(rot_angles).item()
        
        return {
            'RPE_trans_RMSE': rpe_trans,
            'RPE_rot_mean': rpe_rot,
            'RPE_trans_mean': rpe_trans_mean,
            'RPE_trans_std': rpe_trans_std,
            'RPE_trans_median': rpe_trans_median,
            'RPE_rot_mean_deg': np.degrees(rpe_rot_mean),
            'RPE_rot_std': np.degrees(rpe_rot_std),
            'RPE_rot_median': np.degrees(rpe_rot_median),
            'num_relative_poses': m,
            'delta_t': delta_t
        }
    
    def evaluate_trajectory(self, pred_translations: torch.Tensor, pred_rotations: torch.Tensor,
                          gt_translations: torch.Tensor, gt_rotations: torch.Tensor,
                          delta_t_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
        results = {}
        
        # Compute ATE
        try:
            ate_results = self.compute_ate(pred_translations, pred_rotations,
                                         gt_translations, gt_rotations)
            results.update(ate_results)
            logger.info(f"ATE RMSE: {ate_results['ATE_RMSE']:.4f}")
        except Exception as e:
            logger.error(f"Error computing ATE: {e}")
            results.update({'ATE_RMSE': float('inf'), 'ATE_mean': float('inf')})
        
        # Compute RPE for different time intervals
        for delta_t in delta_t_list:
            try:
                rpe_results = self.compute_rpe(pred_translations, pred_rotations,
                                             gt_translations, gt_rotations, delta_t)
                # Add delta_t suffix to keys
                for key, value in rpe_results.items():
                    if key != 'delta_t':
                        results[f"{key}_dt{delta_t}"] = value
                
                logger.info(f"RPE (Δt={delta_t}) - Trans RMSE: {rpe_results['RPE_trans_RMSE']:.4f}, "
                           f"Rot Mean: {rpe_results['RPE_rot_mean_deg']:.2f}°")
            except Exception as e:
                logger.error(f"Error computing RPE for delta_t={delta_t}: {e}")
                results.update({
                    f'RPE_trans_RMSE_dt{delta_t}': float('inf'),
                    f'RPE_rot_mean_dt{delta_t}': float('inf')
                })
        
        return results

def convert_relative_to_absolute_poses(relative_translations: torch.Tensor, 
                                     relative_rotations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert relative poses to absolute poses.
    
    Args:
        relative_translations: [seq_len-1, 3] relative translations
        relative_rotations: [seq_len-1, 3, 3] relative rotation matrices
        
    Returns:
        absolute_translations: [seq_len, 3] absolute translations
        absolute_rotations: [seq_len, 3, 3] absolute rotation matrices
    """
    rel_seq_len = relative_translations.shape[0]  # This is seq_len - 1
    abs_seq_len = rel_seq_len + 1  # Absolute poses include the initial pose
    device = relative_translations.device
    
    absolute_translations = torch.zeros(abs_seq_len, 3, device=device)
    absolute_rotations = torch.zeros(abs_seq_len, 3, 3, device=device)
    
    # Initialize with identity pose
    absolute_translations[0] = torch.zeros(3, device=device)
    absolute_rotations[0] = torch.eye(3, device=device)  # Identity rotation matrix
    
    # Start with identity transformation
    current_transformation = torch.eye(4, device=device)
    
    for i in range(rel_seq_len):
        rel_t = relative_translations[i]  # [3]
        rel_R = relative_rotations[i]     # [3, 3]
        
        # Create relative transformation matrix
        rel_transform = torch.eye(4, device=device)
        rel_transform[:3, :3] = rel_R
        rel_transform[:3, 3] = rel_t
        
        # Accumulate transformation
        current_transformation = torch.matmul(current_transformation, rel_transform)
        
        # Extract absolute pose
        absolute_translations[i + 1] = current_transformation[:3, 3]
        absolute_rotations[i + 1] = current_transformation[:3, :3]
    
    return absolute_translations, absolute_rotations


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to rotation matrix
    More stable than quaternions for neural network training
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)
