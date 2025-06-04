import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TrajectoryEvaluator:
    """
    - Absolute Trajectory Error (ATE): Equation 16-17 from the paper
    - Relative Pose Error (RPE): Equations 18-19 from the paper
    """
    
    def __init__(self, align_trajectories: bool = False):
        """
        Args:
            align_trajectories: Whether to align predicted and ground truth trajectories
                               using rigid body transformation before computing ATE
        """
        self.align_trajectories = align_trajectories
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        # Normalize quaternions
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        R = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
        ], dim=-2)
        
        return R
    
    def poses_to_transformation_matrices(self, translations: torch.Tensor, 
                                       quaternions: torch.Tensor) -> torch.Tensor:
        batch_size = translations.shape[0]
        device = translations.device
        
        R = self.quaternion_to_rotation_matrix(quaternions)  # [N, 3, 3]
        
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
    
    def compute_ate(self, pred_translations: torch.Tensor, pred_quaternions: torch.Tensor,
                   gt_translations: torch.Tensor, gt_quaternions: torch.Tensor) -> Dict[str, float]:
        
        pred_poses = self.poses_to_transformation_matrices(pred_translations, pred_quaternions)
        gt_poses = self.poses_to_transformation_matrices(gt_translations, gt_quaternions)
        
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
    
    def compute_rpe(self, pred_translations: torch.Tensor, pred_quaternions: torch.Tensor,
                   gt_translations: torch.Tensor, gt_quaternions: torch.Tensor,
                   delta_t: int = 1) -> Dict[str, float]:
        n = len(pred_translations)
        m = n - delta_t
        
        if m <= 0:
            logger.warning(f"Not enough poses for RPE computation with delta_t={delta_t}")
            return {'RPE_trans': 0.0, 'RPE_rot': 0.0, 'num_relative_poses': 0}
        
        pred_poses = self.poses_to_transformation_matrices(pred_translations, pred_quaternions)
        gt_poses = self.poses_to_transformation_matrices(gt_translations, gt_quaternions)
        
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
    
    def evaluate_trajectory(self, pred_translations: torch.Tensor, pred_quaternions: torch.Tensor,
                          gt_translations: torch.Tensor, gt_quaternions: torch.Tensor,
                          delta_t_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
        results = {}
        
        # Compute ATE
        try:
            ate_results = self.compute_ate(pred_translations, pred_quaternions,
                                         gt_translations, gt_quaternions)
            results.update(ate_results)
            logger.info(f"ATE RMSE: {ate_results['ATE_RMSE']:.4f}")
        except Exception as e:
            logger.error(f"Error computing ATE: {e}")
            results.update({'ATE_RMSE': float('inf'), 'ATE_mean': float('inf')})
        
        # Compute RPE for different time intervals
        for delta_t in delta_t_list:
            try:
                rpe_results = self.compute_rpe(pred_translations, pred_quaternions,
                                             gt_translations, gt_quaternions, delta_t)
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
                                     relative_quaternions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rel_seq_len = relative_translations.shape[0]  # This is seq_len - 1
    abs_seq_len = rel_seq_len + 1  # Absolute poses include the initial pose
    device = relative_translations.device
    
    absolute_translations = torch.zeros(abs_seq_len, 3, device=device)
    absolute_quaternions = torch.zeros(abs_seq_len, 4, device=device)
    
    absolute_translations[0] = torch.zeros(3, device=device)
    absolute_quaternions[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # Identity quaternion
    
    evaluator = TrajectoryEvaluator()
    
    current_transformation = torch.eye(4, device=device)
    
    for i in range(rel_seq_len):
        rel_t = relative_translations[i]  # [3]
        rel_q = relative_quaternions[i]   # [4]
        
        rel_transform = evaluator.poses_to_transformation_matrices(
            rel_t.unsqueeze(0), rel_q.unsqueeze(0)
        ).squeeze(0)  # [4, 4]
        
        current_transformation = torch.matmul(current_transformation, rel_transform)
        
        absolute_translations[i + 1] = current_transformation[:3, 3]
        
        rot_matrix = current_transformation[:3, :3]
        abs_q = rotation_matrix_to_quaternion(rot_matrix)
        absolute_quaternions[i + 1] = abs_q
    
    return absolute_translations, absolute_quaternions

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    device = R.device
    
    trace = torch.trace(R)
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    quaternion = torch.tensor([qw, qx, qy, qz], device=device)
    
    quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=0)
    
    return quaternion
