import torch
import torch.nn as nn

class CheiralityLayer(nn.Module):
    """
    Differentiable Cheirality Layer that enforces depth positivity constraint
    using normal flow estimates from NFlowNet.
    
    Args:
        max_iter (int): Maximum iterations for optimization
        tol (float): Tolerance for convergence
        lr (float): Learning rate for L-BFGS optimizer
    """
    def __init__(self, max_iter=100, tol=1e-20, lr=0.1):
        super(CheiralityLayer, self).__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        
    def forward(self, normal_flow, image_gradients, init_pose):
        """
        Args:
            normal_flow (torch.Tensor): Estimated normal flow from NFlowNet [B, H, W]
            image_gradients (torch.Tensor): Image spatial gradients [B, 2, H, W]
            init_pose (torch.Tensor): Initial pose estimate from PoseNet [B, 6] (3 translation, 3 rotation)
            
        Returns:
            torch.Tensor: Refined pose estimate [B, 6]
        """
        batch_size = init_pose.size(0)
        refined_poses = []
        
        # Process each sample in batch separately
        for i in range(batch_size):
            nf = normal_flow[i]  # [H, W]
            grad = image_gradients[i]  # [2, H, W]
            init_p = init_pose[i]  # [6]
            
            # Convert to numpy for L-BFGS (paper uses this approach)
            # Note: In practice you might want to keep everything in PyTorch
            refined_p = self.optimize_pose(nf, grad, init_p)
            refined_poses.append(refined_p)
            
        return torch.stack(refined_poses, dim=0)
    
    def optimize_pose(self, normal_flow, image_gradients, init_pose):
        """
        Optimize pose using cheirality constraint with L-BFGS
        
        Args:
            normal_flow (torch.Tensor): [H, W]
            image_gradients (torch.Tensor): [2, H, W]
            init_pose (torch.Tensor): [6]
            
        Returns:
            torch.Tensor: Refined pose [6]
        """
        # Convert to numpy for optimization (as done in paper)
        # Alternatively could implement fully in PyTorch using torch.optim.LBFGS
        import numpy as np
        from scipy.optimize import minimize
        
        # Normalize gradients to get direction (unit vectors)
        grad_norm = torch.norm(image_gradients, dim=0, keepdim=True)
        grad_dir = image_gradients / (grad_norm + 1e-6)  # [2, H, W]
        
        # Get image coordinates grid
        H, W = normal_flow.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
        coords = torch.stack([x_coords, y_coords], dim=0).float().to(normal_flow.device)  # [2, H, W]
        
        # Prepare data for optimization
        nf_np = normal_flow.detach().cpu().numpy().flatten()
        grad_dir_np = grad_dir.detach().cpu().numpy().reshape(2, -1)  # [2, H*W]
        coords_np = coords.detach().cpu().numpy().reshape(2, -1)  # [2, H*W]
        init_pose_np = init_pose.detach().cpu().numpy()
        
        # Define objective function
        def objective(pose_params):
            V = pose_params[:3]  # translation
            Omega = pose_params[3:]  # rotation
            
            # Compute A and B matrices (from paper equations 9-10)
            x, y = coords_np
            ones = np.ones_like(x)
            zeros = np.zeros_like(x)
            
            # A matrix [2, 3] -> [H*W, 2, 3]
            A = np.stack([
                -ones, zeros, x,
                zeros, -ones, y
            ], axis=1).reshape(-1, 2, 3)
            
            # B matrix [2, 3] -> [H*W, 2, 3]
            B = np.stack([
                x*y, -(x**2 + 1), y,
                (y**2 + 1), -x*y, -x
            ], axis=1).reshape(-1, 2, 3)
            
            # Compute rho (equation 12)
            g_dot_A = np.einsum('hwi,hwi->hw', grad_dir_np.T[:, None, :], A)  # [H*W, 3]
            g_dot_B = np.einsum('hwi,hwi->hw', grad_dir_np.T[:, None, :], B)  # [H*W, 3]
            
            term1 = np.dot(g_dot_A, V)  # [H*W]
            term2 = nf_np - np.dot(g_dot_B, Omega)  # [H*W]
            
            rho = term1 * term2  # [H*W]
            
            # GELU activation (smooth ReLU approximation)
            # Negative GELU as in paper equation 13
            loss = -0.5 * rho * (1.0 + torch.erf(rho / np.sqrt(2.0)))  # GELU approximation
            return np.mean(loss)
        
        # Run L-BFGS optimization
        result = minimize(objective, 
                         init_pose_np,
                         method='L-BFGS-B',
                         options={'maxiter': self.max_iter,
                                'ftol': self.tol,
                                'gtol': self.tol})
        
        refined_pose = torch.from_numpy(result.x).float().to(init_pose.device)
        return refined_pose