import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from nflownet.utils import compute_image_gradients
import kornia
import kornia_rs

class CheiralityLayer(nn.Module):
    def __init__(self, posenet, nflownet):
        super().__init__()
        self.posenet = posenet
        self.nflownet = nflownet
        self.nflownet.eval()  # freeze NFlowNet
        for p in self.nflownet.parameters():
            p.requires_grad = False


    def construct_A_B(self, H, W, device):
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x = x.to(device).float()
        y = y.to(device).float()
        A = torch.stack([ -torch.ones_like(x), torch.zeros_like(x), x,
                          torch.zeros_like(y), -torch.ones_like(y), y ], dim=0).reshape(2, 3, H, W)
        B = torch.stack([
            x * y, -(x ** 2 + 1), y,
            y ** 2 + 1, -x * y, -x
        ], dim=0).reshape(2, 3, H, W)
        return A, B

    def cheirality_loss(self, img_pair_shape, pose, grad_dirs, normal_flow, device):
        B, _, H, W = img_pair_shape
        V, W_ = pose[:, :3], pose[:, 3:]

        A, B_mat = self.construct_A_B(H, W, device)

        loss = 0
        for b in range(B):
            gx = grad_dirs[b]  # [2, H, W]
            nf = normal_flow[b]  # [2, H, W]
            AV = torch.einsum("chw,c->hw", A, V[b])
            BW = torch.einsum("chw,c->hw", B_mat, W_[b])
            rho = (gx * AV).sum(0) * (nf.sum(0) - (gx * BW).sum(0))
            gelu_rho = F.gelu(-rho)
            loss += gelu_rho.mean()
        return loss / B

    def refine_pose(self, img_pair_shape, Pec, grad_dirs, normal_flow):
        B = Pec.shape[0]
        Per = Pec.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([Per], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = self.cheirality_loss(img_pair_shape, Per, grad_dirs, normal_flow, Pec.device)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Cheirality loss after refinement
        loss_cheirality = self.cheirality_loss(img_pair_shape, Per, grad_dirs, normal_flow, Pec.device)

        # Compute dL/dPer (gradient of cheirality loss wrt Per)
        grad_cheirality = grad(loss_cheirality, Per, create_graph=True)[0]  # shape: [B, 6]

        return Per, grad_cheirality

    #torch autograd.grad

    def upper_level_loss(self, img_pair_shape, Pec, Per, n_flow_pred, g_x,device):
        B, _, H, W = img_pair_shape
        A, B = self.construct_A_B(H, W, device)  # Shapes: A (2, 3), B (2, 3)

        # Reshape g_x to (B, H, W, 2)
        g_x = g_x.permute(0, 2, 3, 1)  # Now shape (B, H, W, 2)

        # Compute rotational and translational terms for Per (refined pose)
        V_r, Omega_r = Per[:, :3], Per[:, 3:]  # Shapes (B, 3), (B, 3)
        V_c, Omega_c = Pec[:, :3], Pec[:, 3:]

        # Compute (g_x · B) Omega_r (derotation term, Eq. 11)
        g_dot_B = torch.einsum("bhwc,cd->bhwd", g_x, B)  # (B, H, W, 3)
        g_dot_B_Omega_r = torch.einsum("bhwd,bd->bhw", g_dot_B, Omega_r)  # (B, H, W)

        # Compute derotated normal flow: n_x - (g_x · B) Omega_r
        derotated_flow = n_flow_pred - g_dot_B_Omega_r

        # Compute (g_x · A) V_r (translational term, Eq. 11)
        g_dot_A = torch.einsum("bhwc,cd->bhwd", g_x, A)  # (B, H, W, 3)
        g_dot_A_V_r = torch.einsum("bhwd,bd->bhw", g_dot_A, V_r)  # (B, H, W)

        # Compute implicit depth scaling: Z_x = (g_x · A) V_r / derotated_flow
        depth_scale = (g_dot_A_V_r / derotated_flow)  # (B, H, W)

        # Compute normal flow from Pec (coarse pose)
        g_dot_A_V_c = torch.einsum("bhwd,bd->bhw", g_dot_A, V_c)  # (B, H, W)
        g_dot_B_Omega_c = torch.einsum("bhwd,bd->bhw", g_dot_B, Omega_c)  # (B, H, W)

        # Final normal flow prediction: (g_x · A) V_c / Z_x - (g_x · B) Omega_c
        n_flow_pred_from_pose = (g_dot_A_V_c / depth_scale) - g_dot_B_Omega_c

        # Loss: MSE between NFlowNet's prediction and pose-derived normal flow
        loss = F.mse_loss(n_flow_pred_from_pose, n_flow_pred)
        return loss

    def forward(self, img_pair):
        img_pair_shape = img_pair.shape

        # Coarse pose from PoseNet
        translation, rotation = self.posenet(img_pair)
        q_flat = rotation.reshape(-1, 4)
        # Convert to rotation vector (axis-angle): [B * T, 3]
        rotvec_flat = kornia.geometry.quaternion_to_angle_axis(q_flat)

        # Geri reshape: [B, T, 3]
        rotation = rotvec_flat.view(rotation.shape[0], rotation.shape[1], 3)

        Pec = torch.cat([translation, rotation], dim=-1) 
        
        nflow_pred = self.nflownet(img_pair)
        gray = img_pair[:, :3].mean(1, keepdim=True)
        grad_dirs = compute_image_gradients(gray)  # [B, 2, H, W]

        # Lower-level: refine using cheirality
        Per, dL_dPer = self.refine_pose(img_pair_shape, Pec, grad_dirs, nflow_pred)
        
        upper_loss = self.upper_level_loss(img_pair_shape, Pec, Per, nflow_pred, grad_dirs,img_pair.device)
        
        # ∂L_upper / ∂Pec + ∂L_upper / ∂Per * ∂Per / ∂Pec
        dL_dPec_upper = grad(upper_loss, Pec, retain_graph=True, create_graph=True)[0]
        dL_dPer_upper = grad(upper_loss, Per, retain_graph=True, create_graph=True)[0]

        # ∂L_total / ∂Pec = ∂L_upper / ∂Pec - dL_dPer_upper ⊙ d_cheirality_loss / d_Per
        total_grad = dL_dPec_upper - torch.autograd.grad(dL_dPer @ dL_dPer_upper.T, Pec, retain_graph=True)[0]

        # Kayıp gibi davranarak backward yap
        dummy_loss = (Pec * total_grad.detach()).sum()
        return dummy_loss
           