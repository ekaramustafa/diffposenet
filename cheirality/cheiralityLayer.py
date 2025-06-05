import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from nflownet.utils import compute_image_gradients
import kornia

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

        # A and B: [H, W, 2, 3]
        A = torch.stack([
            -torch.ones_like(x), torch.zeros_like(x), x,
            torch.zeros_like(y), -torch.ones_like(y), y
        ], dim=-1).reshape(H, W, 2, 3)

        B = torch.stack([
            x * y, -(x ** 2 + 1), y,
            y ** 2 + 1, -x * y, -x
        ], dim=-1).reshape(H, W, 2, 3)

        # Reshape to [H*W, 2, 3]
        A = A.view(-1, 2, 3)
        B = B.view(-1, 2, 3)

        return A, B

    def cheirality_loss(self, img_pair_shape, pose, grad_dirs, normal_flow, device):
        B, _, H, W = img_pair_shape
        V, W_ = pose[:, :, :3], pose[:, :, 3:]  # [B, 1, 3]

        A, B_mat = self.construct_A_B(H, W, device)  # [H*W, 2, 3]

        # Gradient directions
        grad_x, grad_y = grad_dirs  # each: [B, H, W]
        gx = torch.stack([grad_x, grad_y], dim=-1).view(B, H*W, 2)  # [B, H*W, 2]
        gx_unit = F.normalize(gx, dim=-1)  # [B, H*W, 2]

        # Normal flow: [B, 2, H, W] -> [B, H*W, 2]
        nf = normal_flow.permute(0, 2, 3, 1).view(B, H*W, 2)

        # Compute scalar projection of normal flow onto gradient direction
        nf_scalar = torch.sum(nf * gx_unit, dim=-1)  # [B, H*W]

        # gA = gx · A → [B, H*W, 3]
        gA = torch.einsum('bpi,pij->bpj', gx_unit, A)

        # gB = gx · B → [B, H*W, 3]
        gB = torch.einsum('bpi,pij->bpj', gx_unit, B_mat)

        # term1 = gA @ V → [B, H*W]
        term1 = torch.sum(gA * V, dim=-1)

        # term2 = nf_scalar - (gB @ Omega) → [B, H*W]
        term2 = nf_scalar - torch.sum(gB * W_, dim=-1)

        # rho = term1 * term2
        rho = term1 * term2  # [B, H*W]
        # Penalize negative values of rho
        loss = F.gelu(-rho).mean()
        return loss

    def refine_pose(self, img_pair_shape, Pec, grad_dirs, normal_flow):
        B = Pec.shape[0]
        Per = Pec.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([Per], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = self.cheirality_loss(img_pair_shape, Per, grad_dirs, normal_flow, Pec.device)
            loss.backward() 
            torch.nn.utils.clip_grad_norm_([Per], max_norm=100)
            return loss

        optimizer.step(closure)

        # Cheirality loss after refinement
        loss_cheirality = self.cheirality_loss(img_pair_shape, Per, grad_dirs, normal_flow, Pec.device)

        # Compute dL/dPer (gradient of cheirality loss wrt Per)
        grad_cheirality = grad(loss_cheirality, Per, create_graph=True)[0]  # shape: [B, 6]

        return Per.detach().requires_grad_(), grad_cheirality

    def upper_level_loss(self, img_pair_shape, Pec, Per, n_flow_pred, grad_dirs, device):
        B, _, H, W = img_pair_shape
        A, B_mat = self.construct_A_B(H, W, device)  # [H*W, 2, 3]

        grad_x, grad_y = grad_dirs  # each [B, H, W]

        # Stack and reshape gradient directions: [B, H, W, 2] -> [B, H*W, 2]
        g_x = torch.stack([grad_x, grad_y], dim=-1).view(B, H*W, 2)
        gx_unit = F.normalize(g_x, dim=-1)  # normalize gradient directions

        # Reshape predicted normal flow vector to [B, H*W, 2]
        n_flow_vec = n_flow_pred.permute(0, 2, 3, 1).reshape(B, H*W, 2)  # [B, H*W, 2]

        # Project predicted normal flow vector along gradient directions (scalar)
        n_flow_scalar = torch.sum(n_flow_vec * gx_unit, dim=-1)  # [B, H*W]

        # Pose splits (translation and rotation)
        V_r, Omega_r = Per[:,:, :3], Per[:,:, 3:]  # [B, 3]
        V_c, Omega_c = Pec[:,:, :3], Pec[:,:, 3:]  # [B, 3]

        # Project A and B matrices along gradient directions
        g_dot_A = torch.einsum("bpi,pij->bpj", gx_unit, A)      # [B, H*W, 3]
        g_dot_B = torch.einsum("bpi,pij->bpj", gx_unit, B_mat)  # [B, H*W, 3]

        # Compute derotation term: (gx · B) · Omega_r
        g_dot_B_Omega_r = torch.sum(g_dot_B * Omega_r.unsqueeze(1), dim=-1)  # [B, H*W]

        # Compute translational term: (gx · A) · V_r
        g_dot_A_V_r = torch.sum(g_dot_A * V_r.unsqueeze(1), dim=-1)  # [B, H*W]

        # Compute denominator of depth scale: n_x - (gx · B) · Omega_r
        denom = n_flow_scalar - g_dot_B_Omega_r  # [B, H*W]
        depth_scale = g_dot_A_V_r / (denom)  # [B, H*W]

        # Compute terms for coarse pose Pec
        g_dot_A_V_c = torch.sum(g_dot_A * V_c.unsqueeze(1), dim=-1)  # [B, H*W]
        g_dot_B_Omega_c = torch.sum(g_dot_B * Omega_c.unsqueeze(1), dim=-1)  # [B, H*W]

        # Predict normal flow scalar from coarse pose Pec using depth scale
        n_flow_pred_from_pose = (g_dot_A_V_c / (depth_scale)) - g_dot_B_Omega_c  # [B, H*W]

        # Compute MSE loss between predicted scalar normal flow and model predicted scalar normal flow
        n_flow_pred_from_pose = n_flow_pred_from_pose.squeeze(1)  # now shape [1, 307200]

        loss = F.mse_loss(n_flow_pred_from_pose, n_flow_scalar)
        return loss


    def forward(self, img_pair):

        # Coarse pose from PoseNet
        translation, rotation = self.posenet(img_pair)
        q_flat = rotation.reshape(-1, 4)
        # Convert to rotation vector (axis-angle): [B * T, 3]
        rotvec_flat = kornia.geometry.quaternion_to_axis_angle(q_flat)

        # reshape: [B, T, 3]
        rotation = rotvec_flat.view(rotation.shape[0], rotation.shape[1], 3)
        Pec = torch.cat([translation, rotation], dim=-1) 
        
        # Dont look here because ı am gonna change dataset here for sake of trying ı added these things to pass to nflownet
        img_pair = img_pair.view(1, 2 * 3, 224, 224)
        img_pair = F.interpolate(img_pair, size=(480, 640), mode='bilinear', align_corners=False)

        nflow_pred = self.nflownet(img_pair)
        gray = img_pair[:, :3].mean(1, keepdim=True)
        grad_dirs = compute_image_gradients(gray)  # [B, 2, H, W]
        img_pair_shape = img_pair.shape
        # Lower-level: refine using cheirality
        Per, dL_dPer = self.refine_pose(img_pair_shape, Pec, grad_dirs, nflow_pred)
        upper_loss = self.upper_level_loss(img_pair_shape, Pec, Per, nflow_pred, grad_dirs,img_pair.device)
        
        # ∂L_upper / ∂Pec + ∂L_upper / ∂Per * ∂Per / ∂Pec
        dL_dPec_upper = grad(upper_loss, Pec, retain_graph=True, create_graph=True)[0]
        dL_dPer_upper = grad(upper_loss, Per, retain_graph=True, create_graph=True)[0]

        # ∂L_total / ∂Pec = ∂L_upper / ∂Pec - dL_dPer_upper ⊙ d_cheirality_loss / d_Per
        total_grad = dL_dPec_upper - torch.autograd.grad((dL_dPer * dL_dPer_upper).sum(), Pec, retain_graph=True)[0]

        # Kayıp gibi davranarak backward yap
        dummy_loss = (Pec * total_grad.detach()).sum()
        return dummy_loss
           