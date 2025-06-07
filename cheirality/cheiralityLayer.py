import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from nflownet.utils import compute_image_gradients
import kornia
from torchvision import transforms

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
        grad_x, grad_y = grad_dirs[:, 0, :, :], grad_dirs[:, 1, :, :]
        gx = torch.stack([grad_x, grad_y], dim=-1).view(B, H*W, 2)  # [B, H*W, 2]
        gx_unit = F.normalize(gx, dim=-1)  # [B, H*W, 2]

        # Normal flow: [B, 2, H, W] -> [B, H*W, 2]
        nf = normal_flow.permute(0, 2, 3, 1).reshape(B, H*W, 2)

        n_flow_scalar = torch.norm(nf, dim=-1)  # [B, H*W]

        # gA = gx · A → [B, H*W, 3]
        gA = torch.einsum('bpi,pij->bpj', gx_unit, A)

        # gB = gx · B → [B, H*W, 3]
        gB = torch.einsum('bpi,pij->bpj', gx_unit, B_mat)

        # term1 = gA @ V → [B, H*W]
        term1 = torch.sum(gA * V, dim=-1)

        # term2 = nf_scalar - (gB @ Omega) → [B, H*W]
        term2 = n_flow_scalar - torch.sum(gB * W_, dim=-1)

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

        # Gradients x ve y'yi ayır ve [B, H*W, 2] şekline getir
        grad_x, grad_y = grad_dirs[:, 0, :, :], grad_dirs[:, 1, :, :]
        gx = torch.stack([grad_x, grad_y], dim=-1).view(B, H * W, 2)  # [B, H*W, 2]
        gx_unit = F.normalize(gx, dim=-1)  # Birim vektörler [B, H*W, 2]

        # Normal flow tahminini [B, H*W, 2] şekline getir
        nf = n_flow_pred.permute(0, 2, 3, 1).reshape(B, H * W, 2)  # [B, H*W, 2]

        # Pose vektörlerini çevirme (translation) ve dönme (rotation) olarak ayır
        V_r, Omega_r = Per[:, :, :3], Per[:, :, 3:]  # [B, 3]
        V_c, Omega_c = Pec[:, :, :3], Pec[:, :, 3:]  # [B, 3]

        # A ve B matrislerini gradyan yönlerine projekte et
        gA = torch.einsum('bpi,pij->bpj', gx_unit, A)      # [B, H*W, 3]
        gB = torch.einsum('bpi,pij->bpj', gx_unit, B_mat)  # [B, H*W, 3]

        # Derinlik ölçeğini Per'den hesapla
        g_dot_B_Omega_r = torch.sum(gB * Omega_r.unsqueeze(1), dim=-1)  # [B, H*W]
        g_dot_A_V_r = torch.sum(gA * V_r.unsqueeze(1), dim=-1)          # [B, H*W]
        n_flow_scalar = torch.norm(nf, dim=-1)                          # [B, H*W]

        denom = n_flow_scalar - g_dot_B_Omega_r  # Stabilite için payda
        depth_scale = g_dot_A_V_r / denom        # [B, H*W]

        # Pec'den benzer hesaplamalar
        g_dot_A_V_c = torch.sum(gA * V_c.unsqueeze(1), dim=-1)
        g_dot_B_Omega_c = torch.sum(gB * Omega_c.unsqueeze(1), dim=-1)

        # Derinlik ile normal flow tahmini yeniden oluşturuluyor
        flow_from_pose = g_dot_A_V_c / depth_scale  # [B, H*W, 2]
        flow_from_pose = flow_from_pose + g_dot_B_Omega_c         # [B, H*W, 2]

        # Tahmin edilen normal flow ile rekonstrüksiyon arasındaki MSE kaybını hesapla

        loss = F.mse_loss(flow_from_pose.squeeze(1), n_flow_scalar)

        return loss


    def forward(self, img_pair):
        img1, img2,img3,img4,img5,img6 = torch.chunk(img_pair, chunks=6, dim=1)

        # Remove the singleton dimension (1) at dim=1
        img1 = img1.squeeze(1)  # shape: [1, 3, 480, 640]
        img2 = img2.squeeze(1)
        img3 = img3.squeeze(1)
        img4 = img4.squeeze(1)
        img5 = img5.squeeze(1)
        img6 = img6.squeeze(1)  # [1, 3, 480, 640]
        # Resize & normalize manually
        def preprocess(img):
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)  # [B, 3, 224, 224]
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
            img = (img - mean) / std
            return img

        a_ = preprocess(img1)  # [B, 3, 224, 224]
        b_ = preprocess(img2)
        c_ = preprocess(img3)  # [B, 3, 224, 224]
        d_ = preprocess(img4)
        e_ = preprocess(img5)  # [B, 3, 224, 224]
        f_ = preprocess(img6)  # [B, 3, 224, 224]

        pose_img = torch.stack([a_, b_,c_,d_,e_,f_], dim=1) 

    
        translation, rotation = self.posenet(pose_img)
        translation = translation[:,:1,:]
        rotation = rotation[:,:1,:]
        q_flat = rotation.reshape(-1, 4)
        # Convert to rotation vector (axis-angle): [B * T, 3]
        rotvec_flat = kornia.geometry.quaternion_to_axis_angle(q_flat)
        # reshape: [B, T, 3]
        rotation = rotvec_flat.view(rotation.shape[0], rotation.shape[1], 3)

        Pec = torch.cat([translation, rotation], dim=-1) 
        img_pair = img_pair[:,:2]
        img_pair = img_pair.view(1, 2 * 3, 480, 640)
        nflow_pred = self.nflownet(img_pair)
        nflow_pred = nflow_pred[:, :, 1:-1, 1:-1]  # [1, 2, 478, 638]
        gray = nflow_pred[:, :3].mean(1, keepdim=True)
        grad_dirs = compute_image_gradients(gray)  # [B, 2, H, W]


        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        
        # Compute gradients for both images and average
        grad_x1 = F.conv2d(nflow_pred[:, 0:1, :, :], sobel_x, padding=1)
        grad_y1 = F.conv2d(nflow_pred[:, 0:1, :, :], sobel_y, padding=1)
        grad_x2 = F.conv2d(nflow_pred[:, 1:2, :, :], sobel_x, padding=1)
        grad_y2 = F.conv2d(nflow_pred[:, 1:2, :, :], sobel_y, padding=1)

        image_gradients = 0.5 * (torch.cat([grad_x1, grad_y1], dim=1) + torch.cat([grad_x2, grad_y2], dim=1))  # [B, 2, H, W]
        img_rand = torch.rand(1, 6, 478, 638)
        img_pair_shape = img_rand.shape
        # Lower-level: refine using cheirality
        Per, dL_dPer = self.refine_pose(img_pair_shape, Pec, image_gradients, nflow_pred)       
        upper_loss = self.upper_level_loss(img_pair_shape, Pec, Per, nflow_pred, image_gradients,img_pair.device)

        print("Upper level loss:")
        print(upper_loss)
        # ∂L_upper / ∂Pec + ∂L_upper / ∂Per * ∂Per / ∂Pec
        dL_dPec_upper = grad(upper_loss, Pec, retain_graph=True, create_graph=True)[0]
        dL_dPer_upper = grad(upper_loss, Per, retain_graph=True, create_graph=True)[0]

        # ∂L_total / ∂Pec = ∂L_upper / ∂Pec - dL_dPer_upper ⊙ d_cheirality_loss / d_Per
        total_grad = dL_dPec_upper - torch.autograd.grad((dL_dPer * dL_dPer_upper).sum(), Pec, retain_graph=True)[0]

        # Kayıp gibi davranarak backward yap
        dummy_loss = (Pec * total_grad.detach()).sum()
        return dummy_loss
           