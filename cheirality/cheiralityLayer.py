import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class CheiralityLayer(nn.Module):
    def __init__(self, posenet, nflownet):
        super().__init__()
        self.posenet = posenet
        self.nflownet = nflownet
        self.nflownet.eval()  # freeze NFlowNet
        for p in self.nflownet.parameters():
            p.requires_grad = False

    def get_normal_flow(self, img_pair):
        with torch.no_grad():
            return self.nflownet(img_pair)  # [B, 2, H, W]

    def get_image_gradients(self, gray):
        gx = F.conv2d(gray, torch.tensor([[[[-1, 0, 1]]]], device=gray.device), padding=(0, 1))
        gy = F.conv2d(gray, torch.tensor([[[[-1], [0], [1]]]], device=gray.device), padding=(1, 0))
        norm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return torch.cat([gx / norm, gy / norm], dim=1)

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

    def cheirality_loss(self, img_pair, pose):
        B, _, H, W = img_pair.shape
        device = img_pair.device
        V, W_ = pose[:, :3], pose[:, 3:]

        normal_flow = self.get_normal_flow(img_pair)  # [B, 2, H, W]
        gray = img_pair[:, :3].mean(1, keepdim=True)
        grad_dirs = self.get_image_gradients(gray)  # [B, 2, H, W]
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

    def refine_pose(self, img_pair, Pec):
        Per = Pec.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([Per], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = self.cheirality_loss(img_pair, Per)
            loss.backward()
            return loss

        optimizer.step(closure)
        return Per

    def upper_level_loss(self, img_pair, Pec, Per):
        B, _, H, W = img_pair.shape
        device = img_pair.device
        A, B_mat = self.construct_A_B(H, W, device)

        def motion_flow(pose):
            V, W_ = pose[:, :3], pose[:, 3:]
            AV = torch.einsum("chw,bc->bhw", A, V)
            BW = torch.einsum("chw,bc->bhw", B_mat, W_)
            return AV - BW  # shape: [B, H, W]

        flow_coarse = motion_flow(Pec)
        flow_refine = motion_flow(Per)
        return F.mse_loss(flow_coarse, flow_refine)

    def forward(self, img_pair):
        # Coarse pose from PoseNet
        Pec = self.posenet(img_pair)  # [B, 6]

        # Lower-level: refine using cheirality
        Per = self.refine_pose(img_pair, Pec)

        # Upper-level: compute loss to update PoseNet
        loss = self.upper_level_loss(img_pair, Pec, Per)

        return loss
