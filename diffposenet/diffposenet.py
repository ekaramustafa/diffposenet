import torch
import torch.nn as nn
import torch.functional as F

class DiffPoseNet(nn.Module):

    """
    Complete DiffPoseNet framework integrating:
    - NFlowNet for normal flow estimation
    - PoseNet for initial pose estimation
    - CheiralityLayer for pose refinement
    """
    def __init__(self, nflownet, posenet, cheirality_layer):
        super(DiffPoseNet, self).__init__()
        self.nflownet = nflownet
        self.posenet = posenet
        self.cheirality_layer = cheirality_layer
        
    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): First image [B, C, H, W]
            img2 (torch.Tensor): Second image [B, C, H, W]
            
        Returns:
            tuple: (refined_pose, init_pose, normal_flow)
        """
        # Compute normal flow between images
        with torch.no_grad():
            img = torch.cat((img1, img2), dim=1)
            normal_flow = self.nflownet(img)  # [B, H, W]
        
        # Compute image gradients (for cheirality layer)
        # Using Sobel filters as approximation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        
        # Compute gradients for both images and average
        grad_x1 = F.conv2d(img1.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y1 = F.conv2d(img1.mean(dim=1, keepdim=True), sobel_y, padding=1)
        grad_x2 = F.conv2d(img2.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y2 = F.conv2d(img2.mean(dim=1, keepdim=True), sobel_y, padding=1)
        
        image_gradients = 0.5 * (torch.cat([grad_x1, grad_y1], dim=1) + torch.cat([grad_x2, grad_y2], dim=1))  # [B, 2, H, W]
        
        # Get initial pose estimate
        init_pose = self.posenet(torch.stack([img1, img2], dim=1))  # [B, 6]
        
        # Refine pose using cheirality layer
        refined_pose = self.cheirality_layer(normal_flow, image_gradients, init_pose)
        
        return refined_pose, init_pose, normal_flow