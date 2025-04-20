import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        
        vgg = models.vgg16(pretrained=True)
        self.cnn = nn.Sequential(*list(vgg.features.children()))
        
        self.feature_dim = 512 * 7 * 7  # assuming input image is 224x224
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=250, num_layers=2, batch_first=True)

        # Step 4: Regress 6D pose (translation + rotation)
        self.pose_fc = nn.Linear(250, 7)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        cnn_feats = []

        for t in range(seq_len):
            feat = self.cnn(x[:, t])       # shape: [B, 512, 7, 7]
            feat = feat.view(batch_size, -1)  # flatten
            cnn_feats.append(feat)
        
        feats = torch.stack(cnn_feats, dim=1)  # shape: [B, seq_len, feat_dim]
        lstm_out, _ = self.lstm(feats)         # shape: [B, seq_len, 250]
        pose = self.pose_fc(lstm_out[:, -1])   # use last output for pose
        t = pose[:, :3]
        q = pose[:, 3:]
        q = q / q.norm(dim=1, keepdim=True)     # normalize quaternion

        return t, q

    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=1.0):
        loss_t = F.mse_loss(t_pred, t_gt)
        loss_q = self.quaternion_loss(q_pred = q_pred, q_gt = q_gt)
        return loss_t + lambda_q * loss_q
    
    # geosedic loss
    def quaternion_loss(self, q_pred, q_gt):
        q_pred = F.normalize(q_pred, dim=1)
        q_gt = F.normalize(q_gt, dim=1)
        dot = torch.sum(q_pred * q_gt, dim=1) 
        return torch.mean(1.0 - torch.abs(dot))