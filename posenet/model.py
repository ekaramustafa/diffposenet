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
        self.hidden_dim= 128
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, dropout=0.3)

        # Step 4: Regress 6D pose (translation + rotation)
        self.pose_fc = nn.Linear(self.hidden_dim, 7)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  # [B, 6, 3, H, W]
        cnn_feats = []

        for t in range(seq_len):
            feat = self.cnn(x[:, t])           # [B, 512, 7, 7]
            feat = feat.view(batch_size, -1)   # [B, feat_dim]
            cnn_feats.append(feat)

        feats = torch.stack(cnn_feats, dim=1)  # [B, 6, feat_dim]
        lstm_out, _ = self.lstm(feats)         # [B, 6, hidden_dim]

        lstm_out = lstm_out[:, 1:]             # [B, 5, hidden_dim] – for relative poses
        pose = self.pose_fc(lstm_out)          # [B, 5, 7]

        t = pose[:, :, :3]                     # [B, 5, 3]
        q = pose[:, :, 3:]                     # [B, 5, 4]
        q = q / q.norm(dim=2, keepdim=True)    # normalize quaternion

        return t, q

    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=1.0):
        loss_t = F.mse_loss(t_pred, t_gt.squeeze(1))
        loss_q = self.quaternion_loss(q_pred = q_pred, q_gt = q_gt)
        return loss_t + lambda_q * loss_q
    
    def quaternion_loss(self, q_pred, q_gt):
        q_pred = F.normalize(q_pred, p=2, dim=2)
        q_gt = F.normalize(q_gt, p=2, dim=2)
        
        dot = torch.sum(q_pred * q_gt, dim=2)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        loss = 1.0 - torch.abs(dot)
        return torch.mean(loss)