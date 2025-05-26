import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoImageProcessor

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()

        # Load DINOv2
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base")  # output: [B, 197, 768]
        self.feature_dim = 768  # CLS token dimension

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=250, num_layers=2, batch_first=True, dropout=0.3)

        # Step 4: Regress 6D pose (translation + rotation)
        self.pose_fc = nn.Linear(250, 7)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        dino_feats = []

        for t in range(seq_len):
            frame = x[:, t]  # shape: [B, 3, 224, 224]
            with torch.no_grad():  # freeze DINO if needed
                outputs = self.dino(frame)  # returns a ModelOutput
                cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token: shape [B, 768]

            dino_feats.append(cls_token)

        feats = torch.stack(dino_feats, dim=1)  # shape: [B, seq_len, 768]
        lstm_out, _ = self.lstm(feats)          # shape: [B, seq_len, 250]
        pose = self.pose_fc(lstm_out[:, -1])    # last timestep
        t = pose[:, :3]
        q = pose[:, 3:]
        q = q / q.norm(dim=1, keepdim=True)     # normalize quaternion

        return t, q

    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=1.0):
        loss_t = F.mse_loss(t_pred, t_gt.squeeze(1))
        loss_q = self.quaternion_loss(q_pred=q_pred, q_gt=q_gt)
        return loss_t + lambda_q * loss_q

    def quaternion_loss(self, q_pred, q_gt):
        q_pred = F.normalize(q_pred, p=2, dim=1)
        q_gt = F.normalize(q_gt, p=2, dim=1)
        dot = torch.sum(q_pred * q_gt, dim=1)
        dot = torch.clamp(dot, -1.0, 1.0)
        loss = 1.0 - torch.abs(dot)
        return torch.mean(loss)
