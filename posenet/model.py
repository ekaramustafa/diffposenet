import torch
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
        self.pose_fc = nn.Linear(250, 6)

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
        return pose
