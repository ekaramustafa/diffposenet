import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    #VGG16
    def __init__(self, backbone='vgg16', freeze_cnn=True, hidden_dim=128, num_layers=2, dropout=0.3):
        super(PoseNet, self).__init__()
        
        if backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
        
        if freeze_cnn:
            for param in vgg.features.parameters():
                param.requires_grad = False
        self.cnn = nn.Sequential(*list(vgg.features.children()))
        
        self.feature_dim = 512 * 7 * 7  # assuming input image is 224x224
        self.hidden_dim= hidden_dim
        
        self.lstm = nn.GRU(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # self.pose_fc = nn.Linear(self.hidden_dim, 7)
        # Separate network heads for translation and rotation
        self.translation_fc = nn.Linear(self.hidden_dim, 3)
        self.rotation_fc = nn.Linear(self.hidden_dim, 4)

        # uncertainty parameters
        # self.log_var_t = nn.Parameter(torch.zeros(1))
        # self.log_var_q = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  # [B, 6, 3, H, W]
        cnn_feats = []

        for t in range(seq_len):
            feat = self.cnn(x[:, t])           # [B, 512, 7, 7]
            feat = feat.view(batch_size, -1)   # [B, feat_dim]
            cnn_feats.append(feat)

        feats = torch.stack(cnn_feats, dim=1)  # [B, 6, feat_dim]
        lstm_out, _ = self.lstm(feats)         # [B, 6, hidden_dim]

        lstm_out = lstm_out[:, 1:]             # [B, 5, hidden_dim]
        # self.pose_fc = nn.Linear(self.hidden_dim, 7)
        # pose = self.pose_fc(lstm_out)          # [B, 5, 7]

        # t = pose[:, :, :3]                     # [B, 5, 3]
        # q = pose[:, :, 3:]                     # [B, 5, 4]
        # Use separate heads for translation and rotation
        t = self.translation_fc(lstm_out)      # [B, 5, 3]
        q = self.rotation_fc(lstm_out)         # [B, 5, 4]
        q = q / q.norm(dim=2, keepdim=True)    # normalize quaternion

        return t, q

    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=0.01):
        loss_t = F.mse_loss(t_pred, t_gt.squeeze(1))
        loss_q = self.quaternion_loss(q_pred = q_pred, q_gt = q_gt)
        return loss_t + lambda_q * loss_q,  {
            'translation_loss': loss_t.item(),
            'quaternion_loss': loss_q.item(),
            'total_loss': (loss_t + lambda_q * loss_q).item()
        }

    # def pose_loss(self, t_pred, q_pred, t_gt, q_gt):
    #     loss_t = F.mse_loss(t_pred, t_gt.squeeze(1))
    #     loss_q = self.quaternion_loss(q_pred=q_pred, q_gt=q_gt)
        
    #     # Adaptive weighting based on learned uncertainties
    #     precision_t = torch.exp(-self.log_var_t)
    #     precision_q = torch.exp(-self.log_var_q)
        
    #     weighted_loss_t = precision_t * loss_t + self.log_var_t
    #     weighted_loss_q = precision_q * loss_q + self.log_var_q
        
    #     total_loss = weighted_loss_t + weighted_loss_q
        
    #     # Ensure loss is non-negative
    #     total_loss = torch.clamp(total_loss, min=0.0)
        
    #     return total_loss, {
    #         'translation_loss': loss_t.item(),
    #         'quaternion_loss': loss_q.item(),
    #         'weight_t': precision_t.item(),
    #         'weight_q': precision_q.item(),
    #         'total_loss': total_loss.item()
    #     }
    
    def quaternion_loss(self, q_pred, q_gt):
        q_pred = F.normalize(q_pred, p=2, dim=2)
        q_gt = F.normalize(q_gt, p=2, dim=2)
        
        dot = torch.sum(q_pred * q_gt, dim=2)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        loss = 1.0 - torch.abs(dot)
        return torch.mean(loss)


class PoseNetDino(PoseNet):
    def __init__(self, model_size='base', freeze_dino=True):
        nn.Module.__init__(self)
        
        if model_size == 'small':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.feature_dim = 384
        elif model_size == 'base':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.feature_dim = 768
        elif model_size == 'large':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.feature_dim = 1024
        else:
            raise ValueError("model_size must be 'small', 'base', or 'large'")
        
        if freeze_dino:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        self.hidden_dim = 128
        
        self.lstm = nn.GRU(input_size=self.feature_dim, hidden_size=self.hidden_dim, 
                          num_layers=2, batch_first=True, dropout=0.3)

        # Separate network heads for translation and rotation
        # self.pose_fc = nn.Linear(self.hidden_dim, 7)
        self.translation_fc = nn.Linear(self.hidden_dim, 3)
        self.rotation_fc = nn.Linear(self.hidden_dim, 4)

        # uncertainty parameters
        # self.log_var_t = nn.Parameter(torch.zeros(1))
        # self.log_var_q = nn.Parameter(torch.zeros(1))

    def extract_cls_token(self, x):
        if self.dinov2.training and not self.training:
            self.dinov2.eval()
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.dinov2(x)  # [B, feature_dim]
            return features

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  # [B, 6, 3, H, W]
        dino_feats = []

        for t in range(seq_len):
            feat = self.extract_cls_token(x[:, t])  # [B, feature_dim]
            dino_feats.append(feat)

        feats = torch.stack(dino_feats, dim=1)  # [B, 6, feature_dim]
        lstm_out, _ = self.lstm(feats)          # [B, 6, hidden_dim]

        lstm_out = lstm_out[:, 1:]              # [B, 5, hidden_dim] – for relative poses
        # pose = self.pose_fc(lstm_out)           # [B, 5, 7]

        # t = pose[:, :, :3]                      # [B, 5, 3]
        # q = pose[:, :, 3:]                      # [B, 5, 4]
        # Use separate heads for translation and rotation
        t = self.translation_fc(lstm_out)       # [B, 5, 3]
        q = self.rotation_fc(lstm_out)          # [B, 5, 4]
        q = q / q.norm(dim=2, keepdim=True)     # normalize quaternion

        return t, q