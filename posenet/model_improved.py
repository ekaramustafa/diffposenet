import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SelfAttention(nn.Module):
    """Self-attention mechanism for feature refinement"""
    def __init__(self, feature_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.norm(x + self.dropout(attn_output))

class ImprovedPoseNet(nn.Module):
    def __init__(self, backbone='resnet50', hidden_dim=256, num_layers=2, use_attention=True):
        super(ImprovedPoseNet, self).__init__()
        
        self.use_attention = use_attention
        
        # Improved backbone options
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Remove the last two layers (avgpool and fc)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
            self.spatial_dim = 7  # for 224x224 input
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 512
            self.spatial_dim = 7
        elif backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.cnn = nn.Sequential(*list(vgg.features.children()))
            self.feature_dim = 512
            self.spatial_dim = 7
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze early layers of backbone
        for i, param in enumerate(self.cnn.parameters()):
            if i < 20:  # Freeze first 20 layers
                param.requires_grad = False
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Positional encoding for sequence
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Self-attention for feature refinement
        if self.use_attention:
            self.self_attention = SelfAttention(hidden_dim)
        
        # Improved LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Separate heads for translation and rotation
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  # [B, seq_len, 3, H, W]
        
        # Extract CNN features for each frame
        cnn_feats = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t])  # [B, feature_dim, spatial_h, spatial_w]
            feat = self.adaptive_pool(feat)  # [B, feature_dim, 1, 1]
            feat = feat.view(batch_size, -1)  # [B, feature_dim]
            cnn_feats.append(feat)
        
        # Stack features and project
        feats = torch.stack(cnn_feats, dim=1)  # [B, seq_len, feature_dim]
        feats = self.feature_proj(feats)  # [B, seq_len, hidden_dim]
        
        # Add positional encoding
        feats = feats.transpose(0, 1)  # [seq_len, B, hidden_dim]
        feats = self.pos_encoding(feats)
        feats = feats.transpose(0, 1)  # [B, seq_len, hidden_dim]
        
        # Apply self-attention if enabled
        if self.use_attention:
            feats = self.self_attention(feats)
        
        # LSTM processing
        lstm_out, _ = self.lstm(feats)  # [B, seq_len, hidden_dim]
        
        # Use relative poses (skip first frame)
        lstm_out = lstm_out[:, 1:]  # [B, seq_len-1, hidden_dim]
        
        # Separate prediction heads
        t_pred = self.translation_head(lstm_out)  # [B, seq_len-1, 3]
        q_pred = self.rotation_head(lstm_out)     # [B, seq_len-1, 4]
        
        # Normalize quaternions
        q_pred = F.normalize(q_pred, p=2, dim=2)
        
        return t_pred, q_pred
    
    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=1.0, lambda_smooth=0.1):
        """Improved loss function with smoothness regularization"""
        
        # Basic pose losses
        loss_t = F.mse_loss(t_pred, t_gt)
        loss_q = self.quaternion_loss(q_pred, q_gt)
        
        # Smoothness regularization
        loss_smooth_t = 0.0
        loss_smooth_q = 0.0
        
        if t_pred.size(1) > 1:  # If we have multiple time steps
            # Translation smoothness
            t_diff = t_pred[:, 1:] - t_pred[:, :-1]
            loss_smooth_t = torch.mean(torch.norm(t_diff, dim=2))
            
            # Rotation smoothness (angular velocity)
            q_diff = self._quaternion_angular_distance(q_pred[:, 1:], q_pred[:, :-1])
            loss_smooth_q = torch.mean(q_diff)
        
        total_loss = (loss_t + 
                     lambda_q * loss_q + 
                     lambda_smooth * (loss_smooth_t + loss_smooth_q))
        
        return total_loss, {
            'translation_loss': loss_t.item(),
            'rotation_loss': loss_q.item(),
            'smoothness_t': loss_smooth_t.item() if isinstance(loss_smooth_t, torch.Tensor) else loss_smooth_t,
            'smoothness_q': loss_smooth_q.item() if isinstance(loss_smooth_q, torch.Tensor) else loss_smooth_q,
            'total_loss': total_loss.item()
        }
    
    def quaternion_loss(self, q_pred, q_gt):
        """Improved quaternion loss using geodesic distance"""
        q_pred = F.normalize(q_pred, p=2, dim=2)
        q_gt = F.normalize(q_gt, p=2, dim=2)
        
        # Compute dot product
        dot = torch.sum(q_pred * q_gt, dim=2)
        
        # Handle quaternion double cover (q and -q represent same rotation)
        dot = torch.abs(dot)
        dot = torch.clamp(dot, 0.0, 1.0)
        
        # Geodesic distance on quaternion manifold
        loss = 1.0 - dot
        return torch.mean(loss)
    
    def _quaternion_angular_distance(self, q1, q2):
        """Compute angular distance between quaternions"""
        q1 = F.normalize(q1, p=2, dim=2)
        q2 = F.normalize(q2, p=2, dim=2)
        
        dot = torch.abs(torch.sum(q1 * q2, dim=2))
        dot = torch.clamp(dot, 0.0, 1.0)
        
        # Angular distance
        angle = 2 * torch.acos(dot)
        return angle

# Backward compatibility - alias for the original PoseNet
class PoseNet(ImprovedPoseNet):
    def __init__(self):
        super(PoseNet, self).__init__(backbone='vgg16', hidden_dim=128, num_layers=2, use_attention=False)
    
    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=1.0):
        """Original loss function for backward compatibility"""
        loss_t = F.mse_loss(t_pred, t_gt)
        loss_q = self.quaternion_loss(q_pred, q_gt)
        return loss_t + lambda_q * loss_q 