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
        self.rotation_fc = nn.Linear(self.hidden_dim, 3)

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

    def pose_loss(self, t_pred, r_pred, t_gt, r_gt, lambda_t=1, lambda_q=0.001):

        translation_loss = torch.nn.functional.mse_loss(t_pred, t_gt)
    
        rotation_loss = torch.nn.functional.mse_loss(r_pred, r_gt)
                
        total_loss = translation_loss * lambda_t + rotation_loss * lambda_q  # or with different weights
        
        return total_loss, {
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'total_loss': total_loss
        }


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
        
        self.hidden_dim = 256
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, 
                          num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)

        # Separate network heads for translation and rotation
        # Account for bidirectional LSTM output (hidden_dim * 2)
        lstm_output_dim = self.hidden_dim * 2
        
        self.translation_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        
        self.rotation_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, 6)
        )


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
        
        t = self.translation_fc(lstm_out)       # [B, 5, 3]
        r = self.rotation_fc(lstm_out)          # [B, 5, 6]
        r = rotation_6d_to_matrix(r) # [B, 5, 3, 3]

        return t, r


class PoseNetDinoContrastive(PoseNetDino):
    def __init__(self, model_size='base', freeze_dino=True, projection_dim=128, temperature=0.07):
        super().__init__(model_size=model_size, freeze_dino=freeze_dino)
        
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        self.dino_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        
        self.lstm_projector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.projection_dim),  # Account for bidirectional LSTM
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        
    def forward(self, x, return_contrastive_features=False):
        batch_size, seq_len, C, H, W = x.shape  # [B, 6, 3, H, W]
        dino_feats = []
        raw_dino_feats = []

        for t in range(seq_len):
            feat = self.extract_cls_token(x[:, t])  # [B, feature_dim]
            dino_feats.append(feat)
            raw_dino_feats.append(feat)

        feats = torch.stack(dino_feats, dim=1)  # [B, 6, feature_dim]
        raw_dino_feats = torch.stack(raw_dino_feats, dim=1)  # [B, 6, feature_dim]
        
        lstm_out, _ = self.lstm(feats)          # [B, 6, hidden_dim]

        dino_contrastive = raw_dino_feats[:, 1:]  # [B, 5, feature_dim]
        lstm_contrastive = lstm_out[:, 1:]        # [B, 5, hidden_dim]
        
        dino_projected = self.dino_projector(dino_contrastive.reshape(-1, self.feature_dim))  # [B*5, projection_dim]
        lstm_projected = self.lstm_projector(lstm_contrastive.reshape(-1, self.hidden_dim * 2))   # [B*5, projection_dim]
        
        dino_projected = F.normalize(dino_projected, dim=1)
        lstm_projected = F.normalize(lstm_projected, dim=1)
        
        dino_projected = dino_projected.reshape(batch_size, 5, self.projection_dim)
        lstm_projected = lstm_projected.reshape(batch_size, 5, self.projection_dim)

        t = self.translation_fc(lstm_contrastive)       # [B, 5, 3]
        q = self.rotation_fc(lstm_contrastive)          # [B, 5, 4]
        q = q / q.norm(dim=2, keepdim=True)             # normalize quaternion

        if return_contrastive_features:
            return t, q, dino_projected, lstm_projected
        else:
            return t, q
    
    def contrastive_loss(self, dino_proj, lstm_proj):
        batch_size, seq_len, proj_dim = dino_proj.shape
        
        dino_flat = dino_proj.reshape(-1, proj_dim)  
        lstm_flat = lstm_proj.reshape(-1, proj_dim)  
        
        logits = torch.matmul(dino_flat, lstm_flat.T) / self.temperature 
        
        labels = torch.arange(batch_size * seq_len, device=dino_proj.device)
        
        loss_dino_to_lstm = F.cross_entropy(logits, labels)
        loss_lstm_to_dino = F.cross_entropy(logits.T, labels)
        
        contrastive_loss = (loss_dino_to_lstm + loss_lstm_to_dino) / 2
        
        return contrastive_loss
    
    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, 
                                   dino_proj, lstm_proj, 
                                   lambda_q=0.01, lambda_contrastive=0.1):
        
        loss_t = F.mse_loss(t_pred, t_gt.squeeze(1))
        
        # Handle rotation loss - check if q_gt is 3D or 4D
        if q_gt.shape[-1] == 3:
            # Ground truth is 3D (axis-angle or euler), convert quaternion pred to 3D
            q_pred_3d = q_pred[..., 1:]  # Take the vector part of quaternion [x, y, z]
            loss_q = F.mse_loss(q_pred_3d, q_gt)
        elif q_gt.shape[-1] == 4:
            # Ground truth is quaternion, use proper quaternion loss
            loss_q = self.quaternion_loss(q_pred=q_pred, q_gt=q_gt)
        else:
            raise ValueError(f"Unexpected rotation dimension: {q_gt.shape[-1]}")
            
        pose_loss = loss_t + lambda_q * loss_q
        
        contrastive_loss = self.contrastive_loss(dino_proj, lstm_proj)
        
        total_loss = pose_loss + lambda_contrastive * contrastive_loss
        
        return total_loss, {
            'translation_loss': loss_t.item(),
            'rotation_loss': loss_q.item(),
            'contrastive_loss': contrastive_loss.item(),
            'pose_loss': pose_loss.item(),
            'total_loss': total_loss.item()
        }


class PoseNetDinoImproved(PoseNetDino):
    def __init__(self, model_size='base', freeze_dino=True, hidden_dim=256, use_attention=True, 
                    use_bidirectional_lstm=True, dropout=0.3):
        super().__init__(model_size=model_size, freeze_dino=freeze_dino)
        
        self.use_attention = use_attention
        self.use_bidirectional_lstm = use_bidirectional_lstm
        
        # Override hidden_dim
        self.hidden_dim = hidden_dim
        
        # Improved temporal modeling
        lstm_directions = 2 if use_bidirectional_lstm else 1
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim // lstm_directions,
            num_layers=3,  # Deeper LSTM
            batch_first=True, 
            dropout=dropout,
            bidirectional=use_bidirectional_lstm
        )
        
        # Temporal attention
        if self.use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim, num_heads=8, dropout=dropout
            )
            self.temporal_norm = nn.LayerNorm(self.hidden_dim)
        
        # Enhanced pose regression heads with residual connections
        self.translation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, 6)
        )
        
        # Scale prediction for translation (to handle varying motion scales)
        self.translation_scale = nn.Parameter(torch.ones(1))
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for better convergence"""
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
        batch_size, seq_len, C, H, W = x.shape  # [B, 6, 3, H, W]
        dino_feats = []

        for t in range(seq_len):
            feat = self.extract_cls_token(x[:, t])
            dino_feats.append(feat)

        feats = torch.stack(dino_feats, dim=1)  # [B, 6, feature_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(feats)          # [B, 6, hidden_dim]
        
        # Take relative poses (frames 2-6 relative to previous)
        lstm_out = lstm_out[:, 1:]              # [B, 5, hidden_dim]
        
        # Apply temporal attention if enabled
        if self.use_attention:
            lstm_out_t = lstm_out.transpose(0, 1)  # [5, B, hidden_dim]
            attended_out, _ = self.temporal_attention(
                lstm_out_t, lstm_out_t, lstm_out_t
            )
            lstm_out = self.temporal_norm(attended_out.transpose(0, 1))
        
        # Pose prediction
        t = self.translation_head(lstm_out)     # [B, 5, 3]
        q = self.rotation_head(lstm_out)        # [B, 5, 4]
        
        # Apply learned scale to translation
        t = t * self.translation_scale
        
        # Normalize quaternion
        q = q / q.norm(dim=2, keepdim=True)
        
        return t, q
    
    def pose_loss(self, t_pred, q_pred, t_gt, q_gt, lambda_q=0.1, lambda_smooth=0.01):
        """Enhanced loss function with smoothness regularization"""
        # Main translation loss - using Smooth L1 instead of MSE for robustness
        loss_t = F.smooth_l1_loss(t_pred, t_gt.squeeze(1))
        
        # Quaternion loss
        loss_q = self.quaternion_loss(q_pred=q_pred, q_gt=q_gt)
        
        # Smoothness regularization for translation
        if t_pred.shape[1] > 1:
            t_diff = t_pred[:, 1:] - t_pred[:, :-1]
            loss_smooth_t = torch.mean(torch.norm(t_diff, dim=2))
        else:
            loss_smooth_t = torch.tensor(0.0, device=t_pred.device)
        
        # Smoothness regularization for rotation
        if q_pred.shape[1] > 1:
            q_diff = q_pred[:, 1:] - q_pred[:, :-1]
            loss_smooth_q = torch.mean(torch.norm(q_diff, dim=2))
        else:
            loss_smooth_q = torch.tensor(0.0, device=q_pred.device)
        
        # Total loss
        total_loss = (loss_t + 
                     lambda_q * loss_q + 
                     lambda_smooth * (loss_smooth_t + loss_smooth_q))
        
        return total_loss, {
            'translation_loss': loss_t.item(),
            'quaternion_loss': loss_q.item(),
            'smoothness_loss_t': loss_smooth_t.item(),
            'smoothness_loss_q': loss_smooth_q.item(),
            'total_loss': total_loss.item()
        }


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix
    
    Args:
        q: torch.Tensor of shape [batch_size, 4] 
           Quaternion in format [qx, qy, qz, qw] or [qw, qx, qy, qz]
           
    Returns:
        R: torch.Tensor of shape [batch_size, 3, 3] rotation matrices
    """
    # Normalize quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    # Assuming quaternion format [qx, qy, qz, qw]
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # If your quaternion is [qw, qx, qy, qz], use this instead:
    # qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = qx**2, qy**2, qz**2
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    xw, yw, zw = qx*qw, qy*qw, qz*qw
    
    # Build rotation matrix
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - zw)
    R[:, 0, 2] = 2*(xz + yw)
    
    R[:, 1, 0] = 2*(xy + zw)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - xw)
    
    R[:, 2, 0] = 2*(xz - yw)
    R[:, 2, 1] = 2*(yz + xw)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    
    return R


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to rotation matrix
    More stable than quaternions for neural network training
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)