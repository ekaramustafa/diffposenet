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
        
        self.hidden_dim = 256
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_dim, 
                          num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)

        # Separate network heads for translation and rotation
        # self.pose_fc = nn.Linear(self.hidden_dim, 7)
        # self.translation_fc = nn.Linear(self.hidden_dim, 3)
        # self.rotation_fc = nn.Linear(self.hidden_dim, 4)

        self.translation_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 4, 3)
        )
        
        self.rotation_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 4, 4)
        )

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
            nn.Linear(self.hidden_dim, self.projection_dim),
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
        lstm_projected = self.lstm_projector(lstm_contrastive.reshape(-1, self.hidden_dim))   # [B*5, projection_dim]
        
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
        loss_q = self.quaternion_loss(q_pred=q_pred, q_gt=q_gt)
        pose_loss = loss_t + lambda_q * loss_q
        
        contrastive_loss = self.contrastive_loss(dino_proj, lstm_proj)
        
        total_loss = pose_loss + lambda_contrastive * contrastive_loss
        
        return total_loss, {
            'translation_loss': loss_t.item(),
            'quaternion_loss': loss_q.item(),
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
            nn.Linear(self.hidden_dim // 4, 4)
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


class PoseNetDinoMultiScale(PoseNetDino):
    """PoseNetDino with multi-scale feature extraction for better translation learning"""
    
    def __init__(self, model_size='base', freeze_dino=True, hidden_dim=256):
        super().__init__(model_size=model_size, freeze_dino=freeze_dino)
        
        self.hidden_dim = hidden_dim
        
        # Multi-scale feature extraction
        self.scale_factors = [1.0, 0.75, 0.5]  # Different scales
        
        # Feature aggregation
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * len(self.scale_factors), self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Enhanced LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Pose heads
        self.translation_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 3)
        )
        
        self.rotation_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 4)
        )
    
    def extract_multiscale_features(self, x):
        """Extract features at multiple scales"""
        features = []
        B, C, H, W = x.shape
        
        for scale in self.scale_factors:
            if scale != 1.0:
                # Resize image
                new_size = (int(H * scale), int(W * scale))
                x_scaled = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Extract DINOv2 features
            feat = self.extract_cls_token(x_scaled)
            features.append(feat)
        
        # Concatenate multi-scale features
        multi_scale_feat = torch.cat(features, dim=1)
        
        # Fuse features
        fused_feat = self.multi_scale_fusion(multi_scale_feat)
        
        return fused_feat
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        dino_feats = []

        for t in range(seq_len):
            feat = self.extract_multiscale_features(x[:, t])
            dino_feats.append(feat)

        feats = torch.stack(dino_feats, dim=1)
        lstm_out, _ = self.lstm(feats)
        lstm_out = lstm_out[:, 1:]

        t = self.translation_fc(lstm_out)
        q = self.rotation_fc(lstm_out)
        q = q / q.norm(dim=2, keepdim=True)

        return t, q