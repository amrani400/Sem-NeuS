import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGuidance(nn.Module):
    def __init__(self, dino_dim=1024, proj_dim=64):
        super().__init__()
        # The learnable projection layer (1024 -> 64)
        self.gt_dino_proj = nn.Linear(dino_dim, proj_dim)

    def sample_features(self, pts, intrinsics, pose, feature_map, img_size):
        """
        Projects 3D points onto the 2D feature map and samples DINO features.
        """
        N_rays, N_samples, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        
        # World to Camera
        w2c = torch.inverse(pose)
        rot = w2c[:3, :3]
        trans = w2c[:3, 3]
        pts_cam = torch.matmul(rot, pts_flat.t()) + trans[:, None]
        pts_cam = pts_cam.t()
        
        # Camera to Image Plane
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        u = (fx * x / (z + 1e-6)) + cx
        v = (fy * y / (z + 1e-6)) + cy
        
        # Normalize to [-1, 1] for grid_sample
        H_img, W_img = img_size
        u_norm = 2 * (u / (W_img - 1)) - 1
        v_norm = 2 * (v / (H_img - 1)) - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2)
        
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)
            
        # Bilinear sampling
        sampled_feats = F.grid_sample(feature_map, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        sampled_feats = sampled_feats.squeeze(0).squeeze(-1).permute(1, 0)
        
        return sampled_feats.reshape(N_rays, N_samples, -1)

    def compute_loss(self, pred_embeddings, raw_dino_feats, weights):
        """
        Computes the distillation loss.
        """
        # 1. Project and Normalize Ground Truth (DINO)
        gt_flat = raw_dino_feats.reshape(-1, raw_dino_feats.shape[-1])
        gt_compressed = self.gt_dino_proj(gt_flat)
        gt_normalized = F.normalize(gt_compressed, p=2, dim=-1).view(pred_embeddings.shape)
        
        # 2. Normalize Prediction
        pred_normalized = F.normalize(pred_embeddings, p=2, dim=-1)
        
        # 3. Compute Weighted Cosine Distance
        cos_sim = F.cosine_similarity(pred_normalized, gt_normalized, dim=-1)
        
        # Loss = 1 - CosineSimilarity (Weighted average)
        fg_weights = weights 
        loss_val = (fg_weights * (1.0 - cos_sim)).sum() / (fg_weights.sum() + 1e-5)
        
        return loss_val