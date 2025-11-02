import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAttention(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims
        self.projection_layers = nn.ModuleList([
            nn.Linear(dim, 256) for dim in feature_dims
        ])
        
        self.attention = nn.MultiheadAttention(256, 8)
        self.layer_norm = nn.LayerNorm(256)
    
    def forward(self, modalities):
        projected = []
        for i, modality in enumerate(modalities):
            if len(modality.shape) == 1:
                modality = modality.unsqueeze(0)
            proj = self.projection_layers[i](modality)
            projected.append(proj)
        
        stacked = torch.stack(projected, dim=0)
        attended, _ = self.attention(stacked, stacked, stacked)
        
        fused = attended.mean(dim=0)
        fused = self.layer_norm(fused)
        
        return fused.flatten()