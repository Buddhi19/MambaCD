import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChangeGuidedAttention(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])
        
    def forward(self, features_list, change_map):
        conditioned_features = []
        for i, features in enumerate(features_list):
            # Down sample change_map to match feature size
            _, _, H, W = features.shape
            change_resized = F.interpolate(change_map, size=(H, W), mode='bilinear')
            
            # Compute attention
            attention = torch.sigmoid(self.conv_layers[i](change_resized))
            
            # Condition features
            conditioned_features.append(features * (1 + attention))
        
        return conditioned_features


class MultiScaleChangeGuidedAttention_StageByStage(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.activation = torch.sigmoid
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])

    def forward(self, feature_maps, change_maps):
        conditioned_features = []
        for i in range(len(feature_maps)):
            feature_map = feature_maps[i]
            change_map = change_maps[i]
            # Compute attention
            change_map = self.conv_layers[i](change_map)
            attention = self.activation(change_map)

            # print(f"Attention shape: {attention.shape}, Feature map shape: {feature_map.shape}")
            
            # Condition features
            conditioned_features.append(feature_map * (1 + attention))
        return conditioned_features


class ChangeGuidedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.sigmoid

    def forward(self, feature_map, change_map):
        attention = self.activation(change_map)
        # Condition features
        conditioned_features = feature_map * attention

        return conditioned_features