import torch
import torch.nn as nn
import torch.nn.functional as F
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from RemoteSensing.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from RemoteSensing.changedetection.models.GuidedFusion import PyramidFusion, DepthwiseSeparableConv
from RemoteSensing.changedetection.models.MultiScaleChangeGuidedAttention import ChangeGuidedAttention

class SemanticDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(SemanticDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_4_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_1_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )           

        self.trans_layer_4 = DepthwiseSeparableConv(in_channels=encoder_dims[-1], out_channels=128)
        self.trans_layer_3 = DepthwiseSeparableConv(in_channels=encoder_dims[-2], out_channels=128)
        self.trans_layer_2 = DepthwiseSeparableConv(in_channels=encoder_dims[-3], out_channels=128)
        self.trans_layer_1 = DepthwiseSeparableConv(in_channels=encoder_dims[-4], out_channels=128)


        # Smooth layer
        self.smooth_layer_3_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_0_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features, change_maps):
        feat_1, feat_2, feat_3, feat_4 = features
        change_map_1, change_map_2, change_map_3, change_map_4 = change_maps

        '''
            Stage I
        '''
        p4 = self.trans_layer_4(feat_4)
        p4 = ChangeGuidedAttention()(p4, change_map_4)
        p4 = self.st_block_4_semantic(p4)
        '''
            Stage II
        '''
        p3 = self.trans_layer_3(feat_3)
        p3 = ChangeGuidedAttention()(p3, change_map_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3_semantic(p3)
        p3 = self.st_block_3_semantic(p3)

        '''
            Stage III
        '''
        p2 = self.trans_layer_2(feat_2)
        p2 = ChangeGuidedAttention()(p2, change_map_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2_semantic(p2)
        p2 = self.st_block_2_semantic(p2)

        '''
            Stage IV
        '''
        p1 = self.trans_layer_1(feat_1)
        p1 = ChangeGuidedAttention()(p1, change_map_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1_semantic(p1)
        p1 = self.st_block_1_semantic(p1)
        p1 = self.smooth_layer_0_semantic(p1)
        return p1 