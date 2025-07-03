import torch
import torch.nn as nn
import torch.nn.functional as F
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from RemoteSensing.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from RemoteSensing.changedetection.models.GuidedFusion import PyramidFusion, DepthwiseSeparableConv

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-1]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_44 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-1]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_31 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-2]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_35 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-2]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-3]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_25 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-3]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-4]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_15 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-4]*2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_46 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs.get('ssm_d_state', 16), ssm_ratio=kwargs.get('ssm_ratio', 2.0),
                ssm_dt_rank=kwargs.get('ssm_dt_rank', 32), ssm_act_layer=kwargs.get('ssm_act_layer', nn.SiLU),
                ssm_conv=kwargs.get('ssm_conv', 3), ssm_conv_bias=kwargs.get('ssm_conv_bias', True),
                ssm_drop_rate=kwargs.get('ssm_drop_rate', 0.0), ssm_init=kwargs.get('ssm_init', 'v0'),
                forward_type=kwargs.get('forward_type', 'v0'), mlp_ratio=kwargs.get('mlp_ratio', 4.0),
                mlp_act_layer=kwargs.get('mlp_act_layer', nn.GELU), mlp_drop_rate=kwargs.get('mlp_drop_rate', 0.0),
                gmlp=kwargs.get('gmlp', False), use_checkpoint=kwargs.get('use_checkpoint', False)
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_36 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs.get('ssm_d_state', 16), ssm_ratio=kwargs.get('ssm_ratio', 2.0),
                ssm_dt_rank=kwargs.get('ssm_dt_rank', 32), ssm_act_layer=kwargs.get('ssm_act_layer', nn.SiLU),
                ssm_conv=kwargs.get('ssm_conv', 3), ssm_conv_bias=kwargs.get('ssm_conv_bias', True),
                ssm_drop_rate=kwargs.get('ssm_drop_rate', 0.0), ssm_init=kwargs.get('ssm_init', 'v0'),
                forward_type=kwargs.get('forward_type', 'v0'), mlp_ratio=kwargs.get('mlp_ratio', 4.0),
                mlp_act_layer=kwargs.get('mlp_act_layer', nn.GELU), mlp_drop_rate=kwargs.get('mlp_drop_rate', 0.0),
                gmlp=kwargs.get('gmlp', False), use_checkpoint=kwargs.get('use_checkpoint', False)
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_26 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs.get('ssm_d_state', 16), ssm_ratio=kwargs.get('ssm_ratio', 2.0),
                ssm_dt_rank=kwargs.get('ssm_dt_rank', 32), ssm_act_layer=kwargs.get('ssm_act_layer', nn.SiLU),
                ssm_conv=kwargs.get('ssm_conv', 3), ssm_conv_bias=kwargs.get('ssm_conv_bias', True),
                ssm_drop_rate=kwargs.get('ssm_drop_rate', 0.0), ssm_init=kwargs.get('ssm_init', 'v0'),
                forward_type=kwargs.get('forward_type', 'v0'), mlp_ratio=kwargs.get('mlp_ratio', 4.0),
                mlp_act_layer=kwargs.get('mlp_act_layer', nn.GELU), mlp_drop_rate=kwargs.get('mlp_drop_rate', 0.0),
                gmlp=kwargs.get('gmlp', False), use_checkpoint=kwargs.get('use_checkpoint', False)
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_16 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs.get('ssm_d_state', 16), ssm_ratio=kwargs.get('ssm_ratio', 2.0),
                ssm_dt_rank=kwargs.get('ssm_dt_rank', 32), ssm_act_layer=kwargs.get('ssm_act_layer', nn.SiLU),
                ssm_conv=kwargs.get('ssm_conv', 3), ssm_conv_bias=kwargs.get('ssm_conv_bias', True),
                ssm_drop_rate=kwargs.get('ssm_drop_rate', 0.0), ssm_init=kwargs.get('ssm_init', 'v0'),
                forward_type=kwargs.get('forward_type', 'v0'), mlp_ratio=kwargs.get('mlp_ratio', 4.0),
                mlp_act_layer=kwargs.get('mlp_act_layer', nn.GELU), mlp_drop_rate=kwargs.get('mlp_drop_rate', 0.0),
                gmlp=kwargs.get('gmlp', False), use_checkpoint=kwargs.get('use_checkpoint', False)
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer  
        self.fuse_layer_4 = PyramidFusion(in_channels=128 * 7, out_channels=128)
        self.fuse_layer_3 = PyramidFusion(in_channels=128 * 7, out_channels=128)
        self.fuse_layer_2 = PyramidFusion(in_channels=128 * 7, out_channels=128)
        self.fuse_layer_1 = PyramidFusion(in_channels=128 * 7, out_channels=128)

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
        self.attention_conv_4 = nn.Conv2d(in_channels=encoder_dims[-1], out_channels=128, kernel_size=1)
        self.attention_conv_3 = nn.Conv2d(in_channels=encoder_dims[-2], out_channels=128, kernel_size=1)
        self.attention_conv_2 = nn.Conv2d(in_channels=encoder_dims[-3], out_channels=128, kernel_size=1)
        self.attention_conv_1 = nn.Conv2d(in_channels=encoder_dims[-4], out_channels=128, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        change_maps = []
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat((pre_feat_4, post_feat_4), dim=1))
        B, C, H, W = pre_feat_4.size()
        ct_tensor_41 = torch.empty(B, 2*C, H, W, device=pre_feat_4.device)
        ct_tensor_41[:, ::2, :, :] = pre_feat_4
        ct_tensor_41[:, 1::2, :, :] = post_feat_4
        p45 = self.st_block_44(ct_tensor_41)

        ct_tensor_42 = torch.empty(B, C, H, 2*W, device=pre_feat_4.device)
        ct_tensor_42[:, :, :, ::2] = pre_feat_4
        ct_tensor_42[:, :, :, 1::2] = post_feat_4
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 2*W, device=pre_feat_4.device)
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43)

        diff_feat_4 = torch.abs(pre_feat_4 - post_feat_4)
        p46 = self.st_block_46(diff_feat_4)

        p4 = self.fuse_layer_4(
            torch.cat([p41, p45, p42[:, :, :, ::2], p43[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:], p46], dim=1)
        )
        # Apply change-guided attention
        attention_map_4 = torch.sigmoid(self.attention_conv_4(diff_feat_4))
        p4 = p4 * attention_map_4
        change_maps.append(p4)

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat((pre_feat_3, post_feat_3), dim=1))
        B, C, H, W = pre_feat_3.size()
        ct_tensor_31 = torch.empty(B, 2*C, H, W, device=pre_feat_3.device)
        ct_tensor_31[:, ::2, :, :] = pre_feat_3
        ct_tensor_31[:, 1::2, :, :] = post_feat_3
        p35 = self.st_block_35(ct_tensor_31)

        ct_tensor_32 = torch.empty(B, C, H, 2*W, device=pre_feat_3.device)
        ct_tensor_32[:, :, :, ::2] = pre_feat_3
        ct_tensor_32[:, :, :, 1::2] = post_feat_3
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2*W, device=pre_feat_3.device)
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        diff_feat_3 = torch.abs(pre_feat_3 - post_feat_3)
        p36 = self.st_block_36(diff_feat_3)

        p3 = self.fuse_layer_3(
            torch.cat([p31, p35, p32[:, :, :, ::2], p33[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:], p36], dim=1)
        )
        # Apply change-guided attention
        attention_map_3 = torch.sigmoid(self.attention_conv_3(diff_feat_3))
        p3 = p3 * attention_map_3
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
        change_maps.append(p3)

        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat((pre_feat_2, post_feat_2), dim=1))
        B, C, H, W = pre_feat_2.size()
        ct_tensor_21 = torch.empty(B, 2*C, H, W, device=pre_feat_2.device)
        ct_tensor_21[:, ::2, :, :] = pre_feat_2
        ct_tensor_21[:, 1::2, :, :] = post_feat_2
        p25 = self.st_block_25(ct_tensor_21)

        ct_tensor_22 = torch.empty(B, C, H, 2*W, device=pre_feat_2.device)
        ct_tensor_22[:, :, :, ::2] = pre_feat_2
        ct_tensor_22[:, :, :, 1::2] = post_feat_2
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2*W, device=pre_feat_2.device)
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        diff_feat_2 = torch.abs(pre_feat_2 - post_feat_2)
        p26 = self.st_block_26(diff_feat_2)

        p2 = self.fuse_layer_2(
            torch.cat([p21, p25, p22[:, :, :, ::2], p23[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:], p26], dim=1)
        )
        # Apply change-guided attention
        attention_map_2 = torch.sigmoid(self.attention_conv_2(diff_feat_2))
        p2 = p2 * attention_map_2
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
        change_maps.append(p2)

        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat((pre_feat_1, post_feat_1), dim=1))
        B, C, H, W = pre_feat_1.size()
        ct_tensor_11 = torch.empty(B, 2*C, H, W, device=pre_feat_1.device)
        ct_tensor_11[:, ::2, :, :] = pre_feat_1
        ct_tensor_11[:, 1::2, :, :] = post_feat_1
        p15 = self.st_block_15(ct_tensor_11)

        ct_tensor_12 = torch.empty(B, C, H, 2*W, device=pre_feat_1.device)
        ct_tensor_12[:, :, :, ::2] = pre_feat_1
        ct_tensor_12[:, :, :, 1::2] = post_feat_1
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2*W, device=pre_feat_1.device)
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        diff_feat_1 = torch.abs(pre_feat_1 - post_feat_1)
        p16 = self.st_block_16(diff_feat_1)

        p1 = self.fuse_layer_1(
            torch.cat([p11, p15, p12[:, :, :, ::2], p13[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:], p16], dim=1)
        )
        # Apply change-guided attention
        attention_map_1 = torch.sigmoid(self.attention_conv_1(diff_feat_1))
        p1 = p1 * attention_map_1
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        change_maps.append(p1)

        return p1, change_maps