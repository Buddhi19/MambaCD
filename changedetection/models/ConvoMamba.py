import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv1DMamba(nn.Module):
    """
    with gating applied to output of 2d convolution and 1d convolution
    """
    def __init__(
            self,
            in_channels,
            norm_layer,
            channel_first,
            ssm_act_layer,
            mlp_act_layer,
            **kwargs
    ):
        super(Conv1DMamba, self).__init__()
        self.conv2d1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.SILU_ = nn.SiLU()

        self.conv2d2 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.vssm = VSSBlock(hidden_dim=in_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'])
        
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.linear = nn.Linear(
            in_features=32,
            out_features=in_channels,
            bias=False
        )

    def forward(self,x):
        x1 = self.vssm(x)
        x_conv = x.permute(0, 3, 1, 2)

        x2 = self.conv2d2(self.SILU_(self.conv2d1(x_conv)))
        x2 = x2.permute(0, 2, 3, 1)

        height, width = x2.shape[1], x2.shape[2]
        x_conv_1d = x_conv.view(x_conv.size(0), x_conv.size(1), -1)
        x3 = self.conv1d(x_conv_1d)

        batch_size, channels = x3.size(0), x3.size(1)
        x3 = x3.view(batch_size, channels, height, width)
        x3 = x3.permute(0, 2, 3, 1)


        x3 = self.SILU_(x3)
        x3 = self.linear(x3)

        if x1.shape != x2.shape:
            print("Shapes are not equal")
            x2 = x2.view_as(x1)

        x_out1 = x1*x2
        x_out = x_out1 + x3
        return x_out