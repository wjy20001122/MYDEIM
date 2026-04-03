"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINEr)

reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging
from .common import get_activation

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2', 'SimAM', 'SPPCSPC', 'C2f']


class SimAM(nn.Module):
    """
    Simple Attention Module - 零参数注意力
    论文: SimAM: A Simple, Parameter-Free Attention Module for CNNs
    对FPS几乎无影响，精度提升明显
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda) + 0.5)
        return x * torch.sigmoid(y)


class SPPCSPC(nn.Module):
    """
    SPPCSPC - Spatial Pyramid Pooling Cross Stage Partial Channel
    来自 YOLOv7，多尺度特征提取，对小目标检测有效
    """
    def __init__(self, in_channels, out_channels=None, act='relu'):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        c_ = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, c_, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act1 = get_activation(act)
        self.conv2 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.act2 = get_activation(act)
        self.conv3 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_)
        self.act3 = get_activation(act)
        self.mpool1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.mpool2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.mpool3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.conv_out = nn.Conv2d(c_ * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = get_activation(act)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        y1 = self.act2(self.bn2(self.conv2(x)))
        y2 = self.mpool1(y1)
        y3 = self.mpool2(y1)
        y4 = self.mpool3(y1)
        y5 = self.act3(self.bn3(self.conv3(x)))
        out = torch.cat([y1, y2, y3, y4, y5], dim=1)
        return self.act_out(self.bn_out(self.conv_out(out)))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu'):
        super().__init__()
        c_ = int(c * e)
        self.cv1 = ConvBNAct(c, c_, k[0], 1, use_act=True, act=act)
        self.cv2 = ConvBNAct(c_, c, k[1], 1, groups=g, use_act=True, act=act)
        self.add = shortcut and c == c_

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DWConv(nn.Module):
    """Depth-wise Convolution with Point-wise Convolution"""
    def __init__(self, c1, c2, k=3, s=1, act='silu'):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class LightBottleneck(nn.Module):
    """Lightweight bottleneck using DWConv - much fewer FLOPs"""
    def __init__(self, c, shortcut=True, e=0.5, act='silu'):
        super().__init__()
        c_ = int(c * e)
        self.cv1 = ConvBNAct(c, c_, 1, 1, use_act=True, act=act)
        self.cv2 = DWConv(c_, c, k=3, act=act)
        self.add = shortcut and c == c_

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f module - Cross Stage Partial with Faster implementation
    来自 YOLOv8，比C3更高效，梯度流更好
    
    特点:
    - Split输入为两部分
    - 多个Bottleneck并行处理
    - Concat输出，保留更多信息流
    - 对小目标检测效果显著提升
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', lightweight=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1, use_act=True, act=act)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1, use_act=True, act=act)
        if lightweight:
            self.m = nn.ModuleList([LightBottleneck(self.c, shortcut, e=1.0, act=act) for _ in range(n)])
        else:
            self.m = nn.ModuleList([Bottleneck(self.c, shortcut, g, k=(3, 3), e=1.0, act=act) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Alternative forward method for better gradient flow"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            # self.act = nn.ReLU()
            self.act = get_activation(act)
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            act=act,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act=act,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
            )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
            act='relu',
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act=act,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act=act,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
            act='relu',
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                act=act,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    act=act,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'Atto': {      # only 3 stages
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 256, 1, True, True, 3, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Femto': {      # only 3 stages
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Pico': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 act='relu',
                 use_simam_stages=None,
                 use_sppcspc_stage=-1,
                 use_c2f_stages=None,
                 c2f_n=1,
                 c2f_e=0.5,
                 c2f_lightweight=False,
                 custom_pretrained=None
                 ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        print(f"        ### Backbone.act: {act} ###     ")
        print(f"        ### Backbone.act: {act} ###     ")

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab,
                act=act)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    act=act)
            )

        self.use_simam_stages = use_simam_stages if use_simam_stages is not None else []
        self.use_sppcspc_stage = use_sppcspc_stage

        self.simam_modules = nn.ModuleList()
        if self.use_simam_stages:
            for stage_idx in self.use_simam_stages:
                if stage_idx < len(self.stages):
                    out_ch = stage_config[list(stage_config.keys())[stage_idx]][2]
                    self.simam_modules.append(SimAM())
                else:
                    self.simam_modules.append(nn.Identity())

        self.sppcspc_module = None
        if self.use_sppcspc_stage >= 0 and self.use_sppcspc_stage < len(self.stages):
            sppcspc_idx = self.use_sppcspc_stage
            out_ch = stage_config[list(stage_config.keys())[sppcspc_idx]][2]
            self.sppcspc_module = SPPCSPC(out_ch, act=act)

        self.c2f_modules = nn.ModuleList()
        self.use_c2f_stages = use_c2f_stages if use_c2f_stages is not None else []
        if self.use_c2f_stages and isinstance(self.use_c2f_stages, list):
            for stage_idx in self.use_c2f_stages:
                if stage_idx < len(self.stages):
                    out_ch = stage_config[list(stage_config.keys())[stage_idx]][2]
                    self.c2f_modules.append(C2f(out_ch, out_ch, n=c2f_n, e=c2f_e, act=act, lightweight=c2f_lightweight))
                else:
                    self.c2f_modules.append(nn.Identity())
        else:
            self.use_c2f_stages = []

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if custom_pretrained and os.path.exists(custom_pretrained):
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                state = torch.load(custom_pretrained, map_location='cpu')
                print(f"Loading custom pretrained weights from: {custom_pretrained}")
                
                if 'model' in state:
                    state = state['model']
                elif 'state_dict' in state:
                    state = state['state_dict']
                
                missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
                print(f"Loaded custom pretrained weights from: {custom_pretrained}")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                print(f"Failed to load custom pretrained weights: {e}")
                
        elif pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                if (name in ['Atto', 'Femto', 'Pico']):
                    model_path = local_model_dir + 'PPHGNetV2_' + 'B0' + '_stage1.pth'
                else:
                    model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                    if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        if is_distributed:
                            torch.distributed.barrier()
                    else:
                        if is_distributed:
                            torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                if ('Atto' == name):
                    self.load_partial_state_dict(self, state)
                elif ('Femto' == name) or ('Pico' == name):
                    missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
                    print("Missing keys:", missing_keys)
                    print("Unexpected keys:", unexpected_keys)
                else:
                    self.load_partial_state_dict(self, state)

            except (Exception, KeyboardInterrupt) as e:
                is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()

    @staticmethod
    def load_partial_state_dict(model, state_dict):
        model_dict = model.state_dict()
        # 只保留shape完全一致的参数
        filtered_dict = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}

        # 更新模型参数
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        missing = set(model_dict.keys()) - set(filtered_dict.keys())
        unexpected = set(state_dict.keys()) - set(filtered_dict.keys())
        print("Missing keys:", missing)
        print("   #########################################################")
        print("Unexpected keys:", unexpected)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        simam_idx = 0
        c2f_idx = 0
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_simam_stages and idx in self.use_simam_stages:
                if simam_idx < len(self.simam_modules):
                    x = self.simam_modules[simam_idx](x)
                simam_idx += 1
            if idx == self.use_sppcspc_stage and self.sppcspc_module is not None:
                x = self.sppcspc_module(x)
            if self.use_c2f_stages and idx in self.use_c2f_stages:
                if c2f_idx < len(self.c2f_modules):
                    x = self.c2f_modules[c2f_idx](x)
                c2f_idx += 1
            if idx in self.return_idx:
                outs.append(x)
        return outs
