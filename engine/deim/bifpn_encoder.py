"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
BiFPN Encoder with Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from ..core import register
from .utils import get_activation

__all__ = ['LiteEncoderBiFPN']


class WeightedFeatureFusion(nn.Module):
    """
    BiFPN加权特征融合
    Out = Σ(wi * Ini) / (Σwi + ε)
    """
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        self.weight = nn.Parameter(torch.ones(num_inputs), requires_grad=True)
        
    def forward(self, inputs):
        assert len(inputs) == self.num_inputs, f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        
        weight = F.relu(self.weight)
        weighted_sum = sum(w * inp for w, inp in zip(weight, inputs))
        return weighted_sum / (weight.sum() + self.eps)


class ChannelAttention(nn.Module):
    """
    通道注意力机制 (SE-Block风格)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA-Net)
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020)
    
    特点:
    - 比SE-Net更轻量（无降维）
    - 使用1D卷积捕获跨通道交互信息
    - 自适应卷积核大小
    - 参数量极少，效果优异
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        # 自适应计算卷积核大小
        # k = |log2(C)/gamma + b/gamma|odd
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        
        # 转换维度用于1D卷积: [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D卷积捕获跨通道交互
        y = self.conv(y)  # [B, 1, C]
        
        # 转回原始维度
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 通道注意力加权
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SeparableConv(nn.Module):
    """
    深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act='silu'):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    """
    单层BiFPN块，支持注意力机制
    支持2级或3级特征融合
    """
    def __init__(self, channels, act='silu', use_attention='cbam', attention_reduction=16, num_levels=2):
        super().__init__()
        self.use_attention = use_attention
        self.num_levels = num_levels
        
        self.conv_down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act)
        )
        
        if num_levels == 2:
            self.w1_p3 = WeightedFeatureFusion(2)
            self.w1_p4 = WeightedFeatureFusion(2)
            self.w2_p3 = WeightedFeatureFusion(3)
            
            self.fuse_p3 = SeparableConv(channels, channels, 3, 1, 1, act)
            self.fuse_p4 = SeparableConv(channels, channels, 3, 1, 1, act)
            
            if use_attention == 'cbam':
                self.attention_p3 = CBAM(channels, attention_reduction)
                self.attention_p4 = CBAM(channels, attention_reduction)
            elif use_attention == 'channel':
                self.attention_p3 = ChannelAttention(channels, attention_reduction)
                self.attention_p4 = ChannelAttention(channels, attention_reduction)
            elif use_attention == 'eca':
                self.attention_p3 = ECALayer(channels)
                self.attention_p4 = ECALayer(channels)
            elif use_attention == 'spatial':
                self.attention_p3 = SpatialAttention()
                self.attention_p4 = SpatialAttention()
            else:
                self.attention_p3 = nn.Identity()
                self.attention_p4 = nn.Identity()
        else:
            self.w1_p3 = WeightedFeatureFusion(2)
            self.w1_p4 = WeightedFeatureFusion(2)
            self.w1_p5 = WeightedFeatureFusion(2)
            self.w2_p3 = WeightedFeatureFusion(3)
            self.w2_p4 = WeightedFeatureFusion(3)
            
            self.fuse_p3 = SeparableConv(channels, channels, 3, 1, 1, act)
            self.fuse_p4 = SeparableConv(channels, channels, 3, 1, 1, act)
            self.fuse_p5 = SeparableConv(channels, channels, 3, 1, 1, act)
            
            if use_attention == 'cbam':
                self.attention_p3 = CBAM(channels, attention_reduction)
                self.attention_p4 = CBAM(channels, attention_reduction)
                self.attention_p5 = CBAM(channels, attention_reduction)
            elif use_attention == 'channel':
                self.attention_p3 = ChannelAttention(channels, attention_reduction)
                self.attention_p4 = ChannelAttention(channels, attention_reduction)
                self.attention_p5 = ChannelAttention(channels, attention_reduction)
            elif use_attention == 'eca':
                self.attention_p3 = ECALayer(channels)
                self.attention_p4 = ECALayer(channels)
                self.attention_p5 = ECALayer(channels)
            elif use_attention == 'spatial':
                self.attention_p3 = SpatialAttention()
                self.attention_p4 = SpatialAttention()
                self.attention_p5 = SpatialAttention()
            else:
                self.attention_p3 = nn.Identity()
                self.attention_p4 = nn.Identity()
                self.attention_p5 = nn.Identity()
    
    def forward_2level(self, p3, p4):
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3_fused = self.w1_p3([p3, p4_up])
        p3_out = self.fuse_p3(p3_fused)
        p3_out = self.attention_p3(p3_out)
        
        p3_down = self.conv_down(p3_out)
        p4_fused = self.w1_p4([p4, p3_down])
        p4_out = self.fuse_p4(p4_fused)
        p4_out = self.attention_p4(p4_out)
        
        p4_up_final = F.interpolate(p4_out, scale_factor=2, mode='nearest')
        p3_final = self.w2_p3([p3, p4_up, p4_up_final])
        p3_final = self.fuse_p3(p3_final)
        p3_final = self.attention_p3(p3_final)
        
        return p3_final, p4_out
    
    def forward_3level(self, p2, p3, p4):
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3_fused = self.w1_p4([p3, p4_up])
        p3_out = self.fuse_p4(p3_fused)
        p3_out = self.attention_p4(p3_out)
        
        p3_up = F.interpolate(p3_out, scale_factor=2, mode='nearest')
        p2_fused = self.w1_p3([p2, p3_up])
        p2_out = self.fuse_p3(p2_fused)
        p2_out = self.attention_p3(p2_out)
        
        p2_down = self.conv_down(p2_out)
        p3_fused2 = self.w2_p3([p3, p3_out, p2_down])
        p3_final = self.fuse_p3(p3_fused2)
        p3_final = self.attention_p3(p3_final)
        
        p3_down = self.conv_down(p3_final)
        p4_fused = self.w1_p5([p4, p3_down])
        p4_final = self.fuse_p5(p4_fused)
        p4_final = self.attention_p5(p4_final)
        
        return p2_out, p3_final, p4_final
    
    def forward(self, *feats):
        if self.num_levels == 2:
            return self.forward_2level(feats[0], feats[1])
        else:
            return self.forward_3level(feats[0], feats[1], feats[2])


@register()
class LiteEncoderBiFPN(nn.Module):
    """
    基于BiFPN的轻量级编码器
    支持多种注意力机制
    支持2级或3级输入，可配置输出级数
    支持2D位置编码
    """
    __share__ = ['eval_spatial_size']
    
    def __init__(self,
                 in_channels=[512],
                 feat_strides=[16],
                 hidden_dim=112,
                 expansion=0.34,
                 depth_mult=0.5,
                 act='silu',
                 eval_spatial_size=None,
                 use_attention='cbam',
                 attention_reduction=16,
                 use_global_fusion=True,
                 num_outputs=2,
                 output_strides=None,
                 use_pos_embed=False,
                 pos_embed_type='sincos',
                 pe_temperature=10000.):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size
        self.num_inputs = len(in_channels)
        self.num_outputs = num_outputs
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type
        self.pe_temperature = pe_temperature
        
        if output_strides is None:
            if self.num_inputs == 1:
                self.output_strides = [feat_strides[0], feat_strides[0] * 2]
            elif self.num_inputs == 3:
                self.output_strides = [feat_strides[1], feat_strides[2]]
            else:
                self.output_strides = feat_strides[-num_outputs:]
        else:
            self.output_strides = output_strides
        
        self.out_channels = [hidden_dim for _ in range(num_outputs)]
        self.out_strides = self.output_strides
        self.use_global_fusion = use_global_fusion
        
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))
            self.input_proj.append(proj)
        
        self.down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act)
        )
        
        if use_global_fusion:
            self.global_fusion = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                get_activation(act)
            )
        
        self.bifpn_block = BiFPNBlock(
            hidden_dim, 
            act=act, 
            use_attention=use_attention,
            attention_reduction=attention_reduction,
            num_levels=self.num_inputs if self.num_inputs > 1 else 2
        )
        
        if use_pos_embed and eval_spatial_size:
            self._init_pos_embeds(eval_spatial_size)
        
        self._initialize_weights()
    
    def _init_pos_embeds(self, spatial_size):
        for idx, stride in enumerate(self.output_strides):
            h, w = spatial_size[0] // stride, spatial_size[1] // stride
            pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature)
            self.register_buffer(f'pos_embed{idx}', pos_embed)
    
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :].permute(0, 2, 1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        if self.num_inputs == 1:
            proj_feats.append(self.down_sample(proj_feats[-1]))
            
            if self.use_global_fusion:
                global_feat = self.global_fusion(proj_feats[-1])
                proj_feats[-1] = proj_feats[-1] + global_feat
            
            p3, p4 = proj_feats[0], proj_feats[1]
            p3_out, p4_out = self.bifpn_block(p3, p4)
            outs = [p3_out, p4_out]
        
        elif self.num_inputs == 3:
            if self.use_global_fusion:
                global_feat = self.global_fusion(proj_feats[-1])
                proj_feats[-1] = proj_feats[-1] + global_feat
            
            p2, p3, p4 = proj_feats[0], proj_feats[1], proj_feats[2]
            p2_out, p3_out, p4_out = self.bifpn_block(p2, p3, p4)
            
            all_outs = {'p2': p2_out, 'p3': p3_out, 'p4': p4_out}
            
            stride_to_key = {8: 'p2', 16: 'p3', 32: 'p4'}
            selected_keys = [stride_to_key.get(s, 'p3') for s in self.output_strides]
            
            outs = [all_outs[k] for k in selected_keys]
        
        else:
            if self.use_global_fusion:
                global_feat = self.global_fusion(proj_feats[-1])
                proj_feats[-1] = proj_feats[-1] + global_feat
            
            outs = self.bifpn_block(*proj_feats)
            outs = list(outs[-self.num_outputs:])
        
        if self.use_pos_embed:
            outs = self._add_pos_embed(outs)
        
        return outs
    
    def _add_pos_embed(self, feats):
        outs = []
        for idx, feat in enumerate(feats):
            b, c, h, w = feat.shape
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(w, h, c, self.pe_temperature).to(feat.device)
            else:
                pos_embed = getattr(self, f'pos_embed{idx}', None)
                if pos_embed is not None:
                    pos_embed = pos_embed.to(feat.device)
                    if pos_embed.shape[-1] != h * w:
                        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, self.pe_temperature).to(feat.device)
                else:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, c, self.pe_temperature).to(feat.device)
            
            pos_embed = pos_embed.view(1, c, h, w)
            outs.append(feat + pos_embed)
        
        return outs


@register()
class LiteEncoderBiFPNMultiScale(nn.Module):
    """
    支持多尺度的BiFPN编码器
    """
    __share__ = ['eval_spatial_size']
    
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 num_bifpn_blocks=1,
                 act='silu',
                 eval_spatial_size=None,
                 use_attention='cbam',
                 attention_reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)
        
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(hidden_dim, act=act, use_attention=use_attention, 
                      attention_reduction=attention_reduction) 
            for _ in range(num_bifpn_blocks)
        ])
    
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        for bifpn_block in self.bifpn_blocks:
            proj_feats = bifpn_block(proj_feats)
        
        return proj_feats
