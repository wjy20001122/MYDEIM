"""
测试BiFPN编码器效果
对比LiteEncoder和LiteEncoderBiFPN的性能
"""

import torch
import torch.nn as nn
import time
import argparse
from thop import profile, clever_format

from engine.deim.lite_encoder import LiteEncoder
from engine.deim.bifpn_encoder import LiteEncoderBiFPN


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_tensor, num_runs=100, warmup=10):
    """测量推理时间"""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000
    return avg_time


def test_single_scale():
    """测试单尺度BiFPN"""
    print("=" * 80)
    print("测试单尺度BiFPN (Pico配置)")
    print("=" * 80)
    
    batch_size = 2
    input_tensor = [torch.randn(batch_size, 512, 40, 40).cuda()]
    
    print("\n1. LiteEncoder (原始PAN-FPN)")
    print("-" * 80)
    lite_encoder = LiteEncoder(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        expansion=0.34,
        depth_mult=0.5,
        act='silu'
    ).cuda()
    
    lite_params = count_parameters(lite_encoder)
    print(f"参数量: {lite_params:,}")
    
    flops, params = profile(lite_encoder, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    lite_time = measure_inference_time(lite_encoder, input_tensor)
    print(f"平均推理时间: {lite_time:.3f} ms")
    
    with torch.no_grad():
        lite_output = lite_encoder(input_tensor)
    print(f"输出形状: {[o.shape for o in lite_output]}")
    
    print("\n2. LiteEncoderBiFPN (带CBAM注意力)")
    print("-" * 80)
    bifpn_encoder = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        expansion=0.34,
        depth_mult=0.5,
        act='silu',
        use_attention='cbam',
        attention_reduction=16,
        use_global_fusion=True
    ).cuda()
    
    bifpn_params = count_parameters(bifpn_encoder)
    print(f"参数量: {bifpn_params:,}")
    
    flops, params = profile(bifpn_encoder, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    bifpn_time = measure_inference_time(bifpn_encoder, input_tensor)
    print(f"平均推理时间: {bifpn_time:.3f} ms")
    
    with torch.no_grad():
        bifpn_output = bifpn_encoder(input_tensor)
    print(f"输出形状: {[o.shape for o in bifpn_output]}")
    
    print("\n3. LiteEncoderBiFPN (带通道注意力)")
    print("-" * 80)
    bifpn_channel = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        use_attention='channel',
        attention_reduction=16
    ).cuda()
    
    channel_params = count_parameters(bifpn_channel)
    print(f"参数量: {channel_params:,}")
    
    flops, params = profile(bifpn_channel, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    channel_time = measure_inference_time(bifpn_channel, input_tensor)
    print(f"平均推理时间: {channel_time:.3f} ms")
    
    print("\n4. LiteEncoderBiFPN (带ECA注意力)")
    print("-" * 80)
    bifpn_eca = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        use_attention='eca'
    ).cuda()
    
    eca_params = count_parameters(bifpn_eca)
    print(f"参数量: {eca_params:,}")
    
    flops, params = profile(bifpn_eca, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    eca_time = measure_inference_time(bifpn_eca, input_tensor)
    print(f"平均推理时间: {eca_time:.3f} ms")
    
    with torch.no_grad():
        eca_output = bifpn_eca(input_tensor)
    print(f"输出形状: {[o.shape for o in eca_output]}")
    
    print("\n5. LiteEncoderBiFPN (无注意力)")
    print("-" * 80)
    bifpn_no_attn = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        use_attention='none'
    ).cuda()
    
    no_attn_params = count_parameters(bifpn_no_attn)
    print(f"参数量: {no_attn_params:,}")
    
    flops, params = profile(bifpn_no_attn, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
    
    no_attn_time = measure_inference_time(bifpn_no_attn, input_tensor)
    print(f"平均推理时间: {no_attn_time:.3f} ms")
    
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print(f"{'模型':<30} {'参数量':>12} {'推理时间(ms)':>15} {'相对速度':>12}")
    print("-" * 80)
    print(f"{'LiteEncoder (PAN-FPN)':<30} {lite_params:>12,} {lite_time:>15.3f} {'1.00x':>12}")
    print(f"{'BiFPN + CBAM':<30} {bifpn_params:>12,} {bifpn_time:>15.3f} {lite_time/bifpn_time:>11.2f}x")
    print(f"{'BiFPN + Channel':<30} {channel_params:>12,} {channel_time:>15.3f} {lite_time/channel_time:>11.2f}x")
    print(f"{'BiFPN + ECA':<30} {eca_params:>12,} {eca_time:>15.3f} {lite_time/eca_time:>11.2f}x")
    print(f"{'BiFPN (无注意力)':<30} {no_attn_params:>12,} {no_attn_time:>15.3f} {lite_time/no_attn_time:>11.2f}x")
    print("=" * 80)
    
    print("\n参数量变化:")
    print(f"  BiFPN + CBAM:     +{bifpn_params - lite_params:,} ({(bifpn_params/lite_params-1)*100:+.2f}%)")
    print(f"  BiFPN + Channel:  +{channel_params - lite_params:,} ({(channel_params/lite_params-1)*100:+.2f}%)")
    print(f"  BiFPN + ECA:      +{eca_params - lite_params:,} ({(eca_params/lite_params-1)*100:+.2f}%)")
    print(f"  BiFPN (无注意力): +{no_attn_params - lite_params:,} ({(no_attn_params/lite_params-1)*100:+.2f}%)")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 80)
    print("测试梯度流")
    print("=" * 80)
    
    batch_size = 2
    input_tensor = [torch.randn(batch_size, 512, 40, 40, requires_grad=True).cuda()]
    
    bifpn_encoder = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        use_attention='cbam'
    ).cuda()
    
    output = bifpn_encoder(input_tensor)
    loss = sum(o.sum() for o in output)
    loss.backward()
    
    print(f"输入梯度存在: {input_tensor[0].grad is not None}")
    print(f"输入梯度范数: {input_tensor[0].grad.norm().item():.6f}")
    
    for name, param in bifpn_encoder.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm().item():.6f}")


def test_weighted_fusion():
    """测试加权融合权重"""
    print("\n" + "=" * 80)
    print("测试加权融合权重")
    print("=" * 80)
    
    bifpn_encoder = LiteEncoderBiFPN(
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=112,
        use_attention='cbam'
    ).cuda()
    
    print("\nBiFPN加权融合权重:")
    for name, param in bifpn_encoder.named_parameters():
        if 'weight' in name and 'bifpn' in name.lower():
            print(f"{name}: {param.data}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试BiFPN编码器')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'single', 'gradient', 'weight'],
                       help='测试模式')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU测试（速度会较慢）")
    
    if args.mode in ['all', 'single']:
        test_single_scale()
    
    if args.mode in ['all', 'gradient']:
        test_gradient_flow()
    
    if args.mode in ['all', 'weight']:
        test_weighted_fusion()
    
    print("\n测试完成！")
