"""
DEIMv2 验证/评估脚本
用于在验证集上评估模型性能，支持显示每个类别的精度
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import time
import torch
import numpy as np

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS


def get_model_params_trainable(model):
    """获取模型可训练参数量，正确处理各种包装"""
    # 使用de_parallel解除DDP/DP包装
    model = dist_utils.de_parallel(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_shape=(1, 3, 640, 640)):
    """计算模型FLOPs"""
    try:
        from calflops import calculate_flops as cal_flops
        flops, _, _ = cal_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=True,
            output_precision=4,
            print_detailed=False
        )
        return flops
    except ImportError:
        print("  警告: calflops 未安装，跳过FLOPs计算")
        return None


def calculate_model_params(model):
    """计算模型参数量"""
    # 先解除DDP包装获取原始模型
    if hasattr(model, 'module'):
        model = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters, n_trainable


def benchmark_fps(model, device, input_shape=(1, 3, 640, 640), num_warmup=10, num_runs=100, fps_device='cpu'):
    """测试模型FPS
    
    Args:
        model: 模型
        device: 模型所在设备
        input_shape: 输入形状
        num_warmup: 预热次数
        num_runs: 测试次数
        fps_device: FPS测试设备 ('cpu' 或 'cuda')
    """
    model.eval()
    
    if fps_device == 'cpu':
        test_device = torch.device('cpu')
        model_cpu = model.to(test_device)
        dummy_input = torch.randn(input_shape).to(test_device)
    else:
        test_device = device
        dummy_input = torch.randn(input_shape).to(device)
        model_cpu = model
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(dummy_input)
        if test_device.type == 'cuda':
            torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_cpu(dummy_input)
            if test_device.type == 'cuda':
                torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    fps = num_runs / elapsed_time
    avg_latency = elapsed_time / num_runs * 1000
    
    return fps, avg_latency


def benchmark_onnx_fps(onnx_path, input_shape=(1, 3, 640, 640), num_warmup=10, num_runs=100, device='cpu'):
    """测试ONNX模型FPS
    
    Args:
        onnx_path: ONNX模型路径
        input_shape: 输入形状
        num_warmup: 预热次数
        num_runs: 测试次数
        device: 测试设备 ('cpu' 或 'cuda')
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  警告: onnxruntime 未安装，跳过ONNX FPS测试")
        return None, None
    
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    orig_size = np.array([[input_shape[2], input_shape[3]]], dtype=np.int64)
    
    for _ in range(num_warmup):
        _ = session.run(None, {'images': dummy_input, 'orig_target_sizes': orig_size})
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, {'images': dummy_input, 'orig_target_sizes': orig_size})
    
    elapsed_time = time.perf_counter() - start_time
    fps = num_runs / elapsed_time
    avg_latency = elapsed_time / num_runs * 1000
    
    return fps, avg_latency


def export_onnx(cfg, output_path, input_shape=(1, 3, 640, 640), device='cpu'):
    """导出模型为ONNX格式
    
    Args:
        cfg: YAMLConfig配置对象
        output_path: 输出路径
        input_shape: 输入形状
        device: 设备
    """
    import torch.nn as nn
    
    class Model(nn.Module):
        def __init__(self, cfg) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model(cfg)
    model.eval()
    
    if device == 'cpu':
        model = model.to('cpu')
        dummy_input = torch.randn(input_shape).to('cpu')
    else:
        dummy_input = torch.randn(input_shape).to(device)
    
    orig_size = torch.tensor([[input_shape[2], input_shape[3]]])
    
    torch.onnx.export(
        model,
        (dummy_input, orig_size),
        output_path,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes={
            'images': {0: 'batch'},
            'orig_target_sizes': {0: 'batch'},
            'labels': {0: 'batch', 1: 'num_detections'},
            'boxes': {0: 'batch', 1: 'num_detections'},
            'scores': {0: 'batch', 1: 'num_detections'},
        },
        opset_version=16,
        do_constant_folding=True,
    )
    print(f"  ONNX模型已保存到: {output_path}")
    return output_path


def print_per_class_ap(coco_eval, class_names=None):
    """打印每个类别的 AP，包括 IoU=0.50:0.95 和 IoU=0.50"""
    if coco_eval is None:
        return None, None
    
    # 获取 COCO 评估结果
    coco_eval_obj = coco_eval.coco_eval.get('bbox', None)
    if coco_eval_obj is None:
        return None, None
    
    # 获取每个类别的详细评估结果
    precisions = coco_eval_obj.eval['precision']  # shape: (T, R, K, A, M)
    # T: iou thresholds, R: recall thresholds, K: categories, A: area ranges, M: max detections
    
    # 获取 IoU 阈值索引
    iou_thrs = coco_eval_obj.params.iouThrs  # IoU 阈值列表
    
    # 找到 IoU=0.5 的索引
    iou_50_idx = np.where(np.abs(iou_thrs - 0.5) < 1e-6)[0]
    if len(iou_50_idx) == 0:
        iou_50_idx = np.argmin(np.abs(iou_thrs - 0.5))
    else:
        iou_50_idx = iou_50_idx[0]
    
    num_classes = precisions.shape[2]
    
    # COCO precision数组: (T, R, K, A, M)
    # A=0: all areas, M=2: maxDets=100 (这是COCO计算AP的标准设置)
    # 计算每个类别的 AP (IoU=0.50:0.95) - 对所有IoU阈值取平均
    ap_per_class = []
    for i in range(num_classes):
        # 只使用 A=0 (all areas) 和 M=2 (maxDets=100)
        precision = precisions[:, :, i, 0, 2]
        precision = precision[precision > -1]
        if len(precision) > 0:
            ap = np.mean(precision)
        else:
            ap = 0.0
        ap_per_class.append(ap)
    
    # 计算每个类别的 AP50 (IoU=0.50) - 只取IoU=0.5时的AP
    ap50_per_class = []
    for i in range(num_classes):
        # 只使用 A=0 (all areas) 和 M=2 (maxDets=100)
        precision = precisions[iou_50_idx, :, i, 0, 2]
        precision = precision[precision > -1]
        if len(precision) > 0:
            ap50 = np.mean(precision)
        else:
            ap50 = 0.0
        ap50_per_class.append(ap50)
    
    # 获取类别名称
    if class_names is None:
        cats = coco_eval_obj.cocoGt.loadCats(coco_eval_obj.cocoGt.getCatIds())
        class_names = [cat['name'] for cat in cats]
    
    # 打印每个类别的 AP (IoU=0.50:0.95)
    print('\n' + '=' * 80)
    print('每个类别的 AP (IoU=0.50:0.95):')
    print('-' * 80)
    for i, (ap, name) in enumerate(zip(ap_per_class, class_names)):
        print(f"  类别 {i}: {name:30s} AP = {ap:.4f}")
    print('-' * 80)
    print(f"  平均 AP (mAP): {np.mean(ap_per_class):.4f}")
    print('=' * 80)
    
    # 打印每个类别的 AP50 (IoU=0.50)
    print('\n' + '=' * 80)
    print('每个类别的 AP50 (IoU=0.50):')
    print('-' * 80)
    for i, (ap50, name) in enumerate(zip(ap50_per_class, class_names)):
        print(f"  类别 {i}: {name:30s} AP50 = {ap50:.4f}")
    print('-' * 80)
    print(f"  平均 AP50 (mAP@50): {np.mean(ap50_per_class):.4f}")
    print('=' * 80)
    
    return ap_per_class, ap50_per_class


def main(args) -> None:
    """主函数"""
    # 设置分布式环境（单卡模式下也会初始化）
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # 解析命令行参数更新
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    # 加载配置
    cfg = YAMLConfig(args.config, **update_dict)
    
    # 如果指定了设备，覆盖配置中的设备设置
    if args.device:
        if args.device == 'cpu':
            cfg.device = 'cpu'
            cfg.yaml_cfg['device'] = 'cpu'
        elif args.device.startswith('cuda'):
            cfg.device = args.device
            cfg.yaml_cfg['device'] = args.device
    else:
        # 自动检测：如果CUDA不可用或内存不足，使用CPU
        if not torch.cuda.is_available():
            print("  CUDA不可用，使用CPU进行验证")
            cfg.device = 'cpu'
            cfg.yaml_cfg['device'] = 'cpu'
    
    # 如果指定了检查点，关闭预训练权重加载
    if args.resume:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('=' * 80)
    print('配置信息:')
    print(f"  配置文件: {args.config}")
    print(f"  检查点: {args.resume if args.resume else 'None'}")
    print(f"  验证数据: {cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {}).get('img_folder', 'N/A')}")
    print(f"  类别数: {cfg.yaml_cfg.get('num_classes', 'N/A')}")
    print(f"  设备: {cfg.device if cfg.device else 'auto'}")
    print('=' * 80)

    # 创建 solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # 运行验证，捕获CUDA OOM错误并自动回退到CPU
    print('\n开始验证...')
    try:
        stats = solver.val()
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) and cfg.device != 'cpu':
            print("\n  CUDA内存不足，自动切换到CPU进行验证...")
            torch.cuda.empty_cache()
            cfg.device = 'cpu'
            cfg.yaml_cfg['device'] = 'cpu'
            # 重新创建solver
            solver = TASKS[cfg.yaml_cfg['task']](cfg)
            stats = solver.val()
        else:
            raise
    
    # 确保 stats 是字典
    if stats is None:
        stats = {}

    # 计算模型参数量和FLOPs
    print('\n计算模型信息...')
    # 使用EMA模型进行推理测试，但参数量从原始模型获取（EMA的params.requires_grad=False）
    model_for_info = solver.ema.module if solver.ema else solver.model
    model_for_info.eval()
    # 获取输入尺寸
    base_size = cfg.train_dataloader.collate_fn.base_size if hasattr(cfg.train_dataloader, 'collate_fn') else 640
    input_shape = (1, 3, base_size, base_size)

    # 参数量 - 从原始模型获取可训练参数
    n_params, _ = calculate_model_params(model_for_info)
    # 可训练参数从原始模型获取（EMA模型的参数都是requires_grad=False）
    
    # 调试：检查solver.model的结构
    print(f"  [调试] solver.model类型: {type(solver.model)}")
    if hasattr(solver.model, 'module'):
        print(f"  [调试] solver.model.module类型: {type(solver.model.module)}")
        # 检查是否有requires_grad=True的参数
        trainable_count = sum(1 for p in solver.model.module.parameters() if p.requires_grad)
        print(f"  [调试] solver.model.module中requires_grad=True的参数数量: {trainable_count}")
    
    n_trainable = get_model_params_trainable(solver.model)
    print(f"  总参数量: {n_params:,} ({n_params / 1e6:.4f}M)")
    print(f"  可训练参数量: {n_trainable:,} ({n_trainable / 1e6:.4f}M)")

    # FLOPs
    flops = calculate_flops(model_for_info, input_shape=input_shape)
    if flops is not None:
        print(f"  FLOPs: {flops}")

    # 测试 FPS
    print('\n测试模型推理速度...')
    device = solver.device
    fps_device = args.fps_device
    use_onnx = args.use_onnx
    
    # PyTorch FPS测试
    print(f"\n  PyTorch FPS测试 (设备: {fps_device}):")
    fps, latency = benchmark_fps(model_for_info, device, input_shape=input_shape, fps_device=fps_device)
    print(f"    FPS: {fps:.2f}")
    print(f"    平均延迟: {latency:.2f} ms")
    
    # ONNX FPS测试
    onnx_fps = None
    onnx_latency = None
    if use_onnx:
        print(f"\n  ONNX FPS测试 (设备: {fps_device}):")
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            export_onnx(cfg, onnx_path, input_shape=input_shape, device=fps_device)
            onnx_fps, onnx_latency = benchmark_onnx_fps(onnx_path, input_shape=input_shape, device=fps_device)
            if onnx_fps is not None:
                print(f"    FPS: {onnx_fps:.2f}")
                print(f"    平均延迟: {onnx_latency:.2f} ms")
        except Exception as e:
            print(f"    ONNX导出/测试失败: {e}")
    
    # 打印验证结果
    print('\n' + '=' * 80)
    print('验证结果汇总:')
    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, (list, np.ndarray)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {stats}")
    print('=' * 80)
    
    # 获取每个类别的精度
    if hasattr(solver, 'evaluator') and solver.evaluator is not None:
        # 获取类别名称
        val_dataset = solver.val_dataloader.dataset
        if hasattr(val_dataset, 'categories'):
            class_names = [cat['name'] for cat in val_dataset.categories]
        else:
            class_names = None
        
        # 打印每个类别的 AP
        ap_per_class, ap50_per_class = print_per_class_ap(solver.evaluator, class_names)
        
        # 将每个类别的 AP 添加到 stats
        if ap_per_class is not None:
            stats['ap_per_class'] = ap_per_class
            stats['ap50_per_class'] = ap50_per_class
            stats['map'] = np.mean(ap_per_class)
            stats['map50'] = np.mean(ap50_per_class)
            if class_names:
                stats['class_names'] = class_names
    
    # 添加参数量、FLOPs 和 FPS 到 stats
    stats['params_total'] = n_params
    stats['params_trainable'] = n_trainable
    if flops is not None:
        stats['flops'] = flops
    stats['fps'] = fps
    stats['latency_ms'] = latency
    stats['fps_device'] = fps_device
    if onnx_fps is not None:
        stats['onnx_fps'] = onnx_fps
        stats['onnx_latency_ms'] = onnx_latency
    stats['input_size'] = base_size
    
    # 保存结果到文件
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, 'val_results.json')
        
        # 转换 numpy 类型为 Python 原生类型以便 JSON 序列化
        stats_serializable = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                stats_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                stats_serializable[key] = float(value)
            else:
                stats_serializable[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
        print(f'\n结果已保存到: {output_file}')

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2 验证脚本')

    # 必需参数
    parser.add_argument('-c', '--config', type=str, 
                        default='./configs/deimv2/deimv2_hgnetv2_n_pest.yml',
                        help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='模型检查点路径（必需）')
    
    # 可选参数
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='设备 (cuda/cpu)，默认cpu')
    parser.add_argument('--fps-device', type=str, default='cpu',
                        help='FPS测试设备 (cpu/cuda)，默认cpu')
    parser.add_argument('--use-onnx', action='store_true',
                        help='使用ONNX进行FPS测试')
    parser.add_argument('--seed', type=int, default=0, 
                        help='随机种子')
    parser.add_argument('--output-dir', type=str, 
                        default='./outputs/val_results',
                        help='结果输出目录')
    
    # 分布式相关
    parser.add_argument('--print-method', type=str, default='builtin', 
                        help='打印方法')
    parser.add_argument('--print-rank', type=int, default=0, 
                        help='打印rank ID')
    parser.add_argument('--local-rank', type=int, 
                        help='本地rank ID')
    
    # 配置更新
    parser.add_argument('-u', '--update', nargs='+', 
                        help='更新YAML配置，例如: val_dataloader.dataset.img_folder=/new/path')

    args = parser.parse_args()
    
    main(args)
