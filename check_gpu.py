#!/usr/bin/env python3
"""
GPU 检测工具 - 检查 PyTorch 是否能识别 GPU
"""

import sys

print("=" * 60)
print("GPU 检测工具")
print("=" * 60)

# 检查 PyTorch
try:
    import torch
    print(f"\n✅ PyTorch 已安装: {torch.__version__}")
    
    # 检查 CUDA
    print(f"\n🔍 CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 版本: {torch.version.cuda}")
        print(f"✅ GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n📊 GPU {i}:")
            print(f"   名称: {torch.cuda.get_device_name(i)}")
            print(f"   显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("\n❌ CUDA 不可用！")
        print("\n可能的原因:")
        print("1. 没有安装 CUDA 版本的 PyTorch")
        print("2. NVIDIA 驱动未安装或版本过旧")
        print("3. 没有 NVIDIA GPU")
        
        print("\n解决方案:")
        print("1. 卸载当前 PyTorch:")
        print("   pip uninstall torch")
        print("\n2. 安装 CUDA 版本的 PyTorch:")
        print("   访问: https://pytorch.org/get-started/locally/")
        print("   或运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError:
    print("\n❌ PyTorch 未安装！")
    print("请运行: pip install torch")
    sys.exit(1)

# 检查 transformers
try:
    import transformers
    print(f"\n✅ Transformers 已安装: {transformers.__version__}")
except ImportError:
    print("\n❌ Transformers 未安装！")
    print("请运行: pip install transformers")

print("\n" + "=" * 60)

