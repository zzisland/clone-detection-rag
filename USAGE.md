# 代码克隆检测 RAG 助手 - 使用指南

## 📋 完整使用步骤

### 第一步：环境检查

1. **检查 GPU 是否可用**
```bash
python check_gpu.py
```

如果显示 "CUDA 不可用"，需要安装 CUDA 版本的 PyTorch：
```bash
# 卸载当前版本
pip uninstall torch

# 安装 CUDA 版本（根据您的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 第二步：启动应用

```bash
cd F:\clone-detection-rag
streamlit run app.py
```

### 第三步：选择模型

应用启动后，您会看到两个模型选项：

#### 选项 A：Qwen2.5-Coder-1.5B（推荐新手）
- ✅ 轻量级，加载快（约10秒）
- ✅ 显存需求低（约3GB）
- ✅ 适合大多数用户
- ⚠️ 性能中等

**点击"选择 1.5B 模型"按钮**

#### 选项 B：Qwen2.5-Coder-7B（推荐专业用户）
- ✅ 高性能，回答质量好
- ✅ 理解能力强
- ⚠️ 显存需求高（约14GB）
- ⚠️ 加载较慢（约30秒）

**点击"选择 7B 模型"按钮**

### 第四步：等待模型加载

首次使用会自动下载模型：
- 1.5B 模型：约 3GB，预计 5-10 分钟
- 7B 模型：约 14GB，预计 20-40 分钟

您会看到进度提示：
```
⏳ [1/4] 初始化系统...
⏳ [2/4] 加载 Tokenizer...
⏳ [3/4] 加载 LLM 模型...
✅ [4/4] 系统加载完成！
```

### 第五步：摄取数据

1. 在左侧边栏找到"📚 数据管理"
2. 点击"重新摄取数据"按钮
3. 等待数据处理完成（约1-2分钟）
4. 看到"✅ 数据已摄取"提示

### 第六步：开始提问

#### 方式 1：直接输入
1. 在底部输入框输入问题
2. 按回车键或点击"发送 ✉️"按钮
3. 等待 AI 回答

#### 方式 2：使用示例问题
1. 在左侧边栏"🚀 快速操作"
2. 从下拉框选择示例问题
3. 点击"使用示例问题"按钮
4. 自动发送并获得回答

### 第七步：查看回答

回答会显示：
- 📝 详细的回答内容
- 📚 参考来源（可展开查看）
- 🎯 置信度（high/medium/low）

## 🔧 常见问题解决

### 问题 1：显示使用 CPU 而不是 GPU

**原因**：PyTorch 未正确安装 CUDA 支持

**解决**：
```bash
# 1. 检查 GPU
python check_gpu.py

# 2. 如果显示 CUDA 不可用，重新安装 PyTorch
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2：下载速度很慢

**解决**：
```bash
# 启动前设置镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
streamlit run app.py
```

### 问题 3：显存不足

**解决**：
- 选择 1.5B 模型而不是 7B
- 或关闭其他占用显存的程序

### 问题 4：向量数据库维度不匹配

**解决**：
```bash
# 删除旧的向量数据库
Remove-Item -Recurse -Force .\data\chroma

# 重新启动应用并摄取数据
```

## 📊 系统要求

### 最低配置（1.5B 模型）
- CPU: 4核以上
- 内存: 8GB
- 显存: 3GB（GPU）或使用 CPU
- 硬盘: 10GB 可用空间

### 推荐配置（7B 模型）
- CPU: 8核以上
- 内存: 16GB
- 显存: 16GB（GPU）
- 硬盘: 20GB 可用空间

## 🎯 使用技巧

### 提问技巧
1. **概念询问**：什么是 Type-1 克隆？
2. **工具比较**：比较 NiCad 和 CCFinder
3. **方法解释**：AST 方法的原理是什么？
4. **代码分析**：分析这段代码的克隆特征

### 获得更好回答的方法
- 问题尽量具体明确
- 可以分步骤提问
- 利用参考来源深入了解
- 使用示例问题作为参考

## 📞 获取帮助

如果遇到问题：
1. 查看本文档的常见问题部分
2. 运行 `python check_gpu.py` 检查环境
3. 运行 `python clear_cache.py` 管理缓存
4. 查看终端输出的错误信息

---

**提示**：首次使用建议选择 1.5B 模型，熟悉后再尝试 7B 模型。

