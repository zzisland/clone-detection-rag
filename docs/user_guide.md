# 用户使用指南

## 快速开始

### 环境要求
- Python 3.8+
- 2GB+ 内存
- 10GB+ 存储空间

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/zzisland/clone-detection-rag.git
cd clone-detection-rag
```

#### 2. 创建虚拟环境
```bash
# 使用conda
conda env create -f environment.yml
conda activate clone-rag

# 或使用pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### 3. 启动应用
```bash
# 快速启动
python run.py

# 或直接启动
streamlit run app.py
```

### 首次使用

#### 1. 环境检查
启动脚本会自动检查：
- Python版本
- 必要的包
- 配置文件

#### 2. 数据摄取
首次使用需要摄取数据：
1. 点击"重新摄取数据"按钮
2. 等待数据摄取完成
3. 确认数据统计信息

#### 3. 开始问答
在聊天界面输入问题，例如：
- "什么是代码克隆检测？"
- "比较NiCad和CCFinder工具"
- "如何检测Type-3克隆？"

## 功能使用

### 智能问答

#### 基础概念查询
**适用问题**: 概念解释、定义说明
**示例**:
- "什么是Type-1克隆？"
- "代码克隆检测的重要性是什么？"
- "AST和Token方法的区别"

> **详细概念说明**: 详见 [克隆检测基础](data/papers/clone_detection_basics.txt)

#### 技术细节查询
**适用问题**: 技术原理、实现方法
**示例**:
- "如何评估克隆检测工具的准确性？"
- "NiCad工具的配置参数有哪些？"
- "向量检索的原理是什么？"

#### 工具比较查询
**适用问题**: 工具对比、选择建议
**示例**:
- "CCFinder和SourcererCC有什么区别？"
- "哪个工具适合大规模代码检测？"
- "开源工具和商业工具的对比"

### 代码分析

#### 代码片段分析
**使用方法**: 在对话框中粘贴代码片段
**示例**:
```python
def calculate_sum(a, b):
    return a + b
```

**分析内容**:
- 代码结构特征
- 可能的克隆类型
- 检测建议

> **克隆类型定义**: 详见 [克隆检测基础](data/papers/clone_detection_basics.txt)

#### 克隆类型判断
**系统会自动分析**:
- 代码复杂度
- 结构相似性
- 功能等价性

### 检索模式

#### 通用检索
**特点**: 基于语义相似度的全局检索
**适用**: 一般性问题咨询
**使用**: 默认模式，无需特殊设置

#### 分类检索
**特点**: 按文档类型过滤检索
**适用**: 特定类型文档查询
**选项**:
- 论文文档
- 工具文档
- 项目文档
- 示例代码

#### 混合检索
**特点**: 结合多种策略提高召回率
**适用**: 复杂问题解答
**使用**: 系统自动选择最佳策略

## 高级功能

### 配置管理

#### 环境变量配置
编辑 `.env` 文件：
```bash
# 向量数据库路径
CHROMA_PERSIST_DIRECTORY=./data/chroma

# 文档处理参数
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 检索参数
TOP_K_RETRIEVAL=5
```

#### 模型配置
支持不同的Qwen模型：
- Qwen2.5-Coder-1.5B-Instruct (默认)
- Qwen2.5-Coder-7B-Instruct (更强性能)

### 数据管理

#### 数据摄取
```bash
# 重新摄取所有数据
python -c "from src.ingest import DataIngestor; DataIngestor().ingest_all_data()"

# 摄取特定类型数据
python -c "from src.ingest import DataIngestor; DataIngestor().ingest_data_type('papers')"
```

#### 数据统计
查看系统状态面板了解：
- 文档数量
- 向量数量
- 数据来源分布

#### 数据清理
```bash
# 清理向量数据库
rm -rf data/chroma/

# 重新摄取
python run.py
```

### 自定义数据

#### 添加新文档
1. 将文档放入对应目录：
   - `data/papers/` - 论文文档
   - `data/tools_docs/` - 工具文档
   - `data/project_docs/` - 项目文档
   - `data/examples/` - 示例代码

2. 重新摄取数据：
   ```bash
   python run.py
   ```

#### 支持的文件格式
- **文本**: .txt, .md
- **文档**: .pdf
- **网页**: .html
- **代码**: .py, .java, .cpp, .c, .js

## 故障排除

### 常见问题

#### 1. 启动失败
**问题**: 应用无法启动
**解决方案**:
```bash
# 检查Python版本
python --version

# 检查依赖包
pip list

# 重新安装依赖
pip install -r requirements.txt
```

#### 2. 数据摄取失败
**问题**: 数据摄取过程中断
**解决方案**:
```bash
# 检查数据目录权限
ls -la data/

# 重新摄取
python run.py
```

#### 3. 模型加载失败
**问题**: Qwen模型无法加载
**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 手动下载模型
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"
```

#### 4. 检索结果不准确
**问题**: 检索结果与问题不相关
**解决方案**:
- 调整检索参数
- 增加相关文档
- 重新摄取数据

#### 5. 响应速度慢
**问题**: 系统响应时间过长
**解决方案**:
- 减少TOP_K_RETRIEVAL值
- 使用更小的模型
- 优化数据分块大小

### 错误代码

#### ENV001: 环境变量缺失
**症状**: 启动时提示配置错误
**解决**: 复制 `.env.example` 到 `.env`

#### DATA001: 数据目录不存在
**症状**: 找不到数据文件
**解决**: 确保数据目录结构正确

#### MODEL001: 模型加载失败
**症状**: 无法加载Qwen模型
**解决**: 检查网络连接和模型路径

#### RETRIEVE001: 检索失败
**症状**: 无法检索相关文档
**解决**: 重新摄取数据

## 性能优化

### 系统优化

#### 1. 内存优化
```bash
# 减少分块大小
CHUNK_SIZE=500

# 减少检索数量
TOP_K_RETRIEVAL=3
```

#### 2. 速度优化
```bash
# 使用更小的模型
# 修改 src/rag.py 中的模型名称
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

#### 3. 存储优化
```bash
# 定期清理向量数据库
rm -rf data/chroma/*

# 压缩数据文件
tar -czf data_backup.tar.gz data/
```

### 使用技巧

#### 1. 提问技巧
- **具体化**: 使用具体的问题描述
- **上下文**: 提供相关的背景信息
- **分步骤**: 复杂问题分步提问

#### 2. 代码分析技巧
- **完整代码**: 提供完整的代码片段
- **注释说明**: 添加必要的注释
- **格式规范**: 使用规范的代码格式

#### 3. 检索技巧
- **关键词**: 使用准确的技术关键词
- **同义词**: 尝试不同的表达方式
- **分类检索**: 使用特定的文档类型

## 扩展使用

### API使用

#### RESTful API
```python
import requests

# 发送查询请求
response = requests.post('http://localhost:8501/api/query', json={
    'question': '什么是代码克隆检测？',
    'mode': 'general'
})

result = response.json()
print(result['answer'])
```

#### Python API
```python
from src.rag import CloneDetectionRAG

# 初始化RAG系统
rag = CloneDetectionRAG()

# 查询问题
result = rag.query('什么是代码克隆检测？')
print(result['answer'])
```

### 集成开发

#### IDE插件
- **VS Code**: 开发插件支持
- **IntelliJ**: 开发插件支持
- **Eclipse**: 开发插件支持

#### CI/CD集成
```yaml
# GitHub Actions示例
- name: Run Clone Detection
  run: |
    python scripts/analyze_code.py --path ./src
```

## 最佳实践

### 1. 日常使用
- **定期更新**: 保持数据和模型更新
- **备份配置**: 定期备份配置文件
- **监控性能**: 关注系统性能指标

### 2. 团队协作
- **统一配置**: 团队使用相同的配置
- **共享数据**: 共享有价值的文档
- **版本控制**: 使用版本控制管理配置

### 3. 学习提升
- **阅读文档**: 详细阅读技术文档
- **实践应用**: 在实际项目中应用
- **反馈改进**: 提供使用反馈和建议

---

**文档版本**: v1.0  
**更新日期**: 2026年1月11日  
**维护者**: 项目开发团队

**获取帮助**:
- GitHub Issues: https://github.com/zzisland/clone-detection-rag/issues
- 邮件支持: 19810794281@163.com
