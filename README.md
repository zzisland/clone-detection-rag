# 代码克隆检测RAG助手

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于 RAG（Retrieval-Augmented Generation）技术的代码克隆检测知识助手

## 📖 项目简介

本项目实现了一个专业的代码克隆检测知识问答系统，通过 RAG 技术结合大语言模型，能够：

- 🤖 回答代码克隆检测相关的专业问题
- 📊 分析代码片段的克隆特征
- 🔧 比较不同的克隆检测工具
- 💡 解释克隆检测的核心概念
- 📚 提供基于文档的专业建议

## ✨ 主要特性

- **智能问答**：基于检索增强生成，提供准确的专业回答
- **多模式交互**：支持问答、代码分析、工具比较等多种模式
- **专业知识库**：包含论文、工具文档、项目文档和示例代码
- **友好界面**：基于 Streamlit 的现代化 Web UI
- **灵活检索**：支持通用检索、分类检索和混合检索策略

## 🎬 演示

### 界面截图

![主界面](docs/images/main-interface.png)
*主界面 - 对话交互*

![系统状态](docs/images/system-status.png)
*系统状态 - 数据管理*

### 功能演示

```
用户：什么是代码克隆检测？

助手：代码克隆检测是指在软件系统中识别相似或重复代码片段的过程。
主要分为以下几种类型：

1. Type-1 克隆：完全相同的代码片段（除了空格和注释）
2. Type-2 克隆：语法相同但标识符不同的代码片段
3. Type-3 克隆：语法相似但有部分语句修改的代码片段
4. Type-4 克隆：功能相同但实现方式不同的代码片段

常用的检测方法包括：
- 基于文本的方法
- 基于 Token 的方法
- 基于 AST 的方法
- 基于 PDG 的方法
...
```

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- OpenAI API Key

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/你的用户名/Simple-Rag.git
cd Simple-Rag
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

创建 `.env` 文件并添加以下内容：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
```

4. **运行诊断**（可选）

```bash
python diagnose.py
```

5. **启动应用**

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，访问地址：`http://localhost:8501`

## 📁 项目结构

```
Simple-Rag/
├── app.py                  # Streamlit Web 应用主入口
├── run.py                  # 启动脚本（包含环境检查）
├── diagnose.py             # 诊断工具
├── requirements.txt        # Python 依赖
├── environment.yml         # Conda 环境配置
├── .env.example           # 环境变量示例
├── README.md              # 项目说明
│
├── src/                   # 核心代码
│   ├── config.py         # 配置管理
│   ├── ingest.py         # 数据摄取
│   ├── retriever.py      # 检索管理
│   └── rag.py            # RAG 系统核心
│
├── data/                  # 数据目录
│   ├── papers/           # 论文文档
│   ├── tools_docs/       # 工具文档
│   ├── project_docs/     # 项目文档
│   ├── examples/         # 示例代码
│   └── chroma/           # 向量数据库（运行时生成）
│
└── tests/                 # 测试代码
```

## 🔧 核心模块

### 1. 数据摄取（ingest.py）

负责加载和处理各类文档：

- 支持 `.txt`、`.md`、`.py` 等格式
- 文档分块和向量化
- 元数据管理

### 2. 检索管理（retriever.py）

提供多种检索策略：

- **通用检索**：基于语义相似度
- **分类检索**：按文档类型过滤
- **混合检索**：结合多种策略

### 3. RAG 系统（rag.py）

核心问答逻辑：

- 智能问答
- 代码分析
- 工具比较
- 概念解释

### 4. Web 界面（app.py）

用户交互界面：

- 对话界面
- 系统状态监控
- 数据管理
- 配置说明

## 💡 使用指南

### 基本使用

1. **首次使用**：点击左侧"重新摄取数据"按钮加载知识库
2. **提问**：在输入框中输入问题，点击"发送"
3. **查看来源**：展开"参考来源"查看答案依据
4. **示例问题**：使用左侧快速操作中的示例问题

### 高级功能

#### 代码分析

```
输入：请分析以下代码是否存在克隆...
[代码片段]
```

#### 工具比较

```
输入：比较 NiCad 和 CCFinder 工具
```

#### 概念解释

```
输入：解释什么是 AST 方法
```

## 📊 技术架构

```
┌─────────────────────────────────────────┐
│         Streamlit Web UI                 │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         RAG System (rag.py)              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Q&A      │  │ Analysis │  │Compare ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    Retriever Manager (retriever.py)      │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ General  │  │ By Type  │  │ Hybrid ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      ChromaDB Vector Store               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Knowledge Base                   │
│  Papers | Tools | Projects | Examples   │
└─────────────────────────────────────────┘
```

## 🧪 测试

运行诊断工具检查环境：

```bash
python diagnose.py
```

输出示例：

```
[诊断] 开始诊断 Simple-RAG 运行环境

============================================================
1. 检查Python版本
============================================================
✅ Python版本符合要求 (>= 3.8)

============================================================
2. 检查必要的Python包
============================================================
✅ streamlit 已安装
✅ langchain 已安装
✅ chromadb 已安装
...

[成功] 所有检查通过！
```

## 📚 知识库内容

### 论文文档
- 代码克隆检测基础知识
- 检测方法综述

### 工具文档
- CCFinder 使用指南
- NiCad 使用指南

### 项目文档
- 系统设计概述

### 示例代码
- Type-1/2/3 克隆示例

## 🛠️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 必填 |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `CHROMA_PERSIST_DIRECTORY` | 向量数据库路径 | `./data/chroma` |
| `CHUNK_SIZE` | 文档分块大小 | `1000` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `200` |
| `TOP_K_RETRIEVAL` | 检索文档数量 | `5` |

### 模型配置

在 `src/rag.py` 中可以修改：

```python
self.llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # 可改为 gpt-4
    temperature=0.1,         # 控制随机性
    ...
)
```

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 更新日志

### v1.0.0 (2026-01-10)

- ✨ 初始版本发布
- ✅ 实现基础 RAG 问答功能
- ✅ 支持多种检索策略
- ✅ 提供 Web 交互界面
- ✅ 添加诊断工具

## 🔮 未来计划

- [ ] 支持更多文档格式（PDF、Word）
- [ ] 增加代码文件上传功能
- [ ] 集成实际克隆检测工具
- [ ] 提供可视化分析结果
- [ ] 支持多语言（英文）
- [ ] 部署在线 Demo

## ❓ 常见问题

### Q: 为什么要用 `streamlit run` 而不是 `python app.py`？

A: Streamlit 应用必须通过 `streamlit run` 命令启动，直接用 `python` 运行会报错。

### Q: 如何更换 LLM 模型？

A: 在 `src/rag.py` 中修改 `ChatOpenAI` 的 `model` 参数。

### Q: 数据摄取失败怎么办？

A: 运行 `python diagnose.py` 检查环境，确保所有依赖已安装且 API Key 正确。

### Q: 如何添加自己的文档？

A: 将文档放入 `data/` 对应的子目录，然后点击"重新摄取数据"。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://www.langchain.com/) - RAG 框架
- [Streamlit](https://streamlit.io/) - Web 框架
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [OpenAI](https://openai.com/) - LLM 服务

## 📧 联系方式

- **作者**：[你的姓名]
- **邮箱**：your.email@example.com
- **GitHub**：[@你的用户名](https://github.com/你的用户名)

---

⭐ 如果这个项目对你有帮助，请给个 Star！
