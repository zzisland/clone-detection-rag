import streamlit as st
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from rag import CloneDetectionRAG
from ingest import DataIngestor
from config import Config

# 页面配置
st.set_page_config(
    page_title="代码克隆检测RAG助手",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .source-badge {
        background-color: #e3f2fd;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin: 0.125rem;
        display: inline-block;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """初始化会话状态"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'data_ingested' not in st.session_state:
        st.session_state.data_ingested = False

def load_rag_system():
    """加载RAG系统"""
    if st.session_state.rag_system is None:
        with st.spinner("正在加载RAG系统..."):
            try:
                st.session_state.rag_system = CloneDetectionRAG()
                st.success("RAG系统加载成功！")
                return True
            except Exception as e:
                st.error(f"加载RAG系统失败: {e}")
                return False
    return True

def sidebar():
    """侧边栏"""
    st.sidebar.title("🔧 控制面板")
    
    # 数据摄取部分
    st.sidebar.subheader("📚 数据管理")
    
    if st.sidebar.button("重新摄取数据", type="primary"):
        with st.spinner("正在摄取数据..."):
            try:
                ingestor = DataIngestor()
                vector_store = ingestor.ingest_all_data()
                if vector_store:
                    st.session_state.data_ingested = True
                    st.sidebar.success("数据摄取完成！")
                else:
                    st.sidebar.error("数据摄取失败！")
            except Exception as e:
                st.sidebar.error(f"数据摄取出错: {e}")
    
    # 系统状态
    st.sidebar.subheader("📊 系统状态")
    if st.session_state.rag_system:
        st.sidebar.success("✅ RAG系统已加载")
    else:
        st.sidebar.warning("⚠️ RAG系统未加载")
    
    if st.session_state.data_ingested:
        st.sidebar.success("✅ 数据已摄取")
    else:
        st.sidebar.warning("⚠️ 数据未摄取")
    
    # 快速操作
    st.sidebar.subheader("🚀 快速操作")
    
    sample_questions = [
        "什么是代码克隆检测？",
        "Type-1、Type-2、Type-3克隆的区别",
        "AST和Token方法的比较",
        "如何评估克隆检测工具？",
        "NiCad工具的使用方法"
    ]
    
    selected_question = st.sidebar.selectbox(
        "选择示例问题：",
        sample_questions,
        index=0
    )
    
    if st.sidebar.button("使用示例问题"):
        st.session_state.example_question = selected_question

def chat_interface():
    """聊天界面"""
    st.markdown('<h1 class="main-header">🔍 代码克隆检测RAG助手</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">专业的代码克隆检测知识助手</p>', unsafe_allow_html=True)
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # 助手回复
                    st.markdown(message["content"])
                    
                    # 显示额外信息
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 参考来源"):
                            for source in message["sources"]:
                                st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
                    
                    if "confidence" in message:
                        confidence_class = f"confidence-{message['confidence']}"
                        st.markdown(f'<p class="{confidence_class}">置信度: {message["confidence"]}</p>', unsafe_allow_html=True)

def main():
    """主函数"""
    initialize_session_state()
    
    # 侧边栏
    sidebar()
    
    # 主界面
    tab1, tab2, tab3 = st.tabs(["💬 对话", "📊 系统信息", "🔧 配置"])
    
    with tab1:
        chat_interface()
        
        # 聊天输入区域（使用text_input替代chat_input）
        st.markdown("---")
        
        # 处理示例问题
        example_question = st.session_state.get("example_question", None)
        if example_question:
            del st.session_state.example_question
        
        # 输入区域
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # 如果有示例问题，使用它作为默认值
            default_value = example_question if example_question else ""
            user_input = st.text_input("请输入您的问题...", value=default_value, placeholder="例如：什么是代码克隆检测？")
        
        with col2:
            send_button = st.button("发送", type="primary")
        
        # 处理用户输入（按钮点击）
        if send_button and user_input and user_input.strip():
            # 确保RAG系统已加载
            if not load_rag_system():
                st.error("无法加载RAG系统，请检查配置。")
            else:
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # 生成助手回复
                with st.spinner("正在思考..."):
                    try:
                        result = st.session_state.rag_system.get_chat_response(user_input)
                        
                        # 添加到消息历史
                        assistant_message = {
                            "role": "assistant",
                            "content": result.get("answer", "抱歉，无法生成回答。"),
                            "sources": result.get("sources", []),
                            "confidence": result.get("confidence", "medium")
                        }
                        st.session_state.messages.append(assistant_message)
                        
                    except Exception as e:
                        st.error(f"生成回答时出错: {e}")
                
                # 重新运行以显示新消息
                st.rerun()
    
    with tab2:
        st.subheader("📊 系统信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 数据统计")
            if st.session_state.data_ingested:
                st.success("✅ 数据已摄取")
            else:
                st.warning("⚠️ 数据未摄取，请点击左侧'重新摄取数据'")
            
            st.markdown("### 💬 对话统计")
            st.info(f"对话轮数: {len(st.session_state.messages) // 2}")
        
        with col2:
            st.markdown("### 🤖 模型配置")
            st.json({
                "LLM 模型": "Qwen2.5-Coder-1.5B",
                "Embedding 模型": "BAAI/bge-small-zh-v1.5",
                "运行设备": "GPU/CPU 自动检测",
                "镜像加速": "已启用"
            })
            
            st.markdown("### 🔧 系统配置")
            st.json({
                "Chunk Size": Config.CHUNK_SIZE,
                "Chunk Overlap": Config.CHUNK_OVERLAP,
                "Top K Retrieval": Config.TOP_K_RETRIEVAL,
                "Vector DB": Config.CHROMA_PERSIST_DIRECTORY
            })
        
        st.markdown("---")
        
        # 操作按钮
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ 清除对话历史", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 重新加载系统", use_container_width=True):
                st.session_state.rag_system = None
                st.rerun()
        
        with col3:
            if st.button("📊 查看缓存", use_container_width=True):
                st.info("运行 `python clear_cache.py` 查看和管理模型缓存")
    
    with tab3:
        st.subheader("配置说明")
        
        st.markdown("""
        ### 📋 使用说明
        
        1. **首次使用**：
           - ✅ 无需 API Key，完全本地化运行
           - 首次启动会自动下载模型（约 3GB，使用国内镜像加速）
           - 点击左侧"重新摄取数据"按钮加载知识库
           - 等待模型加载完成（约 10-15 秒）
        
        2. **模型配置**：
           - **LLM 模型**: Qwen2.5-Coder-1.5B-Instruct（代码专用）
           - **Embedding 模型**: BAAI/bge-small-zh-v1.5（中文向量）
           - **运行设备**: 自动检测 GPU/CPU
           - **下载加速**: 已配置 HuggingFace 国内镜像
        
        3. **提问技巧**：
           - 询问克隆检测的基本概念（如：什么是 Type-1 克隆？）
           - 比较不同的检测工具（如：比较 NiCad 和 CCFinder）
           - 询问检测方法和技术细节（如：AST 方法的原理）
           - 提供代码片段进行分析
        
        4. **数据来源**：
           - 📄 经典论文摘要
           - 🔧 工具文档（NiCad、CCFinder 等）
           - 📚 项目文档
           - 💻 示例代码（Type-1/2/3 克隆）
        
        5. **性能优化**：
           - 使用 GPU 可大幅提升速度（自动检测）
           - 模型缓存后启动速度很快
           - 如需更强性能，可在代码中切换为 7B 模型
        """)
        
        st.markdown("""
        ### 🗂️ 数据目录结构
        
        ```
        data/
        ├── papers/          # 论文文档
        ├── tools_docs/      # 工具文档
        ├── project_docs/    # 项目文档
        └── examples/        # 示例代码
        ```
        """)
        
        # 显示环境变量配置
        st.markdown("### 🔑 环境变量配置（可选）")
        st.code("""
# 向量数据库配置
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5

# HuggingFace 镜像加速（已内置）
HF_ENDPOINT=https://hf-mirror.com
        """)
        
        st.markdown("### 💡 常见问题")
        st.markdown("""
        **Q: 首次启动很慢？**  
        A: 首次需要下载模型（约 3GB），已配置国内镜像，预计 5-10 分钟。后续启动很快。
        
        **Q: 如何清除模型缓存？**  
        A: 运行 `python clear_cache.py` 清理工具。
        
        **Q: 显存不足怎么办？**  
        A: 项目默认使用 1.5B 模型（约 3GB 显存），如仍不足可切换到 CPU 模式。
        
        **Q: 如何切换更强的模型？**  
        A: 修改 `src/rag.py` 第 36 行，将模型改为 `Qwen2.5-Coder-7B-Instruct`。
        
        **Q: 向量数据库维度不匹配？**  
        A: 删除 `data/chroma` 文件夹，重新摄取数据即可。
        
        详细说明请查看 [INSTALL.md](https://github.com/你的用户名/clone-detection-rag/blob/main/INSTALL.md)
        """)

if __name__ == "__main__":
    main()
