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
        st.subheader("系统信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 数据统计")
            if st.session_state.data_ingested:
                st.success("✅ 数据已摄取")
                # 这里可以添加更详细的统计信息
            else:
                st.warning("⚠️ 数据未摄取")
        
        with col2:
            st.markdown("### 🔧 系统配置")
            st.json({
                "Chunk Size": Config.CHUNK_SIZE,
                "Chunk Overlap": Config.CHUNK_OVERLAP,
                "Top K Retrieval": Config.TOP_K_RETRIEVAL,
                "Vector DB": Config.CHROMA_PERSIST_DIRECTORY
            })
        
        # 清除对话历史
        if st.button("清除对话历史"):
            st.session_state.messages = []
            st.rerun()
    
    with tab3:
        st.subheader("配置说明")
        
        st.markdown("""
        ### 📋 使用说明
        
        1. **首次使用**：
           - 在 `.env` 文件中配置您的 OpenAI API Key
           - 点击"重新摄取数据"按钮加载数据
        
        2. **提问技巧**：
           - 可以询问克隆检测的基本概念
           - 可以比较不同的检测工具
           - 可以询问检测方法和技术细节
           - 可以提供代码片段进行分析
        
        3. **数据来源**：
           - 经典论文摘要
           - 工具文档
           - 项目文档
           - 示例代码
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
        st.markdown("### 🔑 环境变量配置")
        st.code("""
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
        """)

if __name__ == "__main__":
    main()
