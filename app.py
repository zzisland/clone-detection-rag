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

# 自定义CSS - 优化加载速度
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        text-align: center;
    }
    .source-badge {
        background-color: #e3f2fd;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin: 0.125rem;
        display: inline-block;
    }
    .confidence-high { color: #4caf50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    
    /* 加速渲染 */
    .stApp { animation: none !important; }
    .element-container { animation: none !important; }
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
    if 'system_loading' not in st.session_state:
        st.session_state.system_loading = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_input' not in st.session_state:
        st.session_state.current_input = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "1.5B"
    if 'trigger_send' not in st.session_state:
        st.session_state.trigger_send = False

def load_rag_system(model_size="1.5B"):
    """加载RAG系统（延迟加载）"""
    if st.session_state.rag_system is None:
        # 创建加载界面
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            st.info(f"🚀 正在加载 {model_size} 模型，请稍候...")
            
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 步骤 1: 初始化
                status_text.text("⏳ [1/4] 初始化系统...")
                progress_bar.progress(10)
                
                # 步骤 2: 加载 Tokenizer
                model_info = {
                    "1.5B": "约10MB",
                    "7B": "约10MB"
                }
                status_text.text(f"⏳ [2/4] 加载 Tokenizer（首次需下载，{model_info.get(model_size, '约10MB')}）...")
                progress_bar.progress(25)
                
                # 步骤 3: 加载模型
                model_size_info = {
                    "1.5B": "约3GB",
                    "7B": "约14GB"
                }
                status_text.text(f"⏳ [3/4] 加载 LLM 模型（首次需下载，{model_size_info.get(model_size, '约3GB')}，使用镜像加速）...")
                progress_bar.progress(40)
                
                # 实际加载（传入模型大小参数）
                st.session_state.rag_system = CloneDetectionRAG(model_size=model_size)
                st.session_state.selected_model = model_size
                
                # 步骤 4: 完成
                progress_bar.progress(100)
                status_text.text("✅ [4/4] 系统加载完成！")
                
                # 清除加载界面
                import time
                time.sleep(1)
                loading_placeholder.empty()
                
                st.success(f"✅ RAG 系统已就绪！当前模型: {model_size}")
                return True
                
            except Exception as e:
                loading_placeholder.empty()
                st.error(f"❌ 加载失败: {e}")
                st.info("💡 提示：首次运行需要下载模型，请确保网络连接正常。")
                return False
    return True

def sidebar():
    """侧边栏 - 优化版"""
    st.sidebar.title("🔧 控制面板")
    
    # 系统状态 - 简化显示
    with st.sidebar.expander("📊 系统状态", expanded=True):
        if st.session_state.rag_system:
            st.success(f"✅ 模型: {st.session_state.selected_model}")
        else:
            st.warning("⚠️ 未加载")
        
        if st.session_state.data_ingested:
            st.success("✅ 数据已就绪")
        else:
            st.info("💡 需要摄取数据")
    
    # 数据管理
    with st.sidebar.expander("📚 数据管理"):
        if st.button("重新摄取数据", type="primary", use_container_width=True):
            with st.spinner("处理中..."):
                try:
                    ingestor = DataIngestor()
                    vector_store = ingestor.ingest_all_data()
                    if vector_store:
                        st.session_state.data_ingested = True
                        st.success("完成！")
                    else:
                        st.error("失败！")
                except Exception as e:
                    st.error(f"错误: {str(e)[:50]}...")
    
    # 快速操作 - 简化
    with st.sidebar.expander("🚀 示例问题"):
        questions = [
            "什么是代码克隆检测？",
            "Type-1/2/3克隆的区别",
            "AST和Token方法比较",
            "如何评估检测工具？"
        ]
        
        selected = st.selectbox("选择：", questions, label_visibility="collapsed")
        
        if st.button("使用此问题", use_container_width=True):
            st.session_state.current_input = selected
            st.session_state.trigger_send = True
            st.rerun()

def chat_interface():
    """聊天界面 - 优化版"""
    st.markdown('<h1 class="main-header">🔍 代码克隆检测RAG助手</h1>', unsafe_allow_html=True)
    
    # 检查系统状态
    if st.session_state.rag_system is None:
        st.markdown('<p class="sub-header">👋 欢迎！请选择模型开始使用</p>', unsafe_allow_html=True)
        
        # 模型选择 - 简化版
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("### 🚀 轻量版")
                st.markdown("**1.5B 模型**")
                st.caption("✅ 快速 | 3GB显存 | 推荐")
                if st.button("选择", key="1.5b", type="primary", use_container_width=True):
                    load_rag_system("1.5B")
                    st.rerun()
        
        with col2:
            with st.container():
                st.markdown("### 💪 专业版")
                st.markdown("**7B 模型**")
                st.caption("✅ 高性能 | 14GB显存")
                if st.button("选择", key="7b", use_container_width=True):
                    load_rag_system("7B")
                    st.rerun()
        
        st.info("💡 首次使用需下载模型，已配置国内镜像加速")
        return
    
    # 显示聊天历史 - 优化渲染
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                # 简化来源显示
                if message.get("sources"):
                    with st.expander("📚 来源", expanded=False):
                        for src in message["sources"][:3]:  # 只显示前3个
                            st.caption(src.split("/")[-1])  # 只显示文件名

def main():
    """主函数"""
    initialize_session_state()
    
    # 侧边栏
    sidebar()
    
    # 主界面
    tab1, tab2, tab3 = st.tabs(["💬 对话", "📊 系统信息", "🔧 配置"])
    
    with tab1:
        chat_interface()
        
        # 聊天输入区域
        st.markdown("---")
        
        # 处理示例问题触发
        trigger_input = None
        if st.session_state.get("trigger_send", False):
            trigger_input = st.session_state.get("current_input", "")
            st.session_state.trigger_send = False
            st.session_state.current_input = ""
        
        # 输入区域
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # 不使用 session_state 作为 value，避免 setIn 错误
                user_input = st.text_input(
                    "请输入您的问题...", 
                    placeholder="例如：什么是代码克隆检测？按回车发送",
                    key="user_input_field",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button(
                    "发送 ✉️", 
                    type="primary",
                    use_container_width=True
                )
        
        # 如果有触发的输入，使用它
        if trigger_input:
            user_input = trigger_input
            send_button = True
        
        # 处理用户输入
        if send_button and user_input and user_input.strip():
            # 检查系统是否已加载
            if st.session_state.rag_system is None:
                st.warning("⚠️ 请先选择并初始化模型！点击上方的模型选择按钮开始。")
            else:
                # 设置处理状态
                st.session_state.processing = True
                st.session_state.current_input = ""  # 清空输入
                
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # 创建进度显示
                progress_placeholder = st.empty()
                
                try:
                    with progress_placeholder.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.info("🤔 AI 正在思考中...")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 步骤 1: 检索
                            import time
                            status_text.text("📚 [1/3] 检索相关文档...")
                            progress_bar.progress(20)
                            time.sleep(0.2)
                            
                            progress_bar.progress(40)
                            
                            # 步骤 2: 生成
                            status_text.text("💭 [2/3] 生成回答中...")
                            progress_bar.progress(50)
                            
                            # 实际生成回答
                            result = st.session_state.rag_system.get_chat_response(user_input)
                            
                            progress_bar.progress(90)
                            status_text.text("✨ [3/3] 完成！")
                            progress_bar.progress(100)
                            time.sleep(0.3)
                    
                    # 清除进度显示
                    progress_placeholder.empty()
                    
                    # 添加到消息历史
                    assistant_message = {
                        "role": "assistant",
                        "content": result.get("answer", "抱歉，无法生成回答。"),
                        "sources": result.get("sources", []),
                        "confidence": result.get("confidence", "medium")
                    }
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"❌ 生成回答时出错: {str(e)}")
                    st.info("💡 提示：如果问题持续，请尝试重新加载系统或选择其他模型。")
                    # 移除用户消息（因为失败了）
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                        st.session_state.messages.pop()
                
                finally:
                    # 重置处理状态（确保一定会执行）
                    st.session_state.processing = False
                
                # 重新运行以显示新消息
                st.rerun()
    
    with tab2:
        st.header("📊 系统信息")
        st.write("")  # 添加空行
        
        # 数据统计
        st.subheader("📚 数据统计")
        if st.session_state.data_ingested:
            st.success("✅ 数据已摄取")
        else:
            st.warning("⚠️ 数据未摄取，请点击左侧'重新摄取数据'")
        
        st.write("")
        
        # 对话统计
        st.subheader("💬 对话统计")
        st.info(f"对话轮数: {len(st.session_state.messages) // 2}")
        
        st.write("")
        st.divider()
        
        # 模型配置
        st.subheader("🤖 模型配置")
        model_name = "未加载"
        device_info = "未检测"
        
        if st.session_state.rag_system:
            model_name = f"Qwen2.5-Coder-{st.session_state.selected_model}"
            import torch
            device_info = "CPU 模式（RTX 5060 兼容性问题）"
        
        st.code(f"""
当前模型: {model_name}
Embedding 模型: BAAI/bge-small-zh-v1.5
运行设备: {device_info}
镜像加速: 已启用 (hf-mirror.com)
        """, language="text")
        
        st.write("")
        st.divider()
        
        # 系统配置
        st.subheader("🔧 系统配置")
        st.code(f"""
Chunk Size: {Config.CHUNK_SIZE}
Chunk Overlap: {Config.CHUNK_OVERLAP}
Top K Retrieval: {Config.TOP_K_RETRIEVAL}
Vector DB: {Config.CHROMA_PERSIST_DIRECTORY}
        """, language="text")
        
        st.write("")
        st.divider()
        
        # 操作按钮
        st.subheader("⚙️ 操作")
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
                st.info("💡 运行 `python clear_cache.py` 查看和管理模型缓存")
    
    with tab3:
        st.header("🔧 配置与帮助")
        st.write("")
        
        # 使用指南
        st.subheader("📖 快速开始")
        
        st.markdown("""
        ### 🚀 使用步骤
        
        **1️⃣ 选择模型**
        - 在"对话"标签页选择 1.5B（推荐）或 7B 模型
        - 首次使用需要下载，已配置国内镜像加速
        
        **2️⃣ 摄取数据**（可选）
        - 点击左侧边栏的"重新摄取数据"
        - 等待处理完成（约1-2分钟）
        
        **3️⃣ 开始提问**
        - 直接输入问题，按回车发送
        - 或使用左侧的示例问题快速开始
        """)
        
        st.divider()
        
        # 性能说明
        st.subheader("⚡ 性能说明")
        
        st.info("""
        **当前运行模式：CPU**
        
        由于 RTX 5060 是新显卡，当前 PyTorch 版本不支持，系统使用 CPU 模式运行。
        
        **预期响应时间：**
        - 文档检索：1-2秒
        - 生成回答：15-30秒（1.5B）/ 30-60秒（7B）
        - 总计：约20-35秒
        
        **优化建议：**
        - 使用 1.5B 模型（更快）
        - 问题尽量简洁明确
        - 等待 PyTorch 官方支持 RTX 50 系列后可切换回 GPU
        """)
        
        st.divider()
        
        # 提问技巧
        st.subheader("💡 提问技巧")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ 好的问题：**
            - "什么是 Type-1 克隆？"
            - "比较 NiCad 和 CCFinder"
            - "AST 方法的原理是什么？"
            - "如何评估检测工具的性能？"
            """)
        
        with col2:
            st.markdown("""
            **❌ 避免：**
            - 过于宽泛的问题
            - 多个问题混在一起
            - 与代码克隆检测无关的问题
            - 过长的代码片段
            """)
        
        st.divider()
        
        # 常见问题
        st.subheader("❓ 常见问题")
        
        with st.expander("Q1: 为什么使用 CPU 而不是 GPU？"):
            st.markdown("""
            **原因：** RTX 5060 是 2024/2025 年的新显卡，当前 PyTorch 版本不支持。
            
            **解决：** 等待 PyTorch 官方发布支持 RTX 50 系列的版本，或使用 PyTorch Nightly 版本（实验性）。
            """)
        
        with st.expander("Q2: 响应速度太慢怎么办？"):
            st.markdown("""
            **建议：**
            1. 使用 1.5B 模型（比 7B 快 2-3 倍）
            2. 问题尽量简洁
            3. 耐心等待（CPU 模式确实较慢）
            4. 考虑升级 PyTorch 以使用 GPU
            """)
        
        with st.expander("Q3: 如何重新摄取数据？"):
            st.markdown("""
            **步骤：**
            1. 点击左侧边栏的"数据管理"
            2. 点击"重新摄取数据"按钮
            3. 等待处理完成
            
            **注意：** 如果更改了 data 目录中的文档，需要重新摄取。
            """)
        
        with st.expander("Q4: 如何清除缓存？"):
            st.markdown("""
            **方法 1：** 在终端运行
            ```bash
            python clear_cache.py
            ```
            
            **方法 2：** 手动删除
            ```bash
            Remove-Item -Recurse -Force .\\src\\__pycache__
            ```
            """)
        
        st.divider()
        
        # 数据目录
        st.subheader("📁 数据目录结构")
        
        st.code("""
data/
├── papers/          # 论文文档（PDF、TXT）
├── tools_docs/      # 工具文档
├── project_docs/    # 项目文档
├── examples/        # 示例代码
└── chroma/          # 向量数据库（自动生成）
        """, language="text")
        
        st.divider()
        
        # 系统要求
        st.subheader("💻 系统要求")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **最低配置：**
            - CPU: 4核心
            - 内存: 8GB
            - 硬盘: 10GB 可用空间
            - Python: 3.8+
            """)
        
        with col2:
            st.markdown("""
            **推荐配置：**
            - CPU: 8核心+
            - 内存: 16GB+
            - GPU: 8GB+ 显存（支持的显卡）
            - 硬盘: 20GB+ 可用空间
            """)
        
        st.divider()
        
        # 联系方式
        st.subheader("📞 获取帮助")
        
        st.info("""
        如果遇到问题，可以：
        1. 查看项目 README.md 文档
        2. 查看 USAGE.md 详细使用指南
        3. 提交 GitHub Issue
        4. 联系项目维护者
        """)
        
        st.success("✨ 祝您使用愉快！")

if __name__ == "__main__":
    main()
